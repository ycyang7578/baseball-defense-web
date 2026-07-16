"""實驗：評價用難度模型改用可解釋的 spline GLM（取代 GBM）。

動機（2026-07-12，使用者決定）：不用無法說明的模型，接受準確度損失換可解釋性。
難度模型不搬野手、無反事實需求，故內生性禁令不適用——spray 合法、spline/交互
可自由使用。

特徵工程（對齊 Melville §2 的鏡像處理）：
- spray_rel：左打 spray 翻號，讓「拉打」方向全聯盟對齊（負=拉、正=推）
- splines：spray_rel（8 節點，要容納四守位的 lane 結構）、LA/EV/hp（6 節點）
- 交互：spray×EV（強襲穿洞）、spray×hp（慢滾內野安打的方向性）

配置：
  GBM        現行 HistGradientBoosting（對照）
  D1         主效應 splines + stand_R
  D2         D1 + spray×EV
  D3         D2 + spray×hp
輸出：2025 樣本外 AUC/Brier/校準 + 最佳 GLM 的官方 OAA 相關（qualified R、
分位置、scale），供拍板是否換入生產。

執行：python scripts/experiments/exp_if_difficulty_glm.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DSN
from src.if_dataset import build_gb_dataset
from src.if_eval import aggregate_players, score_test_year
from src.if_model import (DIFFICULTY_FEATURES, make_difficulty_gbm,
                          make_difficulty_glm)

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025


def calibration_max_dev(y, p, bins: int = 10) -> float:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    return float((g["pred"] - g["obs"]).abs().max())


def official_correlations(test: pd.DataFrame, label: str) -> None:
    model = aggregate_players(test)
    with psycopg2.connect(DSN) as conn:
        official = pd.read_sql(
            "SELECT player_id, oaa, n_opp, is_qualified FROM if_oaa_leaderboard "
            "WHERE year = %(y)s", conn, params={"y": TEST_YEAR})
    merged = model.merge(official, left_on="resp_fielder", right_on="player_id")
    q = merged[merged["is_qualified"] == True]  # noqa: E712
    r = np.corrcoef(q["model_oaa"], q["oaa"])[0, 1]
    rho = q["model_oaa"].corr(q["oaa"], method="spearman")
    r_rate = np.corrcoef(q["model_oaa"] / q["n_balls"], q["oaa"] / q["n_opp"])[0, 1]
    print(f"\n[{label}] qualified n={len(q)}: R={r:.3f} Spearman={rho:.3f} "
          f"rate R={r_rate:.3f}  scale SD {q['model_oaa'].std():.1f} vs "
          f"official {q['oaa'].std():.1f}")
    for pos in ("1B", "2B", "3B", "SS"):
        sub = merged[(merged["resp_pos"] == pos) & (merged["n_balls"] >= 100)]
        if len(sub) >= 3:
            rp = np.corrcoef(sub["model_oaa"], sub["oaa"])[0, 1]
            print(f"  {pos}: n={len(sub):>3} R={rp:.3f}")


def main() -> None:
    train = build_gb_dataset(TRAIN_YEARS)
    test = build_gb_dataset([TEST_YEAR])
    y_tr, y_te = train["is_out"].to_numpy(), test["is_out"].to_numpy()
    print(f"full-population GB: n_train={len(train):,} n_test={len(test):,}")

    configs = {
        "GBM (現行)": make_difficulty_gbm,
        "D1 splines": lambda: make_difficulty_glm(spray_ev=False, spray_hp=False),
        "D2 +sprayxEV": lambda: make_difficulty_glm(spray_ev=True, spray_hp=False),
        "D3 +sprayxhp": lambda: make_difficulty_glm(spray_ev=True, spray_hp=True),
    }
    rows = []
    for name, factory in configs.items():
        m = factory()
        m.fit(train[DIFFICULTY_FEATURES], y_tr)
        p = m.predict_proba(test[DIFFICULTY_FEATURES])[:, 1]
        rows.append({"config": name,
                     "auc": round(roc_auc_score(y_te, p), 4),
                     "brier": round(brier_score_loss(y_te, p), 4),
                     "cal_max_dev": round(calibration_max_dev(y_te, p), 4)})
    print("\n2025 out-of-sample (difficulty model):")
    print(pd.DataFrame(rows).to_string(index=False))

    # 官方 OAA 相關：GBM vs 最佳 GLM（D3）
    scored_gbm = score_test_year(TRAIN_YEARS, TEST_YEAR)
    official_correlations(scored_gbm, "GBM")
    scored_glm = score_test_year(
        TRAIN_YEARS, TEST_YEAR,
        model_factory=lambda: make_difficulty_glm(spray_ev=True, spray_hp=True))
    official_correlations(scored_glm, "GLM D3")


if __name__ == "__main__":
    main()
