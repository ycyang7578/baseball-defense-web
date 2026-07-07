"""階段 5：內野站位建議的跨年驗證（2023–24 → 2025）。

對每位合格打者：
- 跨年增益 = E[outs | 用 2023–24 球優化的站位, 2025 球] − E[outs | 聯盟平均, 2025 球]
- 同年上限 = E[outs | 用 2025 球優化的站位, 2025 球] − E[outs | 聯盟平均, 2025 球]
兩者都評估在 2025 資料上，可直接比較（同年增益若評在自身訓練資料上會高估）。
保留率 = 跨年增益 / 同年上限。跨打者對「跨年增益 > 0」做單樣本 t 檢定。

注意：反事實站位下的實際出局結果不可觀測，這裡的成效是模型評分
（optimizer GLM，2023–24 訓練，已在 2025 樣本外驗證 AUC 0.754/校準偏差 0.029）。
跨年設計檢驗的是「打者傾向與模型可遷移性」——站位建議隔年是否仍有效。

執行：python scripts/validate_if_positioning.py [min_train_gb] [min_test_gb]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN
from src.if_optimize import (expected_outs, fetch_batter_gbs,
                             league_average_positions, optimize_infield,
                             positions_to_params)

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "if_gb"
OUT_PATH = _MODEL_DIR / "validation_2025.json"
ROWS_PATH = _MODEL_DIR / "validation_rows_2025.csv"   # checkpoint：逐打者落盤，可續跑
MODEL = _MODEL_DIR / "if_gb_optimizer_glm.joblib"


def qualifying_batters(min_train: int, min_test: int) -> pd.DataFrame:
    sql = """
        SELECT batter, max(stand) AS stand,
               count(*) FILTER (WHERE game_year = ANY(%(tr)s)) AS n_train,
               count(*) FILTER (WHERE game_year = %(te)s) AS n_test
        FROM statcast
        WHERE bb_type = 'ground_ball' AND hc_x IS NOT NULL
          AND game_year = ANY(%(all)s)
        GROUP BY batter
        HAVING count(*) FILTER (WHERE game_year = ANY(%(tr)s)) >= %(mtr)s
           AND count(*) FILTER (WHERE game_year = %(te)s) >= %(mte)s
        ORDER BY batter
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn, params={
            "tr": TRAIN_YEARS, "te": TEST_YEAR, "all": TRAIN_YEARS + [TEST_YEAR],
            "mtr": min_train, "mte": min_test})


def main(min_train: int = 150, min_test: int = 80) -> None:
    model = joblib.load(MODEL)
    avg_angles, avg_depths = league_average_positions(TRAIN_YEARS)
    avg_start = positions_to_params(avg_angles, avg_depths)

    batters = qualifying_batters(min_train, min_test)
    print(f"合格打者（train GB>={min_train}, test GB>={min_test}）: {len(batters)} 位")

    # checkpoint：已算過的打者直接沿用（長批次在這台機器上要能從當機中續跑）
    done: set[int] = set()
    if ROWS_PATH.exists():
        done = set(pd.read_csv(ROWS_PATH)["batter"])
        print(f"  checkpoint: 已有 {len(done)} 位，續跑其餘")

    for i, row in enumerate(batters.itertuples(), 1):
        if row.batter in done:
            continue
        train_balls = fetch_batter_gbs(row.batter, TRAIN_YEARS)
        test_balls = fetch_batter_gbs(row.batter, [TEST_YEAR])
        base = expected_outs(model, test_balls, avg_angles, avg_depths)

        opt_cross = optimize_infield(train_balls, model, n_restarts=16, seed=42,
                                     extra_starts=[avg_start])
        e_cross = expected_outs(model, test_balls, opt_cross["angles"], opt_cross["depths"])

        opt_same = optimize_infield(test_balls, model, n_restarts=16, seed=42,
                                    extra_starts=[avg_start])
        e_same = expected_outs(model, test_balls, opt_same["angles"], opt_same["depths"])

        pd.DataFrame([{"batter": row.batter, "stand": row.stand,
                       "n_train": row.n_train, "n_test": row.n_test,
                       "gain_cross": e_cross - base, "gain_same": e_same - base}]).to_csv(
            ROWS_PATH, mode="a", header=not ROWS_PATH.exists(), index=False)
        if i % 20 == 0:
            print(f"  ...{i}/{len(batters)}")

    df = pd.read_csv(ROWS_PATH)
    df = df[df["batter"].isin(set(batters["batter"]))]
    t, p = stats.ttest_1samp(df["gain_cross"], 0.0)
    retention = df["gain_cross"].sum() / df["gain_same"].sum()

    print(f"\n=== 跨年驗證結果（n={len(df)} 位打者，皆評估於 2025 球）===")
    print(f"跨年增益: 平均 {df['gain_cross'].mean():+.4f}"
          f"（每 450 GB 約 {df['gain_cross'].mean() * 450:+.1f} 出局）, "
          f"正增益比例 {(df['gain_cross'] > 0).mean():.1%}")
    print(f"同年上限: 平均 {df['gain_same'].mean():+.4f}")
    print(f"保留率（Σ跨年/Σ同年）: {retention:.1%}")
    print(f"t 檢定（跨年增益>0）: t={t:.2f}, p={p:.2e}")
    for s in ("L", "R"):
        sub = df[df["stand"] == s]
        if len(sub):
            print(f"  {s} 打（n={len(sub)}）: 跨年 {sub['gain_cross'].mean():+.4f}, "
                  f"同年 {sub['gain_same'].mean():+.4f}")

    summary = {
        "n_batters": len(df), "min_train_gb": min_train, "min_test_gb": min_test,
        "mean_gain_cross": round(df["gain_cross"].mean(), 5),
        "mean_gain_same": round(df["gain_same"].mean(), 5),
        "retention": round(retention, 4),
        "share_positive": round((df["gain_cross"] > 0).mean(), 4),
        "t_stat": round(t, 3), "p_value": float(f"{p:.3e}"),
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
