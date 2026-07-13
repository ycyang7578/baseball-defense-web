"""跨年驗證升級為失分口徑（內外野整合的統一計價，2023–24 → 2025）。

設計同 validate_if_positioning.py（同一批合格打者、聯盟基準=2023–24 平均站位、
全部評估在 2025 球上），但成效以期望失分 E[ΔRE] 計價（無人在壘 0 出局，
權重見 src/if_runvalue.py）。對每位打者：
- 跨年增益(outs 目標) = E[ΔRE | 聯盟平均] − E[ΔRE | 2023–24 出局率目標站位]
  ——生產 cascade 部署的就是這組站位，這是網站實際提供建議的失分口徑成效
- 跨年增益(run 目標)  = 同上，但站位用 run-value 目標優化（整合端點的精修目標）
- 同年上限 = E[ΔRE | 聯盟平均] − E[ΔRE | 2025 球 run-value 目標站位]
  （同年增益評估在驗證年資料上，才能與跨年增益直接比較）
保留率 = Σ跨年 / Σ同年。跨打者對「跨年增益 > 0」做單樣本 t 檢定。

逐打者 checkpoint（這台機器會 BSOD），中斷後重跑同指令續算。

執行：python scripts/validate_if_runvalue.py [min_train_gb] [min_test_gb]
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
from src.if_runvalue import runvalue_ball_weights
from src.re24 import load_re24

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
STATE = (0, 0, 0, 0)          # 無人在壘、0 出局（同 cascade 主範圍）
BASE = Path(__file__).resolve().parent.parent
_MODEL_DIR = BASE / "models" / "if_gb"
OUT_PATH = _MODEL_DIR / "validation_runvalue_2025.json"
ROWS_PATH = _MODEL_DIR / "validation_runvalue_rows_2025.csv"
MODEL = _MODEL_DIR / "bayes" / "if_bayes_group_pipeline.joblib"
XB_MODEL = _MODEL_DIR / "if_gb_xb_model.joblib"
PRE_DIR = BASE / "data" / "precomputed"


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
    xb = joblib.load(XB_MODEL)
    re24, delta_re = load_re24(PRE_DIR)
    avg_angles, avg_depths = league_average_positions(TRAIN_YEARS)
    avg_start = positions_to_params(avg_angles, avg_depths)

    batters = qualifying_batters(min_train, min_test)
    print(f"合格打者（train GB>={min_train}, test GB>={min_test}）: {len(batters)} 位")

    done: set[int] = set()
    if ROWS_PATH.exists():
        done = set(pd.read_csv(ROWS_PATH)["batter"])
        print(f"  checkpoint: 已有 {len(done)} 位，續跑其餘")

    for i, row in enumerate(batters.itertuples(), 1):
        if row.batter in done:
            continue
        tr = fetch_batter_gbs(row.batter, TRAIN_YEARS)
        te = fetch_batter_gbs(row.batter, [TEST_YEAR])
        bw_tr, _ = runvalue_ball_weights(tr, xb, re24, delta_re, STATE)
        bw_te, mean_w_te = runvalue_ball_weights(te, xb, re24, delta_re, STATE)

        def dre_2025(angles, depths) -> float:
            return mean_w_te - expected_outs(model, te, angles, depths,
                                             ball_weights=bw_te)

        dre_league = dre_2025(avg_angles, avg_depths)

        opt_outs = optimize_infield(tr, model, n_restarts=16, seed=42,
                                    extra_starts=[avg_start])
        opt_runs = optimize_infield(tr, model, n_restarts=16, seed=42,
                                    extra_starts=[avg_start], ball_weights=bw_tr)
        opt_same = optimize_infield(te, model, n_restarts=16, seed=42,
                                    extra_starts=[avg_start], ball_weights=bw_te)

        pd.DataFrame([{
            "batter": row.batter, "stand": row.stand,
            "n_train": row.n_train, "n_test": row.n_test,
            "dre_league": dre_league,
            "gain_cross_outs": dre_league - dre_2025(opt_outs["angles"], opt_outs["depths"]),
            "gain_cross_runs": dre_league - dre_2025(opt_runs["angles"], opt_runs["depths"]),
            "gain_same": dre_league - dre_2025(opt_same["angles"], opt_same["depths"]),
        }]).to_csv(ROWS_PATH, mode="a", header=not ROWS_PATH.exists(), index=False)
        if i % 20 == 0 or i == len(batters):
            print(f"  ...{i}/{len(batters)}", flush=True)

    df = pd.read_csv(ROWS_PATH)
    df = df[df["batter"].isin(set(batters["batter"]))]
    t, p = stats.ttest_1samp(df["gain_cross_outs"], 0.0)
    retention = df["gain_cross_outs"].sum() / df["gain_same"].sum()

    print(f"\n=== 跨年驗證・失分口徑（n={len(df)} 位打者，皆評估於 2025 球）===")
    print(f"跨年增益(outs 目標站位): 平均 {df['gain_cross_outs'].mean():+.5f} 分/GB"
          f"（每 450 GB 約 {df['gain_cross_outs'].mean() * 450:+.2f} 分）, "
          f"正增益比例 {(df['gain_cross_outs'] > 0).mean():.1%}")
    print(f"跨年增益(run 目標站位):  平均 {df['gain_cross_runs'].mean():+.5f} 分/GB"
          f"（每 450 GB 約 {df['gain_cross_runs'].mean() * 450:+.2f} 分）")
    print(f"同年上限(run 目標): 平均 {df['gain_same'].mean():+.5f} 分/GB")
    print(f"保留率（Σ跨年(outs)/Σ同年）: {retention:.1%}")
    print(f"t 檢定（跨年增益(outs)>0）: t={t:.2f}, p={p:.2e}")
    for s in ("L", "R"):
        sub = df[df["stand"] == s]
        if len(sub):
            print(f"  {s} 打（n={len(sub)}）: 跨年(outs) {sub['gain_cross_outs'].mean():+.5f}, "
                  f"同年 {sub['gain_same'].mean():+.5f}")

    summary = {
        "n_batters": len(df), "min_train_gb": min_train, "min_test_gb": min_test,
        "state": list(STATE),
        "mean_gain_cross_outs": round(df["gain_cross_outs"].mean(), 6),
        "mean_gain_cross_runs": round(df["gain_cross_runs"].mean(), 6),
        "mean_gain_same": round(df["gain_same"].mean(), 6),
        "runs_per_450gb_cross_outs": round(df["gain_cross_outs"].mean() * 450, 3),
        "retention": round(retention, 4),
        "share_positive": round((df["gain_cross_outs"] > 0).mean(), 4),
        "t_stat": round(t, 3), "p_value": float(f"{p:.3e}"),
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
