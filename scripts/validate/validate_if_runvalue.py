"""Cross-year validation upgraded to run-value denomination (unified pricing integrating
infield and outfield, 2023-24 -> 2025).

Same design as validate_if_positioning.py (same batch of qualifying batters, league
baseline = 2023-24 average positioning, everything evaluated on 2025 balls), but the
effect is denominated in expected runs E[ΔRE] (bases empty, 0 outs; weights are in
src/if_runvalue.py). For each batter:
- Cross-year gain (outs objective) = E[ΔRE | league average] - E[ΔRE | 2023-24
  out-rate-objective positioning] — this is exactly the positioning the production
  cascade deploys, i.e. the run-value-denominated effect of what the site actually
  recommends
- Cross-year gain (run objective) = same as above, but positioning is optimized against
  the run-value objective (the refined objective for the integration endpoint)
- Same-year ceiling = E[ΔRE | league average] - E[ΔRE | 2025-ball run-value-objective
  positioning] (the same-year gain is evaluated on the validation year's data so it's
  directly comparable to the cross-year gain)
Retention = Sum(cross-year) / Sum(same-year). A one-sample t-test is run across batters
against "cross-year gain > 0".

Per-batter checkpointing (this machine BSODs), so an interrupted run can be resumed with
the same command.

Run: python -m scripts.validate.validate_if_runvalue [min_train_gb] [min_test_gb]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DSN
from src.if_optimize import (expected_outs, fetch_batter_gbs,
                             league_average_positions, optimize_infield,
                             positions_to_params)
from src.if_runvalue import runvalue_ball_weights
from src.re24 import load_re24

from scripts._if_validation import qualifying_batters

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
STATE = (0, 0, 0, 0)          # bases empty, 0 outs (same as the cascade's main scope)
BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = BASE / "models" / "if_gb"
OUT_PATH = _MODEL_DIR / "validation_runvalue_2025.json"
ROWS_PATH = _MODEL_DIR / "validation_runvalue_rows_2025.csv"
MODEL = _MODEL_DIR / "bayes" / "if_bayes_group_pipeline.joblib"
XB_MODEL = _MODEL_DIR / "if_gb_xb_model.joblib"
PRE_DIR = BASE / "data" / "precomputed"


def main(min_train: int = 150, min_test: int = 80) -> None:
    model = joblib.load(MODEL)
    xb = joblib.load(XB_MODEL)
    re24, delta_re = load_re24(PRE_DIR)
    avg_angles, avg_depths = league_average_positions(TRAIN_YEARS)
    avg_start = positions_to_params(avg_angles, avg_depths)

    batters = qualifying_batters(DSN, TRAIN_YEARS, TEST_YEAR, min_train, min_test)
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
