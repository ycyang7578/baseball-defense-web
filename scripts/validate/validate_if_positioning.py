"""Stage 5: cross-year validation of infield positioning recommendations (2023-24 -> 2025).

For each qualifying batter:
- Cross-year gain = E[outs | positioning optimized on 2023-24 balls, 2025 balls]
  - E[outs | league average, 2025 balls]
- Same-year ceiling = E[outs | positioning optimized on 2025 balls, 2025 balls]
  - E[outs | league average, 2025 balls]
Both are evaluated on 2025 data so they're directly comparable (evaluating the same-year
gain on its own training data would overestimate it).
Retention = cross-year gain / same-year ceiling. A one-sample t-test is run across batters
against "cross-year gain > 0".

Note: the actual out outcomes under counterfactual positioning are unobservable, so the
effect reported here is a model score (optimizer GLM, trained on 2021-24, already validated
out-of-sample on 2025 with AUC 0.753 / calibration deviation 0.031). The cross-year design
tests "batter tendency and model transferability" — whether the positioning recommendation
still holds up the following year. TRAIN_YEARS is the batter-distribution year range (part
of the cross-year validation design itself) and is unrelated to the GLM's training years;
it stays fixed at 2023-24.

Run: python -m scripts.validate.validate_if_positioning [min_train_gb] [min_test_gb]
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

from scripts._if_validation import qualifying_batters

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "if_gb"
OUT_PATH = _MODEL_DIR / "validation_2025.json"
ROWS_PATH = _MODEL_DIR / "validation_rows_2025.csv"   # checkpoint: written per-batter, resumable
MODEL = _MODEL_DIR / "bayes" / "if_bayes_group_pipeline.joblib"   # group-level = production optimizer


def main(min_train: int = 150, min_test: int = 80) -> None:
    model = joblib.load(MODEL)
    avg_angles, avg_depths = league_average_positions(TRAIN_YEARS)
    avg_start = positions_to_params(avg_angles, avg_depths)

    batters = qualifying_batters(DSN, TRAIN_YEARS, TEST_YEAR, min_train, min_test)
    print(f"合格打者（train GB>={min_train}, test GB>={min_test}）: {len(batters)} 位")

    # checkpoint: reuse batters already computed (long batch jobs on this machine need to be resumable after a crash)
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
