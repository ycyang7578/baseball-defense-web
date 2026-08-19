"""Stage B cross-year validation: runner-on-first (0 outs) double-play-situation
positioning, 2023-24 -> 2025.

Same design as validate_if_runvalue.py (same batch of qualifying batters, everything
evaluated on 2025 balls, run-value denominated), with 1B always pinned to the league
hold-runner position (2023-24 average with a runner on first). For each batter, compare
(all evaluated with the Stage B two-stage model + DPScorer):
- Cross-year gain (DP solution)      = E[ΔRE | league runner-on-first positioning]
  - E[ΔRE | 2023-24 DP-optimized positioning]
- Cross-year gain (no-base-state solution) = same as above, but positioning = the
  2023-24 out-rate-objective optimal solution (current production, i.e. "no base-state
  adjustment") — the difference between the two is Stage B's situational optimization's
  **incremental** cross-year value
- Same-year ceiling (DP solution)    = optimized on 2025 balls, evaluated on 2025 balls

Per-batter checkpointing (this machine BSODs), so an interrupted run can be resumed with
the same command.

Run: python -m scripts.validate.validate_if_dp [min_train_gb] [min_test_gb]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DSN
from src.if_dp_optimize import (DPScorer, dp_delta_re,
                                league_average_positions_on1b,
                                league_median_runner_speed,
                                optimize_infield_dp, positions_to_params_dp)
from src.if_optimize import fetch_batter_gbs, optimize_infield
from src.if_runvalue import gb_miss_costs
from src.re24 import load_re24

from scripts._if_validation import qualifying_batters

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
OUTS = 0                      # runner on first, 0 outs
BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = BASE / "models" / "if_gb" / "on1b"
OUT_PATH = _MODEL_DIR / "validation_dp_2025.json"
ROWS_PATH = _MODEL_DIR / "validation_dp_rows_2025.csv"


def main(min_train: int = 150, min_test: int = 80) -> None:
    out_model = joblib.load(_MODEL_DIR / "if_on1b_out_glm.joblib")
    dp_model = joblib.load(_MODEL_DIR / "if_on1b_dp_glm.joblib")
    xb = joblib.load(BASE / "models/if_gb/if_gb_xb_model.joblib")
    bayes = joblib.load(BASE / "models/if_gb/bayes/if_bayes_group_pipeline.joblib")
    re24, delta_re = load_re24(BASE / "data/precomputed")

    lg_angles, lg_depths = league_average_positions_on1b(TRAIN_YEARS)
    pinned_1b = (float(lg_angles[0]), float(lg_depths[0]))
    lg3 = (lg_angles[1:], lg_depths[1:])
    lg_start = positions_to_params_dp(*lg3)
    runner_hp = league_median_runner_speed(TRAIN_YEARS)
    d1, d2 = dp_delta_re(re24, delta_re, OUTS)

    batters = qualifying_batters(DSN, TRAIN_YEARS, TEST_YEAR, min_train, min_test)
    print(f"合格打者（train GB>={min_train}, test GB>={min_test}）: {len(batters)} 位",
          flush=True)

    done: set[int] = set()
    if ROWS_PATH.exists():
        done = set(pd.read_csv(ROWS_PATH)["batter"])
        print(f"  checkpoint: 已有 {len(done)} 位，續跑其餘", flush=True)

    for i, row in enumerate(batters.itertuples(), 1):
        if row.batter in done:
            continue
        tr = fetch_batter_gbs(row.batter, TRAIN_YEARS)
        te = fetch_batter_gbs(row.batter, [TEST_YEAR])
        tr["runner_hp_to_1b"] = runner_hp
        te["runner_hp_to_1b"] = runner_hp
        w_tr = gb_miss_costs(tr, xb, delta_re, (1, 0, 0, OUTS))
        w_te = gb_miss_costs(te, xb, delta_re, (1, 0, 0, OUTS))
        scorer_te = DPScorer(out_model, dp_model, te, pinned_1b, w_te, d1, d2)

        dre_league = scorer_te.expected_re(*lg3)

        # 2023-24 no-base-state (out-rate objective) optimal solution -> used as the DP
        # optimization warm start and as the control group
        base_opt = optimize_infield(tr, bayes, n_restarts=16, seed=42)
        base3 = (base_opt["angles"][1:], base_opt["depths"][1:])
        base_start = positions_to_params_dp(*base3)

        opt_cross = optimize_infield_dp(tr, out_model, dp_model, pinned_1b, w_tr,
                                        d1, d2, n_restarts=8, seed=42,
                                        extra_starts=[lg_start, base_start])
        opt_same = optimize_infield_dp(te, out_model, dp_model, pinned_1b, w_te,
                                       d1, d2, n_restarts=8, seed=42,
                                       extra_starts=[lg_start, base_start])

        pd.DataFrame([{
            "batter": row.batter, "stand": row.stand,
            "n_train": row.n_train, "n_test": row.n_test,
            "dre_league": dre_league,
            "gain_cross_dp": dre_league - scorer_te.expected_re(
                opt_cross["angles"], opt_cross["depths"]),
            "gain_cross_base": dre_league - scorer_te.expected_re(*base3),
            "gain_same": dre_league - scorer_te.expected_re(
                opt_same["angles"], opt_same["depths"]),
        }]).to_csv(ROWS_PATH, mode="a", header=not ROWS_PATH.exists(), index=False)
        if i % 10 == 0 or i == len(batters):
            print(f"  ...{i}/{len(batters)}", flush=True)

    df = pd.read_csv(ROWS_PATH)
    df = df[df["batter"].isin(set(batters["batter"]))]
    extra = df["gain_cross_dp"] - df["gain_cross_base"]
    t, p = stats.ttest_1samp(df["gain_cross_dp"], 0.0)
    t_x, p_x = stats.ttest_1samp(extra, 0.0)
    retention = df["gain_cross_dp"].sum() / df["gain_same"].sum()

    print(f"\n=== 階段B 跨年驗證（一壘有人 0 出局，n={len(df)}，皆評 2025 球）===")
    print(f"跨年增益(DP 解 vs 聯盟):   平均 {df['gain_cross_dp'].mean():+.5f} 分/GB"
          f"（450 GB 約 {df['gain_cross_dp'].mean() * 450:+.2f} 分）, "
          f"正增益 {(df['gain_cross_dp'] > 0).mean():.1%}, t={t:.2f} (p={p:.1e})")
    print(f"跨年增益(無壘況解 vs 聯盟): 平均 {df['gain_cross_base'].mean():+.5f} 分/GB")
    print(f"情境優化的額外增益:        平均 {extra.mean():+.5f} 分/GB"
          f"（450 GB 約 {extra.mean() * 450:+.2f} 分）, "
          f"正增益 {(extra > 0).mean():.1%}, t={t_x:.2f} (p={p_x:.1e})")
    print(f"同年上限: 平均 {df['gain_same'].mean():+.5f} 分/GB, 保留率 {retention:.1%}")
    for s in ("L", "R"):
        sub = df[df["stand"] == s]
        if len(sub):
            print(f"  {s} 打（n={len(sub)}）: DP {sub['gain_cross_dp'].mean():+.5f}, "
                  f"額外 {(sub['gain_cross_dp'] - sub['gain_cross_base']).mean():+.5f}")

    summary = {
        "n_batters": len(df), "min_train_gb": min_train, "min_test_gb": min_test,
        "state": [1, 0, 0, OUTS],
        "mean_gain_cross_dp": round(df["gain_cross_dp"].mean(), 6),
        "mean_gain_cross_base": round(df["gain_cross_base"].mean(), 6),
        "mean_extra_gain": round(extra.mean(), 6),
        "extra_gain_per_450gb": round(extra.mean() * 450, 3),
        "share_positive_extra": round((extra > 0).mean(), 4),
        "t_extra": round(t_x, 3), "p_extra": float(f"{p_x:.3e}"),
        "mean_gain_same": round(df["gain_same"].mean(), 6),
        "retention": round(retention, 4),
        "t_dp": round(t, 3), "p_dp": float(f"{p:.3e}"),
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
