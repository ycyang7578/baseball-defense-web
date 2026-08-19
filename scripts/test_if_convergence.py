"""Infield optimizer convergence stability test: determines the n_restarts
value to use for the web integration.

30 random batters (a convergence test with sample size <20 produces false
confidence -- a lesson learned from the outfield work). For each batter,
first compute a reference "true" solution with n_restarts=150, then sweep
candidate values 20/16/12/10/8, and tally the miss rate (candidate solution's
expected out rate falling short of the reference solution beyond a tolerance).

The reference and candidate solutions use different LHS seeds, to avoid a
false consistency from overlapping starting-point sets.
All runs include extra_starts=[league average positioning], matching the
usage in the demo / cross-year validation.

Checkpoints to disk per (batter, candidate value), resumable from a crash.

Run: python -m scripts.test_if_convergence [n_batters, default 30]
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2

from src.config import DSN
from src.if_optimize import (POSITIONS, fetch_batter_gbs,
                             league_average_positions, optimize_infield,
                             positions_to_params)

YEARS = [2023, 2024]
MIN_GB = 80          # Relaxed to 80 (cross-year validation uses 150): covers batters
                     # with smaller samples and rougher terrain
REF_RESTARTS = 150
REF_SEED = 999
CAND_RESTARTS = (20, 16, 12, 10, 8)
CAND_SEED = 42       # Same as demo / cross-year validation
SAMPLE_SEED = 7
MISS_TOL = 1e-4      # A shortfall in expected out rate beyond this value counts as a
                     # miss (1% of the typical ~0.01-0.03 gain)
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "if_gb"
MODEL = _MODEL_DIR / "if_gb_optimizer_glm.joblib"
ROWS_PATH = _MODEL_DIR / "convergence_rows.csv"   # checkpoint: written row by row, resumable


def sample_batters(n: int) -> pd.DataFrame:
    with psycopg2.connect(DSN) as conn:
        pool = pd.read_sql(
            "SELECT batter, count(*) AS n_gb "
            "FROM statcast WHERE bb_type='ground_ball' AND game_year = ANY(%(y)s) "
            "  AND hc_x IS NOT NULL "
            "GROUP BY batter HAVING count(*) >= %(m)s ORDER BY batter",
            conn, params={"y": YEARS, "m": MIN_GB})
    return pool.sample(n=n, random_state=SAMPLE_SEED).reset_index(drop=True)


def max_position_gap(res_a: dict, res_b: dict) -> float:
    """Maximum single-fielder distance difference between two solutions (feet)."""
    def xy(res):
        rad = np.radians(res["angles"])
        return np.column_stack([res["depths"] * np.sin(rad),
                                res["depths"] * np.cos(rad)])
    return float(np.hypot(*(xy(res_a) - xy(res_b)).T).max())


def main(n_batters: int = 30) -> None:
    model = joblib.load(MODEL)
    avg_angles, avg_depths = league_average_positions(YEARS)
    avg_start = positions_to_params(avg_angles, avg_depths)

    batters = sample_batters(n_batters)
    print(f"抽樣 {len(batters)} 位打者（pool: 2023-24 GB>={MIN_GB}，seed={SAMPLE_SEED}）")

    done: set[tuple[int, int]] = set()
    if ROWS_PATH.exists():
        prev = pd.read_csv(ROWS_PATH)
        done = set(zip(prev["batter"], prev["n_restarts"]))
        print(f"  checkpoint: 已有 {len(prev)} 列，續跑其餘")

    for i, row in enumerate(batters.itertuples(), 1):
        todo = [n for n in CAND_RESTARTS if (row.batter, n) not in done]
        if not todo:
            continue
        balls = fetch_batter_gbs(row.batter, YEARS)
        ref = optimize_infield(balls, model, n_restarts=REF_RESTARTS,
                               seed=REF_SEED, extra_starts=[avg_start])
        rows = []
        for n in todo:
            cand = optimize_infield(balls, model, n_restarts=n,
                                    seed=CAND_SEED, extra_starts=[avg_start])
            rows.append({
                "batter": row.batter, "n_gb": row.n_gb, "n_restarts": n,
                "ref_exp_outs": ref["exp_outs"], "cand_exp_outs": cand["exp_outs"],
                "shortfall": ref["exp_outs"] - cand["exp_outs"],
                "pos_gap_ft": max_position_gap(ref, cand),
            })
        pd.DataFrame(rows).to_csv(ROWS_PATH, mode="a", index=False,
                                  header=not ROWS_PATH.exists())
        worst = max(r["shortfall"] for r in rows)
        print(f"[{i}/{len(batters)}] 打者 {row.batter}（{row.n_gb} GB）"
              f" 最大落後 {worst:.5f}", flush=True)

    df = pd.read_csv(ROWS_PATH)
    df = df[df["batter"].isin(batters["batter"])]
    print(f"\n=== 收斂測試結果（{df['batter'].nunique()} 位打者，"
          f"參考解 n_restarts={REF_RESTARTS}，miss 容忍 {MISS_TOL}）===")
    print(f"{'n_restarts':>10} {'miss':>6} {'rate':>7} {'中位落後':>10} "
          f"{'最大落後':>10} {'miss時中位站位差(呎)':>18}")
    for n in CAND_RESTARTS:
        sub = df[df["n_restarts"] == n]
        miss = sub[sub["shortfall"] > MISS_TOL]
        med_gap = miss["pos_gap_ft"].median() if len(miss) else 0.0
        print(f"{n:>10} {len(miss):>4}/{len(sub):<3} {len(miss) / len(sub):>6.1%} "
              f"{sub['shortfall'].median():>10.6f} {sub['shortfall'].max():>10.5f} "
              f"{med_gap:>18.1f}")
    neg = df[df["shortfall"] < -MISS_TOL]
    if len(neg):
        print(f"\n[!] 有 {len(neg)} 列候選解優於參考解（參考解本身未收斂），"
              f"最大 {-neg['shortfall'].min():.5f}——參考 n_restarts 可能要再拉高")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
