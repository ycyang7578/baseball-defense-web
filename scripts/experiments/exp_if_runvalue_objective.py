"""Experiment (stage A / bridge to infield-outfield integration): run-value
objective vs. out-rate objective.

Reproduces Melville's comparison (he found that wOBA-optimized positioning was
even better on expected outs, but that was in-sample); this experiment checks it
under a cross-year setup: optimize positioning on 2023-24 balls, evaluate on 2025
balls, bases empty and 0 outs.

Components:
- P(extra-base hit | ground ball) model (src/if_runvalue.py); first report its 2025
  out-of-sample quality and a sideline sanity check (P(XB) for |spray|>40 should be
  much higher than for the middle zone)
- w_j and ΔRE(out) use the same RE24/delta_re table as the outfield model
  (data/precomputed)
- Optimize separately for each objective (n_restarts=50, same budget as
  production), then evaluate both metrics on the same batch of 2025 balls:
  expected out rate, expected runs E[ΔRE]

Per-batter checkpointing (this machine is prone to BSODs), resumable with the
same command after an interruption.

Run: python scripts/experiments/exp_if_runvalue_objective.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.if_optimize import expected_outs, fetch_batter_gbs, optimize_infield
from src.if_runvalue import (XB_FEATURES, delta_re_out, fetch_gb_hits,
                             gb_miss_costs, make_gb_xb_model,
                             runvalue_ball_weights)
from src.re24 import load_re24

BASE = Path(__file__).resolve().parent.parent.parent
MODEL = BASE / "models" / "if_gb" / "bayes" / "if_bayes_group_pipeline.joblib"
VALID_ROWS = BASE / "models" / "if_gb" / "validation_rows_2025.csv"
CKPT = BASE / "models" / "if_gb" / "runvalue_objective_rows.csv"
PRE_DIR = BASE / "data" / "precomputed"

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
STATE = (0, 0, 0, 0)          # bases empty, 0 outs
N_RESTARTS = 50
SEED = 42

COLS = ["batter", "n_tr", "n_te",
        "eo_outsopt", "eo_runsopt",          # 2025 expected out rate
        "dre_outsopt", "dre_runsopt"]        # 2025 expected runs E[ΔRE]


def main() -> None:
    # -- 1. Hit-type model ----------------------------------------
    hits_tr = fetch_gb_hits(TRAIN_YEARS)
    hits_te = fetch_gb_hits([TEST_YEAR])
    xb = make_gb_xb_model().fit(hits_tr[XB_FEATURES], hits_tr["is_xb"])
    p_te = xb.predict_proba(hits_te[XB_FEATURES])[:, 1]
    auc = roc_auc_score(hits_te["is_xb"], p_te)
    print(f"XB model: train hits={len(hits_tr):,} (XB率 {hits_tr['is_xb'].mean():.3f}), "
          f"2025 hits={len(hits_te):,}, AUC={auc:.4f}")
    line = hits_te["spray_deg"].abs() > 40
    mid = hits_te["spray_deg"].abs() < 20
    print(f"  sanity: P(XB|邊線 |spray|>40) 預測 {p_te[line].mean():.3f} / "
          f"實際 {hits_te.loc[line, 'is_xb'].mean():.3f}；"
          f"中間 |spray|<20 預測 {p_te[mid].mean():.3f} / "
          f"實際 {hits_te.loc[mid, 'is_xb'].mean():.3f}")

    re24, delta_re = load_re24(PRE_DIR)
    dre_o = delta_re_out(re24, STATE)
    print(f"ΔRE(out)={dre_o:+.4f}, ΔRE(1B)={delta_re.get(('1B', *STATE)):+.4f}, "
          f"ΔRE(2B)={delta_re.get(('2B', *STATE)):+.4f}")

    # -- 2. Batter sample (reuses the same 212 batters from cross-year validation) --
    batters = pd.read_csv(VALID_ROWS)["batter"].astype(int).tolist()
    done = set()
    if CKPT.exists():
        done = set(pd.read_csv(CKPT)["batter"].astype(int))
        print(f"checkpoint: {len(done)} 位已完成")
    model = joblib.load(MODEL)

    for i, b in enumerate(batters, 1):
        if b in done:
            continue
        tr = fetch_batter_gbs(b, TRAIN_YEARS)
        te = fetch_batter_gbs(b, [TEST_YEAR])
        bw_tr, _ = runvalue_ball_weights(tr, xb, re24, delta_re, STATE)

        opt_o = optimize_infield(tr, model, n_restarts=N_RESTARTS, seed=SEED)
        opt_r = optimize_infield(tr, model, n_restarts=N_RESTARTS, seed=SEED,
                                 ball_weights=bw_tr)

        w_te = gb_miss_costs(te, xb, delta_re, STATE)
        bw_te = w_te - dre_o

        def eval_on_2025(sol):
            eo = expected_outs(model, te, sol["angles"], sol["depths"])
            dre = float(w_te.mean()) - expected_outs(
                model, te, sol["angles"], sol["depths"], ball_weights=bw_te)
            return eo, dre

        eo_o, dre_o_sol = eval_on_2025(opt_o)
        eo_r, dre_r_sol = eval_on_2025(opt_r)
        row = pd.DataFrame([[b, len(tr), len(te),
                             eo_o, eo_r, dre_o_sol, dre_r_sol]], columns=COLS)
        row.to_csv(CKPT, mode="a", header=not CKPT.exists(), index=False)
        if i % 10 == 0 or i == len(batters):
            print(f"  ...{i}/{len(batters)}", flush=True)

    # -- 3. Summary ------------------------------------------------
    df = pd.read_csv(CKPT)
    d_eo = df["eo_runsopt"] - df["eo_outsopt"]        # out rate: run-objective - out-objective
    d_dre = df["dre_runsopt"] - df["dre_outsopt"]     # runs: more negative is better
    print(f"\n=== run-value 目標 vs 出局率目標（n={len(df)}，2025 樣本外）===")
    print(f"期望出局率差:  mean {d_eo.mean():+.5f}  median {d_eo.median():+.5f}  "
          f"run目標較高比例 {(d_eo > 0).mean():.1%}")
    print(f"期望失分差:    mean {d_dre.mean():+.5f}  median {d_dre.median():+.5f}  "
          f"run目標較低比例 {(d_dre < 0).mean():.1%}")
    print(f"換算每 450 GB: 失分 {d_dre.mean() * 450:+.2f} 分、"
          f"出局 {d_eo.mean() * 450:+.2f} 個")


if __name__ == "__main__":
    main()
