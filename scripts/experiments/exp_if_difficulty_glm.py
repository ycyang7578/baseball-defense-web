"""Experiment: switch the difficulty model to an interpretable spline GLM
(replacing GBM).

Motivation (2026-07-12, user decision): avoid an unexplainable model, accept some
accuracy loss in exchange for interpretability. The difficulty model does not
reposition fielders and has no counterfactual requirement, so the endogeneity ban
does not apply -- spray is a legitimate feature, and splines/interactions are
free to use.

Feature engineering (mirrors the treatment in Melville Sec. 2):
- spray_rel: flip the sign of spray for left-handed batters so the "pull" direction
  is aligned league-wide (negative = pull, positive = push/oppo)
- splines: spray_rel (8 knots, needs to accommodate the lane structure across the
  four infield positions), LA/EV/hp (6 knots)
- interactions: spray x EV (hard-hit balls finding gaps), spray x hp (directional
  bias of slow-roller infield hits)

Configurations:
  GBM        current HistGradientBoosting (baseline)
  D1         main-effect splines + stand_R
  D2         D1 + spray x EV
  D3         D2 + spray x hp
Output: 2025 out-of-sample AUC/Brier/calibration + the best GLM's correlation with
official OAA (qualified R, by position, scale), to decide whether to promote it to
production.

Run: python scripts/experiments/exp_if_difficulty_glm.py
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

    # Official OAA correlation: GBM vs best GLM (D3)
    scored_gbm = score_test_year(TRAIN_YEARS, TEST_YEAR)
    official_correlations(scored_gbm, "GBM")
    scored_glm = score_test_year(
        TRAIN_YEARS, TEST_YEAR,
        model_factory=lambda: make_difficulty_glm(spray_ev=True, spray_hp=True))
    official_correlations(scored_glm, "GLM D3")


if __name__ == "__main__":
    main()
