"""Shared logic for infielder player evaluation: difficulty model scoring + credit assignment + per-position centering.

Extracted from scripts/evaluate_if_2025.py (to keep the row-by-row calculation
consistent), so the evaluation report and the web ranking pipeline
(scripts/precompute_if_model_oaa.py) use the same implementation.
See the evaluate_if_2025.py docstring for the methodology.
As of 2026-07-12 the default difficulty model is the interpretable spline GLM
(see the if_model.py docstring).
"""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.if_dataset import INFIELD_COLS, build_gb_dataset
from src.if_model import DIFFICULTY_FEATURES, make_difficulty_glm


def score_test_year(train_years: list[int], test_year: int,
                    model_factory: Callable[[], Pipeline] = make_difficulty_glm) -> pd.DataFrame:
    """Per-ball scoring over the full ground ball set: fit the difficulty model on train_years, score test_year.

    model_factory defaults to the evaluation difficulty GLM; alternative models (e.g. a
    GBM benchmark) can pass in their own factory, which must accept the
    DIFFICULTY_FEATURES columns.
    Returns the test set as a per-ball DataFrame with added columns:
    p_hat (league-average difficulty), resp_fielder/resp_pos (credit assignment),
    oaa_play (is_out - p_hat), oaa_play_c (after per-position centering).
    """
    train = build_gb_dataset(train_years)
    test = build_gb_dataset([test_year])
    print(f"全量滾地球: train n={len(train):,}, test n={len(test):,}")

    diff_model = model_factory()
    diff_model.fit(train[DIFFICULTY_FEATURES], train["is_out"])
    test = test.copy()
    test["p_hat"] = diff_model.predict_proba(test[DIFFICULTY_FEATURES])[:, 1]

    # Credit assignment: outs are credited to the fielder who actually made the play
    # (hit_location 3-6, matching official credit); hits and balls handled by the
    # pitcher/catcher fall back to the nearest infielder by angular distance
    pos_to_col = {v: k for k, v in INFIELD_COLS.items()}
    nearest_ids = np.select(
        [test["nearest_pos"] == p for p in pos_to_col],
        [test[c] for c in (pos_to_col[p] for p in pos_to_col)])
    loc = pd.to_numeric(test["hit_location"], errors="coerce")
    use_loc = (test["is_out"] == 1) & loc.between(3, 6)
    loc_ids = np.select(
        [loc == int(c.split("_")[1]) for c in INFIELD_COLS],
        [test[c] for c in INFIELD_COLS])
    test["resp_fielder"] = np.where(use_loc, loc_ids, nearest_ids).astype(int)
    loc_pos = loc.map({3: "1B", 4: "2B", 5: "3B", 6: "SS"})
    test["resp_pos"] = np.where(use_loc, loc_pos, test["nearest_pos"])
    print(f"出局球以 hit_location 歸責的比例: {use_loc[test['is_out'] == 1].mean():.1%}"
          f"（其中與最近角距不同者 "
          f"{(use_loc & (loc_ids != nearest_ids)).sum() / max(use_loc.sum(), 1):.1%}）")
    test["oaa_play"] = test["is_out"] - test["p_hat"]
    # Per-position centering: official OAA is measured "relative to the same-position
    # average," but p_hat is the all-league lane average, and systematic offsets between
    # positions (e.g. 1B) would drag down the pooled correlation. Subtracting each
    # position's mean per ball makes each position's total sum to zero.
    test["oaa_play_c"] = (test["oaa_play"]
                          - test.groupby("resp_pos")["oaa_play"].transform("mean"))
    return test


def aggregate_players(test: pd.DataFrame) -> pd.DataFrame:
    """Per-ball -> per-player: model_oaa (centered) / model_oaa_raw / n_balls / resp_pos (mode)."""
    model = (test.groupby("resp_fielder")
             .agg(model_oaa=("oaa_play_c", "sum"), model_oaa_raw=("oaa_play", "sum"),
                  n_balls=("oaa_play", "size"))
             .reset_index())
    pos_mode = (test.groupby(["resp_fielder", "resp_pos"]).size()
                .rename("n").reset_index()
                .sort_values("n", ascending=False)
                .drop_duplicates("resp_fielder"))
    return model.merge(pos_mode[["resp_fielder", "resp_pos"]], on="resp_fielder")
