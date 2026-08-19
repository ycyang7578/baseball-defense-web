from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    POSITIONS,
    _polar_to_xy,
    compute_w_j,
    load_model_params,
    objective_re24,
    optimize_positions,
)
from src.re24 import load_re24

# Uses the actually trained model files in the repo + precomputed data, without connecting to the DB or running PyMC
MODELS_DIR = Path(__file__).parent.parent / "models" / "2025"
RE24_DIR = Path(__file__).parent.parent / "data" / "precomputed"

# n_restarts is reduced from 100 in the production pipeline to 10: empirically, for the same
# synthetic ball set, the left-to-right ordering of RF/CF/LF stays stable across 5~100 restarts
# (only LF's exact coordinates can land on a different local optimum below 30 restarts, but the
# directional conclusion is unaffected)
N_RESTARTS_TEST = 10


def _synthetic_right_field_balls(n=8):
    """8 synthetic balls clustered deep in right field, used to test whether the optimizer moves RF toward the ball cluster."""
    balls = pd.DataFrame({
        "ball_x": np.full(n, 200.0),
        "ball_y": np.full(n, 280.0),
        "flight_time": np.full(n, 4.0),
    })
    hit_probs = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))  # treat all as singles, to simplify w_j
    return balls, hit_probs


def test_optimize_positions_moves_rf_toward_ball_cluster():
    balls, hit_probs = _synthetic_right_field_balls()

    result = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )

    lf_x, _ = result["LF"]
    cf_x, _ = result["CF"]
    rf_x, _ = result["RF"]

    # The ball cluster is deep in right field -> RF should be further right than CF, and CF further right than LF
    assert rf_x > cf_x > lf_x


def test_optimize_positions_respects_polar_bounds():
    balls, hit_probs = _synthetic_right_field_balls()

    result = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )

    for pos in POSITIONS:
        x, y = result[pos]
        r = (x ** 2 + y ** 2) ** 0.5
        assert 150 - 1e-6 <= r <= 400 + 1e-6


def test_optimize_positions_returns_ball_and_wall_counts():
    balls, hit_probs = _synthetic_right_field_balls(n=8)

    result = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )

    assert result["n_balls"] == 8
    assert result["n_wall_balls"] == 0  # home_team not specified, so no wall-ball filtering is applied


def test_optimize_positions_beats_a_deliberately_bad_guess():
    """Verify that the optimizer is actually decreasing the objective value, not just returning the starting point."""
    balls, hit_probs = _synthetic_right_field_balls()

    result = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )

    # Rebuild a ctx that's identical to the one used internally by optimize_positions, so objective_re24 can be compared fairly
    _, delta_re = load_re24(RE24_DIR)
    w_j = compute_w_j(balls, None, delta_re, 0, 0, 0, 0, hit_probs=hit_probs)
    assert (w_j > 0).all()  # confirm none of this synthetic data got filtered out by the w_j>0 filter inside optimize_positions

    # optimize_positions uses unified OF parameters (shared across all three positions, finalized 2026-07-13), matched here
    of_scaler, of_mu = load_model_params("OF", MODELS_DIR)
    scalers = {pos: of_scaler for pos in POSITIONS}
    mus = {pos: of_mu for pos in POSITIONS}
    ctx = {
        "ball_x": balls["ball_x"].values,
        "ball_y": balls["ball_y"].values,
        "flight_time": balls["flight_time"].values,
        "w_j": w_j,
        "sc_means": {p: scalers[p].mean_ for p in POSITIONS},
        "sc_scales": {p: scalers[p].scale_ for p in POSITIONS},
        "mus": mus,
    }

    optimized_flat = np.array([*result["LF"], *result["CF"], *result["RF"]])
    optimized_obj = objective_re24(optimized_flat, ctx)
    assert optimized_obj == pytest.approx(result["objective"], abs=1e-6)

    # A deliberately unreasonable positioning: all three fielders crammed at the leftmost edge of the sector, far from the ball cluster deep in right field
    bad_polar = np.array([150.0, -45.0, 150.0, -22.5, 150.0, -22.5])
    bad_flat = _polar_to_xy(bad_polar)
    bad_obj = objective_re24(bad_flat, ctx)

    assert optimized_obj < bad_obj


def test_optimize_positions_warm_start_reaches_same_optimum_with_zero_random_restarts():
    """warm_start_xy should work standalone as the sole starting point (n_restarts=0), verifying it's actually fed into minimize rather than being ignored."""
    balls, hit_probs = _synthetic_right_field_balls()

    reference = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )
    warm = {p: reference[p] for p in POSITIONS}

    warm_started = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=0, seed=999, balls=balls, hit_probs=hit_probs,
        warm_start_xy=warm,
    )

    assert warm_started["objective"] == pytest.approx(reference["objective"], abs=1e-6)


def test_optimize_positions_warm_start_from_related_problem_matches_full_restart_quality():
    """Simulates the actual no_park -> with_park usage pattern: the ball set differs by only a
    few balls (simulating removal of wall balls); using the no_park solution to warm-start
    with_park, n_restarts=2 (1 random + 1 warm start) should still reach a quality comparable
    to 10 random restarts."""
    balls, hit_probs = _synthetic_right_field_balls(n=12)

    no_park_result = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )

    subset_balls = balls.iloc[:-2].reset_index(drop=True)
    subset_hit_probs = hit_probs[:-2]

    reference_with_park = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=subset_balls, hit_probs=subset_hit_probs,
    )

    warm_started_with_park = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=2, seed=999, balls=subset_balls, hit_probs=subset_hit_probs,
        warm_start_xy={p: no_park_result[p] for p in POSITIONS},
    )

    assert warm_started_with_park["objective"] == pytest.approx(
        reference_with_park["objective"], abs=1e-3
    )


def test_optimize_positions_raises_when_batter_has_no_balls():
    empty_balls = pd.DataFrame({"ball_x": [], "ball_y": [], "flight_time": []})

    with pytest.raises(ValueError):
        optimize_positions(
            batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
            years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
            n_restarts=N_RESTARTS_TEST, seed=42, balls=empty_balls,
        )
