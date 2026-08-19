"""Unit tests for the pure-function parts of src/if_dp_optimize.py (no DB access, no optimization runs)."""
import numpy as np
import pandas as pd
import pytest

from src.if_dataset import SECOND_BASE_Y
from src.if_dp_optimize import (FORCE_SHARE, dp_delta_re, dp_geometry,
                                params_to_positions_dp, positions_to_params_dp)


def _re24():
    """Simplified RE24 table (just needs to be roughly the right order of magnitude)."""
    return {(1, 0, 0, 0): 0.90, (1, 0, 0, 1): 0.52, (1, 0, 0, 2): 0.22,
            (0, 1, 0, 1): 0.66, (0, 1, 0, 2): 0.32,
            (0, 0, 0, 2): 0.10}


def test_dp_delta_re_zero_outs():
    """0 outs: d1 = force/at-1st mix, d2 = double play to (0,0,0,2); d2 < d1 < 0."""
    d1, d2 = dp_delta_re(_re24(), {}, 0)
    expect_d1 = FORCE_SHARE * (0.52 - 0.90) + (1 - FORCE_SHARE) * (0.66 - 0.90)
    assert d1 == pytest.approx(expect_d1)
    assert d2 == pytest.approx(0.10 - 0.90)
    assert d2 < d1 < 0


def test_dp_delta_re_one_out_ends_inning():
    """A double play with 1 out ends the half-inning: d2 = -RE(1,0,0,1)."""
    d1, d2 = dp_delta_re(_re24(), {}, 1)
    assert d2 == pytest.approx(-0.52)
    assert d2 < d1 < 0


def _balls():
    return pd.DataFrame({
        "spray_deg": [0.0, 20.0], "launch_speed": [90.0, 80.0],
        "launch_angle": [-5.0, -8.0], "hp_to_1b": [4.4, 4.4],
        "runner_hp_to_1b": [4.5, 4.5], "stand_R": [1.0, 1.0],
    })


def test_dp_geometry_pivot_and_throw_2b_zero_at_bag():
    """2B standing on second base: pivot_dist=0; when spray is 0 degrees and the nearest fielder's
    depth equals the distance to the bag, throw_dist_2b=0."""
    angles4 = np.array([40.0, 0.0, -30.0, -20.0])   # 2B at 0 degrees
    depths4 = np.array([90.0, SECOND_BASE_Y, 115.0, 145.0])
    feats = dp_geometry(_balls(), angles4, depths4)
    assert feats["pivot_dist"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert feats["throw_dist_2b"].iloc[0] == pytest.approx(0.0, abs=1e-9)  # spray 0
    assert feats["ad_min"].iloc[0] == pytest.approx(0.0)


def test_dp_geometry_pivot_uses_closer_of_2b_ss():
    """When SS is closer to the bag than 2B, pivot_dist takes SS's distance."""
    angles4 = np.array([40.0, 30.0, -30.0, -2.0])
    depths4 = np.array([90.0, 140.0, 115.0, SECOND_BASE_Y])
    feats = dp_geometry(_balls(), angles4, depths4)
    ss_dist = np.hypot(SECOND_BASE_Y * np.sin(np.radians(-2.0)) - 0.0,
                       SECOND_BASE_Y * np.cos(np.radians(-2.0)) - SECOND_BASE_Y)
    assert feats["pivot_dist"].iloc[0] == pytest.approx(ss_dist)


def test_dp_params_roundtrip():
    angles = np.array([15.0, -30.0, -10.0])
    depths = np.array([145.0, 120.0, 150.0])
    a, d = params_to_positions_dp(positions_to_params_dp(angles, depths))
    np.testing.assert_allclose(a, angles)
    np.testing.assert_allclose(d, depths)


def test_dp_scorer_fast_path_matches_pipelines():
    """DPScorer's numpy fast path must be numerically equivalent to the two sklearn pipelines."""
    from pathlib import Path

    import joblib

    from src.if_dp_optimize import DPScorer

    model_dir = Path(__file__).resolve().parent.parent / "models" / "if_gb" / "on1b"
    out_model = joblib.load(model_dir / "if_on1b_out_glm.joblib")
    dp_model = joblib.load(model_dir / "if_on1b_dp_glm.joblib")

    rng = np.random.default_rng(7)
    n = 60
    balls = pd.DataFrame({
        "spray_deg": rng.uniform(-50, 50, n),
        "launch_speed": rng.uniform(60, 110, n),
        "launch_angle": rng.uniform(-40, 5, n),
        "hp_to_1b": rng.uniform(4.0, 5.0, n),
        "runner_hp_to_1b": np.full(n, 4.44),
        "stand_R": rng.integers(0, 2, n).astype(float),
    })
    w = rng.uniform(0.3, 0.9, n)
    pinned = (40.6, 88.0)
    scorer = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77)

    angles3 = np.array([12.0, -33.0, -12.0])
    depths3 = np.array([148.0, 118.0, 146.0])
    p1_fast, p2_fast = scorer._probs(angles3, depths3)

    feats = dp_geometry(balls, np.concatenate([[pinned[0]], angles3]),
                        np.concatenate([[pinned[1]], depths3]))
    p1_ref = out_model.predict_proba(feats)[:, 1]
    p2_ref = dp_model.predict_proba(feats)[:, 1]
    np.testing.assert_allclose(p1_fast, p1_ref, atol=1e-10)
    np.testing.assert_allclose(p2_fast, p2_ref, atol=1e-10)

    # Per-ball P(>=1 out) used by the web endpoint: matches _probs's p1, and its mean equals expected_p1
    p1_public = scorer.per_ball_p1(angles3, depths3)
    np.testing.assert_allclose(p1_public, p1_fast)
    assert scorer.expected_p1(angles3, depths3) == pytest.approx(p1_public.mean())
    assert ((p1_public >= 0) & (p1_public <= 1)).all()


def _real_models():
    from pathlib import Path

    import joblib

    model_dir = Path(__file__).resolve().parent.parent / "models" / "if_gb" / "on1b"
    return (joblib.load(model_dir / "if_on1b_out_glm.joblib"),
            joblib.load(model_dir / "if_on1b_dp_glm.joblib"))


def _rand_balls(n=60, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "spray_deg": rng.uniform(-50, 50, n),
        "launch_speed": rng.uniform(60, 110, n),
        "launch_angle": rng.uniform(-40, 5, n),
        "hp_to_1b": rng.uniform(4.0, 5.0, n),
        "runner_hp_to_1b": np.full(n, 4.44),
        "stand_R": rng.integers(0, 2, n).astype(float),
    }), rng


def test_dp_scorer_zero_player_effects_equal_baseline():
    """All-zero effects must be exactly identical to having no effects at all (league-average fielder)."""
    from src.if_dp_optimize import DPScorer

    out_model, dp_model = _real_models()
    balls, rng = _rand_balls()
    w = rng.uniform(0.3, 0.9, len(balls))
    pinned = (40.6, 88.0)
    pe = {"alpha": np.zeros(4), "g": np.zeros(4), "ad_mean": 6.0, "ad_std": 5.0}
    base = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77)
    with_pe = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77, pe)

    angles3 = np.array([12.0, -33.0, -12.0])
    depths3 = np.array([148.0, 118.0, 146.0])
    np.testing.assert_allclose(with_pe.per_ball_p1(angles3, depths3),
                               base.per_ball_p1(angles3, depths3), atol=1e-12)
    assert with_pe.expected_re(angles3, depths3) == pytest.approx(
        base.expected_re(angles3, depths3), abs=1e-12)


def test_dp_scorer_player_effects_shift_stage1_logit_only():
    """Effect = nearest fielder's α_j + g_j×ad_z added to the stage-1 logit (not applied to stage 2),
    checked against a hand-computed p1 without effects."""
    from src.if_dp_optimize import DPScorer

    out_model, dp_model = _real_models()
    balls, rng = _rand_balls(seed=13)
    w = rng.uniform(0.3, 0.9, len(balls))
    pinned = (40.6, 88.0)
    pe = {"alpha": np.array([0.2, -0.1, 0.05, -0.3]),
          "g": np.array([0.15, 0.0, -0.2, 0.1]),
          "ad_mean": 6.0, "ad_std": 5.0}
    base = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77)
    with_pe = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77, pe)

    angles3 = np.array([20.0, -30.0, -8.0])
    depths3 = np.array([150.0, 120.0, 140.0])
    angles4 = np.concatenate([[pinned[0]], angles3])
    depths4 = np.concatenate([[pinned[1]], depths3])
    spray = balls["spray_deg"].to_numpy(float)
    nearest = np.abs(angles4[None, :] - spray[:, None]).argmin(axis=1)
    ad_min = dp_geometry(balls, angles4, depths4)["ad_min"].to_numpy()
    ad_z = (ad_min - pe["ad_mean"]) / pe["ad_std"]

    p1_base = base.per_ball_p1(angles3, depths3)
    logit = np.log(p1_base / (1 - p1_base)) + pe["alpha"][nearest] + pe["g"][nearest] * ad_z
    np.testing.assert_allclose(with_pe.per_ball_p1(angles3, depths3),
                               1 / (1 + np.exp(-logit)), atol=1e-10)
    # Stage 2 has no effects applied: p2 must be unchanged
    np.testing.assert_allclose(with_pe._probs(angles3, depths3)[1],
                               base._probs(angles3, depths3)[1], atol=1e-12)


def test_dp_optimize_personalized_keeps_slot_assignment():
    """When personalized, the 3B/SS slots are bound to fielders with no label reordering, and the
    returned score matches the effects-aware scorer."""
    from src.if_dp_optimize import DPScorer, optimize_infield_dp

    out_model, dp_model = _real_models()
    balls, rng = _rand_balls(seed=21)
    w = rng.uniform(0.3, 0.9, len(balls))
    pinned = (40.6, 88.0)
    pe = {"alpha": np.array([0.0, 0.0, -0.4, 0.4]),
          "g": np.array([0.0, 0.0, -0.3, 0.3]),
          "ad_mean": 6.0, "ad_std": 5.0}
    res = optimize_infield_dp(balls, out_model, dp_model, pinned, w,
                              -0.29, -0.77, n_restarts=4, seed=2,
                              player_effects=pe)
    scorer = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77, pe)
    assert res["exp_re"] == pytest.approx(
        scorer.expected_re(res["angles"], res["depths"]), abs=1e-9)
    # bounds must still hold (2B on the right side, 3B/SS on the left side)
    assert 1.0 <= res["angles"][0] <= 44.0
    assert all(-44.0 <= a <= -1.0 for a in res["angles"][1:])


def test_dp_anchored_refine_never_worse_than_anchor():
    """Personalized anchored refinement (anchored_starts: anchor + jitter) must score >= the anchor
    itself -- the zero-effects solution often gets stuck on a kink and makes L-BFGS-B fail in place,
    so the jittered starting points need to be able to pick up nearby improvements."""
    from src.if_dp_optimize import (DPScorer, anchored_starts,
                                    optimize_infield_dp,
                                    positions_to_params_dp)

    out_model, dp_model = _real_models()
    balls, rng = _rand_balls(n=120, seed=5)
    w = rng.uniform(0.3, 0.9, len(balls))
    pinned = (40.6, 88.0)
    base = optimize_infield_dp(balls, out_model, dp_model, pinned, w,
                               -0.29, -0.77, n_restarts=6, seed=3)
    pe = {"alpha": np.array([0.0, 0.1, -0.15, 0.09]),
          "g": np.array([0.0, -0.1, 0.05, -0.3]),
          "ad_mean": 6.0, "ad_std": 5.0}
    anchor = positions_to_params_dp(base["angles"], base["depths"])
    starts = anchored_starts(anchor)
    assert len(starts) == 9 and np.allclose(starts[0], anchor)
    res = optimize_infield_dp(balls, out_model, dp_model, pinned, w,
                              -0.29, -0.77, n_restarts=0,
                              extra_starts=starts, player_effects=pe)
    scorer = DPScorer(out_model, dp_model, balls, pinned, w, -0.29, -0.77, pe)
    f_anchor = scorer.expected_re(base["angles"], base["depths"])
    assert res["exp_re"] <= f_anchor + 1e-12
    # Anchoring semantics: the shift should be a small correction, not a jump to a different solution family
    assert np.abs(res["angles"] - base["angles"]).max() < 5.0
    assert np.abs(res["depths"] - base["depths"]).max() < 15.0


def test_on1b_constants_json_complete_for_deploy():
    """The offline constants for the /api/if_optimize DP branch (scripts/precompute_if_on1b_constants.py)
    must exist in the repo with all fields present -- the cloud environment doesn't have
    fielder_positioning_on1b / sprint_speed, so if this file is missing, the DP branch silently
    falls back to the no-runner refinement."""
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parent.parent
            / "data" / "precomputed" / "if_on1b_constants.json")
    const = json.loads(path.read_text(encoding="utf-8"))
    assert const["train_years"] == [2023, 2024]   # must match the Phase-B model's training years
    assert set(const["positions"]) == {"1B", "2B", "3B", "SS"}
    for angle, depth in const["positions"].values():
        assert -50 <= angle <= 50 and 60 <= depth <= 160
    assert 4.0 <= const["runner_hp_to_1b"] <= 5.0
