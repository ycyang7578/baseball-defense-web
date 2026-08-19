import numpy as np
import pytest
from scipy.special import expit

from src.optimization import (
    _catch_prob_single_fielder,
    _polar_to_xy,
    compute_w_j,
    objective_re24,
)


def test_polar_to_xy_matches_hand_computed_values():
    # [r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF]
    params = np.array([200.0, 0.0, 300.0, 45.0, 250.0, -30.0])

    xy = _polar_to_xy(params)

    assert xy[0] == pytest.approx(0.0, abs=1e-9)          # LF x = 200*sin(0)
    assert xy[1] == pytest.approx(200.0)                   # LF y = 200*cos(0)
    assert xy[2] == pytest.approx(300 * np.sin(np.radians(45)))
    assert xy[3] == pytest.approx(300 * np.cos(np.radians(45)))
    assert xy[4] == pytest.approx(250 * np.sin(np.radians(-30)))
    assert xy[5] == pytest.approx(250 * np.cos(np.radians(-30)))


def test_catch_prob_single_fielder_isolates_alpha_when_other_betas_zero():
    # sc_mean=0, sc_scale=1 (no standardization), only mu_alpha is nonzero -> catch_prob should always equal sigmoid(mu_alpha)
    # regardless of ball/fielder coordinates, since all other coefficients are 0
    fx, fy = 0.0, 200.0
    ball_x = np.array([0.0, 50.0, -100.0])
    ball_y = np.array([150.0, 100.0, 300.0])
    flight_time = np.array([3.0, 2.0, 4.0])

    sc_mean = np.zeros(4)
    sc_scale = np.ones(4)
    mu = {"mu_alpha": 0.5, "mu_beta_speed": 0.0, "mu_beta_cos": 0.0,
          "mu_beta_sin": 0.0, "mu_beta_dist": 0.0}

    probs = _catch_prob_single_fielder(fx, fy, ball_x, ball_y, flight_time, sc_mean, sc_scale, mu)

    expected = expit(0.5)
    assert probs == pytest.approx(np.full(3, expected))


def test_catch_prob_single_fielder_decreases_with_higher_required_speed():
    # mu_beta_speed is negative, and speed is the only nonzero coefficient: the larger required_speed is (the harder the ball is to catch), the lower the probability should be
    fx, fy = 0.0, 200.0
    sc_mean = np.zeros(4)
    sc_scale = np.ones(4)
    mu = {"mu_alpha": 0.0, "mu_beta_speed": -0.1, "mu_beta_cos": 0.0,
          "mu_beta_sin": 0.0, "mu_beta_dist": 0.0}

    # The ball is directly ahead (rel_angle=0), at distances of 50ft / 200ft respectively, with flight time fixed at 2 seconds
    # -> required_speed is 25 / 100 ft/s respectively
    ball_x = np.array([0.0, 0.0])
    ball_y = np.array([150.0, 0.0])
    flight_time = np.array([2.0, 2.0])

    probs = _catch_prob_single_fielder(fx, fy, ball_x, ball_y, flight_time, sc_mean, sc_scale, mu)

    logit_near = 0.0 + (-0.1) * 25.0
    logit_far = 0.0 + (-0.1) * 100.0
    assert probs[0] == pytest.approx(expit(logit_near))
    assert probs[1] == pytest.approx(expit(logit_far))
    assert probs[0] > probs[1]  # closer distance, lower required speed -> higher catch probability


def test_compute_w_j_weights_hit_probs_by_delta_re():
    # Feed hit_probs directly, skipping KDE inference; balls/hit_bundle are unused on this path
    hit_probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    delta_re = {
        ("1B", 0, 0, 0, 0): 0.5,
        ("2B", 0, 0, 0, 0): 1.0,
        ("3B", 0, 0, 0, 0): 1.5,
    }

    w_j = compute_w_j(
        balls=None, hit_bundle=None, delta_re=delta_re,
        on_1b=0, on_2b=0, on_3b=0, outs=0, hit_probs=hit_probs,
    )

    assert w_j.tolist() == pytest.approx([0.5, 1.0, 1.5])


def test_objective_re24_matches_hand_computed_value_when_only_alpha_varies():
    # All three fielders have only mu_alpha nonzero (all other coefficients are 0), so catch_prob
    # always equals sigmoid(alpha), independent of ball/fielder coordinates, so the expected value can be hand-computed entirely
    mu_common = {"mu_beta_speed": 0.0, "mu_beta_cos": 0.0, "mu_beta_sin": 0.0, "mu_beta_dist": 0.0}
    mus = {
        "LF": {**mu_common, "mu_alpha": -1.0},
        "CF": {**mu_common, "mu_alpha": 2.0},
        "RF": {**mu_common, "mu_alpha": -1.0},
    }
    sc_means = {pos: np.zeros(4) for pos in ("LF", "CF", "RF")}
    sc_scales = {pos: np.ones(4) for pos in ("LF", "CF", "RF")}

    ctx = {
        "ball_x": np.array([0.0]),
        "ball_y": np.array([200.0]),
        "flight_time": np.array([3.0]),
        "w_j": np.array([2.0]),
        "sc_means": sc_means,
        "sc_scales": sc_scales,
        "mus": mus,
    }
    # Coordinates are arbitrary (they don't affect the result since all betas are 0)
    positions_flat = np.array([-100.0, 200.0, 0.0, 250.0, 100.0, 200.0])

    result = objective_re24(positions_flat, ctx)

    p_lf = expit(-1.0)
    p_cf = expit(2.0)
    p_rf = expit(-1.0)
    p_not_caught = (1 - p_lf) * (1 - p_cf) * (1 - p_rf)
    expected = p_not_caught * 2.0

    assert result == pytest.approx(expected)
