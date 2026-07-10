"""src/if_optimize.py 的單元測試（合成模型/資料，不碰 DB）。"""
import numpy as np
import pandas as pd
import pytest

from src.if_dataset import HOME_X, HOME_Y, attach_features
from src.if_model import OPTIMIZER_FEATURES, make_optimizer_glm
from src.if_optimize import (ANGLE_BOUNDS, FRAC_BOUNDS, MIN_DEPTH, POSITIONS,
                             _FastGLMObjective, dirt_max_depth, expected_outs,
                             geometry_features, optimize_infield,
                             params_to_positions, positions_to_params)


class DummyModel:
    """P(out) 隨角差變小而升高的簡化模型。"""

    def predict_proba(self, feats):
        p = 1 / (1 + np.exp(-(1.5 - 0.12 * feats["ad_min"].to_numpy())))
        return np.column_stack([1 - p, p])


def _balls(sprays, ev=90.0):
    return pd.DataFrame({
        "spray_deg": sprays,
        "launch_speed": ev,
        "launch_angle": -5.0,
        "hp_to_1b": 4.3,
        "stand_R": 1,
    })


def test_dirt_max_depth_up_the_middle():
    """0 度方向的土外緣 = 60.5 + 95 = 155.5 呎，且往邊線遞減。"""
    assert dirt_max_depth(0.0) == pytest.approx(155.5)
    assert dirt_max_depth(44.0) < dirt_max_depth(0.0)


def test_params_positions_roundtrip():
    angles = np.array([30.0, 15.0, -30.0, -15.0])
    depths = np.array([110.0, 145.0, 115.0, 140.0])
    x = positions_to_params(angles, depths)
    a2, d2 = params_to_positions(x)
    np.testing.assert_allclose(a2, angles)
    np.testing.assert_allclose(d2, depths)


def test_geometry_features_matches_attach_features():
    """優化端與資料端的幾何公式必須一致（同一顆球算出同樣特徵）。"""
    angles = np.array([35.0, 20.0, -35.0, -20.0])
    depths = np.array([110.0, 150.0, 115.0, 145.0])
    gb = pd.DataFrame([{
        "game_year": 2024, "batter": 1, "stand": "R",
        "hc_x": HOME_X + 30, "hc_y": HOME_Y - 80,
        "launch_speed": 92.0, "launch_angle": -8.0,
        "events": "field_out", "if_fielding_alignment": "Standard",
        "bases_empty": True,
        "fielder_3": 13, "fielder_4": 14, "fielder_5": 15, "fielder_6": 16,
    }])
    positioning = pd.DataFrame(
        [{"fielder_id": 13 + i, "season": 2024, "position": p,
          "depth": depths[i], "angle": angles[i]} for i, p in enumerate(POSITIONS)])
    run_speed = pd.DataFrame([{"player_id": 1, "season": 2024, "hp_to_1b": 4.3}])
    ds = attach_features(gb, positioning, run_speed)

    balls = ds[["spray_deg", "launch_speed", "launch_angle", "hp_to_1b", "stand_R"]]
    feats = geometry_features(balls, angles, depths)
    for col in ("ad_min", "ball_time", "throw_dist"):
        assert feats[col].iloc[0] == pytest.approx(ds[col].iloc[0])


def test_optimize_respects_constraints_and_beats_bad_start():
    """解必須滿足規則約束，且期望出局率不低於任意合法站位。"""
    balls = _balls(np.array([-30.0, -12.0, 5.0, 18.0, 33.0] * 20))
    model = DummyModel()
    result = optimize_infield(balls, model, n_restarts=8, seed=1)

    angles, depths = result["angles"], result["depths"]
    assert (angles[:2] >= ANGLE_BOUNDS[0][0] - 1e-6).all()   # 1B/2B 在右側
    assert (angles[2:] <= ANGLE_BOUNDS[2][1] + 1e-6).all()   # 3B/SS 在左側
    assert (depths >= MIN_DEPTH - 1e-6).all()
    assert (depths <= dirt_max_depth(angles) + 1e-6).all()   # 內野土內

    bad = expected_outs(model, balls, np.array([40.0, 38.0, -40.0, -38.0]),
                        np.array([70.0, 70.0, 70.0, 70.0]))
    assert result["exp_outs"] >= bad


def _fitted_glm():
    """在合成資料上 fit 真的優化用 GLM pipeline（同 test_if_model 的作法）。"""
    rng = np.random.default_rng(7)
    n = 400
    train = pd.DataFrame({
        "ad_min": rng.uniform(0, 25, n),
        "ball_time": rng.uniform(0.5, 2.5, n),
        "launch_angle": rng.uniform(-60, 10, n),
        "launch_speed": rng.uniform(60, 110, n),
        "throw_dist": rng.uniform(20, 130, n),
        "hp_to_1b": rng.uniform(4.0, 4.8, n),
        "stand_R": rng.integers(0, 2, n),
    })
    p = 1 / (1 + np.exp(-(1.2 - 0.08 * train["ad_min"])))
    y = (rng.uniform(size=n) < p).astype(int)
    return make_optimizer_glm().fit(train[OPTIMIZER_FEATURES], y)


def test_fast_objective_matches_pipeline():
    """快速路徑的期望出局率必須跟走完整 pipeline 的通用路徑數值等價。"""
    glm = _fitted_glm()
    rng = np.random.default_rng(3)
    balls = _balls(rng.uniform(-50, 50, 60), ev=rng.uniform(60, 110, 60))
    fast = _FastGLMObjective(glm, balls)

    lo = np.array([b[0] for b in ANGLE_BOUNDS + FRAC_BOUNDS])
    hi = np.array([b[1] for b in ANGLE_BOUNDS + FRAC_BOUNDS])
    for _ in range(20):
        angles, depths = params_to_positions(lo + rng.uniform(size=8) * (hi - lo))
        assert fast.expected_outs(angles, depths) == pytest.approx(
            expected_outs(glm, balls, angles, depths), abs=1e-10)


def test_optimize_with_pipeline_result_consistent():
    """走快速路徑的 optimize 結果，用 pipeline 重算必須得到同樣的 exp_outs。"""
    glm = _fitted_glm()
    balls = _balls(np.array([-30.0, -12.0, 5.0, 18.0, 33.0] * 12))
    result = optimize_infield(balls, glm, n_restarts=6, seed=1)
    recomputed = expected_outs(glm, balls, result["angles"], result["depths"])
    assert result["exp_outs"] == pytest.approx(recomputed, abs=1e-9)


def _effects(alpha, g):
    return {"alpha": np.asarray(alpha, float), "g": np.asarray(g, float),
            "ad_mean": 6.0, "ad_std": 5.0}


def test_fast_objective_matches_pipeline_with_player_effects():
    """帶球員效應時，快速路徑與通用路徑仍須數值等價。"""
    glm = _fitted_glm()
    rng = np.random.default_rng(11)
    balls = _balls(rng.uniform(-50, 50, 60), ev=rng.uniform(60, 110, 60))
    pe = _effects([0.2, -0.1, 0.05, -0.3], [0.15, 0.0, -0.2, 0.1])
    fast = _FastGLMObjective(glm, balls, pe)

    lo = np.array([b[0] for b in ANGLE_BOUNDS + FRAC_BOUNDS])
    hi = np.array([b[1] for b in ANGLE_BOUNDS + FRAC_BOUNDS])
    for _ in range(20):
        angles, depths = params_to_positions(lo + rng.uniform(size=8) * (hi - lo))
        assert fast.expected_outs(angles, depths) == pytest.approx(
            expected_outs(glm, balls, angles, depths, pe), abs=1e-10)


def test_zero_effects_equal_league_average():
    """全零效應必須跟不帶效應完全一樣（聯盟平均野手）。"""
    glm = _fitted_glm()
    balls = _balls(np.array([-30.0, -12.0, 5.0, 18.0, 33.0] * 4))
    pe = _effects([0.0] * 4, [0.0] * 4)
    angles = np.array([30.0, 15.0, -30.0, -15.0])
    depths = np.array([110.0, 145.0, 115.0, 140.0])
    assert expected_outs(glm, balls, angles, depths, pe) == pytest.approx(
        expected_outs(glm, balls, angles, depths), abs=1e-12)


def test_personalized_optimize_keeps_slot_assignment():
    """個人化時槽位綁定野手：好 range 的 SS（g 大）該分到較大的覆蓋責任，
    且結果不做角落重排（angles 順序即 POSITIONS 槽位）。"""
    glm = _fitted_glm()
    balls = _balls(np.array([-35.0, -25.0, -15.0, -5.0, 10.0, 25.0] * 10))
    pe = _effects([0.0] * 4, [0.0, 0.0, -0.4, 0.4])   # SS range 好、3B 差
    result = optimize_infield(balls, glm, n_restarts=6, seed=2, player_effects=pe)
    assert result["exp_outs"] == pytest.approx(
        expected_outs(glm, balls, result["angles"], result["depths"], pe), abs=1e-9)
