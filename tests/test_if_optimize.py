"""src/if_optimize.py 的單元測試（合成模型/資料，不碰 DB）。"""
import numpy as np
import pandas as pd
import pytest

from src.if_dataset import HOME_X, HOME_Y, attach_features
from src.if_optimize import (ANGLE_BOUNDS, MIN_DEPTH, POSITIONS, dirt_max_depth,
                             expected_outs, geometry_features, optimize_infield,
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
