"""src/if_dp_optimize.py 純函式部分的單元測試（不碰 DB、不跑優化）。"""
import numpy as np
import pandas as pd
import pytest

from src.if_dataset import SECOND_BASE_Y
from src.if_dp_optimize import (FORCE_SHARE, dp_delta_re, dp_geometry,
                                params_to_positions_dp, positions_to_params_dp)


def _re24():
    """簡化 RE24 表（量級接近實際值即可）。"""
    return {(1, 0, 0, 0): 0.90, (1, 0, 0, 1): 0.52, (1, 0, 0, 2): 0.22,
            (0, 1, 0, 1): 0.66, (0, 1, 0, 2): 0.32,
            (0, 0, 0, 2): 0.10}


def test_dp_delta_re_zero_outs():
    """0 出局：d1=force/at-1st 混合、d2=雙殺到 (0,0,0,2)；d2 < d1 < 0。"""
    d1, d2 = dp_delta_re(_re24(), {}, 0)
    expect_d1 = FORCE_SHARE * (0.52 - 0.90) + (1 - FORCE_SHARE) * (0.66 - 0.90)
    assert d1 == pytest.approx(expect_d1)
    assert d2 == pytest.approx(0.10 - 0.90)
    assert d2 < d1 < 0


def test_dp_delta_re_one_out_ends_inning():
    """1 出局的雙殺=半局結束：d2 = −RE(1,0,0,1)。"""
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
    """2B 站上二壘壘包：pivot_dist=0；spray 0 度且最近野手深度=壘包距離時
    throw_dist_2b=0。"""
    angles4 = np.array([40.0, 0.0, -30.0, -20.0])   # 2B 在 0 度
    depths4 = np.array([90.0, SECOND_BASE_Y, 115.0, 145.0])
    feats = dp_geometry(_balls(), angles4, depths4)
    assert feats["pivot_dist"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert feats["throw_dist_2b"].iloc[0] == pytest.approx(0.0, abs=1e-9)  # spray 0
    assert feats["ad_min"].iloc[0] == pytest.approx(0.0)


def test_dp_geometry_pivot_uses_closer_of_2b_ss():
    """SS 比 2B 靠近壘包時，pivot_dist 取 SS 的距離。"""
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
    """DPScorer 的 numpy 快速路徑必須與兩條 sklearn pipeline 數值等價。"""
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

    # web 端點用的逐球 P(≥1 出局)：與 _probs 的 p1 一致、平均=expected_p1
    p1_public = scorer.per_ball_p1(angles3, depths3)
    np.testing.assert_allclose(p1_public, p1_fast)
    assert scorer.expected_p1(angles3, depths3) == pytest.approx(p1_public.mean())
    assert ((p1_public >= 0) & (p1_public <= 1)).all()


def test_on1b_constants_json_complete_for_deploy():
    """/api/if_optimize DP 分支的離線常數（scripts/precompute_if_on1b_constants.py）
    必須在 repo 裡且欄位齊全——雲端沒有 fielder_positioning_on1b / sprint_speed，
    這個檔缺了 DP 分支會靜默退回無壘況精修。"""
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parent.parent
            / "data" / "precomputed" / "if_on1b_constants.json")
    const = json.loads(path.read_text(encoding="utf-8"))
    assert const["train_years"] == [2023, 2024]   # 須同階段B 模型訓練年
    assert set(const["positions"]) == {"1B", "2B", "3B", "SS"}
    for angle, depth in const["positions"].values():
        assert -50 <= angle <= 50 and 60 <= depth <= 160
    assert 4.0 <= const["runner_hp_to_1b"] <= 5.0
