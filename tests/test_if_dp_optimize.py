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
    """全零效應必須跟不帶效應完全一樣（聯盟平均野手）。"""
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
    """效應＝最近野手的 α_j + g_j×ad_z 加在階段1 logit（階段2 不掛），
    與不帶效應的 p1 手算對照。"""
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
    # 階段2 不掛效應：p2 必須不變
    np.testing.assert_allclose(with_pe._probs(angles3, depths3)[1],
                               base._probs(angles3, depths3)[1], atol=1e-12)


def test_dp_optimize_personalized_keeps_slot_assignment():
    """個人化時 3B/SS 槽位綁定野手、不做標籤重排，且回傳分數與帶效應
    scorer 一致。"""
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
    # bounds 仍須成立（2B 右側、3B/SS 左側）
    assert 1.0 <= res["angles"][0] <= 44.0
    assert all(-44.0 <= a <= -1.0 for a in res["angles"][1:])


def test_dp_anchored_refine_never_worse_than_anchor():
    """個人化錨定精修（anchored_starts：錨點＋抖動）必須 ≥ 錨點本身的分數——
    零效應解常卡在 kink 上讓 L-BFGS-B 原地失敗，抖動起點要能撿到附近的改善。"""
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
    # 錨定語意：位移該是小幅修正，不是跳去別的解家族
    assert np.abs(res["angles"] - base["angles"]).max() < 5.0
    assert np.abs(res["depths"] - base["depths"]).max() < 15.0


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
