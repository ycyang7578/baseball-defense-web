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

# 用 repo 裡真的已訓練好的模型檔 + 已 precompute 的資料，不連 DB、不跑 PyMC
MODELS_DIR = Path(__file__).parent.parent / "models" / "2025"
RE24_DIR = Path(__file__).parent.parent / "data" / "precomputed"

# n_restarts 從正式管線的 100 降到 10：實測同一組合成球在 5~100 之間 RF/CF/LF 的
# 左右排序穩定不變（只有 LF 精確座標在 <30 時會卡不同 local optimum，但方向性結論不受影響）
N_RESTARTS_TEST = 10


def _synthetic_right_field_balls(n=8):
    """8 顆集中在右外野深遠處的合成球，用來測試優化器是否把 RF 移向球群方向。"""
    balls = pd.DataFrame({
        "ball_x": np.full(n, 200.0),
        "ball_y": np.full(n, 280.0),
        "flight_time": np.full(n, 4.0),
    })
    hit_probs = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))  # 全部視為一壘安打，簡化 w_j
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

    # 球群集中在右外野深遠處 -> RF 應該比 CF 更靠右，CF 應該比 LF 更靠右
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
    assert result["n_wall_balls"] == 0  # 沒指定 home_team，不做打牆過濾


def test_optimize_positions_beats_a_deliberately_bad_guess():
    """驗證優化器真的有在降低目標值，不是隨便回傳起點。"""
    balls, hit_probs = _synthetic_right_field_balls()

    result = optimize_positions(
        batter_id=0, on_1b=0, on_2b=0, on_3b=0, outs=0,
        years=[2025], models_dir=MODELS_DIR, re24_dir=RE24_DIR,
        n_restarts=N_RESTARTS_TEST, seed=42, balls=balls, hit_probs=hit_probs,
    )

    # 重建跟 optimize_positions 內部完全相同的 ctx，才能用 objective_re24 公平比較
    _, delta_re = load_re24(RE24_DIR)
    w_j = compute_w_j(balls, None, delta_re, 0, 0, 0, 0, hit_probs=hit_probs)
    assert (w_j > 0).all()  # 確認這組合成資料沒有被 optimize_positions 內部的 w_j>0 過濾掉任何球

    scalers, mus = {}, {}
    for pos in POSITIONS:
        scalers[pos], mus[pos] = load_model_params(pos, MODELS_DIR)
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

    # 明顯不合理的站位：三個守備員都塞在扇形最左邊界，遠離球群所在的右外野深遠處
    bad_polar = np.array([150.0, -45.0, 150.0, -22.5, 150.0, -22.5])
    bad_flat = _polar_to_xy(bad_polar)
    bad_obj = objective_re24(bad_flat, ctx)

    assert optimized_obj < bad_obj


def test_optimize_positions_warm_start_reaches_same_optimum_with_zero_random_restarts():
    """warm_start_xy 應該可以獨立當唯一起點用（n_restarts=0），驗證它真的有被拿去 minimize，不是被忽略。"""
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
    """模擬 no_park -> with_park 的實際用法：球集只差幾顆球（模擬拿掉打牆球），
    用 no_park 的解 warm start with_park，n_restarts=2（1 隨機+1 warm start）也要能
    達到跟 10 個隨機起點相近的品質。"""
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
