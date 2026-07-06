"""src/if_model.py 的單元測試（合成資料，不碰 DB）。"""
import numpy as np
import pandas as pd

from src.if_model import (DIFFICULTY_FEATURES, OPTIMIZER_FEATURES,
                          FielderGeometryFeatures, make_difficulty_gbm,
                          make_optimizer_glm)


def _synthetic(n=400, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ad_min": rng.uniform(0, 25, n),
        "ball_time": rng.uniform(0.5, 2.5, n),
        "launch_angle": rng.uniform(-60, 10, n),
        "launch_speed": rng.uniform(60, 110, n),
        "throw_dist": rng.uniform(20, 130, n),
        "hp_to_1b": rng.uniform(4.0, 4.8, n),
        "stand_R": rng.integers(0, 2, n),
        "spray_deg": rng.uniform(-50, 50, n),
    })
    # 角差越大越難出局的合成標籤
    p = 1 / (1 + np.exp(-(1.2 - 0.08 * df["ad_min"])))
    df["is_out"] = (rng.uniform(size=n) < p).astype(int)
    return df


def test_feature_builder_shape_and_determinism():
    df = _synthetic()
    fb = FielderGeometryFeatures().fit(df[OPTIMIZER_FEATURES])
    X1 = fb.transform(df[OPTIMIZER_FEATURES])
    X2 = fb.transform(df[OPTIMIZER_FEATURES])
    assert X1.shape[0] == len(df)
    assert np.array_equal(X1, X2)
    assert np.isfinite(X1).all()


def test_optimizer_glm_responds_to_fielder_geometry():
    """counterfactual 檢查:同一顆球，角差變大 → 出局率必須下降。"""
    df = _synthetic()
    glm = make_optimizer_glm().fit(df[OPTIMIZER_FEATURES], df["is_out"])
    ball = df[OPTIMIZER_FEATURES].iloc[[0]].copy()
    near = ball.assign(ad_min=2.0)
    far = ball.assign(ad_min=22.0)
    p_near = glm.predict_proba(near)[0, 1]
    p_far = glm.predict_proba(far)[0, 1]
    assert p_near > p_far


def test_difficulty_gbm_probability_range():
    df = _synthetic()
    gbm = make_difficulty_gbm().fit(df[DIFFICULTY_FEATURES], df["is_out"])
    p = gbm.predict_proba(df[DIFFICULTY_FEATURES])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
