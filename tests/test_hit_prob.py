from pathlib import Path

import pytest

from src.hit_prob import load_hit_prob, predict_hit_probs, predict_hit_probs_batch

_DATA_DIR = Path(__file__).parent.parent / "data" / "precomputed"


@pytest.fixture(scope="module")
def bundle():
    return load_hit_prob(_DATA_DIR)


def test_predict_hit_probs_returns_valid_probability_distribution(bundle):
    probs = predict_hit_probs(
        bundle,
        launch_speed=95.0,
        launch_angle=20.0,
        spray_angle=0.0,
        stand="R",
    )

    assert set(probs.keys()) == {"1B", "2B", "3B"}
    for p in probs.values():
        assert 0.0 <= p <= 1.0
    assert probs["1B"] + probs["2B"] + probs["3B"] == pytest.approx(1.0)


def test_predict_hit_probs_unknown_stand_falls_back_to_uniform(bundle):
    probs = predict_hit_probs(
        bundle,
        launch_speed=95.0,
        launch_angle=20.0,
        spray_angle=0.0,
        stand="does-not-exist",
    )

    assert probs == {"1B": pytest.approx(1 / 3), "2B": pytest.approx(1 / 3), "3B": pytest.approx(1 / 3)}


def test_predict_hit_probs_batch_matches_single_prediction(bundle):
    import pandas as pd

    df = pd.DataFrame({
        "launch_speed": [95.0],
        "launch_angle": [20.0],
        "spray_angle": [0.0],
        "stand": ["R"],
    })

    batch_result = predict_hit_probs_batch(bundle, df)
    single_result = predict_hit_probs(
        bundle, launch_speed=95.0, launch_angle=20.0, spray_angle=0.0, stand="R"
    )

    assert batch_result.shape[0] == 1
    row_sum = batch_result[0].sum()
    assert row_sum == pytest.approx(1.0)
    # batch 與單筆預測應該對同一筆輸入給出相同結果
    assert batch_result[0][0] == pytest.approx(single_result["1B"], abs=1e-6)
