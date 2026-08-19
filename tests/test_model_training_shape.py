from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.model_training import ALL_COLS, build_model, load_training_data


def _synthetic_training_df(n_per_player=10):
    """Build synthetic defensive-opportunity data, skipping the DB/PyMC sampling step, to verify only the computation graph's shape."""
    players = ["Player A", "Player B", "Player C"]
    rows = []
    rng = np.random.default_rng(0)
    for p in players:
        for _ in range(n_per_player):
            rows.append({
                "speed": rng.uniform(10, 40),
                "cos_angle": rng.uniform(-1, 1),
                "sin_angle": rng.uniform(-1, 1),
                "fielder_dist": rng.uniform(0, 50),
                "caught": rng.integers(0, 2),
                "name_fielder": p,
            })
    return pd.DataFrame(rows)


def test_build_model_does_not_trigger_sampling_and_has_expected_structure():
    df = _synthetic_training_df()
    scaler = StandardScaler().fit(df[ALL_COLS])

    model = build_model(df, scaler)

    # build_model only constructs the computation graph and should not trigger any sampling
    # (there's no observable side effect besides free_RVs; this mainly verifies the return
    # type and that all expected variables are present)
    import pymc as pm
    assert isinstance(model, pm.Model)

    var_names = set(model.named_vars.keys())
    expected_vars = {
        "mu_alpha", "mu_beta_speed", "mu_beta_cos", "mu_beta_sin", "mu_beta_dist",
        "alpha", "beta_speed", "beta_cos", "beta_sin", "beta_dist", "y",
    }
    assert expected_vars <= var_names

    assert list(model.coords["player"]) == ["Player A", "Player B", "Player C"]
    assert len(model.coords["obs"]) == len(df)


def test_load_training_data_renames_speed_and_drops_na_and_concats_years():
    # The columns actually returned by get_defender_opportunities only include required_speed, not speed
    year_2024_df = pd.DataFrame({
        "required_speed": [25.0], "cos_angle": [0.5], "sin_angle": [0.5],
        "fielder_dist": [30.0], "caught": [1], "name_fielder": ["Player A"],
    })
    year_2025_df = pd.DataFrame({
        # Row 2's cos_angle is NaN, which should be dropped by dropna
        "required_speed": [20.0, 15.0], "cos_angle": [0.1, np.nan], "sin_angle": [0.2, 0.3],
        "fielder_dist": [10.0, 20.0], "caught": [0, 1], "name_fielder": ["Player B", "Player C"],
    })

    with patch("src.model_training.get_defender_opportunities") as mock_get:
        mock_get.side_effect = [year_2024_df, year_2025_df]

        df = load_training_data("CF", [2024, 2025])

    # required_speed should be renamed to speed
    assert "speed" in df.columns
    assert "required_speed" not in df.columns
    # 3 rows in, 1 row dropped by dropna due to cos_angle=NaN, leaving 2 rows
    assert len(df) == 2
    assert sorted(df["name_fielder"].tolist()) == ["Player A", "Player B"]
    assert mock_get.call_count == 2
