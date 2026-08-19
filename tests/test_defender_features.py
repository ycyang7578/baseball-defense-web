from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.defender_features import get_defender_opportunities, mark_official


def _mock_psycopg2_connect():
    """psycopg2.connect(...) is used in the code as `with psycopg2.connect(DSN) as conn:`,
    so the mocked object needs to work as a context manager."""
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    return mock_conn


@patch("src.defender_features.pd.read_sql")
@patch("src.defender_features.psycopg2.connect")
def test_get_defender_opportunities_computes_physics_columns(mock_connect, mock_read_sql):
    mock_connect.return_value = _mock_psycopg2_connect()

    # Hand-craft a single row of known values so the expected physics-derived columns can be computed by hand
    mock_read_sql.return_value = pd.DataFrame({
        "hc_x": [125.42],           # equals _STATCAST_ORIGIN_X -> hit straight up the middle
        "hc_y": [98.27],            # _STATCAST_ORIGIN_Y(198.27) - 100
        "hit_distance_sc": [300.0],
        "launch_speed": [100.0],
        "launch_angle": [30.0],
        "plate_z": [0.0],
        "avg_norm_start_distance": [280.0],
        "avg_norm_start_angle": [0.0],   # fielder is also positioned straight up the middle
        "events": ["field_out"],
        "game_pk": [1],
        "at_bat_number": [1],
        "fielder_id": [12345],
    })

    df = get_defender_opportunities("CF", 2025)

    assert len(df) == 1
    # Hit straight up the middle, fielder also positioned straight up the middle -> x_coord and fielder_x should both be 0
    assert df["x_coord"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["fielder_x"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # Ball lands at 300ft, fielder is at 280ft, same line -> fielder_dist should be the 20ft difference
    assert df["fielder_dist"].iloc[0] == pytest.approx(20.0)
    assert df["caught"].iloc[0] == 1  # field_out is a putout event


@patch("src.defender_features.pd.read_sql")
@patch("src.defender_features.psycopg2.connect")
def test_get_defender_opportunities_returns_empty_df_when_no_rows(mock_connect, mock_read_sql):
    mock_connect.return_value = _mock_psycopg2_connect()
    mock_read_sql.return_value = pd.DataFrame()

    df = get_defender_opportunities("LF", 2025)

    assert df.empty


@patch("src.defender_features.pd.read_sql")
@patch("src.defender_features.psycopg2.connect")
def test_mark_official_flags_rows_present_in_savant_fielding(mock_connect, mock_read_sql):
    mock_connect.return_value = _mock_psycopg2_connect()
    mock_read_sql.return_value = pd.DataFrame({
        "player_id": [111, 222],
        "game_pk": [1, 2],
        "at_bat_number": [1, 1],
    })

    df = pd.DataFrame({
        "fielder_id": [111, 999],
        "game_pk": [1, 1],
        "at_bat_number": [1, 1],
    })

    result = mark_official(df)

    assert result["is_official"].tolist() == [True, False]
