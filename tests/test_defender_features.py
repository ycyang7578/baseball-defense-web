from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.defender_features import get_defender_opportunities, mark_official


def _mock_psycopg2_connect():
    """psycopg2.connect(...) 在程式碼裡是用 `with psycopg2.connect(DSN) as conn:`，
    所以 mock 出來的物件要能當 context manager 用。"""
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    return mock_conn


@patch("src.defender_features.pd.read_sql")
@patch("src.defender_features.psycopg2.connect")
def test_get_defender_opportunities_computes_physics_columns(mock_connect, mock_read_sql):
    mock_connect.return_value = _mock_psycopg2_connect()

    # 手造一列已知數值，方便手算期望的 physics 衍生欄位
    mock_read_sql.return_value = pd.DataFrame({
        "hc_x": [125.42],           # 等於 _STATCAST_ORIGIN_X -> 打向正中方向
        "hc_y": [98.27],            # _STATCAST_ORIGIN_Y(198.27) - 100
        "hit_distance_sc": [300.0],
        "launch_speed": [100.0],
        "launch_angle": [30.0],
        "plate_z": [0.0],
        "avg_norm_start_distance": [280.0],
        "avg_norm_start_angle": [0.0],   # 守備員也站在正中方向
        "events": ["field_out"],
        "game_pk": [1],
        "at_bat_number": [1],
        "fielder_id": [12345],
    })

    df = get_defender_opportunities("CF", 2025)

    assert len(df) == 1
    # 打向正中、守備員也站正中 -> x_coord 和 fielder_x 都應該是 0
    assert df["x_coord"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["fielder_x"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # 球落點 300ft，守備員站 280ft，同一直線上 -> fielder_dist 應該是差值 20ft
    assert df["fielder_dist"].iloc[0] == pytest.approx(20.0)
    assert df["caught"].iloc[0] == 1  # field_out 是接殺事件


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
