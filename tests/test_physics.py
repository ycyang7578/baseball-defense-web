import numpy as np
import pandas as pd
import pytest

from src.physics import (
    calculate_flight_time,
    compute_relative_angle,
    mark_caught,
    polar_to_fielder_xy,
    transform_coordinates,
)


def test_mark_caught_recognizes_out_events():
    events = pd.Series(["field_out", "single", "sac_fly", "double"])
    result = mark_caught(events)
    assert result.tolist() == [1, 0, 1, 0]


def test_transform_coordinates_straight_to_center():
    # hc_x == _X0 且 hc_y < _Y0：球打向正中外野，x_coord 應為 0
    hc_x = pd.Series([125.42])
    hc_y = pd.Series([98.27])  # _Y0(198.27) - 100
    hit_distance_sc = pd.Series([300.0])

    x_coord, y_coord = transform_coordinates(hc_x, hc_y, hit_distance_sc)

    assert x_coord[0] == pytest.approx(0.0, abs=1e-9)
    assert y_coord[0] == pytest.approx(300.0)


def test_calculate_flight_time_matches_vacuum_projectile_formula():
    # plate_z = 0（接球高度與擊球高度相同）時，飛行時間退化為經典拋體公式 t = 2*v0*sin(theta)/g
    launch_speed = pd.Series([100.0])   # mph
    launch_angle = pd.Series([35.0])    # degrees
    plate_z = pd.Series([0.0])

    result = calculate_flight_time(launch_speed, launch_angle, plate_z)

    v0 = 100.0 * 1.46667
    expected = 2 * v0 * np.sin(np.radians(35.0)) / 32.174
    assert result[0] == pytest.approx(expected)


def test_calculate_flight_time_clamps_negative_discriminant_to_zero():
    # launch_speed 小、launch_angle 小、plate_z 為負值時，
    # vy0**2 - 2*G*(H1-plate_z) 在不 clamp 的情況下會是負數（此處手算約 -64.35）
    # np.maximum(..., 0) 應把它保底在 0，flight_time = vy0 / G，而不是 NaN
    launch_speed = pd.Series([1.0])
    launch_angle = pd.Series([1.0])
    plate_z = pd.Series([-1.0])

    result = calculate_flight_time(launch_speed, launch_angle, plate_z)

    v0 = 1.0 * 1.46667
    vy0 = v0 * np.sin(np.radians(1.0))
    expected = vy0 / 32.174  # sqrt(clamp 後的 0) = 0

    assert not np.isnan(result[0])
    assert result[0] == pytest.approx(expected)


def test_polar_to_fielder_xy_at_zero_angle_points_straight_out():
    distance = pd.Series([250.0])
    angle = pd.Series([0.0])

    x, y = polar_to_fielder_xy(distance, angle)

    assert x[0] == pytest.approx(0.0, abs=1e-9)
    assert y[0] == pytest.approx(250.0)


def test_polar_to_fielder_xy_at_ninety_degrees_points_to_the_side():
    distance = pd.Series([250.0])
    angle = pd.Series([90.0])

    x, y = polar_to_fielder_xy(distance, angle)

    assert x[0] == pytest.approx(250.0)
    assert y[0] == pytest.approx(0.0, abs=1e-9)


def test_compute_relative_angle_ball_beyond_fielder_is_retreat():
    # 球落點比守備員更遠離本壘、方向相同 -> 守備員要向後退（±pi）
    fielder_x = np.array([0.0])
    fielder_y = np.array([200.0])
    ball_x = np.array([0.0])
    ball_y = np.array([300.0])

    rel = compute_relative_angle(fielder_x, fielder_y, ball_x, ball_y)

    assert abs(rel[0]) == pytest.approx(np.pi)


def test_compute_relative_angle_ball_between_fielder_and_home_is_charge():
    # 球落點比守備員更靠近本壘、方向相同 -> 守備員要向前衝（0）
    fielder_x = np.array([0.0])
    fielder_y = np.array([200.0])
    ball_x = np.array([0.0])
    ball_y = np.array([100.0])

    rel = compute_relative_angle(fielder_x, fielder_y, ball_x, ball_y)

    assert rel[0] == pytest.approx(0.0, abs=1e-9)
