"""Pure physics/geometry helpers for outfield catch-probability modeling.

Reimplemented from Baseball_Defense_Model_3's src/core/physics.py logic
(same formulas, fresh code — Model_3 itself is left untouched as the thesis archive).
"""
import numpy as np
import pandas as pd

# Statcast 像素座標原點（與 Model_3 的 config.DATA_PROCESSING['physics'] 相同）
_STATCAST_ORIGIN_X: float = 125.42
_STATCAST_ORIGIN_Y: float = 198.27
_GRAVITY_FT_PER_S2: float = 32.174   # 重力加速度 ft/s^2
_CATCH_HEIGHT_FT: float = 0.0        # 接球高度 (ft)，補償真空模型誤差
_MPH_TO_FT_S: float = 1.46667

_CATCH_EVENTS: set[str] = {
    "field_out", "sac_fly", "double_play", "triple_play",
    "field_error", "sac_fly_double_play",
}


def transform_coordinates(hc_x: pd.Series, hc_y: pd.Series,
                           hit_distance_sc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Statcast 像素座標 -> 以本壘為原點的笛卡爾座標（ft）。"""
    angle_rad = np.arctan2(hc_x - _STATCAST_ORIGIN_X, _STATCAST_ORIGIN_Y - hc_y)
    x_coord = hit_distance_sc * np.sin(angle_rad)
    y_coord = hit_distance_sc * np.cos(angle_rad)
    return x_coord, y_coord


def calculate_flight_time(launch_speed: pd.Series, launch_angle: pd.Series,
                           plate_z: pd.Series) -> pd.Series:
    """拋體公式計算飛行時間（秒）。"""
    v0 = launch_speed * _MPH_TO_FT_S
    vy0 = v0 * np.sin(np.radians(launch_angle))
    discriminant = np.maximum(vy0 ** 2 - 2 * _GRAVITY_FT_PER_S2 * (_CATCH_HEIGHT_FT - plate_z), 0)
    return (vy0 + np.sqrt(discriminant)) / _GRAVITY_FT_PER_S2


def mark_caught(events: pd.Series) -> pd.Series:
    """events 是否屬於接殺類事件 -> 0/1。"""
    return events.isin(_CATCH_EVENTS).astype(int)


def polar_to_fielder_xy(avg_norm_start_distance: pd.Series,
                         avg_norm_start_angle: pd.Series) -> tuple[pd.Series, pd.Series]:
    """球員平均站位（距離+角度）-> 直角座標 (fielder_x, fielder_y)。"""
    angle_rad = np.radians(avg_norm_start_angle)
    fielder_x = avg_norm_start_distance * np.sin(angle_rad)
    fielder_y = avg_norm_start_distance * np.cos(angle_rad)
    return fielder_x, fielder_y


def compute_relative_angle(fielder_x: np.ndarray, fielder_y: np.ndarray,
                            ball_x: np.ndarray, ball_y: np.ndarray) -> np.ndarray:
    """守備員相對「背向本壘」的跑動方向角（弧度，-pi~pi）。0=正前方(charge)，±pi=正後方(retreat)。"""
    run_angle = np.arctan2(ball_x - fielder_x, ball_y - fielder_y)
    pos_angle = np.arctan2(fielder_x, fielder_y)
    rel_angle = run_angle - (pos_angle + np.pi)
    rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
    return rel_angle


