"""Unit tests for the feature engineering in src/if_dataset.py (pure functions, no DB access)."""
import numpy as np
import pandas as pd
import pytest

from src.if_dataset import (HOME_X, HOME_Y, MPH_TO_FTS, SECOND_BASE_Y,
                            attach_dp_features, attach_features)


def _gb_row(**over):
    row = {
        "game_year": 2024, "batter": 1, "stand": "R",
        "hc_x": HOME_X, "hc_y": HOME_Y - 100,  # straight up the middle (spray 0 degrees)
        "launch_speed": 90.0, "launch_angle": -5.0,
        "events": "field_out", "if_fielding_alignment": "Standard",
        "bases_empty": True,
        "fielder_3": 13, "fielder_4": 14, "fielder_5": 15, "fielder_6": 16,
    }
    row.update(over)
    return row


def _positioning():
    rows = [(13, "1B", 110, 35), (14, "2B", 150, 20), (15, "3B", 115, -35),
            (16, "SS", 145, -20)]
    return pd.DataFrame([{"fielder_id": f, "season": 2024, "position": p,
                          "depth": d, "angle": a} for f, p, d, a in rows])


def _run_speed():
    return pd.DataFrame([{"player_id": 1, "season": 2024, "hp_to_1b": 4.3}])


def test_spray_angle_sign_convention():
    """hc_x greater than home plate's x = first-base line side = positive angle (same convention as Savant's positioning angle)."""
    gb = pd.DataFrame([_gb_row(hc_x=HOME_X + 50, hc_y=HOME_Y - 50),
                       _gb_row(hc_x=HOME_X - 50, hc_y=HOME_Y - 50)])
    df = attach_features(gb, _positioning(), _run_speed())
    assert df["spray_deg"].iloc[0] == pytest.approx(45.0)
    assert df["spray_deg"].iloc[1] == pytest.approx(-45.0)


def test_nearest_fielder_and_ball_time():
    """At spray angle 0, the nearest fielder is the middle infielder at ±20 degrees; b_t = depth/(EV x 1.46667)."""
    df = attach_features(pd.DataFrame([_gb_row()]), _positioning(), _run_speed())
    assert df["nearest_pos"].iloc[0] in ("2B", "SS")
    assert df["ad_min"].iloc[0] == pytest.approx(20.0)
    expected_bt = df["near_depth"].iloc[0] / (90.0 * MPH_TO_FTS)
    assert df["ball_time"].iloc[0] == pytest.approx(expected_bt)
    assert df["lat_ft"].iloc[0] == pytest.approx(
        df["near_depth"].iloc[0] * np.sin(np.radians(20.0)))


def test_throw_dist_zero_at_first_base():
    """When hit toward first base and the fielder's depth is exactly 90 feet, the interception point is right at first base, so throw distance is approximately 0."""
    pos = _positioning()
    pos.loc[pos["position"] == "1B", ["depth", "angle"]] = [90, 45]
    gb = pd.DataFrame([_gb_row(hc_x=HOME_X + 50, hc_y=HOME_Y - 50)])  # spray +45
    df = attach_features(gb, pos, _run_speed())
    assert df["nearest_pos"].iloc[0] == "1B"
    assert df["throw_dist"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_run_speed_median_fill_flag():
    """For a batter with no run-speed data: hp_to_1b is filled with that season's median and has_run_speed=False."""
    gb = pd.DataFrame([_gb_row(), _gb_row(batter=999)])
    df = attach_features(gb, _positioning(), _run_speed())
    known = df[df["batter"] == 1].iloc[0]
    filled = df[df["batter"] == 999].iloc[0]
    assert known["has_run_speed"] and known["hp_to_1b"] == pytest.approx(4.3)
    assert not filled["has_run_speed"]
    assert filled["hp_to_1b"] == pytest.approx(4.3)  # the median equals the only known value


def test_label_and_extreme_spray_filter():
    """Out label is correct; balls with |spray| > 55 degrees are removed."""
    gb = pd.DataFrame([
        _gb_row(events="single"),
        _gb_row(events="grounded_into_double_play"),
        _gb_row(hc_x=HOME_X + 100, hc_y=HOME_Y - 20),  # ~79 degrees, foul-territory noise
    ])
    df = attach_features(gb, _positioning(), _run_speed())
    assert len(df) == 2
    assert df["is_out"].tolist() == [0, 1]


# ── Phase B: double-play situation features (attach_dp_features) ──────────────────────

def _on1b_row(**over):
    row = _gb_row(bases_empty=False, on_1b=77, outs_when_up=0)
    row.update(over)
    return row


def _run_speed_with_runner():
    return pd.DataFrame([{"player_id": 1, "season": 2024, "hp_to_1b": 4.3},
                         {"player_id": 77, "season": 2024, "hp_to_1b": 4.6}])


def test_dp_n_outs_label():
    """n_outs: double play=2, force/put-out=1, hit and fielders_choice (no outs)=0."""
    gb = pd.DataFrame([_on1b_row(events=e) for e in
                       ("grounded_into_double_play", "force_out", "field_out",
                        "single", "fielders_choice")])
    df = attach_dp_features(
        attach_features(gb, _positioning(), _run_speed_with_runner()),
        _run_speed_with_runner())
    assert df["n_outs"].tolist() == [2, 1, 1, 0, 0]


def test_dp_pivot_dist_uses_closer_middle_infielder():
    """pivot_dist = the smaller of 2B/SS's distance to second base; 0 when the fielder is standing on the bag."""
    pos = _positioning()
    pos.loc[pos["position"] == "2B", ["depth", "angle"]] = [SECOND_BASE_Y, 0]
    df = attach_dp_features(
        attach_features(pd.DataFrame([_on1b_row()]), pos, _run_speed_with_runner()),
        _run_speed_with_runner())
    assert df["pivot_dist"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_dp_throw_dist_2b_zero_when_intercept_at_bag():
    """When spray is 0 degrees and the nearest fielder's depth equals the distance to second base, the interception point is right on second base."""
    pos = _positioning()
    pos.loc[pos["position"] == "2B", ["depth", "angle"]] = [SECOND_BASE_Y, 5]
    pos.loc[pos["position"] == "SS", ["depth", "angle"]] = [SECOND_BASE_Y, -5]
    df = attach_dp_features(
        attach_features(pd.DataFrame([_on1b_row()]), pos, _run_speed_with_runner()),
        _run_speed_with_runner())
    assert df["throw_dist_2b"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_dp_runner_speed_join_and_fill():
    """runner_hp_to_1b joins to the runner on first (not the batter); unknown runners are filled with that season's median."""
    gb = pd.DataFrame([_on1b_row(), _on1b_row(on_1b=888)])
    df = attach_dp_features(
        attach_features(gb, _positioning(), _run_speed_with_runner()),
        _run_speed_with_runner())
    known, filled = df.iloc[0], df.iloc[1]
    assert known["has_runner_speed"] and known["runner_hp_to_1b"] == pytest.approx(4.6)
    assert not filled["has_runner_speed"]
    assert filled["runner_hp_to_1b"] == pytest.approx(4.6)  # the median equals the only known value
    assert known["hp_to_1b"] == pytest.approx(4.3)          # batter speed is unaffected
