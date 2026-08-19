import numpy as np

from src.stadium_walls import SUPPORTED_TEAMS, get_park_boundary_coords, is_wall_ball


def test_unknown_team_returns_all_false():
    x = np.array([0.0, 1000.0])
    y = np.array([100.0, 500.0])

    flags = is_wall_ball(x, y, "ZZZ")

    assert flags.tolist() == [False, False]


def test_balls_behind_home_plate_are_never_flagged():
    # Balls with y <= 0 (behind/to the side of home plate) are never flagged as wall balls, no matter how extreme the coordinates
    x = np.array([-9999.0, 9999.0])
    y = np.array([0.0, -50.0])

    flags = is_wall_ball(x, y, "BOS")

    assert flags.tolist() == [False, False]


def test_ball_far_beyond_any_outfield_wall_is_flagged():
    # A landing point 500 feet into the outfield, far beyond any MLB park's wall distance
    x = np.array([0.0])
    y = np.array([500.0])

    flags = is_wall_ball(x, y, "BOS")

    assert flags[0] == True  # noqa: E712


def test_ball_shallow_in_play_is_not_flagged():
    # 50 feet from home plate, straight up the middle, well within the infield/shallow outfield range of any park
    x = np.array([0.0])
    y = np.array([50.0])

    flags = is_wall_ball(x, y, "LAD")

    assert flags[0] == False  # noqa: E712


def test_get_park_boundary_coords_returns_xy_dicts_for_supported_team():
    coords = get_park_boundary_coords("NYY")

    assert coords is not None
    assert len(coords) >= 3
    assert all("x" in c and "y" in c for c in coords)


def test_get_park_boundary_coords_returns_none_for_unknown_team():
    assert get_park_boundary_coords("ZZZ") is None


def test_supported_teams_covers_30_mlb_clubs_with_2_alias_codes():
    # 32 codes map to 30 teams: OAK/ATH (Athletics) and AZ/ARI (Diamondbacks) are each two codes for the same team
    assert len(SUPPORTED_TEAMS) == 32
    assert {"OAK", "ATH"} <= set(SUPPORTED_TEAMS)
    assert {"AZ", "ARI"} <= set(SUPPORTED_TEAMS)
