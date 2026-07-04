import numpy as np

from src.stadium_walls import SUPPORTED_TEAMS, get_park_boundary_coords, is_wall_ball


def test_unknown_team_returns_all_false():
    x = np.array([0.0, 1000.0])
    y = np.array([100.0, 500.0])

    flags = is_wall_ball(x, y, "ZZZ")

    assert flags.tolist() == [False, False]


def test_balls_behind_home_plate_are_never_flagged():
    # y <= 0（本壘後方/兩側）的球一律不判定打牆，不管座標多離譜
    x = np.array([-9999.0, 9999.0])
    y = np.array([0.0, -50.0])

    flags = is_wall_ball(x, y, "BOS")

    assert flags.tolist() == [False, False]


def test_ball_far_beyond_any_outfield_wall_is_flagged():
    # 500 呎外野落點，遠超過任何 MLB 球場的圍牆距離
    x = np.array([0.0])
    y = np.array([500.0])

    flags = is_wall_ball(x, y, "BOS")

    assert flags[0] == True  # noqa: E712


def test_ball_shallow_in_play_is_not_flagged():
    # 距本壘 50 呎、正中方向，遠在任何球場的內野/淺外野範圍內
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
    # 32 個代碼對應 30 支球隊：OAK/ATH（運動家）、AZ/ARI（響尾蛇）各是同隊的兩種代碼
    assert len(SUPPORTED_TEAMS) == 32
    assert {"OAK", "ATH"} <= set(SUPPORTED_TEAMS)
    assert {"AZ", "ARI"} <= set(SUPPORTED_TEAMS)
