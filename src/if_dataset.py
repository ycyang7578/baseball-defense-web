"""Ground ball dataset construction: feature engineering for the infield out probability model.

Design basis:
- Melville (2024) §3.1 — a_d (minimum angular difference between the ball and the nearest
  infielder), b_t (time for the ball to reach the fielder = fielder depth ÷ exit velocity,
  ignoring deceleration)
- Tango, Introducing Infield OAA (MLB Technology Blog, 2020) — an out is a three-way race:
  interception + throw vs. the runner's time to first base (runner uses season-average
  speed, not the actual speed on that play)

Coordinate convention (matches Savant fielder_positioning): angle -45°=third base line,
0°=straight up the middle, +45°=first base line; depth is distance from home plate (feet).

Known limitation: for ground balls, hc_x/hc_y record "where the ball was fielded," not
where it landed (an out is recorded where the infielder intercepts it, while a base hit
is recorded where it's picked up in the outfield). So we only take the spray angle from
the position data (angle is roughly invariant along a straight roll) and don't use depth.
"""
import numpy as np
import pandas as pd
import psycopg2

from src.config import DSN

# Ground ball outcome labels: out = defense converted the ball into at least one out (Tango: only look at the first out of the play)
OUT_EVENTS: tuple[str, ...] = ("field_out", "force_out", "grounded_into_double_play",
                                "double_play", "fielders_choice_out")
NONOUT_EVENTS: tuple[str, ...] = ("single", "double", "triple", "field_error")

# Stage B (runner on 1st, <2 outs): the outcome is upgraded to "how many outs were recorded."
# fielders_choice was spot-checked row by row against des (2026-07-13) = batter reaches base,
# all runners advance, no out recorded; only fielders_choice_out records an out.
ON1B_EVENT_OUTS: dict[str, int] = {
    "grounded_into_double_play": 2, "double_play": 2,
    "force_out": 1, "field_out": 1, "fielders_choice_out": 1,
    "single": 0, "double": 0, "triple": 0, "field_error": 0, "fielders_choice": 0,
}

# Home plate position in Savant's hc_x/hc_y coordinate system
HOME_X: float = 125.42
HOME_Y: float = 198.27
MPH_TO_FTS: float = 1.46667          # exit velocity mph -> ft/s
FIRST_BASE_R: float = 90.0           # distance from home plate to first base (feet)
FIRST_BASE_DEG: float = 45.0         # angle of first base relative to home plate

INFIELD_COLS: dict[str, str] = {"fielder_3": "1B", "fielder_4": "2B",
                                 "fielder_5": "3B", "fielder_6": "SS"}
INFIELD_POSITIONS: tuple[str, ...] = tuple(INFIELD_COLS.values())


def _polar_to_xy(
    radius_ft: np.ndarray | float,
    angle_deg: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """(depth, angle) -> planar coordinates; x is positive toward the first base line, y is positive toward center field."""
    angle_rad = np.radians(angle_deg)
    return radius_ft * np.sin(angle_rad), radius_ft * np.cos(angle_rad)


def fetch_raw_gb(years: list[int]) -> pd.DataFrame:
    """Fetch non-bunt ground balls with a clear outcome label for the given years (including base-state and alignment columns)."""
    events = OUT_EVENTS + NONOUT_EVENTS
    # ORDER BY makes the returned row order deterministic (without it, the GBM's internal
    # early-stopping validation split wouldn't be reproducible)
    sql = f"""
        SELECT game_year, batter, stand, hc_x, hc_y, launch_speed, launch_angle,
               events, if_fielding_alignment, hit_location,
               (on_1b IS NULL AND on_2b IS NULL AND on_3b IS NULL) AS bases_empty,
               fielder_3, fielder_4, fielder_5, fielder_6
        FROM statcast
        WHERE bb_type = 'ground_ball'
          AND game_year = ANY(%(years)s)
          AND hc_x IS NOT NULL AND launch_speed IS NOT NULL
          AND events IN {events}
          AND des NOT ILIKE '%%bunt%%'
        ORDER BY game_year, batter, hc_x, hc_y, launch_speed
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn, params={"years": list(years)})


def fetch_positioning(years: list[int]) -> pd.DataFrame:
    """Season-average positioning for the four infield positions (already deduplicated on load by (player, season, position))."""
    sql = """
        SELECT fielder_id, season, position,
               avg_norm_start_distance AS depth, avg_norm_start_angle AS angle
        FROM fielder_positioning
        WHERE position IN %(pos)s AND season = ANY(%(years)s)
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn,
                           params={"pos": INFIELD_POSITIONS, "years": list(years)})


def fetch_positioning_on1b(years: list[int]) -> pd.DataFrame:
    """Season-average positioning split by "runner on first" (fielder_positioning_on1b, available from 2023 on).

    Stage B's geometric proxies need this table: positioning shifts systematically when
    a runner is on first (1B holding the runner moves -26 to -35 ft, 2B/SS play -3 to
    -4.4 ft shallower for the double play), so using the all-situations average would
    compute the geometric features incorrectly."""
    sql = """
        SELECT fielder_id, season, position,
               avg_norm_start_distance AS depth, avg_norm_start_angle AS angle
        FROM fielder_positioning_on1b
        WHERE position IN %(pos)s AND season = ANY(%(years)s)
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn,
                           params={"pos": INFIELD_POSITIONS, "years": list(years)})


def fetch_raw_gb_on1b(years: list[int]) -> pd.DataFrame:
    """Non-bunt ground balls with a runner on first only and <2 outs (Stage B's main double-play scope).

    Compared to fetch_raw_gb, this also carries on_1b (runner id, for joining runner speed)
    and outs_when_up, and the event set includes fielders_choice (0 outs; this event doesn't
    occur with the bases empty)."""
    events = tuple(ON1B_EVENT_OUTS)
    sql = f"""
        SELECT game_year, batter, stand, hc_x, hc_y, launch_speed, launch_angle,
               events, if_fielding_alignment, hit_location,
               on_1b, outs_when_up,
               (on_1b IS NULL AND on_2b IS NULL AND on_3b IS NULL) AS bases_empty,
               fielder_3, fielder_4, fielder_5, fielder_6
        FROM statcast
        WHERE bb_type = 'ground_ball'
          AND game_year = ANY(%(years)s)
          AND hc_x IS NOT NULL AND launch_speed IS NOT NULL
          AND on_1b IS NOT NULL AND on_2b IS NULL AND on_3b IS NULL
          AND outs_when_up < 2
          AND events IN {events}
          AND des NOT ILIKE '%%bunt%%'
        ORDER BY game_year, batter, hc_x, hc_y, launch_speed
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn, params={"years": list(years)})


def fetch_run_speed(years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(
            "SELECT player_id, season, hp_to_1b FROM sprint_speed WHERE season = ANY(%(years)s)",
            conn, params={"years": list(years)})


def attach_features(gb: pd.DataFrame, positioning: pd.DataFrame,
                    run_speed: pd.DataFrame) -> pd.DataFrame:
    """Join positioning/run speed onto ground balls and compute the model features. Pure function (no DB access), for easy testing.

    Feature columns produced:
    - spray_deg      horizontal spray angle of the batted ball (-45=third base line)
    - ad_min         minimum absolute angular difference to the nearest infielder (degrees)
    - near_depth     depth of the nearest infielder (feet)
    - lat_ft         lateral distance from the nearest infielder to the ball's path (feet) = depth x sin(ad_min)
    - ball_time      b_t: time for the ball to reach the nearest infielder's depth (seconds, ignoring deceleration)
    - throw_dist     distance from the interception point (at the nearest infielder's depth) to first base (feet)
    - hp_to_1b       batter's home-to-first time in seconds (season average; missing values filled with that year's median)
    - has_run_speed  whether hp_to_1b is an actual measured value (False = median-filled)
    """
    df = gb.copy()
    df["is_out"] = df["events"].isin(OUT_EVENTS).astype(int)
    df["spray_deg"] = np.degrees(
        np.arctan2(df["hc_x"] - HOME_X, HOME_Y - df["hc_y"]))
    # More than ~10 degrees outside the foul lines is mostly recording noise (Melville uses [-55,55] for the same reason)
    df = df[df["spray_deg"].abs() <= 55].copy()

    pos_map = positioning.set_index(["fielder_id", "season", "position"])[["depth", "angle"]]
    for col, posname in INFIELD_COLS.items():
        idx = pd.MultiIndex.from_arrays(
            [df[col], df["game_year"], np.repeat(posname, len(df))])
        joined = pos_map.reindex(idx).to_numpy()
        df[f"{posname}_depth"] = joined[:, 0]
        df[f"{posname}_angle"] = joined[:, 1]

    have_all = df[[f"{p}_angle" for p in INFIELD_POSITIONS]].notna().all(axis=1)
    df = df[have_all].copy()

    angles = df[[f"{p}_angle" for p in INFIELD_POSITIONS]].to_numpy(float)
    depths = df[[f"{p}_depth" for p in INFIELD_POSITIONS]].to_numpy(float)
    dtheta = np.abs(angles - df["spray_deg"].to_numpy(float)[:, None])
    nearest = dtheta.argmin(axis=1)
    rows = np.arange(len(df))
    df["nearest_pos"] = np.array(INFIELD_POSITIONS)[nearest]
    df["ad_min"] = dtheta[rows, nearest]
    df["near_depth"] = depths[rows, nearest]
    df["lat_ft"] = df["near_depth"] * np.sin(np.radians(df["ad_min"]))
    df["ball_time"] = df["near_depth"] / (df["launch_speed"] * MPH_TO_FTS)

    # Distance from the interception point (along the ball's path, at the nearest fielder's depth) to first base
    ix, iy = _polar_to_xy(df["near_depth"].to_numpy(float),
                          df["spray_deg"].to_numpy(float))
    bx, by = _polar_to_xy(FIRST_BASE_R, FIRST_BASE_DEG)
    df["throw_dist"] = np.hypot(ix - bx, iy - by)

    rs_map = run_speed.set_index(["player_id", "season"])["hp_to_1b"]
    idx = pd.MultiIndex.from_arrays([df["batter"], df["game_year"]])
    df["hp_to_1b"] = rs_map.reindex(idx).to_numpy()
    df["has_run_speed"] = df["hp_to_1b"].notna()
    year_med = df.groupby("game_year")["hp_to_1b"].transform("median")
    df["hp_to_1b"] = df["hp_to_1b"].fillna(year_med)

    df["stand_R"] = (df["stand"] == "R").astype(int)
    # launch_angle has a small number of missing values (tracking gaps), filled with that year's median; launch_speed is already forced non-null in the SQL
    la_med = df.groupby("game_year")["launch_angle"].transform("median")
    df["launch_angle"] = df["launch_angle"].fillna(la_med)
    return df


SECOND_BASE_X: float = 0.0
SECOND_BASE_Y: float = 90.0 * np.sqrt(2.0)  # second base bag


def attach_dp_features(df: pd.DataFrame, run_speed: pd.DataFrame) -> pd.DataFrame:
    """Additional Stage B (double-play scenario) features. Pure function, applied after attach_features.

    - n_outs           number of outs recorded on this play (0/1/2, ON1B_EVENT_OUTS)
    - throw_dist_2b    distance from the interception point (along the ball's path, at the
                       nearest fielder's depth) to second base (feet) -- with a runner on
                       first, the primary throw target is the force at second; this is the
                       same kind of counterfactual-valid geometry as throw_dist (to first)
                       and is used only linearly (lesson learned about functional-form
                       endogeneity)
    - pivot_dist       the smaller of the 2B/SS fielders' distances to the second base bag
                       (feet) = double-play pivot geometry, a pure function of positioning
                       (moving a fielder changes the prediction accordingly)
    - runner_hp_to_1b  the runner on first's season hp_to_1b (proxy for runner speed;
                       missing values filled with that year's median)
    - has_runner_speed whether it's an actual measured value
    """
    df = df.copy()
    df["n_outs"] = df["events"].map(ON1B_EVENT_OUTS)

    ix, iy = _polar_to_xy(df["near_depth"].to_numpy(float),
                          df["spray_deg"].to_numpy(float))
    df["throw_dist_2b"] = np.hypot(ix - SECOND_BASE_X, iy - SECOND_BASE_Y)

    dists = []
    for pos in ("2B", "SS"):
        px, py = _polar_to_xy(df[f"{pos}_depth"].to_numpy(float),
                              df[f"{pos}_angle"].to_numpy(float))
        dists.append(np.hypot(px - SECOND_BASE_X, py - SECOND_BASE_Y))
    df["pivot_dist"] = np.minimum(*dists)

    rs_map = run_speed.set_index(["player_id", "season"])["hp_to_1b"]
    idx = pd.MultiIndex.from_arrays([df["on_1b"], df["game_year"]])
    df["runner_hp_to_1b"] = rs_map.reindex(idx).to_numpy()
    df["has_runner_speed"] = df["runner_hp_to_1b"].notna()
    year_med = df.groupby("game_year")["runner_hp_to_1b"].transform("median")
    df["runner_hp_to_1b"] = df["runner_hp_to_1b"].fillna(year_med)
    return df


def build_gb_on1b_dataset(years: list[int],
                          alignment: str | None = "Standard") -> pd.DataFrame:
    """One-stop (Stage B): ground balls with a runner on first only, <2 outs, plus double-play features.

    Positioning proxy uses fielder_positioning_on1b (the runner-on-first split); the label
    n_outs is in {0,1,2}.
    """
    run_speed = fetch_run_speed(years)
    df = attach_features(fetch_raw_gb_on1b(years), fetch_positioning_on1b(years),
                         run_speed)
    df = attach_dp_features(df, run_speed)
    if alignment is not None:
        df = df[df["if_fielding_alignment"] == alignment]
    return df.reset_index(drop=True)


def build_gb_dataset(years: list[int], bases_empty: bool | None = None,
                     alignment: str | None = None) -> pd.DataFrame:
    """One-stop: fetch data, attach features, filter by scope.

    bases_empty=True keeps only bases-empty plays (a runner on base pulls positioning,
    e.g. the first baseman holding the runner); alignment='Standard' keeps only standard
    alignment (season-average positioning has more error against non-standard alignments).
    """
    df = attach_features(fetch_raw_gb(years), fetch_positioning(years),
                         fetch_run_speed(years))
    if bases_empty is not None:
        df = df[df["bases_empty"] == bases_empty]
    if alignment is not None:
        df = df[df["if_fielding_alignment"] == alignment]
    return df.reset_index(drop=True)
