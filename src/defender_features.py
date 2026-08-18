"""Defender opportunity feature engineering: query Postgres, apply physics, return per-ball features.

Queries raw statcast + fielder_positioning directly, no precomputed per-position CSVs.
"""
import numpy as np
import pandas as pd
import psycopg2

from . import physics
from .config import DSN

# LF/CF/RF 對應 Statcast 的 fielder_N 欄位與 hit_location 編號（標準棒球位置編號，剛好一致）
_FIELDER_COLUMN: dict[str, str] = {"LF": "fielder_7", "CF": "fielder_8", "RF": "fielder_9"}
_HIT_LOCATION: dict[str, int] = {"LF": 7, "CF": 8, "RF": 9}

_EXCLUDED_ALIGNMENTS: tuple[str, ...] = ("Strategic", "4th outfielder")
_VALID_BB_TYPES: tuple[str, ...] = ("fly_ball", "line_drive")

_QUERY = """
    SELECT
        s.game_year AS season,
        s.game_pk, s.at_bat_number,
        s.game_date, s.player_name, s.batter, s.pitcher, s.inning,
        s.events, s.hit_distance_sc, s.launch_speed, s.launch_angle,
        s.hc_x, s.hc_y, s.plate_z, s.home_team,
        s.{fielder_col} AS fielder_id,
        p.name_fielder,
        p.avg_norm_start_distance,
        p.avg_norm_start_angle
    FROM statcast s
    JOIN fielder_positioning p
        ON p.fielder_id = s.{fielder_col}
        AND p.season = s.game_year
        AND p.position = %(position)s
    WHERE s.game_year = %(year)s
        AND s.type = 'X'
        AND s.hit_location = %(hit_location)s
        AND s.bb_type IN %(valid_bb_types)s
        AND s.of_fielding_alignment NOT IN %(excluded_alignments)s
        AND s.events != 'home_run'
        AND s.hit_distance_sc IS NOT NULL
        AND s.launch_speed IS NOT NULL
        AND s.launch_angle IS NOT NULL
        AND s.hc_x IS NOT NULL
        AND s.hc_y IS NOT NULL
        AND s.plate_z IS NOT NULL
"""


def get_defender_opportunities(position: str, year: int) -> pd.DataFrame:
    """單一守備位置、單一年度的有效守備機會（required_speed, cos_angle, sin_angle, fielder_dist, caught）。"""
    fielder_col = _FIELDER_COLUMN[position]
    query = _QUERY.format(fielder_col=fielder_col)

    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(
            query, conn,
            params={
                "position": position,
                "year": year,
                "hit_location": _HIT_LOCATION[position],
                "valid_bb_types": _VALID_BB_TYPES,
                "excluded_alignments": _EXCLUDED_ALIGNMENTS,
            },
        )

    if df.empty:
        return df

    df["x_coord"], df["y_coord"] = physics.transform_coordinates(
        df["hc_x"], df["hc_y"], df["hit_distance_sc"]
    )
    df["flight_time"] = physics.calculate_flight_time(
        df["launch_speed"], df["launch_angle"], df["plate_z"]
    )
    df["fielder_x"], df["fielder_y"] = physics.polar_to_fielder_xy(
        df["avg_norm_start_distance"], df["avg_norm_start_angle"]
    )
    df["fielder_dist"] = ((df["x_coord"] - df["fielder_x"]) ** 2
                           + (df["y_coord"] - df["fielder_y"]) ** 2) ** 0.5
    df["required_speed"] = df["fielder_dist"] / df["flight_time"].clip(lower=0.1)
    angle = physics.compute_relative_angle(
        df["fielder_x"].values, df["fielder_y"].values,
        df["x_coord"].values, df["y_coord"].values,
    )
    df["cos_angle"] = np.cos(angle)
    df["sin_angle"] = np.sin(angle)
    df["caught"] = physics.mark_caught(df["events"])

    return df


def mark_official(df: pd.DataFrame) -> pd.DataFrame:
    """標記每一列是否為Savant官方計算OAA時真正算進去的球（savant_fielding比對，game_pk+at_bat_number+fielder_id）。"""
    with psycopg2.connect(DSN) as conn:
        sf = pd.read_sql(
            "SELECT DISTINCT player_id, game_pk, at_bat_number FROM savant_fielding",
            conn,
        )
    sf["_key"] = sf["player_id"].astype(str) + "_" + sf["game_pk"].astype(str) + "_" + sf["at_bat_number"].astype(str)
    official_keys = set(sf["_key"])

    df = df.copy()
    df_key = (df["fielder_id"].astype(str) + "_" + df["game_pk"].astype(str)
              + "_" + df["at_bat_number"].astype(str))
    df["is_official"] = df_key.isin(official_keys)
    return df
