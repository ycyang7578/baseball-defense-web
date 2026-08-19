"""Generate the lean deployment precomputed tables: precomputed_batter_balls / precomputed_batter_stand.

Pulls data from the source DB with full statcast data, computes derived columns via
src/physics.py, and writes them to the target DB. For local testing, source=target
is the same DB; when deploying to a cloud free tier, pass --target-dsn with the cloud
DSN so only these two lean tables (not the full 5GB statcast table) get loaded.

The filter conditions must stay in sync with _BATTER_QUERY in src/optimization.py
(used by prepare_batter_balls) and get_batter_stand (no bb_type restriction, all
plate appearances for the season), otherwise the two sides' data won't line up.

Usage:
    python -m scripts.precompute.precompute_batter_balls
    python -m scripts.precompute.precompute_batter_balls --years 2024 2025 --target-dsn "postgresql://..."
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from src import physics
from src.config import DSN

from scripts._pg_load import copy_dataframe

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

# Stay in sync with the filter conditions in _BATTER_QUERY in src/optimization.py (balls in play)
_BALLS_QUERY = """
    SELECT batter, game_year, stand, hit_distance_sc, launch_speed, launch_angle, hc_x, hc_y, plate_z,
           bb_type
    FROM statcast
    WHERE game_year = ANY(%(years)s)
      AND game_type = 'R'
      AND type = 'X'
      AND bb_type IN ('fly_ball', 'line_drive')
      AND events != 'home_run'
      AND hit_distance_sc IS NOT NULL
      AND launch_speed    IS NOT NULL
      AND launch_angle    IS NOT NULL
      AND hc_x            IS NOT NULL
      AND hc_y            IS NOT NULL
      AND plate_z         IS NOT NULL
"""

# Stay in sync with the filter conditions in get_batter_stand in src/optimization.py (no bb_type restriction)
_STAND_QUERY = """
    SELECT batter, game_year, stand, COUNT(*) AS n
    FROM statcast
    WHERE game_year = ANY(%(years)s)
    GROUP BY batter, game_year, stand
"""


def build_batter_balls(source_dsn: str, years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(source_dsn) as conn:
        df = pd.read_sql(_BALLS_QUERY, conn, params={"years": years})
    print(f"[balls] 原始筆數: {len(df):,}")

    df["ball_x"], df["ball_y"] = physics.transform_coordinates(
        df["hc_x"], df["hc_y"], df["hit_distance_sc"]
    )
    df["flight_time"] = physics.calculate_flight_time(
        df["launch_speed"], df["launch_angle"], df["plate_z"]
    )
    df["spray_angle"] = np.degrees(
        np.arctan2(df["hc_x"] - physics._STATCAST_ORIGIN_X, physics._STATCAST_ORIGIN_Y - df["hc_y"])
    )
    df = df[df["flight_time"] > 0.5].copy()
    print(f"[balls] flight_time>0.5 篩選後: {len(df):,}")

    return df[["batter", "game_year", "stand", "ball_x", "ball_y", "flight_time",
               "launch_speed", "launch_angle", "spray_angle", "bb_type"]].reset_index(drop=True)


def build_batter_stand(source_dsn: str, years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(source_dsn) as conn:
        df = pd.read_sql(_STAND_QUERY, conn, params={"years": years})
    # For each (batter, game_year), take the most frequently occurring stand
    df = df.sort_values("n", ascending=False).drop_duplicates(subset=["batter", "game_year"])
    print(f"[stand] (batter, year) 組合數: {len(df):,}")
    return df[["batter", "game_year", "stand"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2026)))
    parser.add_argument("--source-dsn", default=DSN)
    parser.add_argument("--target-dsn", default=None)
    args = parser.parse_args()

    source_dsn = args.source_dsn
    target_dsn = args.target_dsn or source_dsn

    balls_df = build_batter_balls(source_dsn, args.years)
    stand_df = build_batter_stand(source_dsn, args.years)

    with psycopg2.connect(target_dsn) as conn:
        with conn.cursor() as cur:
            # DROP instead of TRUNCATE: the schema may have changed (e.g. bb_type column
            # added on 2026-07-14), and CREATE IF NOT EXISTS won't add missing columns
            cur.execute("DROP TABLE IF EXISTS precomputed_batter_balls")
            cur.execute((SQL_DIR / "create_precomputed_batter_balls_table.sql").read_text(encoding="utf-8"))
            cur.execute((SQL_DIR / "create_precomputed_batter_stand_table.sql").read_text(encoding="utf-8"))
            cur.execute("TRUNCATE precomputed_batter_stand")
        conn.commit()
        copy_dataframe(conn, "precomputed_batter_balls", balls_df)
        copy_dataframe(conn, "precomputed_batter_stand", stand_df)

    target_label = target_dsn.split("@")[-1] if "@" in target_dsn else target_dsn
    print(f"完成：precomputed_batter_balls {len(balls_df):,} 筆、"
          f"precomputed_batter_stand {len(stand_df):,} 筆 → {target_label}")


if __name__ == "__main__":
    main()
