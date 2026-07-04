"""生成部署用的精簡預計算表：precomputed_batter_balls / precomputed_batter_stand。

從有完整 statcast 的來源 DB 撈資料、用 src/physics.py 算好衍生欄位，寫入目標 DB。
本機測試時 source=target 是同一顆 DB；部署到雲端免費方案時用 --target-dsn 帶雲端 DSN，
把這兩張精簡表（而不是整個 5GB 的 statcast）灌進去。

篩選條件跟 src/optimization.py 的 _BATTER_QUERY（prepare_batter_balls 用）與
get_batter_stand（不限 bb_type，整季所有打席）保持一致，否則兩邊資料會對不上。

Usage:
    python scripts/precompute_batter_balls.py
    python scripts/precompute_batter_balls.py --years 2024 2025 --target-dsn "postgresql://..."
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psycopg2

from src import physics
from src.config import DSN

SQL_DIR = Path(__file__).resolve().parent / "sql"

# 跟 src/optimization.py 的 _BATTER_QUERY 篩選條件保持一致（打出去的球）
_BALLS_QUERY = """
    SELECT batter, game_year, stand, hit_distance_sc, launch_speed, launch_angle, hc_x, hc_y, plate_z
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

# 跟 src/optimization.py 的 get_batter_stand 篩選條件保持一致（不限 bb_type）
_STAND_QUERY = """
    SELECT batter, game_year, stand, COUNT(*) AS n
    FROM statcast
    WHERE game_year = ANY(%(years)s)
    GROUP BY batter, game_year, stand
"""


def _copy(conn, table: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    conn.commit()


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
        np.arctan2(df["hc_x"] - physics._X0, physics._Y0 - df["hc_y"])
    )
    df = df[df["flight_time"] > 0.5].copy()
    print(f"[balls] flight_time>0.5 篩選後: {len(df):,}")

    return df[["batter", "game_year", "stand", "ball_x", "ball_y", "flight_time",
               "launch_speed", "launch_angle", "spray_angle"]].reset_index(drop=True)


def build_batter_stand(source_dsn: str, years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(source_dsn) as conn:
        df = pd.read_sql(_STAND_QUERY, conn, params={"years": years})
    # 每個 (batter, game_year) 取出現次數最多的 stand
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
            cur.execute((SQL_DIR / "create_precomputed_batter_balls_table.sql").read_text(encoding="utf-8"))
            cur.execute((SQL_DIR / "create_precomputed_batter_stand_table.sql").read_text(encoding="utf-8"))
            cur.execute("TRUNCATE precomputed_batter_balls")
            cur.execute("TRUNCATE precomputed_batter_stand")
        conn.commit()
        _copy(conn, "precomputed_batter_balls", balls_df)
        _copy(conn, "precomputed_batter_stand", stand_df)

    target_label = target_dsn.split("@")[-1] if "@" in target_dsn else target_dsn
    print(f"完成：precomputed_batter_balls {len(balls_df):,} 筆、"
          f"precomputed_batter_stand {len(stand_df):,} 筆 → {target_label}")


if __name__ == "__main__":
    main()
