"""Generate the lean deployment precomputed table: precomputed_batter_popups
(infield pop-ups for display on the combined page).

Popup positioning has no leverage (98.6% out rate) so it's excluded from
optimization, but the combined page needs to show all of a batter's balls in
play — without popups (~7% of balls in play) the chart would be missing a
chunk. Pulled from the source DB with full statcast data, converted to
display coordinates, and loaded into the target DB; deploy to the cloud with
--target-dsn (same deployment pattern as precompute_batter_balls.py).

Coordinates = hc x 2.5 ft (same display-coordinate convention as
precomputed_if_gbs; popup landing spots are mostly in the infield, so using
the same conversion as ground balls keeps the chart aligned).
Missing launch_speed values are kept as NULL (the frontend only uses it for
display, not computation).

Usage:
    python -m scripts.precompute.precompute_batter_popups
    python -m scripts.precompute.precompute_batter_popups --years 2023 2024 2025 --target-dsn "postgresql://..."
"""
import argparse
from pathlib import Path

import pandas as pd
import psycopg2

from src.config import DSN
from src.if_dataset import HOME_X, HOME_Y

from scripts._pg_load import copy_dataframe

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
FT_PER_UNIT = 2.5   # Same display-coordinate conversion as scripts/precompute_if_optimize.py

_POPUPS_QUERY = """
    SELECT batter, game_year, hc_x, hc_y, launch_speed, events
    FROM statcast
    WHERE game_year = ANY(%(years)s)
      AND game_type = 'R'
      AND type = 'X'
      AND bb_type = 'popup'
      AND hc_x IS NOT NULL AND hc_y IS NOT NULL
"""

# These are the only four non-out events for popups (all other events like
# field_out/force_out/double_play/sac_fly/... record at least one out);
# using a whitelist for the inverse check is the least error-prone approach
_NONOUT_EVENTS = ("single", "double", "triple", "field_error")


def build_batter_popups(source_dsn: str, years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(source_dsn) as conn:
        df = pd.read_sql(_POPUPS_QUERY, conn, params={"years": years})
    print(f"[popups] 原始筆數: {len(df):,}")
    out = pd.DataFrame({
        "batter": df["batter"],
        "game_year": df["game_year"],
        "ball_x": ((df["hc_x"] - HOME_X) * FT_PER_UNIT).round(1),
        "ball_y": ((HOME_Y - df["hc_y"]) * FT_PER_UNIT).round(1),
        "launch_speed": df["launch_speed"],
        "is_out": ~df["events"].isin(_NONOUT_EVENTS),
    })
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2026)))
    parser.add_argument("--source-dsn", default=DSN)
    parser.add_argument("--target-dsn", default=None)
    args = parser.parse_args()

    df = build_batter_popups(args.source_dsn, args.years)

    target_dsn = args.target_dsn or args.source_dsn
    with psycopg2.connect(target_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute((SQL_DIR / "create_precomputed_batter_popups_table.sql")
                        .read_text(encoding="utf-8"))
            cur.execute("TRUNCATE precomputed_batter_popups")
        conn.commit()
        copy_dataframe(conn, "precomputed_batter_popups", df)

    target_label = target_dsn.split("@")[-1] if "@" in target_dsn else target_dsn
    print(f"完成：precomputed_batter_popups {len(df):,} 筆 → {target_label}")


if __name__ == "__main__":
    main()
