"""Generate the if_model_oaa table (infield model OAA) used by the rankings
page / fielder selector.

The scoring logic is shared with scripts/evaluate_if_2025.py via src/if_eval.py
(difficulty GLM trained on 2023-2024, scores a specified year, attributes via
hit_location, centered per position) — this script is only responsible for
writing the table. Player names are joined in from if_oaa_leaderboard (players
that don't match the official leaderboard get NULL and aren't shown on the
rankings page).

Warning: scoring 2023/2024 is in-sample (the scored year falls within the
training years), so the numbers skew optimistic — this is only for menu
labels/display purposes (an informed decision made by the user on 2026-07-14)
and must not be cited as validation; only 2025 is out-of-sample.

Usage:
    python -m scripts.precompute.precompute_if_model_oaa                     # 2025 (out-of-sample)
    python -m scripts.precompute.precompute_if_model_oaa --test-year 2023    # in-sample
    python -m scripts.precompute.precompute_if_model_oaa --target-dsn "postgresql://..."
"""
import argparse
from pathlib import Path

import pandas as pd
import psycopg2

from src.config import DSN
from src.if_eval import aggregate_players, score_test_year

from scripts._pg_load import copy_dataframe

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
TRAIN_YEARS = [2023, 2024]
COLS = ["player_id", "player_name", "position", "year", "model_oaa", "n_balls"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-year", type=int, default=2025,
                        help="評分年（2025=樣本外；2023/2024=in-sample，僅展示用）")
    parser.add_argument("--target-dsn", default=None,
                        help="要灌入的 DB（預設本機；部署時帶 Neon DSN）")
    args = parser.parse_args()
    test_year = args.test_year

    players = aggregate_players(score_test_year(TRAIN_YEARS, test_year))
    with psycopg2.connect(DSN) as conn:
        official = pd.read_sql(
            "SELECT player_id, player_name FROM if_oaa_leaderboard "
            "WHERE year = %(y)s", conn, params={"y": test_year})
    df = players.merge(official, left_on="resp_fielder", right_on="player_id",
                       how="left")
    df["player_id"] = df["resp_fielder"]
    df["position"] = df["resp_pos"]
    df["year"] = test_year
    df["model_oaa"] = df["model_oaa"].round(3)
    df = df[COLS]
    print(f"{len(df)} 位內野手（有官方姓名 {df['player_name'].notna().sum()} 位）")

    with psycopg2.connect(args.target_dsn or DSN) as conn:
        with conn.cursor() as cur:
            cur.execute((SQL_DIR / "create_if_model_oaa_table.sql")
                        .read_text(encoding="utf-8"))
            cur.execute("DELETE FROM if_model_oaa WHERE year = %s", (test_year,))
        conn.commit()
        copy_dataframe(conn, "if_model_oaa", df)
    print(f"完成：if_model_oaa {len(df)} 筆（year={test_year}）")


if __name__ == "__main__":
    main()
