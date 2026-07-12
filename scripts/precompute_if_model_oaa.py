"""生成排名頁用的 if_model_oaa 表（2025 樣本外內野手 model OAA）。

計算邏輯與 scripts/evaluate_if_2025.py 共用 src/if_eval.py（難度 GLM 2023–2024
訓練、2025 評分、hit_location 歸責、分位置中心化），這裡只負責落表。
球員姓名從 if_oaa_leaderboard 帶入（對不上官方 leaderboard 者為 NULL，
排名頁不顯示）。

Usage:
    python scripts/precompute_if_model_oaa.py
    python scripts/precompute_if_model_oaa.py --target-dsn "postgresql://..."
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import psycopg2

from src.config import DSN
from src.if_eval import aggregate_players, score_test_year

SQL_DIR = Path(__file__).resolve().parent / "sql"
TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
COLS = ["player_id", "player_name", "position", "year", "model_oaa", "n_balls"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dsn", default=None,
                        help="要灌入的 DB（預設本機；部署時帶 Neon DSN）")
    args = parser.parse_args()

    players = aggregate_players(score_test_year(TRAIN_YEARS, TEST_YEAR))
    with psycopg2.connect(DSN) as conn:
        official = pd.read_sql(
            "SELECT player_id, player_name FROM if_oaa_leaderboard "
            "WHERE year = %(y)s", conn, params={"y": TEST_YEAR})
    df = players.merge(official, left_on="resp_fielder", right_on="player_id",
                       how="left")
    df["player_id"] = df["resp_fielder"]
    df["position"] = df["resp_pos"]
    df["year"] = TEST_YEAR
    df["model_oaa"] = df["model_oaa"].round(3)
    df = df[COLS]
    print(f"{len(df)} 位內野手（有官方姓名 {df['player_name'].notna().sum()} 位）")

    with psycopg2.connect(args.target_dsn or DSN) as conn:
        with conn.cursor() as cur:
            cur.execute((SQL_DIR / "create_if_model_oaa_table.sql")
                        .read_text(encoding="utf-8"))
            cur.execute("DELETE FROM if_model_oaa WHERE year = %s", (TEST_YEAR,))
        conn.commit()
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        with conn.cursor() as cur:
            cur.copy_expert(
                "COPY if_model_oaa FROM STDIN WITH (FORMAT csv, NULL '')", buf)
        conn.commit()
    print(f"完成：if_model_oaa {len(df)} 筆（year={TEST_YEAR}）")


if __name__ == "__main__":
    main()
