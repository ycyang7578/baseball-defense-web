"""Fetch OAA leaderboard from Baseball Savant and load into oaa_leaderboard table.

Source: GET /leaderboard/outfield_directional_outs_above_average?year={year}
Stores per-player OAA, total opportunities, and 0~5-star breakdown.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import requests

BASE_URL = "https://baseballsavant.mlb.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

POS_MAP = {"7": "LF", "8": "CF", "9": "RF"}

INT_COLS = [
    "oaa",
    "n_opp",
    "n_opp_0stars", "n_opp_1stars", "n_opp_2stars",
    "n_opp_3stars", "n_opp_4stars", "n_opp_5stars",
    "n_fieldout_0stars", "n_fieldout_1stars", "n_fieldout_2stars",
    "n_fieldout_3stars", "n_fieldout_4stars", "n_fieldout_5stars",
]


def fetch_leaderboard(year: int) -> pd.DataFrame:
    r = requests.get(
        f"{BASE_URL}/leaderboard/outfield_directional_outs_above_average",
        params={"year": year}, headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    match = re.search(r"var data\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not match:
        raise ValueError("找不到 leaderboard var data，頁面結構可能已更新")
    raw = pd.DataFrame(json.loads(match.group(1)))

    df = pd.DataFrame({
        "player_id":          raw["resp_fielder_id"].astype(int),
        "year":               year,
        "player_name":        raw["name_display_last_first"],
        "position":           raw["resp_fielder_pos"].astype(str).map(POS_MAP),
        "oaa":                pd.to_numeric(raw["n_outs_above_average"], errors="coerce"),
        "n_opp":              pd.to_numeric(raw["n"],                    errors="coerce"),
        "n_opp_0stars":       pd.to_numeric(raw["n_opp_0stars"],         errors="coerce"),
        "n_opp_1stars":       pd.to_numeric(raw["n_opp_1stars"],         errors="coerce"),
        "n_opp_2stars":       pd.to_numeric(raw["n_opp_2stars"],         errors="coerce"),
        "n_opp_3stars":       pd.to_numeric(raw["n_opp_3stars"],         errors="coerce"),
        "n_opp_4stars":       pd.to_numeric(raw["n_opp_4stars"],         errors="coerce"),
        "n_opp_5stars":       pd.to_numeric(raw["n_opp_5stars"],         errors="coerce"),
        "n_fieldout_0stars":  pd.to_numeric(raw["n_fieldout_0stars"],    errors="coerce"),
        "n_fieldout_1stars":  pd.to_numeric(raw["n_fieldout_1stars"],    errors="coerce"),
        "n_fieldout_2stars":  pd.to_numeric(raw["n_fieldout_2stars"],    errors="coerce"),
        "n_fieldout_3stars":  pd.to_numeric(raw["n_fieldout_3stars"],    errors="coerce"),
        "n_fieldout_4stars":  pd.to_numeric(raw["n_fieldout_4stars"],    errors="coerce"),
        "n_fieldout_5stars":  pd.to_numeric(raw["n_fieldout_5stars"],    errors="coerce"),
        "is_qualified":       raw["is_qualified_leaderboard"].astype(str).map({"1": True, "0": False}),
    })

    for col in INT_COLS:
        df[col] = df[col].astype("Int64")

    return df


def load_to_postgres(df: pd.DataFrame, year: int) -> None:
    conn = psycopg2.connect(host="localhost", dbname="baseball",
                            user="postgres", password="postgres")
    cur = conn.cursor()

    cur.execute("DELETE FROM oaa_leaderboard WHERE year = %s", (year,))
    deleted = cur.rowcount
    if deleted:
        print(f"[delete] 已清除 {year} 年舊資料 ({deleted} rows)")

    rows = [
        (
            int(r.player_id), int(r.year), r.player_name, r.position,
            None if pd.isna(r.oaa) else int(r.oaa),
            None if pd.isna(r.n_opp) else int(r.n_opp),
            None if pd.isna(r.n_opp_0stars) else int(r.n_opp_0stars),
            None if pd.isna(r.n_opp_1stars) else int(r.n_opp_1stars),
            None if pd.isna(r.n_opp_2stars) else int(r.n_opp_2stars),
            None if pd.isna(r.n_opp_3stars) else int(r.n_opp_3stars),
            None if pd.isna(r.n_opp_4stars) else int(r.n_opp_4stars),
            None if pd.isna(r.n_opp_5stars) else int(r.n_opp_5stars),
            None if pd.isna(r.n_fieldout_0stars) else int(r.n_fieldout_0stars),
            None if pd.isna(r.n_fieldout_1stars) else int(r.n_fieldout_1stars),
            None if pd.isna(r.n_fieldout_2stars) else int(r.n_fieldout_2stars),
            None if pd.isna(r.n_fieldout_3stars) else int(r.n_fieldout_3stars),
            None if pd.isna(r.n_fieldout_4stars) else int(r.n_fieldout_4stars),
            None if pd.isna(r.n_fieldout_5stars) else int(r.n_fieldout_5stars),
            bool(r.is_qualified) if r.is_qualified is not None else None,
        )
        for r in df.itertuples()
    ]

    cur.executemany("""
        INSERT INTO oaa_leaderboard (
            player_id, year, player_name, position, oaa, n_opp,
            n_opp_0stars, n_opp_1stars, n_opp_2stars, n_opp_3stars, n_opp_4stars, n_opp_5stars,
            n_fieldout_0stars, n_fieldout_1stars, n_fieldout_2stars,
            n_fieldout_3stars, n_fieldout_4stars, n_fieldout_5stars,
            is_qualified
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, rows)

    conn.commit()
    conn.close()
    print(f"[insert] {len(rows)} rows → oaa_leaderboard")


def main(years: list[int]) -> None:
    for year in years:
        print(f"\n=== {year} ===")
        df = fetch_leaderboard(year)
        qualified = df["is_qualified"].sum()
        print(f"[fetch] {len(df)} 位外野手（qualified: {qualified}）")
        load_to_postgres(df, year)


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2025]
    main(years)
