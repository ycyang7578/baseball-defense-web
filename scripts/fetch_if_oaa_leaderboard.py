"""Fetch infield OAA leaderboard from Baseball Savant into if_oaa_leaderboard table.

Source: GET /leaderboard/outs_above_average?type=Fielder&pos=if&startYear=&endYear=
The CSV export omits attempt counts, so we parse the page's `var data` JSON instead
(same approach as fetch_oaa_leaderboard.py). The JSON's year field is empty; we fill
it from the requested year.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN

BASE_URL = "https://baseballsavant.mlb.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# (df欄位, JSON欄位) 對照；INSERT 欄位順序也依這裡
COLUMNS = [
    ("player_id", "entity_id"),
    ("player_name", "entity_name"),
    ("primary_pos", "primary_pos_formatted"),
    ("oaa", "outs_above_average"),
    ("fielding_runs_prevented", "fielding_runs_prevented"),
    ("n_opp", "n"),
    ("actual_success_rate", "actual_success_rate"),
    ("adj_estimated_success_rate", "adj_estimated_success_rate"),
    ("n_pos3", "n_pos3"), ("n_pos4", "n_pos4"),
    ("n_pos5", "n_pos5"), ("n_pos6", "n_pos6"),
    ("oaa_role3", "outs_above_average_role3"), ("oaa_role4", "outs_above_average_role4"),
    ("oaa_role5", "outs_above_average_role5"), ("oaa_role6", "outs_above_average_role6"),
    ("oaa_infront", "outs_above_average_infront"),
    ("oaa_behind", "outs_above_average_behind"),
    ("oaa_toward3b", "outs_above_average_lateral_toward3bline"),
    ("oaa_toward1b", "outs_above_average_lateral_toward1bline"),
    ("oaa_lhh", "outs_above_average_lhh"),
    ("oaa_rhh", "outs_above_average_rhh"),
]
FLOAT_COLS = {"actual_success_rate", "adj_estimated_success_rate"}


def fetch_leaderboard(year: int) -> pd.DataFrame:
    r = requests.get(
        f"{BASE_URL}/leaderboard/outs_above_average",
        params={"type": "Fielder", "startYear": year, "endYear": year, "split": "no",
                "team": "", "range": "year", "min": 1, "pos": "if", "roles": "",
                "viz": "hide"},
        headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    match = re.search(r"var data\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not match:
        raise ValueError("找不到 leaderboard var data，頁面結構可能已更新")
    raw = pd.DataFrame(json.loads(match.group(1)))

    df = pd.DataFrame({dst: pd.to_numeric(raw[src], errors="coerce")
                       if dst not in ("player_name", "primary_pos") else raw[src]
                       for dst, src in COLUMNS})
    df["year"] = year
    df["is_qualified"] = raw["is_qualified_leaderboard"].astype(str).map(
        {"1": True, "0": False})
    return df


def load_to_postgres(df: pd.DataFrame, year: int) -> None:
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    cur.execute("DELETE FROM if_oaa_leaderboard WHERE year = %s", (year,))
    if cur.rowcount:
        print(f"[delete] 已清除 {year} 年舊資料 ({cur.rowcount} rows)")

    int_like = [dst for dst, _ in COLUMNS
                if dst not in FLOAT_COLS and dst not in ("player_name", "primary_pos")]
    rows = []
    for r in df.itertuples():
        vals = {"year": int(r.year), "is_qualified": bool(r.is_qualified)}
        for dst, _ in COLUMNS:
            v = getattr(r, dst)
            if pd.isna(v):
                vals[dst] = None
            elif dst in int_like:
                vals[dst] = int(v)
            else:
                vals[dst] = v
        rows.append(vals)

    col_names = [dst for dst, _ in COLUMNS] + ["year", "is_qualified"]
    placeholders = ", ".join(f"%({c})s" for c in col_names)
    cur.executemany(
        f"INSERT INTO if_oaa_leaderboard ({', '.join(col_names)}) VALUES ({placeholders})",
        rows)

    conn.commit()
    conn.close()
    print(f"[insert] {len(rows)} rows → if_oaa_leaderboard")


def main(years: list[int]) -> None:
    for year in years:
        print(f"\n=== {year} ===")
        df = fetch_leaderboard(year)
        print(f"[fetch] {len(df)} 位內野手（qualified: {int(df['is_qualified'].sum())}）")
        load_to_postgres(df, year)


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else [2025]
    main(years)
