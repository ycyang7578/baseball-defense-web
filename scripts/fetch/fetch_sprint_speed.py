"""Fetch batter running speed from the Baseball Savant sprint speed leaderboard.

Provides sprint_speed (ft/s) and hp_to_1b (home-to-first seconds), the runner-side
inputs for ground-ball out probability. min=5 keeps part-time runners; the CSV is
small so we keep everyone and let consumers decide their own thresholds.
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sprint_speed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_year(year: int) -> None:
    out_path = OUTPUT_DIR / f"{year}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    params = {"min_season": year, "max_season": year, "position": "", "team": "",
              "min": 5, "csv": "true"}
    r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8-sig"
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"last_name, first_name": "name_runner"})
    df["season"] = year
    df.to_parquet(out_path, index=False)
    print(f"[done] {year}: {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    import sys
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else list(range(2020, 2026))
    for y in years:
        fetch_year(y)
        time.sleep(0.5)
