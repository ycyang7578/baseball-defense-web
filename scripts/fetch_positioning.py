"""Fetch average fielder starting position data from Baseball Savant (no pybaseball wrapper exists).

API endpoint found via browser DevTools network capture (2026-06-23), not officially documented.
Saved per year (like fetch_statcast.py), each file covering LF/CF/RF combined.
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "positioning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://baseballsavant.mlb.com/visuals/position_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

POSITIONS = {"LF": 7, "CF": 8, "RF": 9}


def fetch_one(year: int, position_code: int) -> pd.DataFrame:
    params = {"type": "player", "teamId": "", "season": year, "position": position_code,
              "attempts": 1, "csv": "true"}
    r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8-sig"
    import io
    return pd.read_csv(io.StringIO(r.text))


def fetch_year(year: int) -> None:
    out_path = OUTPUT_DIR / f"{year}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    frames = []
    for pos_name, pos_code in POSITIONS.items():
        df = fetch_one(year, pos_code)
        df["position"] = pos_name  # API回傳的position欄位用縮寫，這裡統一覆寫成LF/CF/RF避免混淆
        frames.append(df)
        time.sleep(0.5)  # 對Savant伺服器友善一點，不要過快連續請求

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"[done] {year}: {len(combined):,} rows -> {out_path}")


if __name__ == "__main__":
    import sys
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else list(range(2020, 2026))
    for y in years:
        fetch_year(y)
