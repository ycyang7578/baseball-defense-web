"""Fetch average fielder starting position data from Baseball Savant (no pybaseball wrapper exists).

API endpoint found via browser DevTools network capture (2026-06-23), not officially documented.
Saved per year (like fetch_statcast.py), each file covering all seven non-battery
positions (1B/2B/3B/SS/LF/CF/RF) combined.
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "positioning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://baseballsavant.mlb.com/visuals/position_data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

POSITIONS = {"1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}


def fetch_one(year: int, position_code: int, retries: int = 3,
              extra_params: dict | None = None) -> pd.DataFrame:
    params = {"type": "player", "teamId": "", "season": year, "position": position_code,
              "attempts": 1, "csv": "true"}
    if extra_params:
        params.update(extra_params)
    for attempt in range(retries):
        r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
        if r.status_code < 500:
            break
        time.sleep(3 * (attempt + 1))  # Savant occasionally returns 502, wait and retry
    r.raise_for_status()
    r.encoding = "utf-8-sig"
    import io
    return pd.read_csv(io.StringIO(r.text))


IF_POSITIONS = {"1B": 3, "2B": 4, "3B": 5, "SS": 6}


def fetch_year_on1b(year: int) -> None:
    """Infield four-position starting positions split by "runner on first (1B Only)" (the geometric proxy for the Phase B double-play model).

    Endpoint parameter gotcha (verified 2026-07-13): the runner-state filter only
    takes effect if batSide + firstBase + shift are all supplied together --
    passing firstBase alone gets ignored and falls back to the full dataset, and
    passing shift alone returns an anomalously small subset. So L/R are fetched
    separately and merged with PA weighting into one row per fielder (consistent
    with the main table's "one row per fielder per year per position" convention).
    shift=1 only has scattered PA left in the shift-ban era (verified 337 vs 9),
    so shift=0 is used.
    Outputs {year}_on1b.parquet, with the same columns as the main table.
    """
    out_path = OUTPUT_DIR / f"{year}_on1b.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    frames = []
    for pos_name, pos_code in IF_POSITIONS.items():
        for side in ("L", "R"):
            df = fetch_one(year, pos_code,
                           extra_params={"firstBase": 1, "shift": 0, "batSide": side})
            df["position"] = pos_name
            frames.append(df)
            time.sleep(0.5)

    combined = pd.concat(frames, ignore_index=True)
    for col in ("pa", "avg_norm_start_distance", "avg_norm_start_angle"):
        combined[col] = pd.to_numeric(combined[col])

    def _merge(g: pd.DataFrame) -> pd.Series:
        w = g["pa"]
        top = g.loc[w.idxmax()]
        return pd.Series({
            "name_fielder": top["name_fielder"],
            "fld_name_display_club": top["fld_name_display_club"],
            "pa": int(w.sum()),
            "avg_norm_start_distance": float((g["avg_norm_start_distance"] * w).sum() / w.sum()),
            "avg_norm_start_angle": float((g["avg_norm_start_angle"] * w).sum() / w.sum()),
        })

    merged = (combined.groupby(["fielder_id", "season", "position"])
              .apply(_merge, include_groups=False).reset_index())
    merged = merged[["name_fielder", "fielder_id", "fld_name_display_club",
                     "season", "position", "pa",
                     "avg_norm_start_distance", "avg_norm_start_angle"]]
    merged.to_parquet(out_path, index=False)
    print(f"[done] {year} on1b: {len(merged):,} rows -> {out_path}")


def fetch_year(year: int) -> None:
    out_path = OUTPUT_DIR / f"{year}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    frames = []
    for pos_name, pos_code in POSITIONS.items():
        df = fetch_one(year, pos_code)
        df["position"] = pos_name  # The API's position column uses abbreviations; overwrite it uniformly to LF/CF/RF to avoid confusion
        frames.append(df)
        time.sleep(0.5)  # Be nice to the Savant server, don't fire requests too fast back-to-back

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"[done] {year}: {len(combined):,} rows -> {out_path}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    on1b = "--on1b" in args
    year_args = [a for a in args if a != "--on1b"]
    years = [int(y) for y in year_args] if year_args else list(range(2020, 2026))
    for y in years:
        fetch_year_on1b(y) if on1b else fetch_year(y)
