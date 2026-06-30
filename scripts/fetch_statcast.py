"""Fetch full Statcast play-level data by year (unfiltered) and save as Parquet.

Usage:
    python fetch_statcast.py [year ...]
    (no args = fetch 2020-2025)
"""
import sys
import time
from pathlib import Path

import pybaseball as pb

pb.cache.enable()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "statcast"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_year(year: int) -> None:
    out_path = OUTPUT_DIR / f"{year}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    print(f"[fetch] {year} ...")
    t0 = time.time()
    df = pb.statcast(f"{year}-01-01", f"{year}-12-31")
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[done] {year}: {len(df):,} rows, {size_mb:.1f} MB, {elapsed:.0f}s -> {out_path}")


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or list(range(2020, 2026))
    for y in years:
        fetch_year(y)
