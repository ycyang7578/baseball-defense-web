"""Incrementally add specific years to statcast and fielder_positioning tables.

Unlike load_to_postgres.py (which TRUNCATEs), this script deletes only the
target years before re-inserting, leaving other years intact.

Usage:
    python append_years_to_db.py 2017 2018 2019
"""
import io
import sys
from pathlib import Path

import pandas as pd
import psycopg2

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DSN = "host=localhost dbname=baseball user=postgres password=postgres"


def _copy(conn, table: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    conn.commit()


def append_statcast(conn, year: int) -> None:
    path = DATA_DIR / "statcast" / f"{year}.parquet"
    if not path.exists():
        print(f"[skip statcast] {path} not found")
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM statcast WHERE game_year = %s", (year,))
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        print(f"[delete statcast] {year}: removed {deleted} existing rows")
    df = pd.read_parquet(path)
    _copy(conn, "statcast", df)
    print(f"[loaded statcast] {year}: {len(df):,} rows")


def append_positioning(conn, year: int) -> None:
    path = DATA_DIR / "positioning" / f"{year}.parquet"
    if not path.exists():
        print(f"[skip positioning] {path} not found")
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM fielder_positioning WHERE season = %s", (year,))
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        print(f"[delete positioning] {year}: removed {deleted} existing rows")
    df = pd.read_parquet(path)
    df = df.sort_values("pa", ascending=False).drop_duplicates(
        subset=["fielder_id", "season", "position"], keep="first"
    )
    _copy(conn, "fielder_positioning", df)
    print(f"[loaded positioning] {year}: {len(df):,} rows")


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]]
    if not years:
        print("Usage: python append_years_to_db.py 2017 2018 2019")
        sys.exit(1)

    conn = psycopg2.connect(DSN)
    try:
        for year in years:
            print(f"\n--- {year} ---")
            append_statcast(conn, year)
            append_positioning(conn, year)
    finally:
        conn.close()
    print("\nDone.")
