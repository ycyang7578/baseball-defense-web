"""Bulk-load raw Parquet files into PostgreSQL via COPY (fast path, not row-by-row INSERT).

Re-runnable: each table is TRUNCATEd before reload, so this always rebuilds the DB
from data/raw/ from scratch (e.g. on a fresh machine).
"""
import io
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
YEARS = range(2020, 2026)


def _copy(conn, table: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    conn.commit()


def load_statcast(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE statcast;")
    conn.commit()
    for year in YEARS:
        path = DATA_DIR / "statcast" / f"{year}.parquet"
        df = pd.read_parquet(path)
        _copy(conn, "statcast", df)
        print(f"[loaded] statcast {year}: {len(df):,} rows")


def load_fielder_positioning(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fielder_positioning;")
    conn.commit()
    for year in YEARS:
        path = DATA_DIR / "positioning" / f"{year}.parquet"
        df = pd.read_parquet(path)
        # 同名球員轉隊會有多筆同位置同年資料，保留出賽數較多的球隊紀錄
        df = df.sort_values("pa", ascending=False).drop_duplicates(
            subset=["fielder_id", "season", "position"], keep="first"
        )
        _copy(conn, "fielder_positioning", df)
        print(f"[loaded] fielder_positioning {year}: {len(df):,} rows")


def load_sprint_speed(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE sprint_speed;")
    conn.commit()
    for year in YEARS:
        path = DATA_DIR / "sprint_speed" / f"{year}.parquet"
        if not path.exists():
            continue  # 尚未抓取的年份直接跳過
        df = pd.read_parquet(path)
        _copy(conn, "sprint_speed", df)
        print(f"[loaded] sprint_speed {year}: {len(df):,} rows")


def load_savant_fielding(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE savant_fielding;")
    conn.commit()
    for year in YEARS:
        path = DATA_DIR / "savant_fielding" / f"{year}.parquet"
        if not path.exists():
            continue  # 目前只抓了部分年份，尚未覆蓋的年份直接跳過
        df = pd.read_parquet(path)
        _copy(conn, "savant_fielding", df)
        print(f"[loaded] savant_fielding {year}: {len(df):,} rows")


if __name__ == "__main__":
    conn = psycopg2.connect(DSN)
    try:
        load_statcast(conn)
        load_fielder_positioning(conn)
        load_sprint_speed(conn)
        load_savant_fielding(conn)
    finally:
        conn.close()
