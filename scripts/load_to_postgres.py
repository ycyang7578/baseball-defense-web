"""Bulk-load raw Parquet files into PostgreSQL via COPY (fast path, not row-by-row INSERT).

Re-runnable: each table is TRUNCATEd before reload, so this always rebuilds the DB
from data/raw/ from scratch (e.g. on a fresh machine).
"""
import sys
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN

from _pg_load import copy_dataframe as _copy
from _pg_load import dedupe_positioning

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
YEARS = range(2020, 2026)


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
        df = dedupe_positioning(df)
        _copy(conn, "fielder_positioning", df)
        print(f"[loaded] fielder_positioning {year}: {len(df):,} rows")


def load_fielder_positioning_on1b(conn) -> None:
    """一壘有人切分（{year}_on1b.parquet，2023 起才有；階段B 雙殺模型用）。"""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fielder_positioning_on1b;")
    conn.commit()
    for year in YEARS:
        path = DATA_DIR / "positioning" / f"{year}_on1b.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = dedupe_positioning(df)
        _copy(conn, "fielder_positioning_on1b", df)
        print(f"[loaded] fielder_positioning_on1b {year}: {len(df):,} rows")


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
    # 無參數=全部重載；帶表名（如 `on1b`）只載該表，避免為單表跑全量 TRUNCATE 重灌
    _LOADERS = {
        "statcast": load_statcast,
        "positioning": load_fielder_positioning,
        "on1b": load_fielder_positioning_on1b,
        "sprint_speed": load_sprint_speed,
        "savant_fielding": load_savant_fielding,
    }
    targets = sys.argv[1:] or list(_LOADERS)
    conn = psycopg2.connect(DSN)
    try:
        for t in targets:
            _LOADERS[t](conn)
    finally:
        conn.close()
