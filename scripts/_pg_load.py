"""Shared: batch-import logic for loading DataFrames into PostgreSQL.

load_to_postgres.py, append_years_to_db.py, and fetch_savant_fielding.py all need
to push fetched parquet/DataFrame data into the database via COPY. Each script used
to duplicate the same io.StringIO -> to_csv -> copy_expert logic; it has been
factored out here into copy_dataframe.

dedupe_positioning is the cleanup rule for fielder_positioning /
fielder_positioning_on1b: if a player has multiple rows for the same season and
position (e.g. due to a trade), keep the one with the higher plate-appearance (pa)
count.
"""
import io

import pandas as pd
from psycopg2.extensions import connection as PgConnection


def copy_dataframe(conn: PgConnection, table: str, df: pd.DataFrame) -> None:
    """Bulk-write df into table via COPY (much faster than row-by-row INSERT), then commit."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    conn.commit()


def dedupe_positioning(df: pd.DataFrame) -> pd.DataFrame:
    """If there are multiple rows for the same (fielder_id, season, position), e.g. due to a trade, keep the one with the higher plate-appearance (pa) count."""
    return df.sort_values("pa", ascending=False).drop_duplicates(
        subset=["fielder_id", "season", "position"], keep="first"
    )
