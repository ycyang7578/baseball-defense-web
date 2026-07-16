"""共用：DataFrame 灌進 PostgreSQL 的批次匯入邏輯。

load_to_postgres.py、append_years_to_db.py、fetch_savant_fielding.py 都要把
抓下來的 parquet/DataFrame 用 COPY 塞進資料庫，原本各自複製一份同樣的
io.StringIO -> to_csv -> copy_expert 邏輯，抽成這裡的 copy_dataframe。

dedupe_positioning 是 fielder_positioning／fielder_positioning_on1b 的清理規則：
同球員同季同位置若因轉隊等原因有多筆，保留出賽數（pa）較多的那筆。
"""
import io

import pandas as pd


def copy_dataframe(conn, table: str, df: pd.DataFrame) -> None:
    """用 COPY（比逐列 INSERT 快很多）把 df 整批寫進 table，並 commit。"""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    conn.commit()


def dedupe_positioning(df: pd.DataFrame) -> pd.DataFrame:
    """同 (fielder_id, season, position) 若因轉隊等原因有多筆，保留出賽數（pa）較多者。"""
    return df.sort_values("pa", ascending=False).drop_duplicates(
        subset=["fielder_id", "season", "position"], keep="first"
    )
