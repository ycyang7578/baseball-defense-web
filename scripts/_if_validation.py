"""共用：跨年驗證腳本的合格打者查詢。

validate_if_positioning.py／validate_if_runvalue.py／validate_if_dp.py 都用同一套
「訓練年滾地球數 >= min_train 且測試年滾地球數 >= min_test」篩選邏輯抓合格打者，
原本各自複製一份同樣的 SQL，抽成這裡的 qualifying_batters。
"""
import pandas as pd
import psycopg2


def qualifying_batters(dsn: str, train_years: list[int], test_year: int,
                       min_train: int, min_test: int) -> pd.DataFrame:
    sql = """
        SELECT batter, max(stand) AS stand,
               count(*) FILTER (WHERE game_year = ANY(%(tr)s)) AS n_train,
               count(*) FILTER (WHERE game_year = %(te)s) AS n_test
        FROM statcast
        WHERE bb_type = 'ground_ball' AND hc_x IS NOT NULL
          AND game_year = ANY(%(all)s)
        GROUP BY batter
        HAVING count(*) FILTER (WHERE game_year = ANY(%(tr)s)) >= %(mtr)s
           AND count(*) FILTER (WHERE game_year = %(te)s) >= %(mte)s
        ORDER BY batter
    """
    with psycopg2.connect(dsn) as conn:
        return pd.read_sql(sql, conn, params={
            "tr": train_years, "te": test_year, "all": train_years + [test_year],
            "mtr": min_train, "mte": min_test})
