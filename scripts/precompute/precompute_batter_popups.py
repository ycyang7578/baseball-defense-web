"""生成部署用的精簡預計算表：precomputed_batter_popups（整合頁展示用內野高飛）。

popup 站位無槓桿（出局率 98.6%）不參與優化，但整合頁要呈現打者的所有場內球，
缺了 popup（場內球的 ~7%）圖會少一塊。從有完整 statcast 的來源 DB 撈出、
換算展示座標後灌進目標 DB；部署到雲端用 --target-dsn（同
precompute_batter_balls.py 的部署模式）。

座標＝hc × 2.5 呎（同 precomputed_if_gbs 的展示座標慣例，popup 落點多在
內野，跟滾地球用同一套換算圖上才對得齊）。
launch_speed 缺值保留 NULL（前端只當展示，不做計算）。

Usage:
    python -m scripts.precompute.precompute_batter_popups
    python -m scripts.precompute.precompute_batter_popups --years 2023 2024 2025 --target-dsn "postgresql://..."
"""
import argparse
from pathlib import Path

import pandas as pd
import psycopg2

from src.config import DSN
from src.if_dataset import HOME_X, HOME_Y

from scripts._pg_load import copy_dataframe

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
FT_PER_UNIT = 2.5   # 同 scripts/precompute_if_optimize.py 的展示座標換算

_POPUPS_QUERY = """
    SELECT batter, game_year, hc_x, hc_y, launch_speed, events
    FROM statcast
    WHERE game_year = ANY(%(years)s)
      AND game_type = 'R'
      AND type = 'X'
      AND bb_type = 'popup'
      AND hc_x IS NOT NULL AND hc_y IS NOT NULL
"""

# popup 的非出局事件只有這四種（其餘 field_out/force_out/double_play/
# sac_fly/... 全都至少拿到一個出局），用白名單反向判定最不易漏
_NONOUT_EVENTS = ("single", "double", "triple", "field_error")


def build_batter_popups(source_dsn: str, years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(source_dsn) as conn:
        df = pd.read_sql(_POPUPS_QUERY, conn, params={"years": years})
    print(f"[popups] 原始筆數: {len(df):,}")
    out = pd.DataFrame({
        "batter": df["batter"],
        "game_year": df["game_year"],
        "ball_x": ((df["hc_x"] - HOME_X) * FT_PER_UNIT).round(1),
        "ball_y": ((HOME_Y - df["hc_y"]) * FT_PER_UNIT).round(1),
        "launch_speed": df["launch_speed"],
        "is_out": ~df["events"].isin(_NONOUT_EVENTS),
    })
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2026)))
    parser.add_argument("--source-dsn", default=DSN)
    parser.add_argument("--target-dsn", default=None)
    args = parser.parse_args()

    df = build_batter_popups(args.source_dsn, args.years)

    target_dsn = args.target_dsn or args.source_dsn
    with psycopg2.connect(target_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute((SQL_DIR / "create_precomputed_batter_popups_table.sql")
                        .read_text(encoding="utf-8"))
            cur.execute("TRUNCATE precomputed_batter_popups")
        conn.commit()
        copy_dataframe(conn, "precomputed_batter_popups", df)

    target_label = target_dsn.split("@")[-1] if "@" in target_dsn else target_dsn
    print(f"完成：precomputed_batter_popups {len(df):,} 筆 → {target_label}")


if __name__ == "__main__":
    main()
