"""Fetch Baseball Savant's official per-play outfield fielding records (the basis of official OAA).

API endpoints found via reverse-engineering (not pybaseball, not officially documented):
  GET /leaderboard/outfield_directional_outs_above_average?year={year}
      -> all qualifying outfielders (player_id, position, n_plays)
  GET /player-services/gamelogs?playerId={id}&playerType={pos_code}&viewType=statcastGameLogsFielding&season={year}
      -> one player's play-by-play fielding chances (game_pk, ab_num, ev, la, dist, catch_prob, ...)

This is the play set that Baseball Savant's own OAA computation is based on -- used to restrict
our own evaluation to the same population when comparing against real_2025_oaa.csv.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

from _savant_leaderboard import BASE_URL, HEADERS, fetch_leaderboard_raw

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "savant_fielding"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_leaderboard(year: int) -> pd.DataFrame:
    raw = fetch_leaderboard_raw("outfield_directional_outs_above_average", {"year": year})
    return pd.DataFrame({
        "player_id": raw["resp_fielder_id"].astype(int),
        "player_name": raw["name_display_last_first"],
        "pos_code": raw["resp_fielder_pos"].astype(str),
        "n_plays": raw["n"].astype(int),
    })


def fetch_player_fielding(player_id: int, pos_code: str, year: int) -> list[dict]:
    r = requests.get(f"{BASE_URL}/player-services/gamelogs",
                      params={"playerId": player_id, "playerType": pos_code,
                              "viewType": "statcastGameLogsFielding", "season": year},
                      headers=HEADERS, timeout=30)
    r.raise_for_status()
    return json.loads(r.text)


def parse_plays(raw: list[dict], player_id: int) -> pd.DataFrame:
    rows = []
    for play in raw:
        if not play.get("ev"):
            continue  # 跳過無EV的非守備機會（三振、保送等）
        rows.append({
            "player_id": player_id,
            "game_pk": int(play["game_pk"]) if play.get("game_pk") else None,
            "play_id": play.get("play_id"),
            "at_bat_number": int(play["ab_num"]) if play.get("ab_num") else None,
            "game_date": play.get("gd"),
            "ev": float(play["ev"]),
            "la": float(play["la"]) if play.get("la") else None,
            "dist": float(play["dist"]) if play.get("dist") else None,
            "catch_prob": float(play["xban"]) if play.get("xban") else None,
            "star_savant": play.get("xbad"),
        })
    return pd.DataFrame(rows)


def fetch_year(year: int, sleep_sec: float = 1.0) -> None:
    out_path = OUTPUT_DIR / f"{year}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    lb = fetch_leaderboard(year)
    print(f"[leaderboard] {len(lb)} 位外野手")

    frames = []
    for i, row in lb.iterrows():
        try:
            raw = fetch_player_fielding(int(row["player_id"]), row["pos_code"], year)
            df = parse_plays(raw, int(row["player_id"]))
            frames.append(df)
            if (i + 1) % 50 == 0:
                print(f"[fetch] {i + 1}/{len(lb)} 完成")
        except Exception as e:
            print(f"[fetch] player_id={row['player_id']} 失敗: {e}")
        time.sleep(sleep_sec)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"[done] {year}: {len(combined):,} rows -> {out_path}")


def reload_db(years: list[int]) -> None:
    """將指定年份的 parquet 追加進 savant_fielding（不 TRUNCATE，只插入缺少的年份）。"""
    import sys
    import psycopg2
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import DSN
    from _pg_load import copy_dataframe
    with psycopg2.connect(DSN) as conn:
        for year in years:
            path = OUTPUT_DIR / f"{year}.parquet"
            if not path.exists():
                print(f"[skip db] {year}.parquet 不存在")
                continue
            df = pd.read_parquet(path)
            # 刪除該年份舊資料再插入（避免重複）
            with conn.cursor() as cur:
                cur.execute("DELETE FROM savant_fielding WHERE game_date >= %s AND game_date < %s",
                            (f"{year}-01-01", f"{year+1}-01-01"))
            copy_dataframe(conn, "savant_fielding", df)
            print(f"[db] savant_fielding {year}: {len(df):,} rows loaded")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024],
                        help="要抓取的年份（預設 2021-2024）")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--no-db", action="store_true", help="只抓 parquet，不寫入 DB")
    args = parser.parse_args()

    for year in args.years:
        print(f"\n===== {year} =====")
        fetch_year(year, sleep_sec=args.sleep)

    if not args.no_db:
        print("\n===== 寫入 PostgreSQL =====")
        reload_db(args.years)
