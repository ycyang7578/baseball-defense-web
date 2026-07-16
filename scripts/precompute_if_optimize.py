"""離線預算內野站位最佳化，服務 web 內野頁（線上即時算不可行，見 DDL 註解）。

對每個 (打者, 年份)（該年滾地球 ≥ --min-gb）：
- 用該年歷史滾地球優化四內野手站位（n_restarts 預設 50，離線不必省；
  收斂測試見 scripts/test_if_convergence.py，n=20 已是線上下限）
- 記錄該年聯盟平均站位的期望出局率當對照
- 逐球存兩組站位下的出局機率（前端上色用）

逐 (打者, 年份) checkpoint 落盤（這台機器會 BSOD），positions 列是 commit
marker：balls 先寫、positions 後寫，續跑時先清掉沒有 marker 的孤兒 balls。
算完後把兩張 CSV 灌進 PostgreSQL；--target-dsn 可指定雲端 DB（同
precompute_batter_balls.py 的部署模式）。聯盟平均站位另存
data/precomputed/if_league_positions.json（API startup 讀）。

Usage:
    python scripts/precompute_if_optimize.py                     # 算 + 灌本機 DB
    python scripts/precompute_if_optimize.py --load-only --target-dsn "postgresql://..."
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import psycopg2

from src.config import DSN
from src.if_dataset import HOME_X, HOME_Y, OUT_EVENTS
from src.if_optimize import (POSITIONS, expected_outs, fetch_batter_gbs,
                             geometry_features, league_average_positions,
                             optimize_infield, positions_to_params)

from _pg_load import copy_dataframe

# Savant hc 座標單位 → 呎（慣用換算；出局球換算後中位深度 ~118 呎落在內野手
# 深度帶、對得上歸責野手位置，經驗上成立）
FT_PER_UNIT = 2.5

BASE = Path(__file__).resolve().parent.parent
SQL_DIR = BASE / "scripts" / "sql"
PRE_DIR = BASE / "data" / "precomputed"
POS_CSV = PRE_DIR / "if_positions_rows.csv"
GBS_CSV = PRE_DIR / "if_gbs_rows.csv"
LEAGUE_JSON = PRE_DIR / "if_league_positions.json"
# 2026-07-10 起優化器 = 貝葉斯群體層 pipeline（後驗平均，介面同 GLM；
# 見 scripts/export_if_bayes.py）。此處預算零效應（聯盟平均野手）解
MODEL = BASE / "models" / "if_gb" / "bayes" / "if_bayes_group_pipeline.joblib"
SEED = 42

POS_COLS = ["batter", "game_year", "stand", "n_gb", "hp_to_1b",
            "exp_outs_league", "exp_outs_opt",
            "angle_1b", "depth_1b", "angle_2b", "depth_2b",
            "angle_3b", "depth_3b", "angle_ss", "depth_ss"]
GBS_COLS = ["batter", "game_year", "spray_deg", "ball_x", "ball_y",
            "launch_speed", "launch_angle", "is_out", "p_out_league", "p_out_opt"]


def gbs_frame(batter: int, year: int, balls: pd.DataFrame,
              p_league, p_opt) -> pd.DataFrame:
    """逐球表的一個 (打者, 年份) 區塊。ball_x/ball_y 是 hc 換算成呎的
    「被處理/撿起位置」（展示用；語意與內生性見 DDL 註解）。"""
    return pd.DataFrame({
        "batter": batter, "game_year": year,
        "spray_deg": balls["spray_deg"].round(2),
        "ball_x": ((balls["hc_x"] - HOME_X) * FT_PER_UNIT).round(1),
        "ball_y": ((HOME_Y - balls["hc_y"]) * FT_PER_UNIT).round(1),
        "launch_speed": balls["launch_speed"],
        "launch_angle": balls["launch_angle"],
        "is_out": balls["events"].isin(OUT_EVENTS),
        "p_out_league": np.round(p_league, 4), "p_out_opt": np.round(p_opt, 4),
    })[GBS_COLS]


def candidate_batters(year: int, min_gb: int) -> list[int]:
    """粗篩（基本欄位條件，是 fetch_batter_gbs 精確篩選的超集，不會漏人）。"""
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(
            "SELECT batter FROM statcast "
            "WHERE bb_type='ground_ball' AND game_year=%(y)s AND hc_x IS NOT NULL "
            "GROUP BY batter HAVING count(*) >= %(m)s ORDER BY batter",
            conn, params={"y": year, "m": min_gb})
    return df["batter"].tolist()


def load_done() -> set[tuple[int, int]]:
    """已完成的 (batter, year)；順便清掉沒有 positions marker 的孤兒 balls 列。"""
    done = set()
    if POS_CSV.exists():
        pos = pd.read_csv(POS_CSV)
        done = set(zip(pos["batter"], pos["game_year"]))
    if GBS_CSV.exists() and done:
        gbs = pd.read_csv(GBS_CSV)
        keep = pd.Series(list(zip(gbs["batter"], gbs["game_year"]))).isin(done)
        if not keep.all():
            print(f"  清掉 {(~keep).sum()} 列孤兒 balls（上次當機的未完成打者）")
            gbs[keep.to_numpy()].to_csv(GBS_CSV, index=False)
    elif GBS_CSV.exists() and not done:
        GBS_CSV.unlink()
    return done


def compute(years: list[int], min_gb: int, n_restarts: int) -> None:
    model = joblib.load(MODEL)
    done = load_done()
    if done:
        print(f"checkpoint: 已完成 {len(done)} 個 (打者, 年份)，續跑其餘")

    league = {}
    for year in years:
        avg_angles, avg_depths = league_average_positions([year])
        league[year] = {p: [round(a, 2), round(d, 2)]
                        for p, a, d in zip(POSITIONS, avg_angles, avg_depths)}
        avg_start = positions_to_params(avg_angles, avg_depths)

        batters = [b for b in candidate_batters(year, min_gb)
                   if (b, year) not in done]
        print(f"=== {year}：候選 {len(batters)} 位（粗篩 GB>={min_gb}）===")
        t0 = time.perf_counter()
        for i, batter in enumerate(batters, 1):
            balls = fetch_batter_gbs(batter, [year])
            if len(balls) < min_gb:
                continue
            res = optimize_infield(balls, model, n_restarts=n_restarts,
                                   seed=SEED, extra_starts=[avg_start])
            exp_league = expected_outs(model, balls, avg_angles, avg_depths)
            p_league = model.predict_proba(
                geometry_features(balls, avg_angles, avg_depths))[:, 1]
            p_opt = model.predict_proba(
                geometry_features(balls, res["angles"], res["depths"]))[:, 1]

            gbs = gbs_frame(batter, year, balls, p_league, p_opt)
            gbs.to_csv(GBS_CSV, mode="a", index=False, header=not GBS_CSV.exists())

            row = {"batter": batter, "game_year": year,
                   "stand": balls["stand"].mode().iloc[0], "n_gb": len(balls),
                   "hp_to_1b": round(float(balls["hp_to_1b"].iloc[0]), 3),
                   "exp_outs_league": round(exp_league, 6),
                   "exp_outs_opt": round(res["exp_outs"], 6)}
            for p, a, d in zip(POSITIONS, res["angles"], res["depths"]):
                row[f"angle_{p.lower()}"] = round(float(a), 2)
                row[f"depth_{p.lower()}"] = round(float(d), 2)
            pd.DataFrame([row])[POS_COLS].to_csv(
                POS_CSV, mode="a", index=False, header=not POS_CSV.exists())

            if i % 10 == 0 or i == len(batters):
                rate = (time.perf_counter() - t0) / i
                print(f"  [{i}/{len(batters)}] 平均 {rate:.1f}s/位，"
                      f"本年剩餘約 {rate * (len(batters) - i) / 60:.0f} 分", flush=True)

    LEAGUE_JSON.write_text(json.dumps(league, indent=2), encoding="utf-8")
    print(f"聯盟平均站位 → {LEAGUE_JSON.name}")


def refresh_gbs() -> None:
    """只重算逐球表：沿用 DB 已存的最佳化站位與該年聯盟平均，重抓球、重算
    兩組站位的出局率（不重跑優化）。改逐球表 schema（如新增 ball_x/ball_y）
    或展示欄位時用這個，幾十分鐘就能重建，不用重跑 3 小時的優化。"""
    model = joblib.load(MODEL)
    with psycopg2.connect(DSN) as conn:
        pos = pd.read_sql(
            "SELECT * FROM precomputed_if_positions ORDER BY game_year, batter", conn)
    print(f"重算 {len(pos)} 個 (打者, 年份) 的逐球表")
    if GBS_CSV.exists():
        GBS_CSV.unlink()

    league = {}
    t0 = time.perf_counter()
    for i, row in enumerate(pos.itertuples(), 1):
        if row.game_year not in league:
            league[row.game_year] = league_average_positions([row.game_year])
        avg_angles, avg_depths = league[row.game_year]
        balls = fetch_batter_gbs(row.batter, [row.game_year])
        if len(balls) != row.n_gb:
            print(f"  [!] 打者 {row.batter} {row.game_year}: 球數 {len(balls)} != "
                  f"已存 n_gb {row.n_gb}（statcast 資料變動？）")
        opt_angles = np.array([row.angle_1b, row.angle_2b, row.angle_3b, row.angle_ss])
        opt_depths = np.array([row.depth_1b, row.depth_2b, row.depth_3b, row.depth_ss])
        p_league = model.predict_proba(
            geometry_features(balls, avg_angles, avg_depths))[:, 1]
        p_opt = model.predict_proba(
            geometry_features(balls, opt_angles, opt_depths))[:, 1]
        gbs = gbs_frame(row.batter, row.game_year, balls, p_league, p_opt)
        gbs.to_csv(GBS_CSV, mode="a", index=False, header=not GBS_CSV.exists())
        if i % 100 == 0 or i == len(pos):
            rate = (time.perf_counter() - t0) / i
            print(f"  [{i}/{len(pos)}] 剩餘約 {rate * (len(pos) - i) / 60:.0f} 分", flush=True)


def load_to_db(target_dsn: str) -> None:
    pos = pd.read_csv(POS_CSV)[POS_COLS]
    gbs = pd.read_csv(GBS_CSV)[GBS_COLS]
    # 只灌有 marker 的球（防孤兒），bool 轉 PostgreSQL 認的 t/f 由 to_csv 的 True/False 處理
    key = set(zip(pos["batter"], pos["game_year"]))
    gbs = gbs[[k in key for k in zip(gbs["batter"], gbs["game_year"])]]
    with psycopg2.connect(target_dsn) as conn:
        with conn.cursor() as cur:
            # 純衍生表：直接重建，schema 變動（如新增欄位）才不會卡在 IF NOT EXISTS
            cur.execute("DROP TABLE IF EXISTS precomputed_if_positions")
            cur.execute("DROP TABLE IF EXISTS precomputed_if_gbs")
            for ddl in ("create_precomputed_if_positions_table.sql",
                        "create_precomputed_if_gbs_table.sql"):
                cur.execute((SQL_DIR / ddl).read_text(encoding="utf-8"))
        conn.commit()
        copy_dataframe(conn, "precomputed_if_positions", pos)
        copy_dataframe(conn, "precomputed_if_gbs", gbs)
    label = target_dsn.split("@")[-1] if "@" in target_dsn else "本機"
    print(f"完成：precomputed_if_positions {len(pos):,} 筆、"
          f"precomputed_if_gbs {len(gbs):,} 筆 → {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--min-gb", type=int, default=50)
    parser.add_argument("--n-restarts", type=int, default=50)
    parser.add_argument("--load-only", action="store_true",
                        help="跳過計算，直接把現有 CSV 灌進 DB")
    parser.add_argument("--refresh-gbs", action="store_true",
                        help="只重算逐球表（沿用 DB 已存站位，不重跑優化）")
    parser.add_argument("--target-dsn", default=None,
                        help="要灌入的 DB（預設本機；部署時帶 Neon DSN）")
    args = parser.parse_args()

    if args.refresh_gbs:
        refresh_gbs()
    elif not args.load_only:
        compute(args.years, args.min_gb, args.n_restarts)
    load_to_db(args.target_dsn or DSN)


if __name__ == "__main__":
    main()
