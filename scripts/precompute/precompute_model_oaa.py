"""
計算指定年度各外野手在我方模型下的 OAA 與星級分布（全量守備機會）。

Usage:
    python -m scripts.precompute.precompute_model_oaa [target_year]   # default 2025
    python -m scripts.precompute.precompute_model_oaa 2024

使用合併外野手模型 models/{target_year}/OF/，LF+CF+RF 共用同一 scaler 與群體層參數。
API 端以跨位置統一中心化（avg_oaa_per_ball = LF+CF+RF 合計）。
輸出：
  data/precomputed/model_oaa_{target_year}.csv
  PostgreSQL: model_oaa, model_star_stats
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.defender_features import get_defender_opportunities, mark_official
from src.config import DSN

_parser = argparse.ArgumentParser()
_parser.add_argument("target_year", type=int, nargs="?", default=2025)
_args = _parser.parse_args()

TARGET_YEAR  = _args.target_year
MODELS_DIR   = Path(__file__).resolve().parent.parent.parent / "models" / str(TARGET_YEAR)
OUT_PATH     = Path(__file__).resolve().parent.parent.parent / "data" / "precomputed" / f"model_oaa_{TARGET_YEAR}.csv"
POSITIONS    = ["LF", "CF", "RF"]
FEATURE_COLS = ["speed", "cos_angle", "sin_angle", "fielder_dist"]


STAR_THRESHOLDS = [0.25, 0.50, 0.75, 0.90, 0.95]  # 5★ ≤ 0.25, 4★ ≤ 0.50, ..., 0★ > 0.95

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def assign_star(catch_prob: float) -> int:
    for star, threshold in zip([5, 4, 3, 2, 1], STAR_THRESHOLDS):
        if catch_prob <= threshold:
            return star
    return 0


# 載入合併 OF 模型（一次即可，三個位置共用）
of_dir   = MODELS_DIR / "OF"
scaler   = joblib.load(of_dir / "OF_scaler.joblib")
group    = pd.read_csv(of_dir / "OF_summary_group.csv", encoding="utf-8-sig", index_col=0)
mu       = group["mean"]
sc_mean  = scaler.mean_
sc_scale = scaler.scale_

rows = []
all_balls = []  # 全位置合併，用於算星級
print(f"目標年份: {TARGET_YEAR}，模型路徑: {MODELS_DIR}")
for pos in POSITIONS:
    print(f"Computing {pos}...", flush=True)
    df = get_defender_opportunities(pos, TARGET_YEAR)
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=FEATURE_COLS + ["caught", "name_fielder"])
    df = mark_official(df)
    df = df[df["is_official"]].copy()

    X = (df[FEATURE_COLS].values - sc_mean) / sc_scale
    logit = (mu["mu_alpha"]
             + mu["mu_beta_speed"] * X[:, 0]
             + mu["mu_beta_cos"]   * X[:, 1]
             + mu["mu_beta_sin"]   * X[:, 2]
             + mu["mu_beta_dist"]  * X[:, 3])
    df["catch_prob"] = sigmoid(logit)
    df["oaa_play"]   = df["caught"] - df["catch_prob"]
    df["star"]       = df["catch_prob"].map(assign_star)

    per_player = (
        df.groupby("name_fielder")
          .agg(model_oaa=("oaa_play", "sum"), n_opp=("oaa_play", "count"))
          .reset_index()
    )
    per_player["position"] = pos
    rows.append(per_player)
    all_balls.append(df[["name_fielder", "caught", "star"]])
    print(f"  {pos}: {len(per_player)} players, {len(df)} balls", flush=True)

result = pd.concat(rows, ignore_index=True)
result["model_oaa"] = result["model_oaa"].round(2)
result.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"\nSaved {len(result)} rows → {OUT_PATH}")

# UPSERT to PostgreSQL
import psycopg2
with psycopg2.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM model_oaa WHERE year = %s", (TARGET_YEAR,))
        for _, row in result.iterrows():
            cur.execute(
                "INSERT INTO model_oaa (name_fielder, position, year, model_oaa, n_opp) VALUES (%s,%s,%s,%s,%s)",
                (row["name_fielder"], row["position"], TARGET_YEAR, float(row["model_oaa"]), int(row["n_opp"]))
            )
    conn.commit()
print(f"Upserted {len(result)} rows into model_oaa")

# ── 星級彙總（跨位置合併，同一球員 LF+CF+RF 合計）───────────────────
balls_all = pd.concat(all_balls, ignore_index=True)
star_rows = []
for name, grp in balls_all.groupby("name_fielder"):
    rec = {"name_fielder": name, "year": TARGET_YEAR}
    for s in range(6):
        sub = grp[grp["star"] == s]
        rec[f"n_opp_{s}stars"]      = len(sub)
        rec[f"n_fieldout_{s}stars"] = int(sub["caught"].sum())
    star_rows.append(rec)
star_df = pd.DataFrame(star_rows)

with psycopg2.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM model_star_stats WHERE year = %s", (TARGET_YEAR,))
        for _, r in star_df.iterrows():
            cur.execute("""
                INSERT INTO model_star_stats
                  (name_fielder, year,
                   n_opp_0stars, n_fieldout_0stars,
                   n_opp_1stars, n_fieldout_1stars,
                   n_opp_2stars, n_fieldout_2stars,
                   n_opp_3stars, n_fieldout_3stars,
                   n_opp_4stars, n_fieldout_4stars,
                   n_opp_5stars, n_fieldout_5stars)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                r["name_fielder"], int(r["year"]),
                int(r["n_opp_0stars"]), int(r["n_fieldout_0stars"]),
                int(r["n_opp_1stars"]), int(r["n_fieldout_1stars"]),
                int(r["n_opp_2stars"]), int(r["n_fieldout_2stars"]),
                int(r["n_opp_3stars"]), int(r["n_fieldout_3stars"]),
                int(r["n_opp_4stars"]), int(r["n_fieldout_4stars"]),
                int(r["n_opp_5stars"]), int(r["n_fieldout_5stars"]),
            ))
    conn.commit()
print(f"Upserted {len(star_df)} rows into model_star_stats")

print("\nTop 10 by model_oaa:")
top10 = result.sort_values("model_oaa", ascending=False).head(10)
for _, r in top10.iterrows():
    print(f"  {r['name_fielder']:30s} {r['position']}  {r['model_oaa']:+.1f}  ({r['n_opp']} opp)")
