"""球員個人化站位的量級實驗：極端 range 陣容 vs 聯盟平均，站位差多少。

貝葉斯球員層已通過樣本外驗收（alpha/g 各自改善 2025 logloss，見
train_if_bayes.py 輸出）。這裡回答產品問題：把 g（range 斜率）P10/P90 的
野手擺滿四個位置，最佳站位相對聯盟平均野手移動幾度/幾呎、期望出局率差多少。
25 位 2025 年 GB 最多的打者（收斂穩定性教訓：樣本至少 25）。

執行：python scripts/exp_if_personalized_positions.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN
from src.if_optimize import (POSITIONS, dirt_max_depth, fetch_batter_gbs,
                             league_average_positions, optimize_infield,
                             positions_to_params)

BAYES_DIR = Path(__file__).resolve().parent.parent / "models" / "if_gb" / "bayes"
YEAR = 2025
N_BATTERS = 25
N_RESTARTS = 20
MIN_BALLS = 200   # g 估計夠穩的野手才拿來當「極端值」


def main() -> None:
    meta = json.loads((BAYES_DIR / "IF_meta.json").read_text(encoding="utf-8"))
    model = joblib.load(BAYES_DIR / "if_bayes_group_pipeline.joblib")
    eff = pd.read_csv(BAYES_DIR / "IF_player_effects.csv")

    # 用歸責球數過濾（訓練資料的 nearest 次數沒存，這裡用主範圍球數 proxy：
    # 直接以效應絕對值的分位數呈現，並附上有多少野手支撐）
    g_lo, g_hi = eff["g"].quantile([0.1, 0.9])
    print(f"g 分布（{len(eff)} 位野手）：P10={g_lo:+.3f}  P90={g_hi:+.3f}  "
          f"SD={eff['g'].std():.3f}")

    with psycopg2.connect(DSN) as conn:
        batters = pd.read_sql(
            "SELECT batter, count(*) AS n FROM statcast "
            "WHERE bb_type='ground_ball' AND game_year=%(y)s AND hc_x IS NOT NULL "
            "GROUP BY batter ORDER BY n DESC LIMIT %(k)s",
            conn, params={"y": YEAR, "k": N_BATTERS})

    avg_angles, avg_depths = league_average_positions([YEAR])
    avg_start = positions_to_params(avg_angles, avg_depths)
    base_kw = dict(n_restarts=N_RESTARTS, seed=42, extra_starts=[avg_start])

    def effects(g_val):
        return {"alpha": np.zeros(4), "g": np.full(4, g_val),
                "ad_mean": meta["ad_mean"], "ad_std": meta["ad_std"]}

    rows = []
    for i, b in enumerate(batters.itertuples(), 1):
        balls = fetch_batter_gbs(b.batter, [YEAR])
        if len(balls) < 50:
            continue
        r0 = optimize_infield(balls, model, **base_kw)
        r_hi = optimize_infield(balls, model, player_effects=effects(g_hi), **base_kw)
        r_lo = optimize_infield(balls, model, player_effects=effects(g_lo), **base_kw)
        for tag, r in (("hi", r_hi), ("lo", r_lo)):
            d_ang = np.abs(r["angles"] - r0["angles"])
            lat0 = r0["depths"] * np.sin(np.radians(r0["angles"]))
            lat1 = r["depths"] * np.sin(np.radians(r["angles"]))
            move_ft = np.hypot(
                r["depths"] * np.cos(np.radians(r["angles"]))
                - r0["depths"] * np.cos(np.radians(r0["angles"])), lat1 - lat0)
            rows.append({"batter": b.batter, "lineup": tag,
                         "d_exp": r["exp_outs"] - r0["exp_outs"],
                         "max_d_angle": d_ang.max(), "mean_d_angle": d_ang.mean(),
                         "max_move_ft": move_ft.max(),
                         "mean_move_ft": move_ft.mean()})
        if i % 5 == 0:
            print(f"  [{i}/{len(batters)}]", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== 極端 range 陣容 vs 聯盟平均野手（站位差異）===")
    for tag, label in (("hi", f"P90 range（g={g_hi:+.3f}）"),
                       ("lo", f"P10 range（g={g_lo:+.3f}）")):
        sub = df[df["lineup"] == tag]
        print(f"{label}：")
        print(f"  期望出局率差 Δ: 中位 {sub['d_exp'].median():+.4f} "
              f"（P10 {sub['d_exp'].quantile(.1):+.4f} / "
              f"P90 {sub['d_exp'].quantile(.9):+.4f}）")
        print(f"  站位移動（呎）: 中位單槽平均 {sub['mean_move_ft'].median():.1f}、"
              f"中位最大槽 {sub['max_move_ft'].median():.1f}、"
              f"跨打者最大 {sub['max_move_ft'].max():.1f}")
        print(f"  角度移動（度）: 中位平均 {sub['mean_d_angle'].median():.2f}、"
              f"中位最大 {sub['max_d_angle'].median():.2f}")


if __name__ == "__main__":
    main()
