"""球員個人化站位的量級實驗 v2（v1 的跨模型比較無效，見教訓）。

正確的比較（全部在同一個效應設定的模型內）：
  repositioning gain = E[outs | 個人化最佳站位] − E[outs | 零效應最佳站位]
  兩者都用「帶該陣容效應」的模型評估 → 回答「為這套陣容重新優化值多少」。
v1 教訓：不同效應設定的 exp_outs 不可直接相減（g 的 ad_z 中心化會整體平移
預測水準，好陣容會出現假的負差）。

同時做外插診斷：g·ad_z 是線性項，訓練支撐外（nearest ad_min 大於 ~25°）仍
線性成長，可能被優化器鑽漏洞。記錄各設定最佳解下的 ad_min 分布對照。

陣容設定：全隊 g=P90 / 全隊 g=P10 / 只有 SS 槽 g=P90（單槽解讀）。
25 位 2025 年 GB 最多的打者。

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
from src.if_optimize import (expected_outs, fetch_batter_gbs,
                             league_average_positions, optimize_infield,
                             positions_to_params)

BAYES_DIR = Path(__file__).resolve().parent.parent / "models" / "if_gb" / "bayes"
YEAR = 2025
N_BATTERS = 25
N_RESTARTS = 20


def ad_stats(balls, angles):
    spray = balls["spray_deg"].to_numpy(float)
    ad = np.abs(np.asarray(angles)[None, :] - spray[:, None]).min(axis=1)
    return float(np.mean(ad)), float(np.quantile(ad, 0.95))


def main() -> None:
    meta = json.loads((BAYES_DIR / "IF_meta.json").read_text(encoding="utf-8"))
    model = joblib.load(BAYES_DIR / "if_bayes_group_pipeline.joblib")
    eff = pd.read_csv(BAYES_DIR / "IF_player_effects.csv")
    g_lo, g_hi = eff["g"].quantile([0.1, 0.9])
    print(f"g 分布（{len(eff)} 位野手）：P10={g_lo:+.3f}  P90={g_hi:+.3f}")

    def pe(g4):
        return {"alpha": np.zeros(4), "g": np.asarray(g4, float),
                "ad_mean": meta["ad_mean"], "ad_std": meta["ad_std"]}

    lineups = {
        "全隊P90": pe([g_hi] * 4),
        "全隊P10": pe([g_lo] * 4),
        "只SS P90": pe([0.0, 0.0, 0.0, g_hi]),
    }

    with psycopg2.connect(DSN) as conn:
        batters = pd.read_sql(
            "SELECT batter, count(*) AS n FROM statcast "
            "WHERE bb_type='ground_ball' AND game_year=%(y)s AND hc_x IS NOT NULL "
            "GROUP BY batter ORDER BY n DESC LIMIT %(k)s",
            conn, params={"y": YEAR, "k": N_BATTERS})

    avg_angles, avg_depths = league_average_positions([YEAR])
    avg_start = positions_to_params(avg_angles, avg_depths)
    base_kw = dict(n_restarts=N_RESTARTS, seed=42, extra_starts=[avg_start])

    rows = []
    for i, b in enumerate(batters.itertuples(), 1):
        balls = fetch_batter_gbs(b.batter, [YEAR])
        if len(balls) < 50:
            continue
        r0 = optimize_infield(balls, model, **base_kw)
        ad0_mean, ad0_p95 = ad_stats(balls, r0["angles"])
        for name, effects in lineups.items():
            r1 = optimize_infield(balls, model, player_effects=effects, **base_kw)
            # 零效應最佳解在「這套陣容的模型」下重新評分（同模型比較）
            e_r0 = expected_outs(model, balls, r0["angles"], r0["depths"], effects)
            d_ang = np.abs(r1["angles"] - r0["angles"])
            x0 = r0["depths"] * np.sin(np.radians(r0["angles"]))
            y0 = r0["depths"] * np.cos(np.radians(r0["angles"]))
            x1 = r1["depths"] * np.sin(np.radians(r1["angles"]))
            y1 = r1["depths"] * np.cos(np.radians(r1["angles"]))
            move = np.hypot(x1 - x0, y1 - y0)
            ad1_mean, ad1_p95 = ad_stats(balls, r1["angles"])
            rows.append({"batter": b.batter, "lineup": name,
                         "gain": r1["exp_outs"] - e_r0,
                         "mean_move_ft": move.mean(), "max_move_ft": move.max(),
                         "mean_d_ang": d_ang.mean(), "max_d_ang": d_ang.max(),
                         "ad_mean_shift": ad1_mean - ad0_mean,
                         "ad_p95_shift": ad1_p95 - ad0_p95})
        if i % 5 == 0:
            print(f"  [{i}/{len(batters)}]", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== 為陣容重新優化的增益與站位移動（同模型比較）===")
    for name in lineups:
        s = df[df["lineup"] == name]
        print(f"{name}：")
        print(f"  repositioning gain: 中位 {s['gain'].median():+.5f} "
              f"（P90 {s['gain'].quantile(.9):+.5f}，×450GB≈"
              f"{s['gain'].median() * 450:+.1f} 出局/季）")
        print(f"  移動（呎）: 中位單槽平均 {s['mean_move_ft'].median():.1f} / "
              f"中位最大槽 {s['max_move_ft'].median():.1f} / "
              f"跨打者最大 {s['max_move_ft'].max():.1f}")
        print(f"  外插診斷 ad_min 位移: mean 中位 {s['ad_mean_shift'].median():+.2f}°、"
              f"P95 中位 {s['ad_p95_shift'].median():+.2f}°")


if __name__ == "__main__":
    main()
