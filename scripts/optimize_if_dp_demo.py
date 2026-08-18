"""階段B 示範：一壘有人的雙殺情境優化，對照聯盟站位與無人在壘最佳解。

對每位示範打者（2023–24 滾地球、0 與 1 出局兩種情境）比較三組站位的
E[ΔRE] / E[outs]（皆以階段B 兩段模型評估、1B 釘死在 hold-runner 位置）：
  A. 聯盟一壘有人平均站位（fielder_positioning_on1b）
  B. 無人在壘的出局率最佳解（現行 production 優化器，代表「不管壘況」）
  C. 階段B 雙殺優化解（optimize_infield_dp）

執行：python -m scripts.optimize_if_dp_demo [batter_id ...]
"""
import sys
from pathlib import Path

import joblib
import numpy as np

from src.if_dp_optimize import (DP_POSITIONS, DPScorer, dp_delta_re,
                                league_average_positions_on1b,
                                league_median_runner_speed, optimize_infield_dp,
                                positions_to_params_dp)
from src.if_optimize import fetch_batter_gbs, optimize_infield
from src.if_runvalue import gb_miss_costs
from src.re24 import load_re24

BASE = Path(__file__).resolve().parent.parent
YEARS = [2023, 2024]

out_model = joblib.load(BASE / "models/if_gb/on1b/if_on1b_out_glm.joblib")
dp_model = joblib.load(BASE / "models/if_gb/on1b/if_on1b_dp_glm.joblib")
xb = joblib.load(BASE / "models/if_gb/if_gb_xb_model.joblib")
bayes = joblib.load(BASE / "models/if_gb/bayes/if_bayes_group_pipeline.joblib")
re24, delta_re = load_re24(BASE / "data/precomputed")

lg_angles, lg_depths = league_average_positions_on1b(YEARS)
pinned_1b = (lg_angles[0], lg_depths[0])
runner_hp = league_median_runner_speed(YEARS)
print(f"聯盟一壘有人平均: " + "  ".join(
    f"{p}({a:.1f}°,{d:.0f}ft)" for p, a, d in
    zip(("1B",) + DP_POSITIONS, lg_angles, lg_depths))
    + f" | 跑者速度中位 {runner_hp:.2f}s")

batters = [int(b) for b in sys.argv[1:]] or [650333, 592450]

for b in batters:
    balls = fetch_batter_gbs(b, YEARS)
    balls["runner_hp_to_1b"] = runner_hp
    base_opt = optimize_infield(balls, bayes, n_restarts=16, seed=42)

    print(f"\n=== batter {b}  n_gb={len(balls)} ===")
    for outs in (0, 1):
        w = gb_miss_costs(balls, xb, delta_re, (1, 0, 0, outs))
        d1, d2 = dp_delta_re(re24, delta_re, outs)
        scorer = DPScorer(out_model, dp_model, balls, pinned_1b, w, d1, d2)

        warm = positions_to_params_dp(base_opt["angles"][1:], base_opt["depths"][1:])
        opt = optimize_infield_dp(balls, out_model, dp_model, pinned_1b, w, d1, d2,
                                  n_restarts=16, seed=42, extra_starts=[warm])

        rows = {
            "A 聯盟一壘有人": (lg_angles[1:], lg_depths[1:]),
            "B 無壘況最佳解": (base_opt["angles"][1:], base_opt["depths"][1:]),
            "C 雙殺優化解  ": (opt["angles"], opt["depths"]),
        }
        print(f"-- 一壘有人 {outs} 出局 (d1={d1:+.3f}, d2={d2:+.3f}, "
              f"mean w={w.mean():.3f})")
        for name, (a, d) in rows.items():
            e_re = scorer.expected_re(a, d)
            e_o = scorer.expected_outs(a, d)
            pos = "  ".join(f"{p}({aa:.1f}°,{dd:.0f}ft)"
                            for p, aa, dd in zip(DP_POSITIONS, a, d))
            print(f"  {name}: E[ΔRE]={e_re:+.4f}/GB  E[outs]={e_o:.3f}  {pos}")
