"""階段B 增益分解：+0.090 分/GB 的額外增益裡，多少是「補 1B 被釘走的洞」、
多少是「雙殺感知」？

對 validate_if_dp.py 同一批打者，加一個中間基準：
- p1 解＝釘死 1B、優化 2B/3B/SS、但只最大化 P(≥1 出局)（懂幾何、不懂雙殺計價）
分解（皆 2023–24 優化、2025 球失分評估）：
  聯盟 → 無壘況解：−（負值＝破洞）      [validate_if_dp 已算]
  聯盟 → p1 解    ：補洞＋幾何適應
  p1 解 → DP 解   ：雙殺感知的淨貢獻    [關鍵數字]

逐打者 checkpoint，中斷後重跑續算。
執行：python scripts/experiments/exp_if_dp_decompose.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.if_dp_optimize import (DPScorer, dp_delta_re,
                                league_average_positions_on1b,
                                league_median_runner_speed,
                                optimize_infield_dp, positions_to_params_dp)
from src.if_optimize import fetch_batter_gbs
from src.if_runvalue import gb_miss_costs
from src.re24 import load_re24

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
OUTS = 0
BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = BASE / "models" / "if_gb" / "on1b"
VALID_ROWS = _MODEL_DIR / "validation_dp_rows_2025.csv"
CKPT = _MODEL_DIR / "decompose_p1_rows.csv"


def main() -> None:
    out_model = joblib.load(_MODEL_DIR / "if_on1b_out_glm.joblib")
    dp_model = joblib.load(_MODEL_DIR / "if_on1b_dp_glm.joblib")
    xb = joblib.load(BASE / "models/if_gb/if_gb_xb_model.joblib")
    re24, delta_re = load_re24(BASE / "data/precomputed")

    lg_angles, lg_depths = league_average_positions_on1b(TRAIN_YEARS)
    pinned_1b = (float(lg_angles[0]), float(lg_depths[0]))
    lg_start = positions_to_params_dp(lg_angles[1:], lg_depths[1:])
    runner_hp = league_median_runner_speed(TRAIN_YEARS)
    d1, d2 = dp_delta_re(re24, delta_re, OUTS)

    valid = pd.read_csv(VALID_ROWS)
    batters = valid["batter"].astype(int).tolist()
    done: set[int] = set()
    if CKPT.exists():
        done = set(pd.read_csv(CKPT)["batter"])
        print(f"checkpoint: {len(done)} 位已完成", flush=True)

    for i, b in enumerate(batters, 1):
        if b in done:
            continue
        tr = fetch_batter_gbs(b, TRAIN_YEARS)
        te = fetch_batter_gbs(b, [TEST_YEAR])
        tr["runner_hp_to_1b"] = runner_hp
        te["runner_hp_to_1b"] = runner_hp
        w_tr = gb_miss_costs(tr, xb, delta_re, (1, 0, 0, OUTS))
        w_te = gb_miss_costs(te, xb, delta_re, (1, 0, 0, OUTS))
        scorer_te = DPScorer(out_model, dp_model, te, pinned_1b, w_te, d1, d2)

        opt_p1 = optimize_infield_dp(tr, out_model, dp_model, pinned_1b, w_tr,
                                     d1, d2, n_restarts=8, seed=42,
                                     extra_starts=[lg_start], objective="p1")
        dre_p1 = scorer_te.expected_re(opt_p1["angles"], opt_p1["depths"])
        pd.DataFrame([{"batter": b, "dre_p1sol": dre_p1}]).to_csv(
            CKPT, mode="a", header=not CKPT.exists(), index=False)
        if i % 20 == 0 or i == len(batters):
            print(f"  ...{i}/{len(batters)}", flush=True)

    dec = pd.read_csv(CKPT).merge(valid, on="batter")
    gain_p1 = dec["dre_league"] - dec["dre_p1sol"]          # 聯盟 → p1 解
    dp_net = dec["gain_cross_dp"] - gain_p1                 # p1 解 → DP 解
    print(f"\n=== 增益分解（n={len(dec)}，2025 球、分/GB）===")
    print(f"聯盟 → 無壘況解:  {dec['gain_cross_base'].mean():+.5f}（破洞）")
    print(f"聯盟 → p1 解:     {gain_p1.mean():+.5f}（補洞＋幾何適應）")
    print(f"p1 解 → DP 解:    {dp_net.mean():+.5f}（雙殺感知淨貢獻, "
          f"450 GB 約 {dp_net.mean() * 450:+.2f} 分, 正比例 {(dp_net > 0).mean():.1%}）")
    print(f"合計（聯盟 → DP）: {dec['gain_cross_dp'].mean():+.5f}")


if __name__ == "__main__":
    main()
