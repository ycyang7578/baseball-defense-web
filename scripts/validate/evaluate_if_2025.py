"""Stage 2: infield player evaluation — model OAA vs. official infield OAA (2025 out-of-sample).

p̂ comes from the evaluation-only difficulty GLM (spray + batted-ball quality + runner,
no fielder info -> no circularity; replaced the GBM as of 2026-07-12, prioritizing
interpretability — see the if_model.py docstring), trained on 2023-2024 and scored on
2025 over the full ground-ball population (no restriction on base state / alignment),
matched to the official ball population (all situations).
Each ball is attributed to the infielder nearest in angle/depth (mirrors the official
"slice" concept).
Player model OAA = Sum(is_out - p̂).

The computation logic lives in src/if_eval.py (shares the same implementation with the
web ranking budget script scripts/precompute_if_model_oaa.py); this script is responsible
for the statistical report comparing against the official numbers.

Known ball-population differences (expected to depress the correlation): official infield
OAA also includes bunts, infield line drives, etc., and uses the actual per-play starting
position; we only have non-bunt ground balls + season-average positioning.

Run: python -m scripts.validate.evaluate_if_2025
"""
import numpy as np
import pandas as pd
import psycopg2

from src.config import DSN
from src.if_eval import aggregate_players, score_test_year

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025


def main() -> None:
    test = score_test_year(TRAIN_YEARS, TEST_YEAR)
    model = aggregate_players(test)

    with psycopg2.connect(DSN) as conn:
        official = pd.read_sql(
            "SELECT player_id, player_name, oaa, n_opp, is_qualified "
            "FROM if_oaa_leaderboard WHERE year = %(y)s",
            conn, params={"y": TEST_YEAR})

    merged = model.merge(official, left_on="resp_fielder", right_on="player_id")
    print(f"\n對得上官方 leaderboard 的球員: {len(merged)}")

    for label, sub in [
        ("qualified (官方門檻)", merged[merged["is_qualified"] == True]),  # noqa: E712
        ("n_balls >= 100", merged[merged["n_balls"] >= 100]),
        ("n_balls >= 200", merged[merged["n_balls"] >= 200]),
    ]:
        if len(sub) < 3:
            continue
        r = np.corrcoef(sub["model_oaa"], sub["oaa"])[0, 1]
        r_raw = np.corrcoef(sub["model_oaa_raw"], sub["oaa"])[0, 1]
        rho = sub["model_oaa"].corr(sub["oaa"], method="spearman")
        r_rate = np.corrcoef(sub["model_oaa"] / sub["n_balls"],
                             sub["oaa"] / sub["n_opp"])[0, 1]
        print(f"  {label:<22} n={len(sub):>3}  Pearson R={r:.3f}（中心化前 {r_raw:.3f}）  "
              f"Spearman={rho:.3f}  每球率 R={r_rate:.3f}")

    # Per-position correlation (the position with the most attributed balls is taken as the player's position)
    print("\n分位置（n_balls >= 100）：")
    for pos in ("1B", "2B", "3B", "SS"):
        sub = merged[(merged["resp_pos"] == pos) & (merged["n_balls"] >= 100)]
        if len(sub) < 3:
            continue
        r = np.corrcoef(sub["model_oaa"], sub["oaa"])[0, 1]
        print(f"  {pos}: n={len(sub):>3}  R={r:.3f}  "
              f"model SD={sub['model_oaa'].std():.1f} vs 官方 SD={sub['oaa'].std():.1f}")

    # Bases-empty subset: test the 1B hold-runner hypothesis (with a runner on, 1B holds close to the bag, so the season-average position is distorted)
    empty = test[test["bases_empty"]]
    m2 = (empty.groupby("resp_fielder")
          .agg(model_oaa_e=("oaa_play", "sum"), n_e=("oaa_play", "size"))
          .reset_index().merge(official, left_on="resp_fielder", right_on="player_id")
          .merge(model[["resp_fielder", "resp_pos"]], on="resp_fielder"))
    print("\n無人在壘子集，分位置（n_e >= 60）：")
    for pos in ("1B", "2B", "3B", "SS"):
        sub = m2[(m2["resp_pos"] == pos) & (m2["n_e"] >= 60)]
        if len(sub) < 3:
            continue
        r = np.corrcoef(sub["model_oaa_e"], sub["oaa"])[0, 1]
        print(f"  {pos}: n={len(sub):>3}  R={r:.3f}")

    q = merged[merged["is_qualified"] == True]  # noqa: E712
    print(f"\nscale 檢查（qualified）：model OAA SD={q['model_oaa'].std():.1f} vs "
          f"官方 SD={q['oaa'].std():.1f}")
    print("\nmodel OAA 前 10（qualified）：")
    cols = ["player_name", "model_oaa", "n_balls", "oaa", "n_opp"]
    print(q.nlargest(10, "model_oaa")[cols].round(1).to_string(index=False))
    print("\nmodel OAA 後 10（qualified）：")
    print(q.nsmallest(10, "model_oaa")[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
