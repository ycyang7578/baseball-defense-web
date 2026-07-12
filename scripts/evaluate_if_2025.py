"""階段 2：內野手球員評價 —— model OAA 對照官方內野 OAA（2025 樣本外）。

p̂ 用評價用 GBM（spray+球質+跑者，無野手資訊 → 無循環論證），在全量滾地球
（不限壘況/佈陣）上以 2023–2024 訓練、2025 評分，跟官方球群（所有情境）對齊。
逐球歸責給最近角距的內野手（對應官方「slice」概念）。
球員 model OAA = Σ(is_out − p̂)。

計算邏輯在 src/if_eval.py（與 web 排名預算 scripts/precompute_if_model_oaa.py
共用同一份實作），此腳本負責對照官方數字的統計報表。

已知的球群差異（會壓低相關係數，屬預期）：官方內野 OAA 另含觸擊、內野平飛等，
且用逐球實際起始位置；我們只有非觸擊滾地球＋賽季平均站位。

執行：python scripts/evaluate_if_2025.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

    # 分位置相關（歸責球最多的位置當作該球員的位置）
    print("\n分位置（n_balls >= 100）：")
    for pos in ("1B", "2B", "3B", "SS"):
        sub = merged[(merged["resp_pos"] == pos) & (merged["n_balls"] >= 100)]
        if len(sub) < 3:
            continue
        r = np.corrcoef(sub["model_oaa"], sub["oaa"])[0, 1]
        print(f"  {pos}: n={len(sub):>3}  R={r:.3f}  "
              f"model SD={sub['model_oaa'].std():.1f} vs 官方 SD={sub['oaa'].std():.1f}")

    # 無人在壘子集：檢驗 1B hold runner 假說（壘上有人時 1B 貼壘，賽季平均位置失真）
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
