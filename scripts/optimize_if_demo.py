"""內野站位最佳化 demo：對滾地球最多的打者跑優化，對照聯盟平均站位。

打者分布用 2023–2024 歷史滾地球（訓練年，不碰 2025——留給之後的跨年驗證）。

執行：python scripts/optimize_if_demo.py [打者數，預設 6]
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN
from src.if_optimize import (POSITIONS, expected_outs, fetch_batter_gbs,
                             league_average_positions, optimize_infield,
                             positions_to_params)

YEARS = [2023, 2024]
MODEL = Path(__file__).resolve().parent.parent / "models" / "if_gb" / "if_gb_optimizer_glm.joblib"


def top_gb_batters(n: int) -> pd.DataFrame:
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(
            "SELECT batter, max(stand) AS stand, count(*) AS n_gb "
            "FROM statcast WHERE bb_type='ground_ball' AND game_year = ANY(%(y)s) "
            "  AND hc_x IS NOT NULL "
            "GROUP BY batter ORDER BY count(*) DESC LIMIT %(n)s",
            conn, params={"y": YEARS, "n": n})


def main(n_batters: int) -> None:
    model = joblib.load(MODEL)
    avg_angles, avg_depths = league_average_positions(YEARS)
    print("聯盟平均站位:", dict(zip(POSITIONS, zip(avg_angles.round(1), avg_depths.round(1)))))
    avg_start = positions_to_params(avg_angles, avg_depths)

    for row in top_gb_batters(n_batters).itertuples():
        balls = fetch_batter_gbs(row.batter, YEARS)
        base = expected_outs(model, balls, avg_angles, avg_depths)
        result = optimize_infield(balls, model, n_restarts=20, seed=42,
                                  extra_starts=[avg_start])
        gain = result["exp_outs"] - base
        print(f"\n打者 {row.batter}（{row.stand}打，{len(balls)} 顆 GB）")
        print(f"  平均站位期望出局率 {base:.4f} → 最佳化 {result['exp_outs']:.4f}"
              f"（+{gain:.4f}，每 450 顆滾地球約 {gain * 450:+.1f} 個出局）")
        print(f"  最佳站位: {result['positions']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
