"""階段B 線上常數的離線預算：聯盟一壘有人站位＋跑者中位速度 → JSON。

/api/if_optimize 的 DP 分支（僅一壘有人、<2 出局）需要兩組常數：
- 聯盟一壘有人平均站位（fielder_positioning_on1b，本機才有這張表）——
  1B 的 hold-runner 釘死位置＋2B/3B/SS 的對照基準
- 聯盟跑者 hp_to_1b 中位數（優化情境的一壘跑者未知，見 src/if_dp_optimize.py）

照 if_league_positions.json 的模式存 data/precomputed/if_on1b_constants.json
（API startup 讀），雲端因此不需要 fielder_positioning_on1b / sprint_speed。
年份須與階段B 模型的訓練年一致（scripts/train_if_on1b.py 的 2023–24）。

Usage:
    python scripts/precompute_if_on1b_constants.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.if_dp_optimize import (DP_POSITIONS, league_average_positions_on1b,
                                league_median_runner_speed)

TRAIN_YEARS = [2023, 2024]
BASE = Path(__file__).resolve().parent.parent
OUT_PATH = BASE / "data" / "precomputed" / "if_on1b_constants.json"


def main() -> None:
    angles, depths = league_average_positions_on1b(TRAIN_YEARS)
    runner_hp = league_median_runner_speed(TRAIN_YEARS)
    payload = {
        "train_years": TRAIN_YEARS,
        "positions": {p: [round(float(a), 2), round(float(d), 2)]
                      for p, a, d in zip(("1B",) + DP_POSITIONS, angles, depths)},
        "runner_hp_to_1b": round(runner_hp, 3),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {OUT_PATH}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
