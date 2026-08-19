"""Offline precomputation of Stage B's online constants: league runner-on-1B
positioning + median runner speed -> JSON.

The DP branch of /api/if_optimize (runner on 1B only, <2 outs) needs two sets
of constants:
- League average runner-on-1B positioning (fielder_positioning_on1b, a table
  that only exists locally) -- 1B's hold-runner pinned position plus the
  2B/3B/SS reference baseline
- League median runner hp_to_1b (the runner on 1B is unknown in the
  optimization scenario, see src/if_dp_optimize.py)

Saved to data/precomputed/if_on1b_constants.json following the same pattern
as if_league_positions.json (read at API startup), so the cloud deployment
doesn't need fielder_positioning_on1b / sprint_speed.
The years must match the training years of the Stage B model
(2023-24, as in scripts/train_if_on1b.py).

Usage:
    python -m scripts.precompute.precompute_if_on1b_constants
"""
import json
from pathlib import Path

from src.if_dp_optimize import (DP_POSITIONS, league_average_positions_on1b,
                                league_median_runner_speed)

TRAIN_YEARS = [2023, 2024]
BASE = Path(__file__).resolve().parent.parent.parent
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
