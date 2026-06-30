"""Precompute RE24 table and hit-type KDE from statcast 2021-2024.

Run once before using the optimizer:
  python scripts/precompute.py

Outputs:
  data/precomputed/re24_table.json
  data/precomputed/delta_re.json
  data/precomputed/hit_type_kde.joblib
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.re24 import build_re24_table, build_delta_re, save_re24
from src.hit_prob import fit_hit_type_kde, save_hit_prob

YEARS = [2021, 2022, 2023, 2024]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed"


def main():
    print("=== RE24 Table ===")
    re24 = build_re24_table(YEARS)
    print(f"States: {len(re24)}/24")
    for k in sorted(re24)[:4]:
        print(f"  {k}: {re24[k]:.4f}")

    print("\n=== Delta RE (empirical advancement) ===")
    delta = build_delta_re(re24, YEARS)
    print(f"Keys: {len(delta)}  (expected 72 = 3 hit types × 24 states)")

    save_re24(re24, delta, OUT_DIR)
    print(f"Saved → {OUT_DIR}/re24_table.json + delta_re.json")

    print("\n=== Hit Type KDE ===")
    bundle = fit_hit_type_kde(YEARS)
    for stand in ("L", "R"):
        for ht in ("1B", "2B", "3B"):
            p = bundle["priors"].get((stand, ht), 0.0)
            has_model = (stand, ht) in bundle["models"]
            print(f"  {stand}/{ht}: prior={p:.3f}  model={'yes' if has_model else 'MISSING'}")

    save_hit_prob(bundle, OUT_DIR)
    print(f"Saved → {OUT_DIR}/hit_type_kde.joblib")
    print("\nDone.")


if __name__ == "__main__":
    main()
