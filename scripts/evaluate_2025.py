"""Evaluate the definitive model (unified OF) on 2025 held-out data (official subset)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _of_validation import POSITIONS, compute_model_oaa, load_of_model, normalize_name

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "2025"


def main():
    scaler, mu = load_of_model(MODELS_DIR)
    all_plays = pd.concat([compute_model_oaa(p, 2025, scaler, mu) for p in POSITIONS],
                          ignore_index=True)
    print(f"官方子集逐球資料總數: {len(all_plays):,}")

    bs = float(np.mean((all_plays["catch_prob"] - all_plays["caught"]) ** 2))
    print(f"Brier Score: {bs:.5f}")

    model_oaa = all_plays.groupby("name_fielder", as_index=False)["oaa_play"].sum()
    model_oaa["name_key"] = model_oaa["name_fielder"].apply(normalize_name)

    real = pd.read_csv(Path(__file__).resolve().parent.parent.parent
                        / "Baseball_Defense_Model_3" / "data" / "raw" / "validation" / "real_2025_oaa.csv",
                        encoding="utf-8-sig")
    name_col = next(c for c in real.columns if "name" in c.lower())
    real["name_key"] = real[name_col].apply(normalize_name)
    real["oaa"] = pd.to_numeric(real["oaa"], errors="coerce")

    merged = real.merge(model_oaa, on="name_key", how="inner").dropna(subset=["oaa", "oaa_play"])
    r, p = stats.pearsonr(merged["oaa"], merged["oaa_play"])
    print(f"n={len(merged)}  R={r:.4f}  p={p:.2e}")


if __name__ == "__main__":
    main()
