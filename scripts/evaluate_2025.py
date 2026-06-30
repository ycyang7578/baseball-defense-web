"""Evaluate the definitive model on 2025 held-out data (official subset)."""
import sys
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defender_features import get_defender_opportunities, mark_official

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "2025"
POSITIONS = ["LF", "CF", "RF"]
FEATURE_COLS = ["speed", "cos_angle", "sin_angle", "fielder_dist"]


def normalize_player_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_position(pos: str) -> pd.DataFrame:
    df = get_defender_opportunities(pos, 2025)
    df = mark_official(df)
    df = df[df["is_official"]].copy()
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=FEATURE_COLS + ["caught", "name_fielder"])

    scaler = joblib.load(MODELS_DIR / pos / f"{pos}_scaler.joblib")
    X = scaler.transform(df[FEATURE_COLS])

    group = pd.read_csv(MODELS_DIR / pos / f"{pos}_summary_group.csv", encoding="utf-8-sig", index_col=0)
    mu = group["mean"]

    logit = (mu["mu_alpha"]
             + mu["mu_beta_speed"] * X[:, 0]
             + mu["mu_beta_cos"] * X[:, 1]
             + mu["mu_beta_sin"] * X[:, 2]
             + mu["mu_beta_dist"] * X[:, 3])

    df["catch_prob"] = sigmoid(logit)
    df["oaa_play"] = df["caught"] - df["catch_prob"]
    return df[["name_fielder", "caught", "catch_prob", "oaa_play"]]


def main():
    all_plays = pd.concat([evaluate_position(p) for p in POSITIONS], ignore_index=True)
    print(f"官方子集逐球資料總數: {len(all_plays):,}")

    bs = float(np.mean((all_plays["catch_prob"] - all_plays["caught"]) ** 2))
    print(f"Brier Score: {bs:.5f}")

    model_oaa = all_plays.groupby("name_fielder", as_index=False)["oaa_play"].sum()
    model_oaa["name_key"] = model_oaa["name_fielder"].apply(normalize_player_name)

    real = pd.read_csv(Path(__file__).resolve().parent.parent.parent
                        / "Baseball_Defense_Model_3" / "data" / "raw" / "validation" / "real_2025_oaa.csv",
                        encoding="utf-8-sig")
    name_col = next(c for c in real.columns if "name" in c.lower())
    real["name_key"] = real[name_col].apply(normalize_player_name)
    real["oaa"] = pd.to_numeric(real["oaa"], errors="coerce")

    merged = real.merge(model_oaa, on="name_key", how="inner").dropna(subset=["oaa", "oaa_play"])
    r, p = stats.pearsonr(merged["oaa"], merged["oaa_play"])
    print(f"n={len(merged)}  R={r:.4f}  p={p:.2e}")


if __name__ == "__main__":
    main()
