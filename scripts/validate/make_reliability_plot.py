"""
Generate reliability diagram (calibration plot) for web model, 2025 out-of-sample.
Output: data/reliability_diagram.png

Run: python -m scripts.validate.make_reliability_plot
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

from src.defender_features import get_defender_opportunities

from scripts._of_validation import FEATURE_COLS, POSITIONS, load_of_model, sigmoid

MODELS_DIR   = Path(__file__).resolve().parent.parent.parent / "models" / "2025"
OUT_PATH     = Path(__file__).resolve().parent.parent.parent / "figures" / "reliability_diagram.png"
N_BINS       = 20

# Unified OF model
scaler, _mu_raw = load_of_model(MODELS_DIR)


def get_probs(pos: str) -> pd.DataFrame:
    df = get_defender_opportunities(pos, 2025)
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=FEATURE_COLS + ["caught"])

    X = scaler.transform(df[FEATURE_COLS])
    logit = (_mu_raw["mu_alpha"]
             + _mu_raw["mu_beta_speed"] * X[:, 0]
             + _mu_raw["mu_beta_cos"]   * X[:, 1]
             + _mu_raw["mu_beta_sin"]   * X[:, 2]
             + _mu_raw["mu_beta_dist"]  * X[:, 3])

    df["catch_prob"] = sigmoid(logit)
    return df[["caught", "catch_prob"]]


def main():
    all_df = pd.concat([get_probs(p) for p in POSITIONS], ignore_index=True)
    y      = all_df["caught"].astype(float).values
    p      = all_df["catch_prob"].values
    print(f"Total balls: {len(y):,}")

    brier   = brier_score_loss(y, p)
    ll      = log_loss(y, p)
    mae     = float(np.mean(np.abs(y - p)))
    print(f"Brier={brier:.4f}  LogLoss={ll:.4f}  MAE={mae:.4f}")

    # ── Equal-width binning (N_BINS = 20) ──────────────────────────────────
    edges  = np.linspace(0, 1, N_BINS + 1)
    bins   = np.digitize(p, edges[1:-1])   # 0 … N_BINS-1

    rows = []
    for b in range(N_BINS):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append({
            "mean_pred":   float(p[mask].mean()),
            "actual_rate": float(y[mask].mean()),
            "n":           int(mask.sum()),
        })
    cal = pd.DataFrame(rows)

    # ── Plotting ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("white")

    # perfect calibration
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.8, label="Perfect Calibration")

    # model calibration curve
    ax.plot(cal["mean_pred"], cal["actual_rate"],
            "o-", color="#3a7ebf", lw=2, ms=6, zorder=4, label="Model")

    # Per-point annotation
    for _, row in cal.iterrows():
        px, ay, n = row["mean_pred"], row["actual_rate"], int(row["n"])
        ax.annotate(
            f"P:{px*100:.1f}%\nA:{ay*100:.1f}%\n(n={n:,})",
            xy=(px, ay),
            xytext=(6, 4), textcoords="offset points",
            fontsize=6.8, color="#1a3a5c", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#aac4e0", alpha=0.85, lw=0.6),
        )

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Catch Rate", fontsize=12)
    ax.set_title(
        f"Reliability Diagram — 2025 (out-of-sample)\n"
        f"Brier={brier:.4f}  LogLoss={ll:.4f}  MAE={mae:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.6, color="#dddddd")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"儲存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
