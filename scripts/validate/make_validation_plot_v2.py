"""
Generate improved Model OAA vs Official OAA validation scatter plot (v2).

Does not modify make_validation_plot.py and does not write to the DB — it just
re-plots with a better visualization design. Data source and computation logic
are identical to the original script.

Usage:
    python -m scripts.validate.make_validation_plot_v2 [year]   # default 2025
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

from src.config import DSN

from scripts._of_validation import load_model_oaa, load_official_oaa

_parser = argparse.ArgumentParser()
_parser.add_argument("year", type=int, nargs="?", default=2025)
_args = _parser.parse_args()
TARGET_YEAR = _args.year

BASE         = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = BASE / "models" / str(TARGET_YEAR)
OUT_PATH     = BASE / "figures" / "validation_scatter_v2.png"

# Okabe-Ito colorblind-safe palette
COLOR_POINTS = "#0072B2"   # blue
COLOR_FIT    = "#D55E00"   # vermillion
COLOR_REF    = "#888888"   # neutral gray (reference line, de-emphasized)


def main():
    # ── Compute model OAA (group-level mu, never player-level) ──────────
    model_oaa = load_model_oaa(MODELS_DIR, TARGET_YEAR)
    print(f"Model OAA 球員數: {len(model_oaa)}")

    # ── Official OAA (is_qualified=True, summed across positions) ────────────────
    official = load_official_oaa(DSN, TARGET_YEAR)
    print(f"官方 OAA (qualified) 球員數: {len(official)}")

    # ── Merge ─────────────────────────────────────────────────────
    merged = official.merge(model_oaa, on="key", how="inner").dropna(subset=["oaa", "oaa_play"])
    x = merged["oaa_play"].values
    y = merged["oaa"].values
    r, p = stats.pearsonr(x, y)
    n = len(merged)
    m, b = np.polyfit(x, y, 1)
    print(f"n={n}  R={r:.4f}  p={p:.2e}  slope={m:.3f}  "
          f"model_sd={x.std():.2f}  official_sd={y.std():.2f}")

    # ── Layout: main scatter + top/right marginal distributions, sharing one axis range (equal aspect) ──
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.08
    lims = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(9.5, 9.5), facecolor="white")
    gs = GridSpec(4, 4, figure=fig, hspace=0.06, wspace=0.06)
    ax_top   = fig.add_subplot(gs[0, 0:3])
    ax_main  = fig.add_subplot(gs[1:4, 0:3])
    ax_right = fig.add_subplot(gs[1:4, 3])

    # Main plot: y = x reference line (gray, dashed, representing "perfect agreement") —
    # this is the key baseline for judging whether the model is merely "rank-correlated"
    # or also "numerically matches official OAA"
    ax_main.plot(lims, lims, "--", color=COLOR_REF, lw=1.6, zorder=2,
                 label="Perfect Agreement (y = x)")

    # OLS regression line + 95% CI
    x_line = np.linspace(lims[0], lims[1], 300)
    y_line = m * x_line + b
    n_pts = len(x)
    x_mean = x.mean()
    se = np.sqrt(np.sum((y - (m * x + b)) ** 2) / (n_pts - 2))
    t_val = stats.t.ppf(0.975, df=n_pts - 2)
    ci = t_val * se * np.sqrt(1 / n_pts + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    ax_main.fill_between(x_line, y_line - ci, y_line + ci, color=COLOR_FIT, alpha=0.15, zorder=3)
    ax_main.plot(x_line, y_line, color=COLOR_FIT, lw=2.2, zorder=4,
                 label=f"OLS Fit (r = {r:.3f})")

    # Scatter points (white outline so overlapping points don't blur together)
    ax_main.scatter(x, y, color=COLOR_POINTS, s=55, alpha=0.85,
                     edgecolors="white", linewidths=0.5, zorder=5)

    ax_main.axhline(0, color=COLOR_REF, linewidth=0.7, zorder=1)
    ax_main.axvline(0, color=COLOR_REF, linewidth=0.7, zorder=1)
    ax_main.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)

    # Label outliers (largest residual or highest model OAA)
    merged["resid"] = np.abs(y - (m * x + b))
    merged["model_oaa"] = x
    top_resid = merged.nlargest(4, "resid")
    top_model = merged.nlargest(3, "model_oaa")
    to_label = pd.concat([top_resid, top_model]).drop_duplicates(subset="key")
    for _, row in to_label.iterrows():
        ax_main.annotate(
            row["player_name"],
            xy=(row["model_oaa"], row["oaa"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8.5, color="#1a1a2e",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75),
            zorder=6,
        )

    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlabel("Model OAA", fontsize=12)
    ax_main.set_ylabel(f"Official OAA ({TARGET_YEAR})", fontsize=12)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.legend(loc="lower right", fontsize=9.5, framealpha=0.9)

    # Top margin: model OAA distribution
    ax_top.hist(x, bins=20, color=COLOR_POINTS, alpha=0.75, range=lims)
    ax_top.set_xlim(lims)
    ax_top.axis("off")

    # Right margin: official OAA distribution
    ax_right.hist(y, bins=20, orientation="horizontal", color=COLOR_FIT, alpha=0.6, range=lims)
    ax_right.set_ylim(lims)
    ax_right.axis("off")

    fig.suptitle(
        f"Model OAA Tracks Official OAA Rankings (r={r:.3f}, n={n})\n"
        f"but Overestimates Magnitude — slope={m:.2f}, "
        f"model SD={x.std():.1f} vs official SD={y.std():.1f}",
        fontsize=13, fontweight="bold", y=0.975,
    )

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"儲存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
