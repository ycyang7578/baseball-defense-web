"""
Generate Model OAA vs Official OAA validation scatter plot.

Usage:
    python make_validation_plot.py [year]   # default 2025
    python make_validation_plot.py 2024
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DSN

from _of_validation import load_model_oaa, load_official_oaa

_parser = argparse.ArgumentParser()
_parser.add_argument("year", type=int, nargs="?", default=2025)
_args = _parser.parse_args()
TARGET_YEAR = _args.year

BASE         = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE / "models" / str(TARGET_YEAR)
OUT_PATH     = BASE / "figures" / f"validation_scatter_{TARGET_YEAR}.png"


def main():
    # ── 計算 model OAA（群體層 mu，絕不用 player-level）──────────
    model_oaa = load_model_oaa(MODELS_DIR, TARGET_YEAR)
    print(f"Model OAA 球員數: {len(model_oaa)}")

    # ── 官方 OAA（is_qualified=True，跨位置加總）────────────────
    official = load_official_oaa(DSN, TARGET_YEAR)
    print(f"官方 OAA (qualified) 球員數: {len(official)}")

    # ── 合併 ─────────────────────────────────────────────────────
    merged = official.merge(model_oaa, on="key", how="inner").dropna(subset=["oaa", "oaa_play"])
    x = merged["oaa_play"].values
    y = merged["oaa"].values
    r, p = stats.pearsonr(x, y)
    n = len(merged)
    print(f"n={n}  R={r:.4f}  p={p:.2e}")

    # ── 繪圖 ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # 散點
    ax.scatter(x, y, color="#4a8ab5", s=45, alpha=0.85, edgecolors="none", zorder=3)

    # 迴歸線（含信賴區間）
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 1, x.max() + 1, 300)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color="red", lw=2, zorder=4, label=f"r = {r:.3f}")

    # 信賴區間（手動算 95% CI）
    n_pts = len(x)
    x_mean = x.mean()
    se = np.sqrt(np.sum((y - (m * x + b)) ** 2) / (n_pts - 2))
    t_val = stats.t.ppf(0.975, df=n_pts - 2)
    ci = t_val * se * np.sqrt(1/n_pts + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    ax.fill_between(x_line, y_line - ci, y_line + ci, color="red", alpha=0.15, zorder=2)

    # 標記離群點：residual 最大 or model OAA 最高
    merged["resid"] = np.abs(y - (m * x + b))
    merged["model_oaa"] = x
    top_resid = merged.nlargest(4, "resid")
    top_model = merged.nlargest(3, "model_oaa")
    to_label  = pd.concat([top_resid, top_model]).drop_duplicates(subset="key")
    for _, row in to_label.iterrows():
        ax.annotate(
            row["player_name"],
            xy=(row["model_oaa"], row["oaa"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=8.5, color="#1a1a2e",
        )

    # 格線
    ax.grid(True, linestyle="--", linewidth=0.7, color="#cccccc", zorder=0)
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)

    ax.set_xlabel("Model OAA", fontsize=12)
    ax.set_ylabel(f"Official OAA ({TARGET_YEAR})", fontsize=12)
    ax.set_title(
        f"Model OAA vs. Baseball Savant Official OAA\n"
        f"{TARGET_YEAR} Out-of-Sample  (n={n},  r={r:.3f})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"儲存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
