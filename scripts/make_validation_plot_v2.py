"""
Generate improved Model OAA vs Official OAA validation scatter plot (v2).

不修改 make_validation_plot.py，也不寫入 DB，只是套用更好的視覺化設計重新畫圖。
資料來源與計算邏輯與原腳本完全相同。

Usage:
    python make_validation_plot_v2.py [year]   # default 2025
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
OUT_PATH     = BASE / "figures" / "validation_scatter_v2.png"

# Okabe-Ito colorblind-safe palette
COLOR_POINTS = "#0072B2"   # blue
COLOR_FIT    = "#D55E00"   # vermillion
COLOR_REF    = "#888888"   # neutral gray (reference line, de-emphasized)


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
    m, b = np.polyfit(x, y, 1)
    print(f"n={n}  R={r:.4f}  p={p:.2e}  slope={m:.3f}  "
          f"model_sd={x.std():.2f}  official_sd={y.std():.2f}")

    # ── 版面：主散點圖 + 上/右邊際分布，共用同一組軸範圍（等比例）──
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.08
    lims = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(9.5, 9.5), facecolor="white")
    gs = GridSpec(4, 4, figure=fig, hspace=0.06, wspace=0.06)
    ax_top   = fig.add_subplot(gs[0, 0:3])
    ax_main  = fig.add_subplot(gs[1:4, 0:3])
    ax_right = fig.add_subplot(gs[1:4, 3])

    # 主圖：y = x 參考線（灰、虛線，代表「完全一致」）─ 這是判斷模型是否
    # 只是「排名相關」還是「數值上也吻合官方 OAA」的關鍵基準線
    ax_main.plot(lims, lims, "--", color=COLOR_REF, lw=1.6, zorder=2,
                 label="Perfect Agreement (y = x)")

    # OLS 迴歸線 + 95% CI
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

    # 散點（白色描邊避免重疊點糊成一團）
    ax_main.scatter(x, y, color=COLOR_POINTS, s=55, alpha=0.85,
                     edgecolors="white", linewidths=0.5, zorder=5)

    ax_main.axhline(0, color=COLOR_REF, linewidth=0.7, zorder=1)
    ax_main.axvline(0, color=COLOR_REF, linewidth=0.7, zorder=1)
    ax_main.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", zorder=0)

    # 標記離群點（residual 最大 or model OAA 最高）
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

    # 上邊際：model OAA 分布
    ax_top.hist(x, bins=20, color=COLOR_POINTS, alpha=0.75, range=lims)
    ax_top.set_xlim(lims)
    ax_top.axis("off")

    # 右邊際：official OAA 分布
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
