"""
Generate Model OAA vs Official OAA validation scatter plot (web model, 2025 out-of-sample).
Output: data/validation_scatter.png
"""
import sys
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.defender_features import get_defender_opportunities, mark_official


MODELS_DIR   = Path(__file__).resolve().parent.parent / "models" / "2025"
OUT_PATH     = Path(__file__).resolve().parent.parent / "figures" / "validation_scatter.png"
POSITIONS    = ["LF", "CF", "RF"]
FEATURE_COLS = ["speed", "cos_angle", "sin_angle", "fielder_dist"]
DSN          = "host=localhost dbname=baseball user=postgres password=postgres"

# 統一 OF 模型
OF_DIR   = MODELS_DIR / "OF"
_scaler  = joblib.load(OF_DIR / "OF_scaler.joblib")
_mu_raw  = pd.read_csv(OF_DIR / "OF_summary_group.csv", encoding="utf-8-sig", index_col=0)["mean"]


def norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_model_oaa(pos: str) -> pd.DataFrame:
    df = get_defender_opportunities(pos, 2025)
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=FEATURE_COLS + ["caught", "name_fielder"])
    df = mark_official(df)
    df = df[df["is_official"]].copy()

    X = _scaler.transform(df[FEATURE_COLS])
    logit = (_mu_raw["mu_alpha"]
             + _mu_raw["mu_beta_speed"] * X[:, 0]
             + _mu_raw["mu_beta_cos"]   * X[:, 1]
             + _mu_raw["mu_beta_sin"]   * X[:, 2]
             + _mu_raw["mu_beta_dist"]  * X[:, 3])

    df["catch_prob"] = sigmoid(logit)
    df["oaa_play"] = df["caught"].astype(float) - df["catch_prob"]
    return df[["name_fielder", "oaa_play"]]


def main():
    # ── 計算 model OAA（群體層 mu，絕不用 player-level）──────────
    all_plays = pd.concat([compute_model_oaa(p) for p in POSITIONS], ignore_index=True)
    model_oaa = all_plays.groupby("name_fielder", as_index=False)["oaa_play"].sum()
    model_oaa["key"] = model_oaa["name_fielder"].apply(norm)
    print(f"Model OAA 球員數: {len(model_oaa)}")

    # ── 官方 OAA（is_qualified=True，跨位置加總）────────────────
    with psycopg2.connect(DSN) as conn:
        raw_off = pd.read_sql(
            "SELECT player_name, oaa FROM oaa_leaderboard WHERE year=2025 AND is_qualified=TRUE",
            conn,
        )
    # 同一球員可能在多個位置達到 qualified，加總後視為整體外野 OAA
    official = raw_off.groupby("player_name", as_index=False)["oaa"].sum()
    official["key"] = official["player_name"].apply(norm)
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
    ax.set_ylabel("Official OAA (2025)", fontsize=12)
    ax.set_title(
        f"Model OAA vs. Baseball Savant Official OAA\n"
        f"2025 Out-of-Sample  (n={n},  r={r:.3f})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"儲存至 {OUT_PATH}")


if __name__ == "__main__":
    main()
