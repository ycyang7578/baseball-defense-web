"""Infield model metrics summary (matches the outfield acceptance approach: AUC / Brier /
BSS / calibration / correlation with official OAA).

BSS (Brier Skill Score) = 1 - Brier_model / Brier_ref,
Brier_ref = climatology reference (predicting the training-set average out rate for every
ball) — same approach as the paper's outfield Model_3.

Scope notes:
- The optimizer GLM and Bayesian models (group-level / +player-level) are evaluated on the
  main scope (bases empty + Standard alignment)
- The difficulty GLM and GBM benchmark are evaluated on the full ball population (the
  population actually used for evaluation)
Each scope's BSS uses its own training set's baseline rate.

Run: python -m scripts.validate.report_if_metrics (2025 out-of-sample)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.if_dataset import INFIELD_COLS, build_gb_dataset
from src.if_model import (DIFFICULTY_FEATURES, OPTIMIZER_FEATURES,
                          make_difficulty_glm, make_difficulty_gbm)

from scripts._metrics import calibration_table

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "if_gb"


def calibration_max_dev(y, p, bins: int = 10) -> float:
    g = calibration_table(y, p, bins)
    return float((g["pred"] - g["actual"]).abs().max())


def metrics_row(name, y, p, ref_rate):
    brier = brier_score_loss(y, p)
    brier_ref = np.mean((y - ref_rate) ** 2)
    return {"model": name,
            "auc": round(roc_auc_score(y, p), 4),
            "brier": round(brier, 4),
            "bss": round(1 - brier / brier_ref, 4),
            "logloss": round(log_loss(y, p), 4),
            "cal_max_dev": round(calibration_max_dev(y, p), 4)}


def bayes_player_logit_adjust(test: pd.DataFrame) -> np.ndarray:
    """Same as the production personalization path: alpha[nearest fielder] + g[nearest fielder]*ad_z (unseen fielder = 0)."""
    eff = pd.read_csv(MODEL_DIR / "bayes" / "IF_player_effects.csv")
    meta = json.loads((MODEL_DIR / "bayes" / "IF_meta.json").read_text("utf-8"))
    id_col = eff.columns[0]
    eff = eff.set_index(id_col)
    pos_to_col = {v: k for k, v in INFIELD_COLS.items()}
    nearest_ids = np.select(
        [test["nearest_pos"] == p for p in pos_to_col],
        [test[c] for c in (pos_to_col[p] for p in pos_to_col)]).astype(int)
    alpha = eff["alpha"].reindex(nearest_ids).fillna(0.0).to_numpy()
    g = eff["g"].reindex(nearest_ids).fillna(0.0).to_numpy()
    ad_z = (test["ad_min"].to_numpy() - meta["ad_mean"]) / meta["ad_std"]
    return alpha + g * ad_z


def main() -> None:
    rows = []

    # ── Main scope (optimizer-side models) ────────────────────────────────
    tr = build_gb_dataset(TRAIN_YEARS, bases_empty=True, alignment="Standard")
    te = build_gb_dataset([TEST_YEAR], bases_empty=True, alignment="Standard")
    y = te["is_out"].to_numpy()
    ref = tr["is_out"].mean()
    print(f"主範圍: train n={len(tr):,} (out率 {ref:.3f}), test n={len(te):,} "
          f"(out率 {y.mean():.3f})")

    glm = joblib.load(MODEL_DIR / "if_gb_optimizer_glm.joblib")
    rows.append(metrics_row("優化用 GLM", y,
                            glm.predict_proba(te[OPTIMIZER_FEATURES])[:, 1], ref))

    pipe = joblib.load(MODEL_DIR / "bayes" / "if_bayes_group_pipeline.joblib")
    p_grp = pipe.predict_proba(te[OPTIMIZER_FEATURES])[:, 1]
    rows.append(metrics_row("貝葉斯 群體層", y, p_grp, ref))

    logit_grp = np.log(p_grp / (1 - p_grp))
    p_ply = 1 / (1 + np.exp(-(logit_grp + bayes_player_logit_adjust(te))))
    rows.append(metrics_row("貝葉斯 ＋球員層", y, p_ply, ref))

    # ── Full ball population (evaluation-side models) ──────────────────────────────
    tr_f = build_gb_dataset(TRAIN_YEARS)
    te_f = build_gb_dataset([TEST_YEAR])
    y_f = te_f["is_out"].to_numpy()
    ref_f = tr_f["is_out"].mean()
    print(f"全量球群: train n={len(tr_f):,} (out率 {ref_f:.3f}), "
          f"test n={len(te_f):,} (out率 {y_f.mean():.3f})")

    dglm = make_difficulty_glm().fit(tr_f[DIFFICULTY_FEATURES], tr_f["is_out"])
    rows.append(metrics_row("難度 GLM（評價用）", y_f,
                            dglm.predict_proba(te_f[DIFFICULTY_FEATURES])[:, 1],
                            ref_f))
    gbm = make_difficulty_gbm().fit(tr_f[DIFFICULTY_FEATURES], tr_f["is_out"])
    rows.append(metrics_row("GBM benchmark", y_f,
                            gbm.predict_proba(te_f[DIFFICULTY_FEATURES])[:, 1],
                            ref_f))

    print("\n=== 2025 樣本外指標總表 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n外野對照（論文/ARCHITECTURE 記錄）：Model_3 BSS=0.7507、"
          "定案模型 Brier=0.0632、R vs 官方 OAA=0.795")


if __name__ == "__main__":
    main()
