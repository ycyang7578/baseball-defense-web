"""Train the infield ground-ball out probability model (stage 1).

Two models, two roles (see the docstring in src/if_model.py for details):
- Optimization GLM: fielder-relative geometry (counterfactual-capable --
  predictions respond to moving fielders)
- Difficulty GLM for evaluation: a league-average difficulty model over
  spray + batted-ball quality + runner (p-hat for a fixed positioning
  scenario; replaced the GBM as of 2026-07-12, prioritizing
  interpretability; the GBM is kept around as a benchmark metric)

Trained on 2023-2024 (post-shift-ban years, so the season-average positioning
doesn't mix in shift alignments; the 2021-22 season averages mix in shifted
balls -- causing systematic bias in 2B depth / 3B angle -- Melville 2024
similarly uses only post-ban data. A 2026-07-09 experiment confirmed the
three 2025 out-of-sample metrics differ within noise between the two year
configurations, so the cleaner one was chosen), tested on 2025 (out-of-sample).
The interaction configuration was selected using train 2023 -> validate 2024;
2025 is touched only once, here.
Main scope: bases empty + Standard alignment (Melville similarly excludes
runners on base; a 1B hold-runner situation pulls positioning).

Run: python -m scripts.train.train_if_gb
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.if_dataset import build_gb_dataset
from src.if_model import (DIFFICULTY_FEATURES, OPTIMIZER_FEATURES,
                          make_difficulty_gbm, make_difficulty_glm,
                          make_optimizer_glm)
from src.if_runvalue import XB_FEATURES, fetch_gb_hits, make_gb_xb_model

from scripts._metrics import calibration_table

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "if_gb"


def evaluate(name: str, model, feats: list[str],
             train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model.fit(train[feats], train["is_out"])
    p = model.predict_proba(test[feats])[:, 1]
    auc = roc_auc_score(test["is_out"], p)
    brier = brier_score_loss(test["is_out"], p)
    cal = calibration_table(test["is_out"].to_numpy(), p)
    max_dev = float((cal["pred"] - cal["actual"]).abs().max())
    print(f"  {name:<24} AUC={auc:.4f}  Brier={brier:.4f}  校準最大偏差={max_dev:.3f}")
    return {"auc": round(auc, 4), "brier": round(brier, 4),
            "calibration_max_dev": round(max_dev, 4)}


def main() -> None:
    train = build_gb_dataset(TRAIN_YEARS, bases_empty=True, alignment="Standard")
    test = build_gb_dataset([TEST_YEAR], bases_empty=True, alignment="Standard")
    print(f"主範圍（無人在壘+Standard）: train n={len(train):,} "
          f"(out率 {train['is_out'].mean():.3f}), test n={len(test):,} "
          f"(out率 {test['is_out'].mean():.3f})")

    glm = make_optimizer_glm()
    dglm = make_difficulty_glm()
    gbm = make_difficulty_gbm()  # benchmark, not shipped to production
    report = {
        "train_years": TRAIN_YEARS, "test_year": TEST_YEAR,
        "n_train": len(train), "n_test": len(test),
        "optimizer_glm": {"features": OPTIMIZER_FEATURES},
        "difficulty_glm": {"features": DIFFICULTY_FEATURES},
        "difficulty_gbm_benchmark": {"features": DIFFICULTY_FEATURES},
    }
    report["optimizer_glm"]["metrics"] = evaluate(
        "優化用 GLM", glm, OPTIMIZER_FEATURES, train, test)
    report["difficulty_glm"]["metrics"] = evaluate(
        "評價用難度 GLM", dglm, DIFFICULTY_FEATURES, train, test)
    report["difficulty_gbm_benchmark"]["metrics"] = evaluate(
        "GBM benchmark", gbm, DIFFICULTY_FEATURES, train, test)

    print("\n優化用 GLM 校準（2025 十分位）：")
    p = glm.predict_proba(test[OPTIMIZER_FEATURES])[:, 1]
    print(calibration_table(test["is_out"].to_numpy(), p).round(3).to_string())
    print("\n評價用難度 GLM 校準（2025 十分位）：")
    p = dglm.predict_proba(test[DIFFICULTY_FEATURES])[:, 1]
    print(calibration_table(test["is_out"].to_numpy(), p).round(3).to_string())

    # Hit-type model (P(extra-base hit | ground-ball hit), used for run-value
    # pricing -- see src/if_runvalue.py. The training domain is league-wide
    # ground-ball hits (no base state/alignment restriction), unrelated to
    # the out model's main scope restriction)
    hits_tr = fetch_gb_hits(TRAIN_YEARS)
    hits_te = fetch_gb_hits([TEST_YEAR])
    xb = make_gb_xb_model().fit(hits_tr[XB_FEATURES], hits_tr["is_xb"])
    p = xb.predict_proba(hits_te[XB_FEATURES])[:, 1]
    xb_auc = roc_auc_score(hits_te["is_xb"], p)
    print(f"\n  {'安打類型模型 (XB)':<24} AUC={xb_auc:.4f}  "
          f"(train hits={len(hits_tr):,}, XB率 {hits_tr['is_xb'].mean():.3f})")
    report["gb_xb_model"] = {
        "features": XB_FEATURES,
        "metrics": {"auc": round(xb_auc, 4), "n_train_hits": len(hits_tr),
                    "n_test_hits": len(hits_te)},
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(glm, MODEL_DIR / "if_gb_optimizer_glm.joblib")
    joblib.dump(dglm, MODEL_DIR / "if_gb_difficulty_glm.joblib")
    joblib.dump(xb, MODEL_DIR / "if_gb_xb_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {MODEL_DIR}\\if_gb_optimizer_glm.joblib / "
          f"if_gb_difficulty_glm.joblib / if_gb_xb_model.joblib / metrics.json")


if __name__ == "__main__":
    main()
