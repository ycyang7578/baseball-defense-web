"""Train Stage B's two-stage out model (runner on 1B, <2 outs).

E[outs] = P(>=1 out) x (1 + P(double play | >=1 out))
- Stage 1: candidate = base geometry vs +throw_dist_2b, selected via
  2023->2024 validation
- Stage 2: candidate = base vs +hp x ball_time interaction, selected via
  2023->2024 validation
- Final: retrained on 2023-24, evaluated once on 2025 out-of-sample
  (AUC/Brier/calibration + E[outs] decile calibration)

Main scope: runner on 1B (1B only), <2 outs, Standard, non-bunt, |spray|<=55
degrees; positioning proxy = fielder_positioning_on1b (runner-on-1B split).

Run: python -m scripts.train.train_if_on1b
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.if_dataset import build_gb_on1b_dataset
from src.if_model import (ON1B_DP_FEATURES, ON1B_OUT_FEATURES,
                          OPTIMIZER_FEATURES, make_on1b_dp_glm,
                          make_on1b_out_glm, make_optimizer_glm)

from scripts._metrics import calibration_table

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "if_gb" / "on1b"


def metrics(y, p) -> dict:
    cal = calibration_table(np.asarray(y), p)
    return {"auc": round(roc_auc_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "calibration_max_dev": round(float((cal["pred"] - cal["actual"]).abs().max()), 4)}


def main() -> None:
    tr = build_gb_on1b_dataset(TRAIN_YEARS)
    te = build_gb_on1b_dataset([TEST_YEAR])
    print(f"主範圍（一壘有人+<2出局+Standard）: train n={len(tr):,} "
          f"(E[outs] {tr['n_outs'].mean():.3f}), test n={len(te):,} "
          f"(E[outs] {te['n_outs'].mean():.3f})")

    # -- Structure selection (train 2023 -> validate 2024, 2025 untouched) --
    t23, t24 = tr[tr.game_year == 2023], tr[tr.game_year == 2024]
    print("\n結構選擇（2023→2024）:")

    y1_23, y1_24 = (t23["n_outs"] >= 1).astype(int), (t24["n_outs"] >= 1).astype(int)
    base1 = make_optimizer_glm().fit(t23[OPTIMIZER_FEATURES], y1_23)
    cand1 = make_on1b_out_glm().fit(t23[ON1B_OUT_FEATURES], y1_23)
    auc_base1 = roc_auc_score(y1_24, base1.predict_proba(t24[OPTIMIZER_FEATURES])[:, 1])
    auc_cand1 = roc_auc_score(y1_24, cand1.predict_proba(t24[ON1B_OUT_FEATURES])[:, 1])
    use_2b = auc_cand1 >= auc_base1
    print(f"  階段1 基準幾何 AUC={auc_base1:.4f} vs +throw_dist_2b {auc_cand1:.4f} "
          f"→ {'採用' if use_2b else '不採用'} throw_dist_2b")

    d23, d24 = t23[t23["n_outs"] >= 1], t24[t24["n_outs"] >= 1]
    y2_23, y2_24 = (d23["n_outs"] == 2).astype(int), (d24["n_outs"] == 2).astype(int)
    dp_base = make_on1b_dp_glm(interactions=False).fit(d23[ON1B_DP_FEATURES], y2_23)
    dp_int = make_on1b_dp_glm(interactions=True).fit(d23[ON1B_DP_FEATURES], y2_23)
    auc_dp_base = roc_auc_score(y2_24, dp_base.predict_proba(d24[ON1B_DP_FEATURES])[:, 1])
    auc_dp_int = roc_auc_score(y2_24, dp_int.predict_proba(d24[ON1B_DP_FEATURES])[:, 1])
    use_int = auc_dp_int >= auc_dp_base
    print(f"  階段2 base AUC={auc_dp_base:.4f} vs +hp×bt 交互 {auc_dp_int:.4f} "
          f"→ {'採用' if use_int else '不採用'} 交互")

    # -- Final model (retrained on 2023-24, 2025 touched only once, here) --
    feats1 = ON1B_OUT_FEATURES if use_2b else OPTIMIZER_FEATURES
    m1 = (make_on1b_out_glm() if use_2b else make_optimizer_glm())
    y1_tr, y1_te = (tr["n_outs"] >= 1).astype(int), (te["n_outs"] >= 1).astype(int)
    m1.fit(tr[feats1], y1_tr)
    p1_te = m1.predict_proba(te[feats1])[:, 1]

    dp_tr, dp_te = tr[tr["n_outs"] >= 1], te[te["n_outs"] >= 1]
    y2_tr, y2_te = (dp_tr["n_outs"] == 2).astype(int), (dp_te["n_outs"] == 2).astype(int)
    m2 = make_on1b_dp_glm(interactions=use_int)
    m2.fit(dp_tr[ON1B_DP_FEATURES], y2_tr)
    p2_dp_te = m2.predict_proba(dp_te[ON1B_DP_FEATURES])[:, 1]

    rep1 = metrics(y1_te, p1_te)
    rep2 = metrics(y2_te, p2_dp_te)
    print(f"\n2025 樣本外:")
    print(f"  階段1 P(≥1 出局)  AUC={rep1['auc']:.4f}  Brier={rep1['brier']:.4f}  "
          f"校準最大偏差={rep1['calibration_max_dev']:.3f}")
    print(f"  階段2 P(DP|≥1out) AUC={rep2['auc']:.4f}  Brier={rep2['brier']:.4f}  "
          f"校準最大偏差={rep2['calibration_max_dev']:.3f}")

    # E[outs] calibration (full sample): E = p1 x (1 + p2), p2 must be computed for every ball
    p2_all_te = m2.predict_proba(te[ON1B_DP_FEATURES])[:, 1]
    e_outs = p1_te * (1.0 + p2_all_te)
    cal = pd.DataFrame({"pred": e_outs, "actual": te["n_outs"].to_numpy(float)})
    cal["bin"] = pd.qcut(cal["pred"], 10, duplicates="drop")
    tbl = cal.groupby("bin", observed=True).agg(pred=("pred", "mean"),
                                                actual=("actual", "mean"),
                                                n=("actual", "size"))
    print(f"\nE[outs] 2025 全體: 預測 {e_outs.mean():.4f} vs 實際 {te['n_outs'].mean():.4f}")
    print("E[outs] 十分位校準:")
    print(tbl.round(3).to_string())
    e_outs_dev = float((tbl["pred"] - tbl["actual"]).abs().max())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(m1, MODEL_DIR / "if_on1b_out_glm.joblib")
    joblib.dump(m2, MODEL_DIR / "if_on1b_dp_glm.joblib")
    report = {
        "train_years": TRAIN_YEARS, "test_year": TEST_YEAR,
        "n_train": len(tr), "n_test": len(te),
        "selection_2023_to_2024": {
            "stage1_base_auc": round(auc_base1, 4), "stage1_plus_2b_auc": round(auc_cand1, 4),
            "use_throw_dist_2b": bool(use_2b),
            "stage2_base_auc": round(auc_dp_base, 4), "stage2_interact_auc": round(auc_dp_int, 4),
            "use_interactions": bool(use_int),
        },
        "stage1_out": {"features": feats1, "metrics": rep1},
        "stage2_dp": {"features": ON1B_DP_FEATURES,
                      "interactions": bool(use_int), "metrics": rep2},
        "e_outs_2025": {"pred_mean": round(float(e_outs.mean()), 4),
                        "actual_mean": round(float(te["n_outs"].mean()), 4),
                        "decile_max_dev": round(e_outs_dev, 4)},
    }
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {MODEL_DIR}\\if_on1b_out_glm.joblib / if_on1b_dp_glm.joblib / metrics.json")


if __name__ == "__main__":
    main()
