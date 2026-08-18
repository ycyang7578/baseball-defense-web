"""訓練內野滾地球 out probability 模型（階段 1）。

兩個模型、兩種角色（詳見 src/if_model.py 的 docstring）：
- 優化用 GLM：野手相對幾何（可反事實——搬動野手預測會跟著變）
- 評價用難度 GLM：spray+球質+跑者的聯盟平均難度模型（位置固定情境的 p̂；
  2026-07-12 起取代 GBM，可解釋性優先；GBM 續留當 benchmark 指標）

訓練 2023–2024（禁令後，賽季平均站位不混 shift 佈陣；2021–22 的整季平均混入
shift 球——2B 深度/3B 角度有系統性偏差——Melville 2024 同理只用禁令後資料。
2026-07-09 實驗確認兩種年份配置 2025 樣本外三指標差異皆在雜訊內，取乾淨者），
測試 2025（樣本外）。
交互作用配置是先用 train 2023 → validate 2024 選定的，2025 只在這裡碰一次。
主範圍：無人在壘 + Standard 佈陣（Melville 同樣排除壘上有人；1B hold runner 會拉動站位）。

執行：python -m scripts.train.train_if_gb
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
    gbm = make_difficulty_gbm()  # benchmark，不進生產
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

    # 安打類型模型（P(長打|滾地安打)，run-value 計價用——見 src/if_runvalue.py。
    # 訓練域是聯盟滾地安打（不限壘況/佈陣），與 out 模型的主範圍限制無關）
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
