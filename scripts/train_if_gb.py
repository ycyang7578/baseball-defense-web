"""訓練內野滾地球 out probability 模型（階段 1）。

主模型：logistic regression（spline 基底）。結構對應 Tango 的內野 OAA 競速觀點：
sigmoid(時間差) 的線性項就是 throw_dist 與 hp_to_1b，幾何項（ad_min、ball_time）
沿用 Melville §3.1，再加球質（EV、LA）。
Benchmark：HistGradientBoosting（無約束），用來偵測 GLM 還漏多少非線性訊號——
兩者差距大才值得升級模型結構。

訓練 2023–2024（禁趨位後規則同代），測試 2025（樣本外）。
主範圍：無人在壘 + Standard 佈陣（賽季平均站位在此誤差最小）；
同時報告全量滾地球的結果供對照。

執行：python scripts/train_if_gb.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.if_dataset import build_gb_dataset

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "if_gb"

FEATURES = ["ad_min", "ball_time", "launch_angle", "launch_speed",
            "throw_dist", "hp_to_1b", "stand_R"]

# spline 欄位：效果明顯非線性（ad_min 有內生性造成的小角差凹陷、ball_time 有最佳深度、
# LA 有滾地球型態差異）；其餘欄位維持線性（競速時間差的線性結構）
SPLINE_COLS = ["ad_min", "ball_time", "launch_angle"]
LINEAR_COLS = ["launch_speed", "throw_dist", "hp_to_1b", "stand_R"]


def make_glm() -> Pipeline:
    ct = ColumnTransformer([
        ("spline", SplineTransformer(n_knots=6, degree=3), SPLINE_COLS),
        ("linear", StandardScaler(), LINEAR_COLS),
    ])
    return Pipeline([("features", ct),
                     ("lr", LogisticRegression(max_iter=3000, C=1.0))])


def make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                          early_stopping=True, random_state=42)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))


def evaluate(name: str, model, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model.fit(train[FEATURES], train["is_out"])
    p = model.predict_proba(test[FEATURES])[:, 1]
    auc = roc_auc_score(test["is_out"], p)
    brier = brier_score_loss(test["is_out"], p)
    cal = calibration_table(test["is_out"].to_numpy(), p)
    max_dev = float((cal["pred"] - cal["actual"]).abs().max())
    print(f"  {name:<10} AUC={auc:.4f}  Brier={brier:.4f}  校準最大偏差={max_dev:.3f}")
    return {"auc": round(auc, 4), "brier": round(brier, 4),
            "calibration_max_dev": round(max_dev, 4)}


def run_scope(scope_name: str, bases_empty, alignment) -> tuple[dict, Pipeline, pd.DataFrame]:
    train = build_gb_dataset(TRAIN_YEARS, bases_empty=bases_empty, alignment=alignment)
    test = build_gb_dataset([TEST_YEAR], bases_empty=bases_empty, alignment=alignment)
    print(f"\n=== {scope_name}: train n={len(train):,} (out率 {train['is_out'].mean():.3f}), "
          f"test n={len(test):,} (out率 {test['is_out'].mean():.3f}) ===")
    glm = make_glm()
    metrics = {
        "n_train": len(train), "n_test": len(test),
        "glm": evaluate("GLM", glm, train, test),
        "gbm": evaluate("GBM bench", make_gbm(), train, test),
    }
    return metrics, glm, test


def main() -> None:
    report = {"train_years": TRAIN_YEARS, "test_year": TEST_YEAR, "features": FEATURES}

    metrics, glm, test = run_scope("主範圍：無人在壘 + Standard", True, "Standard")
    report["primary"] = metrics
    print("\n主範圍 GLM 校準（2025 十分位）：")
    p = glm.predict_proba(test[FEATURES])[:, 1]
    print(calibration_table(test["is_out"].to_numpy(), p).round(3).to_string())

    report["all_gb"], _, _ = run_scope("對照：全量滾地球", None, None)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(glm, MODEL_DIR / "if_gb_glm.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {MODEL_DIR}\\if_gb_glm.joblib + metrics.json")


if __name__ == "__main__":
    main()
