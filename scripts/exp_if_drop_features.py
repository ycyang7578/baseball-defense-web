"""消融：現行 GLM 拿掉 stand_R / launch_speed（含其 ad×EV 交互）的代價。

使用者提議精簡特徵（2026-07-12）。訓練 2023-24、2025 樣本外，
判準沿用結構特徵實驗：AUC 差 <0.003 視為等價、校準不得惡化。

執行：python scripts/exp_if_drop_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.if_dataset import build_gb_dataset

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025


class VariantFeatures(BaseEstimator, TransformerMixin):
    """FielderGeometryFeatures 的可開關版：include_ev / include_stand / spline_all。

    spline_all=True 時 launch_speed/throw_dist/hp_to_1b 的主效應也上 spline
    （交互項仍用 z 分數，隔離「主效應形狀」這一個變因）。
    """

    def __init__(self, include_ev=True, include_stand=True, spline_all=False,
                 spline_extra=()):
        self.include_ev = include_ev
        self.include_stand = include_stand
        self.spline_all = spline_all
        self.spline_extra = spline_extra

    def _extra_spline_cols(self):
        if self.spline_all:
            return ["throw_dist", "hp_to_1b"] + (
                ["launch_speed"] if self.include_ev else [])
        return list(self.spline_extra)

    def fit(self, X, y=None):
        spline_cols = ["ad_min", "ball_time", "launch_angle"] + self._extra_spline_cols()
        self.splines_ = {c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
                         for c in spline_cols}
        lin = ["throw_dist", "hp_to_1b"] + (["launch_speed"] if self.include_ev else [])
        self.lin_cols_ = lin
        self.scaler_ = StandardScaler().fit(X[lin])
        return self

    def transform(self, X):
        a = self.splines_["ad_min"].transform(X[["ad_min"]])
        b = self.splines_["ball_time"].transform(X[["ball_time"]])
        la = self.splines_["launch_angle"].transform(X[["launch_angle"]])
        z = self.scaler_.transform(X[self.lin_cols_])
        throw, hp = z[:, [0]], z[:, [1]]
        n = len(X)
        extra = self._extra_spline_cols()
        main = []  # 主效應：extra 名單內用 spline，其餘用 z 分數線性
        for i, c in enumerate(self.lin_cols_):
            main.append(self.splines_[c].transform(X[[c]]) if c in extra
                        else z[:, [i]])
        parts = [*main, a, b, la,
                 (a[:, :, None] * b[:, None, :]).reshape(n, -1),  # tensor ad×bt
                 hp * throw, hp * b]                              # 跑者交互
        if self.include_ev:
            parts.append(a * z[:, [2]])                           # ad×EV
        if self.include_stand:
            parts.append(X[["stand_R"]].to_numpy(float))
        return np.hstack(parts)


def calibration_max_dev(y, p, bins: int = 10) -> float:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    return float((g["pred"] - g["obs"]).abs().max())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", type=int, nargs="+", default=TRAIN_YEARS)
    ap.add_argument("--test-year", type=int, default=TEST_YEAR)
    args = ap.parse_args()

    cols = ["ad_min", "ball_time", "launch_angle", "launch_speed",
            "throw_dist", "hp_to_1b", "stand_R"]
    train = build_gb_dataset(args.train_years, bases_empty=True,
                             alignment="Standard")
    test = build_gb_dataset([args.test_year], bases_empty=True,
                            alignment="Standard")
    print(f"train={args.train_years} test={args.test_year}")
    y_tr, y_te = train["is_out"].to_numpy(), test["is_out"].to_numpy()
    print(f"n_train={len(train):,}  n_test={len(test):,}")

    configs = {
        "full (現行)": dict(include_ev=True, include_stand=True),
        "-stand_R": dict(include_ev=True, include_stand=False),
        "-launch_speed": dict(include_ev=False, include_stand=True),
        "-both": dict(include_ev=False, include_stand=False),
        "all_spline": dict(include_ev=True, include_stand=True, spline_all=True),
        "all_spline_-stand": dict(include_ev=True, include_stand=False,
                                  spline_all=True),
        "ev_spline_-stand": dict(include_ev=True, include_stand=False,
                                 spline_extra=("launch_speed",)),
        "ev_hp_spline_-stand": dict(include_ev=True, include_stand=False,
                                    spline_extra=("launch_speed", "hp_to_1b")),
    }
    rows = []
    for name, kw in configs.items():
        model = Pipeline([("features", VariantFeatures(**kw)),
                          ("lr", LogisticRegression(max_iter=8000, C=1.0))])
        model.fit(train[cols], y_tr)
        p = model.predict_proba(test[cols])[:, 1]
        rows.append({"config": name,
                     "auc": round(roc_auc_score(y_te, p), 4),
                     "brier": round(brier_score_loss(y_te, p), 4),
                     "cal_max_dev": round(calibration_max_dev(y_te, p), 4)})
    print("\n2025 out-of-sample:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
