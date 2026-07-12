"""實驗：Tango 式結構特徵 vs 現行 GLM（訓練 2023-24、評估 2025 樣本外）。

動機（2026-07-12，文獻精讀後）：官方內野 OAA 的競速結構
race_margin = hp_to_1b - (球到野手時間 + 出手時間 + 傳球飛行時間)
其中出手時間為常數被截距吸收；ball_time 是無減速 proxy、跟 hp_to_1b 的實測秒
時間軸不對齊，故引入校正係數 c 與傳球速度 v（皆只用訓練集擬合，不能照搬官方
1.72s/100ft/s——那是配實測時間的參數）。

required_speed = lat_ft / ball_time：野手到位所需橫移速度（官方「到位機率」段）。

配置（累加消融）：
  A  現行 production GLM（splines + tensor，7 特徵）
  S2 required_speed + race_margin（純結構）
  S3 S2 + launch_angle
  S4 S3 + launch_speed（球質海綿：慢滾球減速誤差與處理難度）

判準（事先定）：結構配置落後 A 在 0.003 AUC 內且校準不惡化 → 可採納精簡版；
S4-S3 < 0.003 → launch_speed 可拿掉。

執行：python scripts/exp_if_structural_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.if_dataset import build_gb_dataset
from src.if_model import OPTIMIZER_FEATURES, make_optimizer_glm

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025

C_GRID = np.arange(0.8, 3.01, 0.1)     # ball_time 校正係數（>1 = 減速讓真實時間更長）
V_GRID = np.arange(60.0, 141.0, 5.0)   # 傳球飛行速度 ft/s


def calibration_max_dev(y, p, bins: int = 10) -> float:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    return float((g["pred"] - g["obs"]).abs().max())


def add_structural(df: pd.DataFrame, c: float, v: float) -> pd.DataFrame:
    out = df.copy()
    out["required_speed"] = out["lat_ft"] / out["ball_time"]
    out["race_margin"] = out["hp_to_1b"] - (c * out["ball_time"]
                                            + out["throw_dist"] / v)
    return out


def fit_margin_params(train: pd.DataFrame) -> tuple[float, float]:
    """在訓練集上 grid search (c, v)：單特徵 logistic(race_margin) 的 logloss 最小。"""
    y = train["is_out"].to_numpy()
    best = (np.inf, None, None)
    for c in C_GRID:
        for v in V_GRID:
            m = (train["hp_to_1b"] - (c * train["ball_time"]
                                      + train["throw_dist"] / v)).to_numpy()[:, None]
            lr = LogisticRegression(max_iter=1000).fit(m, y)
            ll = log_loss(y, lr.predict_proba(m)[:, 1])
            if ll < best[0]:
                best = (ll, c, v)
    return best[1], best[2]


def make_plain_glm() -> Pipeline:
    # 鏡射 production LR 設定（C=1.0），只是設計矩陣換成結構特徵
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


def main() -> None:
    train = build_gb_dataset(TRAIN_YEARS, bases_empty=True, alignment="Standard")
    test = build_gb_dataset([TEST_YEAR], bases_empty=True, alignment="Standard")
    print(f"n_train={len(train):,}  n_test={len(test):,}")

    c, v = fit_margin_params(train)
    print(f"fitted margin params: c={c:.1f} (ball_time correction), "
          f"v={v:.0f} ft/s (throw flight)")
    train = add_structural(train, c, v)
    test = add_structural(test, c, v)

    # margin<0 的球該不該留在訓練集：看它們的實際出局率
    neg = train[train["race_margin"] < 0]
    print(f"\nrace_margin < 0 in train: n={len(neg):,} "
          f"({len(neg) / len(train):.1%}), actual out rate = "
          f"{neg['is_out'].mean():.3f}  (kept in training)")
    for lo, hi in [(-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2)]:
        sub = train[(train["race_margin"] >= lo) & (train["race_margin"] < hi)]
        if len(sub):
            print(f"  margin [{lo:+.1f},{hi:+.1f}): n={len(sub):>6,} "
                  f"out_rate={sub['is_out'].mean():.3f}")

    configs = {
        "A_production": None,  # 特例：production pipeline
        "S2_structural": ["required_speed", "race_margin"],
        "S3_+LA": ["required_speed", "race_margin", "launch_angle"],
        "S4_+LA+EV": ["required_speed", "race_margin", "launch_angle",
                      "launch_speed"],
        # backlog 的原始問題：結構特徵加在現行 GLM 上有沒有殘餘增益
        "A+req_speed": "aug_rs",
        "A+rs+margin": "aug_rs_rm",
    }
    y_tr = train["is_out"].to_numpy()
    y_te = test["is_out"].to_numpy()
    rows = []
    for name, feats in configs.items():
        if feats is None:
            model = make_optimizer_glm()
            model.fit(train[OPTIMIZER_FEATURES], y_tr)
            p = model.predict_proba(test[OPTIMIZER_FEATURES])[:, 1]
            nf = "7+splines"
        elif isinstance(feats, str):  # production 設計矩陣 + 追加結構欄位
            extra = (["required_speed"] if feats == "aug_rs"
                     else ["required_speed", "race_margin"])
            fg = make_optimizer_glm().named_steps["features"].fit(
                train[OPTIMIZER_FEATURES])
            sc = StandardScaler().fit(train[extra])
            Xtr = np.hstack([fg.transform(train[OPTIMIZER_FEATURES]),
                             sc.transform(train[extra])])
            Xte = np.hstack([fg.transform(test[OPTIMIZER_FEATURES]),
                             sc.transform(test[extra])])
            model = LogisticRegression(max_iter=8000, C=1.0).fit(Xtr, y_tr)
            p = model.predict_proba(Xte)[:, 1]
            nf = f"7+splines+{len(extra)}"
        else:
            model = make_plain_glm()
            model.fit(train[feats], y_tr)
            p = model.predict_proba(test[feats])[:, 1]
            nf = len(feats)
        rows.append({"config": name, "n_features": nf,
                     "auc": round(roc_auc_score(y_te, p), 4),
                     "brier": round(brier_score_loss(y_te, p), 4),
                     "cal_max_dev": round(calibration_max_dev(y_te, p), 4)})
    print("\n2025 out-of-sample:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
