"""內野手球員評價的共用邏輯：難度模型評分 + 歸責 + 分位置中心化。

從 scripts/evaluate_if_2025.py 抽出（計算逐行保持一致），讓評估報表與
web 排名預算（scripts/precompute_if_model_oaa.py）用同一份實作。
方法論說明見 evaluate_if_2025.py docstring。
2026-07-12 起難度模型預設為可解釋的 spline GLM（見 if_model.py docstring）。
"""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.if_dataset import INFIELD_COLS, build_gb_dataset
from src.if_model import DIFFICULTY_FEATURES, make_difficulty_glm


def score_test_year(train_years: list[int], test_year: int,
                    model_factory: Callable[[], Pipeline] = make_difficulty_glm) -> pd.DataFrame:
    """全量滾地球逐球評分：train_years 訓練難度模型、test_year 評分。

    model_factory 預設為評價用難度 GLM；替代模型（如 GBM benchmark）可傳入
    自己的 factory，需吃 DIFFICULTY_FEATURES 欄位。
    回傳 test 逐球 DataFrame，附加欄位：
    p_hat（聯盟平均難度）、resp_fielder/resp_pos（歸責）、
    oaa_play（is_out − p̂）、oaa_play_c（分位置中心化後）。
    """
    train = build_gb_dataset(train_years)
    test = build_gb_dataset([test_year])
    print(f"全量滾地球: train n={len(train):,}, test n={len(test):,}")

    diff_model = model_factory()
    diff_model.fit(train[DIFFICULTY_FEATURES], train["is_out"])
    test = test.copy()
    test["p_hat"] = diff_model.predict_proba(test[DIFFICULTY_FEATURES])[:, 1]

    # 歸責：出局球給實際處理者（hit_location 3~6，對齊官方 credit）；
    # 安打球與投手/捕手處理的球退回最近角距的內野手
    pos_to_col = {v: k for k, v in INFIELD_COLS.items()}
    nearest_ids = np.select(
        [test["nearest_pos"] == p for p in pos_to_col],
        [test[c] for c in (pos_to_col[p] for p in pos_to_col)])
    loc = pd.to_numeric(test["hit_location"], errors="coerce")
    use_loc = (test["is_out"] == 1) & loc.between(3, 6)
    loc_ids = np.select(
        [loc == int(c.split("_")[1]) for c in INFIELD_COLS],
        [test[c] for c in INFIELD_COLS])
    test["resp_fielder"] = np.where(use_loc, loc_ids, nearest_ids).astype(int)
    loc_pos = loc.map({3: "1B", 4: "2B", 5: "3B", 6: "SS"})
    test["resp_pos"] = np.where(use_loc, loc_pos, test["nearest_pos"])
    print(f"出局球以 hit_location 歸責的比例: {use_loc[test['is_out'] == 1].mean():.1%}"
          f"（其中與最近角距不同者 "
          f"{(use_loc & (loc_ids != nearest_ids)).sum() / max(use_loc.sum(), 1):.1%}）")
    test["oaa_play"] = test["is_out"] - test["p_hat"]
    # 分位置中心化：官方 OAA 是「跟同位置平均比」，p̂ 是全聯盟 lane 平均，位置間有
    # 系統性偏移（例如 1B）會拖低合併相關。每球減去該位置的平均後，各位置總和歸零。
    test["oaa_play_c"] = (test["oaa_play"]
                          - test.groupby("resp_pos")["oaa_play"].transform("mean"))
    return test


def aggregate_players(test: pd.DataFrame) -> pd.DataFrame:
    """逐球 → 每位球員：model_oaa（中心化）/model_oaa_raw/n_balls/resp_pos（眾數）。"""
    model = (test.groupby("resp_fielder")
             .agg(model_oaa=("oaa_play_c", "sum"), model_oaa_raw=("oaa_play", "sum"),
                  n_balls=("oaa_play", "size"))
             .reset_index())
    pos_mode = (test.groupby(["resp_fielder", "resp_pos"]).size()
                .rename("n").reset_index()
                .sort_values("n", ascending=False)
                .drop_duplicates("resp_fielder"))
    return model.merge(pos_mode[["resp_fielder", "resp_pos"]], on="resp_fielder")
