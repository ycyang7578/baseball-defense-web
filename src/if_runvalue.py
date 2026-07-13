"""內野 run-value 目標（階段A）的零件：滾地安打類型模型 + 失分計價。

對齊外野 objective_re24 的慣例（src/optimization.py compute_w_j）：
- w_j（漏接代價）= Σ_k P(k|球 j) × ΔRE(k, 壘況)
- 外野的 P(k|j) 來自落點 KDE；滾地球的落點內生（hc 是被處理位置），改用
  出手參數建 P(長打|安打)：邊線球 = 二壘打風險、EV 高穿得深、跑者快有腿安打
  的延伸壘打。三壘打在滾地球極罕見，併入長打、以 ΔRE(2B) 計價（低估極小）。
- 出局價值 ΔRE(out)：滾地球出局以「跑者不推進、出局數+1」近似（無人在壘時精確）。

此模型與難度 GLM 同性質：評價/計價用，不搬野手、無反事實需求，spray 合法。
目標函數的換算（見 if_optimize.expected_outs docstring）：
    E[ΔRE] = mean(w_j) − mean(p_j × (w_j − ΔRE_out))
optimize_infield(ball_weights = w_j − ΔRE_out) 最大化第二項 ⇔ 最小化期望失分。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.config import DSN
from src.if_dataset import HOME_X, HOME_Y

XB_FEATURES = ["spray_deg", "launch_speed", "hp_to_1b"]


class XBRateFeatures(BaseEstimator, TransformerMixin):
    """P(長打|滾地安打) 的設計矩陣。

    spray 用**絕對角度** spline（兩條邊線都是二壘打區，場地幾何不隨打者鏡像）；
    launch_speed / hp_to_1b 線性。
    """

    def fit(self, X, y=None):
        self.spl_spray_ = SplineTransformer(n_knots=8, degree=3).fit(
            X[["spray_deg"]])
        self.scaler_ = StandardScaler().fit(X[["launch_speed", "hp_to_1b"]])
        return self

    def transform(self, X):
        s = self.spl_spray_.transform(X[["spray_deg"]])
        z = self.scaler_.transform(X[["launch_speed", "hp_to_1b"]])
        return np.hstack([s, z])


def make_gb_xb_model() -> Pipeline:
    return Pipeline([("features", XBRateFeatures()),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


def fetch_gb_hits(years: list[int]) -> pd.DataFrame:
    """聯盟滾地安打（單/二/三安），含打者該季 hp_to_1b（缺值以年中位數填補）。"""
    sql = """
        SELECT s.game_year, s.hc_x, s.hc_y, s.launch_speed, s.events,
               sp.hp_to_1b
        FROM statcast s
        LEFT JOIN sprint_speed sp
               ON sp.player_id = s.batter AND sp.season = s.game_year
        WHERE s.bb_type = 'ground_ball'
          AND s.game_year = ANY(%(years)s)
          AND s.hc_x IS NOT NULL AND s.launch_speed IS NOT NULL
          AND s.events IN ('single', 'double', 'triple')
          AND s.des NOT ILIKE '%%bunt%%'
        ORDER BY s.game_year, s.hc_x, s.hc_y
    """
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(sql, conn, params={"years": list(years)})
    df["spray_deg"] = np.degrees(
        np.arctan2(df["hc_x"] - HOME_X, HOME_Y - df["hc_y"]))
    df = df[df["spray_deg"].abs() <= 55].copy()
    med = df.groupby("game_year")["hp_to_1b"].transform("median")
    df["hp_to_1b"] = df["hp_to_1b"].fillna(med)
    df["is_xb"] = df["events"].isin(["double", "triple"]).astype(int)
    return df.reset_index(drop=True)


def train_gb_xb_model(years: list[int]) -> Pipeline:
    hits = fetch_gb_hits(years)
    model = make_gb_xb_model()
    model.fit(hits[XB_FEATURES], hits["is_xb"])
    return model


def delta_re_out(re24: dict, state: tuple[int, int, int, int]) -> float:
    """滾地球出局的 ΔRE：跑者不推進、出局+1（無人在壘時精確）。"""
    b1, b2, b3, outs = state
    if outs >= 2:
        return -re24.get(state, 0.0)          # 第三個出局：半局結束
    return re24.get((b1, b2, b3, outs + 1), 0.0) - re24.get(state, 0.0)


def gb_miss_costs(balls: pd.DataFrame, xb_model: Pipeline, delta_re: dict,
                  state: tuple[int, int, int, int]) -> np.ndarray:
    """每球的漏接代價 w_j（對齊外野 compute_w_j 的鍵值慣例）。"""
    p_xb = xb_model.predict_proba(balls[XB_FEATURES])[:, 1]
    dre_1b = delta_re.get(("1B", *state), 0.0)
    dre_2b = delta_re.get(("2B", *state), 0.0)
    return (1.0 - p_xb) * dre_1b + p_xb * dre_2b


def runvalue_ball_weights(balls: pd.DataFrame, xb_model: Pipeline,
                          re24: dict, delta_re: dict,
                          state: tuple[int, int, int, int]
                          ) -> tuple[np.ndarray, float]:
    """回傳 (optimize_infield 用的 ball_weights, mean_miss_cost)。

    E[ΔRE](angles, depths) = mean_miss_cost − score(ball_weights 版 scorer)。
    """
    w = gb_miss_costs(balls, xb_model, delta_re, state)
    dre_o = delta_re_out(re24, state)
    return w - dre_o, float(w.mean())
