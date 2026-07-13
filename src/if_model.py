"""內野滾地球 out probability 的兩個模型定義（角色不同，勿混用）。

1. make_optimizer_glm() — 站位優化用（counterfactual）。
   只用「野手相對幾何」特徵（ad_min/ball_time/throw_dist...），刻意**排除** raw
   spray_deg 項：spray 的位置特定出局率模式反映的是「現在聯盟都站在哪」（內生性），
   搬動野手之後那些模式不會保留；混入會讓優化器重複計算（例如把 SS 移進 5.5 洞後，
   spray 項仍預測那裡出局率低）。
2. make_difficulty_glm() — 球員評價用（階段 2 的 p̂）。
   位置固定情境下的聯盟平均難度模型（xBA 式），spray 位置模式合法且應該用；
   評價不搬野手、無反事實需求，內生性禁令（raw spray/彈性形狀）不適用。
   **2026-07-12 起改用可解釋的 spline GLM 取代 GBM**（使用者決定：不用無法
   說明的模型，接受準確度代價）。實測代價（exp_if_difficulty_glm.py，
   全量滾地球 2023-24→2025）：逐球 AUC 0.770 vs GBM 0.824，但評價是數百球
   加總，qualified R 僅 0.525→0.514、scale 同樣健康（SD 8.3 vs 官方 6.9）。
   特徵工程：spray 左打鏡像（Melville 同款）+ spray/LA/EV/hp splines +
   spray×EV、spray×hp 交互（spray×EV 對校準是必要的：無交互 cal 0.043→0.026）。
3. make_difficulty_gbm() — GBM benchmark（保留供對照，不進生產）。
   2023→2024 驗證：AUC 0.807，只用球質+跑者+spray，加野手特徵無增益（+0.001）——
   證據：Standard 佈陣下賽季平均站位近似 spray 的確定函數。

結構選擇流程（2026-07-06）：train 2023 → validate 2024 挑交互作用配置，2025 只在
最終評估碰一次。GLM 交互配置驗證結果：base 0.7460 → +tensor(ad×bt)+ad×EV+跑者交互
0.7536。
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

# 優化用 GLM 的輸入欄位（野手相對幾何 + 球質 + 跑者）
# stand_R 於 2026-07-12 移除：hp_to_1b 是實測本壘到一壘秒數、已含左打站位優勢，
# 消融顯示 stand_R 冗餘（2025 AUC −0.0006、校準略好，exp_if_drop_features.py）
OPTIMIZER_FEATURES = ["ad_min", "ball_time", "launch_angle", "launch_speed",
                      "throw_dist", "hp_to_1b"]
# 評價用 GBM 的輸入欄位（無任何野手資訊）
DIFFICULTY_FEATURES = ["spray_deg", "launch_angle", "launch_speed",
                       "hp_to_1b", "stand_R"]


class FielderGeometryFeatures(BaseEstimator, TransformerMixin):
    """優化用 GLM 的設計矩陣。

    spline 基底：ad_min、ball_time、launch_angle（非線性效果）。
    線性項：launch_speed、throw_dist、hp_to_1b（競速時間差的線性結構）。
    throw_dist/hp_to_1b **刻意維持線性**：throw_dist 是攔截幾何（spray×深度）的
    確定函數，給彈性形狀會從後門重新編碼被排除的 raw spray 區域訊號（2026-07-12
    實驗：throw_dist spline 單獨 +0.012 AUC，但偏效應呈守位身分的波浪、105 呎谷
    對應 2B/SS 交界的中間地帶——優化器會追假凸起，見 exp_if_drop_features.py）。
    交互作用（2024 驗證選定）：
    - tensor spline(ad_min)×spline(ball_time)：強襲球正面 vs 慢滾球遠處是不同的 play
    - spline(ad_min)×launch_speed
    - hp_to_1b×throw_dist、hp_to_1b×spline(ball_time)：跑者速度只在 close play 起作用
    """

    def fit(self, X, y=None):
        self.splines_ = {c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
                         for c in ("ad_min", "ball_time", "launch_angle")}
        self.scaler_ = StandardScaler().fit(
            X[["launch_speed", "throw_dist", "hp_to_1b"]])
        return self

    def transform(self, X):
        a = self.splines_["ad_min"].transform(X[["ad_min"]])
        b = self.splines_["ball_time"].transform(X[["ball_time"]])
        la = self.splines_["launch_angle"].transform(X[["launch_angle"]])
        z = self.scaler_.transform(X[["launch_speed", "throw_dist", "hp_to_1b"]])
        ev, throw, hp = z[:, [0]], z[:, [1]], z[:, [2]]
        n = len(X)
        return np.hstack([
            a, b, la, z,
            (a[:, :, None] * b[:, None, :]).reshape(n, -1),  # tensor ad×bt
            a * ev,                                          # ad×EV
            hp * throw, hp * b,                              # 跑者交互
        ])


def make_optimizer_glm() -> Pipeline:
    return Pipeline([("features", FielderGeometryFeatures()),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


class DifficultyGLMFeatures(BaseEstimator, TransformerMixin):
    """評價用難度 GLM 的設計矩陣（無野手資訊）。

    spray 先做左打鏡像（負=拉打側，左右打共享形狀，Melville §2 同款），
    上 8 節點 spline（要容納四守位的 lane 結構）；LA/EV/hp 各 6 節點。
    交互：spray×EV（強襲穿洞）、spray×hp（慢滾內野安打的方向性）。
    """

    def __init__(self, spray_ev=True, spray_hp=True):
        self.spray_ev = spray_ev
        self.spray_hp = spray_hp

    @staticmethod
    def _spray_rel(X):
        sign = np.where(X["stand_R"].to_numpy(float) == 1, 1.0, -1.0)
        return (X["spray_deg"].to_numpy(float) * sign)[:, None]

    def fit(self, X, y=None):
        self.spl_spray_ = SplineTransformer(n_knots=8, degree=3).fit(
            self._spray_rel(X))
        self.spl_ = {c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
                     for c in ("launch_angle", "launch_speed", "hp_to_1b")}
        self.scaler_ = StandardScaler().fit(X[["launch_speed", "hp_to_1b"]])
        return self

    def transform(self, X):
        s = self.spl_spray_.transform(self._spray_rel(X))
        la = self.spl_["launch_angle"].transform(X[["launch_angle"]])
        ev = self.spl_["launch_speed"].transform(X[["launch_speed"]])
        hp = self.spl_["hp_to_1b"].transform(X[["hp_to_1b"]])
        z = self.scaler_.transform(X[["launch_speed", "hp_to_1b"]])
        parts = [s, la, ev, hp, X[["stand_R"]].to_numpy(float)]
        if self.spray_ev:
            parts.append(s * z[:, [0]])
        if self.spray_hp:
            parts.append(s * z[:, [1]])
        return np.hstack(parts)


def make_difficulty_glm(**kw) -> Pipeline:
    return Pipeline([("features", DifficultyGLMFeatures(**kw)),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


def make_difficulty_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                          early_stopping=True, random_state=42)


# ── 階段B：一壘有人（<2 出局）的兩段式優化用 GLM ──────────────────
# E[outs] = P(≥1 出局) × (1 + P(雙殺 | ≥1 出局))，兩段皆 counterfactual
# （只用野手相對幾何＋球質＋跑速，排除 raw spray，同 make_optimizer_glm 的紀律）。

ON1B_OUT_FEATURES = OPTIMIZER_FEATURES + ["throw_dist_2b"]
ON1B_DP_FEATURES = ["ad_min", "ball_time", "launch_speed", "hp_to_1b",
                    "runner_hp_to_1b", "pivot_dist", "throw_dist_2b"]


class On1bOutFeatures(FielderGeometryFeatures):
    """階段1 P(≥1 出局)：基準幾何設計矩陣＋throw_dist_2b 線性項。

    一壘有人時最近的 force 目標是二壘，throw_dist（到一壘）之外補傳二壘距離；
    與 throw_dist 同理**維持線性**（攔截幾何的確定函數，彈性形狀=raw spray 後門）。
    """

    def fit(self, X, y=None):
        super().fit(X, y)
        self.scaler_2b_ = StandardScaler().fit(X[["throw_dist_2b"]])
        return self

    def transform(self, X):
        return np.hstack([super().transform(X),
                          self.scaler_2b_.transform(X[["throw_dist_2b"]])])


class On1bDPFeatures(BaseEstimator, TransformerMixin):
    """階段2 P(雙殺 | ≥1 出局) 的設計矩陣。

    spline：ad_min、ball_time（正面接到快球才來得及轉傳）。
    線性：launch_speed、hp_to_1b（打者到一壘＝雙殺第二個 out 的競速）、
    runner_hp_to_1b（跑者到二壘＝force 的競速）、pivot_dist（軸心幾何）、
    throw_dist_2b——幾何項一律線性（函數形式內生性紀律）。
    interactions=True 加 hp_to_1b×spline(ball_time)（跑者速度只在 close play
    起作用，沿用階段1 的結論；是否採用由 2023→2024 驗證決定）。
    """

    def __init__(self, interactions=True):
        self.interactions = interactions

    _LINEAR = ["launch_speed", "hp_to_1b", "runner_hp_to_1b",
               "pivot_dist", "throw_dist_2b"]

    def fit(self, X, y=None):
        self.splines_ = {c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
                         for c in ("ad_min", "ball_time")}
        self.scaler_ = StandardScaler().fit(X[self._LINEAR])
        return self

    def transform(self, X):
        a = self.splines_["ad_min"].transform(X[["ad_min"]])
        b = self.splines_["ball_time"].transform(X[["ball_time"]])
        z = self.scaler_.transform(X[self._LINEAR])
        parts = [a, b, z]
        if self.interactions:
            hp = z[:, [self._LINEAR.index("hp_to_1b")]]
            parts.append(hp * b)
        return np.hstack(parts)


def make_on1b_out_glm() -> Pipeline:
    return Pipeline([("features", On1bOutFeatures()),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


def make_on1b_dp_glm(**kw) -> Pipeline:
    return Pipeline([("features", On1bDPFeatures(**kw)),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])
