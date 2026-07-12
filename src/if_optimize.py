"""內野站位最佳化（2023 禁趨位規則下的約束優化）。

規則約束的處理方式（全部化成 box bounds，讓 L-BFGS-B 直接吃）：
- 二壘兩側各至少兩名內野手、不可換邊 → 1B/2B 角度限 [1°, 44°]、3B/SS 限 [-44°, -1°]
- 站在內野土上 → 深度重參數化為「60 呎到土外緣的比例 f∈[0,1]」，
  土外緣近似為以投手板（距本壘 60.5 呎）為圓心、半徑 95 呎的弧：
  r_max(θ) = 60.5·cosθ + sqrt(95² − (60.5·sinθ)²)
  （角落方向的土實際延伸得更遠，此近似在邊線側偏保守）

目標：max 打者歷史滾地球樣本上的平均 P(out)，P(out) 用優化用 GLM
（models/if_gb/if_gb_optimizer_glm.joblib；只含野手相對幾何，可反事實）。
打者分布直接用歷史球（角度不受站位污染，見 src/if_dataset.py docstring），
不需要重建 KDE。

球員個人化（貝葉斯球員層，見 scripts/train_if_bayes.py）：`player_effects`
指定四個位置野手的 (alpha_j, g_j) 後驗平均，逐球加到最近野手身上：
logit += alpha_j + g_j × ad_z。個人化時同側標籤不再可互換（槽位綁定特定
野手），因此不做角落正規化。格式：
{"alpha": ndarray(4), "g": ndarray(4), "ad_mean": float, "ad_std": float}
（依 POSITIONS 順序；聯盟平均野手 = alpha=g=0）。

已知限制：GLM 是在實際（賽季平均）站位附近的資料上訓練的，離常態很遠的
候選站位屬外插，解讀時要保守。
"""
import copy

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from scipy.stats import qmc

from src.config import DSN
from src.if_dataset import MPH_TO_FTS, OUT_EVENTS, NONOUT_EVENTS, HOME_X, HOME_Y

MOUND_DIST = 60.5
DIRT_RADIUS = 95.0
# 深度下限取訓練資料的支撐範圍邊緣（賽季平均站位約 100–155 呎）：更淺屬 GLM 外插，
# 實測會讓優化器把「閒置野手」（打者冷區的那側）停在外插最樂觀的假象位置
MIN_DEPTH = 75.0
POSITIONS = ("1B", "2B", "3B", "SS")
# box bounds：前 4 維是角度（度）、後 4 維是深度比例 f
ANGLE_BOUNDS = [(1.0, 44.0), (1.0, 44.0), (-44.0, -1.0), (-44.0, -1.0)]
FRAC_BOUNDS = [(0.0, 1.0)] * 4
_FIRST_BASE_XY = (90.0 * np.sin(np.radians(45.0)), 90.0 * np.cos(np.radians(45.0)))


def dirt_max_depth(angle_deg):
    """指定角度下內野土外緣的深度（呎）。"""
    rad = np.radians(angle_deg)
    return MOUND_DIST * np.cos(rad) + np.sqrt(
        DIRT_RADIUS ** 2 - (MOUND_DIST * np.sin(rad)) ** 2)


def params_to_positions(x):
    """8 維參數向量 → (angles[4], depths[4])，依 POSITIONS 順序。"""
    angles = np.asarray(x[:4], dtype=float)
    fracs = np.asarray(x[4:], dtype=float)
    depths = MIN_DEPTH + fracs * (dirt_max_depth(angles) - MIN_DEPTH)
    return angles, depths


def positions_to_params(angles, depths) -> np.ndarray:
    """(angles, depths) → 8 維參數向量（params_to_positions 的反函數）。"""
    angles = np.asarray(angles, dtype=float)
    depths = np.asarray(depths, dtype=float)
    fracs = (depths - MIN_DEPTH) / (dirt_max_depth(angles) - MIN_DEPTH)
    return np.concatenate([angles, np.clip(fracs, 0.0, 1.0)])


def geometry_features(balls: pd.DataFrame, angles, depths) -> pd.DataFrame:
    """給定站位，重算每顆球的野手相對特徵（與 if_dataset.attach_features 同公式）。"""
    spray = balls["spray_deg"].to_numpy(float)
    dtheta = np.abs(np.asarray(angles)[None, :] - spray[:, None])
    nearest = dtheta.argmin(axis=1)
    rows = np.arange(len(balls))
    ad_min = dtheta[rows, nearest]
    near_depth = np.asarray(depths)[nearest]
    rad = np.radians(spray)
    ix, iy = near_depth * np.sin(rad), near_depth * np.cos(rad)
    throw_dist = np.hypot(ix - _FIRST_BASE_XY[0], iy - _FIRST_BASE_XY[1])
    return pd.DataFrame({
        "ad_min": ad_min,
        "ball_time": near_depth / (balls["launch_speed"].to_numpy(float) * MPH_TO_FTS),
        "launch_angle": balls["launch_angle"].to_numpy(float),
        "launch_speed": balls["launch_speed"].to_numpy(float),
        "throw_dist": throw_dist,
        "hp_to_1b": balls["hp_to_1b"].to_numpy(float),
        "stand_R": balls["stand_R"].to_numpy(float),
    })


def predict_p_out(model, balls: pd.DataFrame, angles, depths,
                  player_effects: dict | None = None) -> np.ndarray:
    """逐球 P(out)。player_effects 時把 alpha/g 加在最近野手身上（logit 空間）。"""
    feats = geometry_features(balls, angles, depths)
    p = model.predict_proba(feats)[:, 1]
    if player_effects is not None:
        spray = balls["spray_deg"].to_numpy(float)
        nearest = np.abs(np.asarray(angles)[None, :] - spray[:, None]).argmin(axis=1)
        ad_z = ((feats["ad_min"].to_numpy()
                 - player_effects["ad_mean"]) / player_effects["ad_std"])
        logit = (np.log(p / (1.0 - p))
                 + np.asarray(player_effects["alpha"])[nearest]
                 + np.asarray(player_effects["g"])[nearest] * ad_z)
        p = 1.0 / (1.0 + np.exp(-logit))
    return p


def expected_outs(model, balls: pd.DataFrame, angles, depths,
                  player_effects: dict | None = None) -> float:
    return float(predict_p_out(model, balls, angles, depths, player_effects).mean())


def league_average_positions(years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """聯盟平均站位（PA 加權），依 POSITIONS 順序回傳 (angles, depths)。"""
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(
            "SELECT position, "
            "       sum(avg_norm_start_angle * pa)::float / sum(pa) AS angle, "
            "       sum(avg_norm_start_distance * pa)::float / sum(pa) AS depth "
            "FROM fielder_positioning "
            "WHERE position IN %(pos)s AND season = ANY(%(years)s) "
            "GROUP BY position",
            conn, params={"pos": POSITIONS, "years": list(years)})
    df = df.set_index("position").loc[list(POSITIONS)]
    return df["angle"].to_numpy(), df["depth"].to_numpy()


def fetch_batter_gbs(batter_id: int, years: list[int]) -> pd.DataFrame:
    """打者的歷史滾地球（優化的樣本分布）＋他自己的 hp_to_1b。"""
    events = OUT_EVENTS + NONOUT_EVENTS
    sql = f"""
        SELECT hc_x, hc_y, launch_speed, launch_angle, stand, events
        FROM statcast
        WHERE bb_type = 'ground_ball' AND batter = %(b)s
          AND game_year = ANY(%(years)s)
          AND hc_x IS NOT NULL AND launch_speed IS NOT NULL
          AND events IN {events}
          AND des NOT ILIKE '%%bunt%%'
        ORDER BY hc_x, hc_y, launch_speed
    """
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(sql, conn, params={"b": batter_id, "years": list(years)})
        hp = pd.read_sql(
            "SELECT avg(hp_to_1b) AS hp FROM sprint_speed "
            "WHERE player_id = %(b)s AND season = ANY(%(years)s)",
            conn, params={"b": batter_id, "years": list(years)})["hp"].iloc[0]
        if hp is None or pd.isna(hp):
            hp = pd.read_sql(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY hp_to_1b) AS hp "
                "FROM sprint_speed WHERE season = ANY(%(years)s)",
                conn, params={"years": list(years)})["hp"].iloc[0]

    df["spray_deg"] = np.degrees(np.arctan2(df["hc_x"] - HOME_X, HOME_Y - df["hc_y"]))
    df = df[df["spray_deg"].abs() <= 55].copy()
    df["launch_angle"] = df["launch_angle"].fillna(df["launch_angle"].median())
    df["hp_to_1b"] = float(hp)
    df["stand_R"] = (df["stand"] == "R").astype(int)
    return df.reset_index(drop=True)


class _FastGLMObjective:
    """把優化用 GLM pipeline 展開成純 numpy 的期望出局率評估。

    多起點優化的瓶頸在每次目標函數評估都要重建 DataFrame、重跑整條 sklearn
    pipeline。這裡預先算好不隨站位變動的部分（launch_angle spline、
    launch_speed/hp_to_1b 的 z 分數與其係數貢獻、截距），每次評估
    只重算隨站位變動的 ad_min / ball_time / throw_dist 相關項，logit 直接用
    numpy 組出來。與 model.predict_proba 數值等價（tests/test_if_optimize.py
    驗證到 1e-10）。欄位切段順序必須跟 FielderGeometryFeatures.transform 一致。
    """

    def __init__(self, model, balls: pd.DataFrame,
                 player_effects: dict | None = None):
        if player_effects is not None:
            self._pe_alpha = np.asarray(player_effects["alpha"], dtype=float)
            self._pe_g = np.asarray(player_effects["g"], dtype=float)
            self._pe_admean = float(player_effects["ad_mean"])
            self._pe_adstd = float(player_effects["ad_std"])
        else:
            self._pe_alpha = None
        feats = model.named_steps["features"]
        lr = model.named_steps["lr"]
        # spline 是用 DataFrame fit 的；複製一份去掉 feature 名檢查，讓
        # transform 能直接吃 ndarray（不然每次評估都觸發名稱驗證與警告）
        self._spl_ad = copy.deepcopy(feats.splines_["ad_min"])
        self._spl_bt = copy.deepcopy(feats.splines_["ball_time"])
        for spl in (self._spl_ad, self._spl_bt):
            if hasattr(spl, "feature_names_in_"):
                del spl.feature_names_in_

        spray = balls["spray_deg"].to_numpy(float)
        rad = np.radians(spray)
        self._spray = spray
        self._sin_spray, self._cos_spray = np.sin(rad), np.cos(rad)
        self._speed_fts = balls["launch_speed"].to_numpy(float) * MPH_TO_FTS
        self._rows = np.arange(len(balls))

        mean, scale = feats.scaler_.mean_, feats.scaler_.scale_
        ev_z = (balls["launch_speed"].to_numpy(float) - mean[0]) / scale[0]
        hp_z = (balls["hp_to_1b"].to_numpy(float) - mean[2]) / scale[2]
        self._throw_mean, self._throw_scale = mean[1], scale[1]
        self._ev_z, self._hp_z = ev_z, hp_z

        la = feats.splines_["launch_angle"].transform(balls[["launch_angle"]])
        k_a = self._spl_ad.n_features_out_
        k_b = self._spl_bt.n_features_out_
        k_la = la.shape[1]

        coef = lr.coef_[0]
        pos = 0

        def take(k):
            nonlocal pos
            seg = coef[pos:pos + k]
            pos += k
            return seg

        self._c_a, self._c_b, c_la = take(k_a), take(k_b), take(k_la)
        c_ev, self._c_throw, c_hp = take(3)
        self._C_ab = take(k_a * k_b).reshape(k_a, k_b)
        self._c_aev = take(k_a)
        self._c_ht = take(1)[0]
        self._c_hpb = take(k_b)
        assert pos == len(coef), "係數切段與 FielderGeometryFeatures 欄位順序不符"

        self._const = (la @ c_la + ev_z * c_ev + hp_z * c_hp
                       + lr.intercept_[0])

    def expected_outs(self, angles, depths) -> float:
        dtheta = np.abs(np.asarray(angles)[None, :] - self._spray[:, None])
        nearest = dtheta.argmin(axis=1)
        ad_min = dtheta[self._rows, nearest]
        near_depth = np.asarray(depths)[nearest]
        ball_time = near_depth / self._speed_fts
        ix = near_depth * self._sin_spray
        iy = near_depth * self._cos_spray
        throw_z = (np.hypot(ix - _FIRST_BASE_XY[0], iy - _FIRST_BASE_XY[1])
                   - self._throw_mean) / self._throw_scale
        a = self._spl_ad.transform(ad_min[:, None])
        b = self._spl_bt.transform(ball_time[:, None])
        logit = (self._const + a @ self._c_a + b @ self._c_b
                 + throw_z * self._c_throw
                 + ((a @ self._C_ab) * b).sum(axis=1)
                 + (a @ self._c_aev) * self._ev_z
                 + self._c_ht * self._hp_z * throw_z
                 + (b @ self._c_hpb) * self._hp_z)
        if self._pe_alpha is not None:
            ad_z = (ad_min - self._pe_admean) / self._pe_adstd
            logit = logit + self._pe_alpha[nearest] + self._pe_g[nearest] * ad_z
        return float(np.mean(1.0 / (1.0 + np.exp(-logit))))


def _make_scorer(model, balls: pd.DataFrame, player_effects: dict | None = None):
    """回傳 (angles, depths) → 期望出局率。GLM pipeline 走快速路徑，其他模型退回通用路徑。"""
    if hasattr(model, "named_steps") and {"features", "lr"} <= set(model.named_steps):
        return _FastGLMObjective(model, balls, player_effects).expected_outs
    return lambda angles, depths: expected_outs(model, balls, angles, depths,
                                                player_effects)


def optimize_infield(balls: pd.DataFrame, model, n_restarts: int = 20,
                     seed: int = 42, extra_starts: list[np.ndarray] | None = None,
                     player_effects: dict | None = None) -> dict:
    """LHS 多起點 + L-BFGS-B。回傳最佳站位與期望出局率。"""
    bounds = ANGLE_BOUNDS + FRAC_BOUNDS
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    score = _make_scorer(model, balls, player_effects)

    def neg_exp_outs(x):
        angles, depths = params_to_positions(x)
        return -score(angles, depths)

    starts = []
    if n_restarts > 0:
        sampler = qmc.LatinHypercube(d=8, seed=seed)
        starts = [lo + s * (hi - lo) for s in sampler.random(n_restarts)]
    starts += [np.clip(s, lo, hi) for s in (extra_starts or [])]
    if not starts:
        raise ValueError("n_restarts=0 時必須提供 extra_starts（錨定式優化）")

    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(neg_exp_outs, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-8, "gtol": 1e-6})
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    angles, depths = params_to_positions(best_x)
    if player_effects is None:
        # 同側兩人的標籤在模型裡可互換，正規化成慣例：角落位置（1B/3B）靠邊線。
        # 個人化時槽位綁定特定野手，不可重排。
        right = np.argsort(-angles[:2])          # 角度大者為 1B
        left = 2 + np.argsort(angles[2:])        # 角度最負者為 3B
        order = np.concatenate([right, left])
        angles, depths = angles[order], depths[order]
    return {"angles": angles, "depths": depths, "exp_outs": -best_val,
            "positions": dict(zip(POSITIONS, zip(angles.round(1), depths.round(1))))}
