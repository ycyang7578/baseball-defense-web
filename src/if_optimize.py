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

已知限制：GLM 是在實際（賽季平均）站位附近的資料上訓練的，離常態很遠的
候選站位屬外插，解讀時要保守。
"""
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


def expected_outs(model, balls: pd.DataFrame, angles, depths) -> float:
    feats = geometry_features(balls, angles, depths)
    return float(model.predict_proba(feats)[:, 1].mean())


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
        SELECT hc_x, hc_y, launch_speed, launch_angle, stand
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


def optimize_infield(balls: pd.DataFrame, model, n_restarts: int = 20,
                     seed: int = 42, extra_starts: list[np.ndarray] | None = None) -> dict:
    """LHS 多起點 + L-BFGS-B。回傳最佳站位與期望出局率。"""
    bounds = ANGLE_BOUNDS + FRAC_BOUNDS
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def neg_exp_outs(x):
        angles, depths = params_to_positions(x)
        return -expected_outs(model, balls, angles, depths)

    sampler = qmc.LatinHypercube(d=8, seed=seed)
    starts = [lo + s * (hi - lo) for s in sampler.random(n_restarts)]
    starts += [np.clip(s, lo, hi) for s in (extra_starts or [])]

    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(neg_exp_outs, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-8, "gtol": 1e-6})
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    angles, depths = params_to_positions(best_x)
    # 同側兩人的標籤在模型裡可互換，正規化成慣例：角落位置（1B/3B）靠邊線
    right = np.argsort(-angles[:2])          # 角度大者為 1B
    left = 2 + np.argsort(angles[2:])        # 角度最負者為 3B
    order = np.concatenate([right, left])
    angles, depths = angles[order], depths[order]
    return {"angles": angles, "depths": depths, "exp_outs": -best_val,
            "positions": dict(zip(POSITIONS, zip(angles.round(1), depths.round(1))))}
