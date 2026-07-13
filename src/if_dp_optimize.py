"""階段B：一壘有人（<2 出局）的內野站位最佳化。

與 if_optimize 的差異：
- **1B 釘死在 hold-runner 位置**（聯盟一壘有人平均，−26~−35 呎的大位移是壘上
  跑者的規則性行為，不是站位優化的自由變數），只優化 2B/3B/SS（6 維）。
- 出局模型＝兩段式（P(≥1 出局)×P(雙殺|≥1 出局)，src/if_model.py 階段B 模型），
  幾何特徵含 throw_dist_2b 與 pivot_dist（隨候選站位重算）。
- 目標＝期望失分 E[ΔRE]（無「出局率目標」版本——雙殺讓出局數不再是 0/1，
  失分口徑才能正確計價一球換兩個出局）：
      E[ΔRE] = (1−p1)·w + p1(1−p2)·d1 + p1·p2·d2
  w  = 漏接代價（XB 模型計價，同 src/if_runvalue.gb_miss_costs）
  d1 = 單出局 ΔRE = force at 2nd 與 out at 1st 的實證混合（見 FORCE_SHARE）
  d2 = 雙殺 ΔRE（跑者清空、出局+2；o=1 時半局結束）

跑者速度：優化情境的跑者未知，用該年聯盟中位數（runner_hp_to_1b 常數欄）。
打者分布：與現行優化器同一份歷史滾地球（fetch_batter_gbs），打者的滾地輪廓
不隨壘況變，且逐打者的一壘有人子樣本太小。
"""
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from scipy.stats import qmc

from src.config import DSN
from src.if_dataset import MPH_TO_FTS, SECOND_BASE_X, SECOND_BASE_Y
from src.if_optimize import (ANGLE_BOUNDS, FRAC_BOUNDS, MIN_DEPTH,
                             _FIRST_BASE_XY, dirt_max_depth)

DP_POSITIONS = ("2B", "3B", "SS")          # 優化變數；1B 釘死
# 單出局事件中 force at 2nd 的占比（2023–24 一壘有人主範圍：
# force_out 4,254 / (force_out + field_out 3,092 + fielders_choice_out 4)）
FORCE_SHARE = 0.579


def league_average_positions_on1b(years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """一壘有人切分的聯盟平均站位（PA 加權），依 ("1B",)+DP_POSITIONS 順序。"""
    sql = """
        SELECT position,
               sum(avg_norm_start_angle * pa) / sum(pa) AS angle,
               sum(avg_norm_start_distance * pa) / sum(pa) AS depth
        FROM fielder_positioning_on1b
        WHERE season = ANY(%(years)s)
        GROUP BY position
    """
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql(sql, conn, params={"years": list(years)}).set_index("position")
    order = ("1B",) + DP_POSITIONS
    return (np.array([df.loc[p, "angle"] for p in order], dtype=float),
            np.array([df.loc[p, "depth"] for p in order], dtype=float))


def league_median_runner_speed(years: list[int]) -> float:
    with psycopg2.connect(DSN) as conn:
        return float(pd.read_sql(
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY hp_to_1b) AS hp "
            "FROM sprint_speed WHERE season = ANY(%(years)s)",
            conn, params={"years": list(years)})["hp"].iloc[0])


def dp_delta_re(re24: dict, delta_re: dict, outs: int) -> tuple[float, float]:
    """(d1, d2)：一壘有人 outs 出局時，單出局與雙殺的 ΔRE。

    d1 = FORCE_SHARE × [force at 2nd：打者上一壘、跑者出局 → (1,0,0,o+1)]
       + (1−FORCE_SHARE) × [out at 1st：跑者上二壘 → (0,1,0,o+1)]
    d2 = 雙殺：o=0 → (0,0,0,2)；o=1 → 半局結束（−RE）
    """
    s = (1, 0, 0, outs)
    re_s = re24.get(s, 0.0)
    d1 = (FORCE_SHARE * (re24.get((1, 0, 0, outs + 1), 0.0) - re_s)
          + (1 - FORCE_SHARE) * (re24.get((0, 1, 0, outs + 1), 0.0) - re_s))
    d2 = (re24.get((0, 0, 0, outs + 2), 0.0) - re_s) if outs == 0 else -re_s
    return d1, d2


def dp_geometry(balls: pd.DataFrame, angles4, depths4) -> pd.DataFrame:
    """給定四人站位（含釘死的 1B），重算階段B 兩段模型需要的全部特徵。

    順序慣例：angles4/depths4 依 ("1B",)+DP_POSITIONS。與
    if_dataset.attach_features/attach_dp_features 同公式。
    """
    angles4 = np.asarray(angles4, dtype=float)
    depths4 = np.asarray(depths4, dtype=float)
    spray = balls["spray_deg"].to_numpy(float)
    dtheta = np.abs(angles4[None, :] - spray[:, None])
    nearest = dtheta.argmin(axis=1)
    rows = np.arange(len(balls))
    ad_min = dtheta[rows, nearest]
    near_depth = depths4[nearest]
    rad = np.radians(spray)
    ix, iy = near_depth * np.sin(rad), near_depth * np.cos(rad)

    # pivot_dist：2B（index 1）與 SS（index 3）到二壘壘包的較小距離
    px = depths4 * np.sin(np.radians(angles4))
    py = depths4 * np.cos(np.radians(angles4))
    pivot = min(np.hypot(px[1] - SECOND_BASE_X, py[1] - SECOND_BASE_Y),
                np.hypot(px[3] - SECOND_BASE_X, py[3] - SECOND_BASE_Y))

    return pd.DataFrame({
        "ad_min": ad_min,
        "ball_time": near_depth / (balls["launch_speed"].to_numpy(float) * MPH_TO_FTS),
        "launch_angle": balls["launch_angle"].to_numpy(float),
        "launch_speed": balls["launch_speed"].to_numpy(float),
        "throw_dist": np.hypot(ix - _FIRST_BASE_XY[0], iy - _FIRST_BASE_XY[1]),
        "throw_dist_2b": np.hypot(ix - SECOND_BASE_X, iy - SECOND_BASE_Y),
        "hp_to_1b": balls["hp_to_1b"].to_numpy(float),
        "runner_hp_to_1b": balls["runner_hp_to_1b"].to_numpy(float),
        "pivot_dist": np.full(len(balls), pivot),
        "stand_R": balls["stand_R"].to_numpy(float),
    })


class DPScorer:
    """(angles3, depths3) → (E[ΔRE], E[outs])，1B 釘死。

    通用路徑（直接跑兩條 sklearn pipeline）；瓶頸可承受（單次評估 ~ms 級，
    研究批次夠用；若日後上線再做 numpy 快速路徑）。
    """

    def __init__(self, out_model, dp_model, balls: pd.DataFrame,
                 pinned_1b: tuple[float, float], w: np.ndarray,
                 d1: float, d2: float):
        self._out = out_model
        self._dp = dp_model
        self._balls = balls
        self._1b = pinned_1b
        self._w = np.asarray(w, dtype=float)
        self._d1, self._d2 = float(d1), float(d2)

    def _probs(self, angles3, depths3):
        angles4 = np.concatenate([[self._1b[0]], np.asarray(angles3, dtype=float)])
        depths4 = np.concatenate([[self._1b[1]], np.asarray(depths3, dtype=float)])
        feats = dp_geometry(self._balls, angles4, depths4)
        p1 = self._out.predict_proba(feats)[:, 1]
        p2 = self._dp.predict_proba(feats)[:, 1]
        return p1, p2

    def expected_re(self, angles3, depths3) -> float:
        p1, p2 = self._probs(angles3, depths3)
        return float(np.mean((1 - p1) * self._w
                             + p1 * (1 - p2) * self._d1 + p1 * p2 * self._d2))

    def expected_outs(self, angles3, depths3) -> float:
        p1, p2 = self._probs(angles3, depths3)
        return float(np.mean(p1 * (1 + p2)))


def params_to_positions_dp(x):
    """6 維參數（2B/3B/SS 的角度+深度比例）→ (angles3, depths3)。"""
    angles = np.asarray(x[:3], dtype=float)
    fracs = np.asarray(x[3:], dtype=float)
    depths = MIN_DEPTH + fracs * (dirt_max_depth(angles) - MIN_DEPTH)
    return angles, depths


def positions_to_params_dp(angles3, depths3) -> np.ndarray:
    angles3 = np.asarray(angles3, dtype=float)
    depths3 = np.asarray(depths3, dtype=float)
    fracs = (depths3 - MIN_DEPTH) / (dirt_max_depth(angles3) - MIN_DEPTH)
    return np.concatenate([angles3, np.clip(fracs, 0.0, 1.0)])


def optimize_infield_dp(balls: pd.DataFrame, out_model, dp_model,
                        pinned_1b: tuple[float, float], w: np.ndarray,
                        d1: float, d2: float, n_restarts: int = 16,
                        seed: int = 42,
                        extra_starts: list[np.ndarray] | None = None) -> dict:
    """LHS 多起點 + L-BFGS-B，最小化 E[ΔRE]。

    bounds：2B 角度 [1°,44°]（右側，不與釘死的 1B 換邊）、3B/SS [-44°,-1°]，
    深度重參數化同 if_optimize（內野土約束）。
    """
    bounds = [ANGLE_BOUNDS[1], ANGLE_BOUNDS[2], ANGLE_BOUNDS[3]] + FRAC_BOUNDS[:3]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    scorer = DPScorer(out_model, dp_model, balls, pinned_1b, w, d1, d2)

    def obj(x):
        angles, depths = params_to_positions_dp(x)
        return scorer.expected_re(angles, depths)

    starts = []
    if n_restarts > 0:
        sampler = qmc.LatinHypercube(d=6, seed=seed)
        starts = [lo + s * (hi - lo) for s in sampler.random(n_restarts)]
    starts += [np.clip(s, lo, hi) for s in (extra_starts or [])]
    if not starts:
        raise ValueError("n_restarts=0 時必須提供 extra_starts")

    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-8, "gtol": 1e-6})
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    angles, depths = params_to_positions_dp(best_x)
    # 3B/SS 標籤可互換，正規化：角度最負者為 3B（2B 單獨在右側無此問題）
    if angles[1] > angles[2]:
        angles[[1, 2]] = angles[[2, 1]]
        depths[[1, 2]] = depths[[2, 1]]
    return {"angles": angles, "depths": depths, "exp_re": float(best_val),
            "exp_outs": scorer.expected_outs(angles, depths),
            "positions": dict(zip(DP_POSITIONS, zip(angles.round(1), depths.round(1))))}
