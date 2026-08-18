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
import copy
from typing import TypedDict

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.pipeline import Pipeline

from src.config import DSN
from src.if_dataset import MPH_TO_FTS, SECOND_BASE_X, SECOND_BASE_Y
from src.if_optimize import (ANGLE_BOUNDS, FRAC_BOUNDS, MIN_DEPTH,
                             _FIRST_BASE_XY, PlayerEffects, dirt_max_depth)
from src.re24 import BaseOutState, HitDeltaKey


class DPOptimizeResult(TypedDict):
    angles: np.ndarray
    depths: np.ndarray
    exp_re: float
    exp_outs: float
    positions: dict[str, tuple[float, float]]


DP_POSITIONS: tuple[str, ...] = ("2B", "3B", "SS")          # 優化變數；1B 釘死
# 單出局事件中 force at 2nd 的占比（2023–24 一壘有人主範圍：
# force_out 4,254 / (force_out + field_out 3,092 + fielders_choice_out 4)）
FORCE_SHARE: float = 0.579


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


def dp_delta_re(re24: dict[BaseOutState, float], delta_re: dict[HitDeltaKey, float],
                outs: int) -> tuple[float, float]:
    """(single_out_delta_re, double_play_delta_re)：一壘有人 outs 出局時，單出局與雙殺的 ΔRE。

    single_out_delta_re = FORCE_SHARE × [force at 2nd：打者上一壘、跑者出局 → (1,0,0,o+1)]
       + (1−FORCE_SHARE) × [out at 1st：跑者上二壘 → (0,1,0,o+1)]
    double_play_delta_re = 雙殺：o=0 → (0,0,0,2)；o=1 → 半局結束（−RE）
    """
    state = (1, 0, 0, outs)
    re_state = re24.get(state, 0.0)
    single_out_delta_re = (
        FORCE_SHARE * (re24.get((1, 0, 0, outs + 1), 0.0) - re_state)
        + (1 - FORCE_SHARE) * (re24.get((0, 1, 0, outs + 1), 0.0) - re_state)
    )
    double_play_delta_re = (re24.get((0, 0, 0, outs + 2), 0.0) - re_state) if outs == 0 else -re_state
    return single_out_delta_re, double_play_delta_re


def dp_geometry(balls: pd.DataFrame, angles4: np.ndarray, depths4: np.ndarray) -> pd.DataFrame:
    """給定四人站位（含釘死的 1B），重算階段B 兩段模型需要的全部特徵。

    順序慣例：angles4/depths4 依 ("1B",)+DP_POSITIONS。與
    if_dataset.attach_features/attach_dp_features 同公式。
    """
    angles4 = np.asarray(angles4, dtype=float)
    depths4 = np.asarray(depths4, dtype=float)
    spray = balls["spray_deg"].to_numpy(float)
    dtheta = np.abs(angles4[None, :] - spray[:, None])
    nearest = dtheta.argmin(axis=1)
    row_indices = np.arange(len(balls))
    ad_min = dtheta[row_indices, nearest]
    near_depth = depths4[nearest]
    spray_rad = np.radians(spray)
    ix, iy = near_depth * np.sin(spray_rad), near_depth * np.cos(spray_rad)

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

    快速路徑：把兩段 GLM pipeline 展開成純 numpy（同 if_optimize.
    _FastGLMObjective 的模式），預先算好不隨站位變動的部分，每次評估只重算
    幾何相關項。與 predict_proba 數值等價（tests/test_if_dp_optimize.py
    驗證到 1e-10）。係數切段順序必須跟 On1bOutFeatures / On1bDPFeatures
    的 transform 一致。

    player_effects：格式與語意同 if_optimize（{"alpha","g","ad_mean","ad_std"}，
    依 ("1B","2B","3B","SS") 順序，逐球加在最近野手身上：logit += α_j + g_j×ad_z）。
    效應來自無人在壘的貝葉斯球員層（scripts/train_if_bayes.py）——野手轉換力
    是截距性質、不隨壘況變，移植到階段1 的 P(≥1 出局)；階段2（雙殺轉換）
    沒有球員層資料，不掛。
    """

    def __init__(self, out_model: Pipeline, dp_model: Pipeline, balls: pd.DataFrame,
                 pinned_1b: tuple[float, float], miss_cost: np.ndarray,
                 single_out_delta_re: float, double_play_delta_re: float,
                 player_effects: PlayerEffects | None = None) -> None:
        self._pinned_1b: tuple[float, float] = (float(pinned_1b[0]), float(pinned_1b[1]))
        self._miss_cost: np.ndarray = np.asarray(miss_cost, dtype=float)
        self._single_out_delta_re: float = float(single_out_delta_re)
        self._double_play_delta_re: float = float(double_play_delta_re)
        if player_effects is not None:
            self._player_alpha: np.ndarray | None = np.asarray(player_effects["alpha"], dtype=float)
            self._player_g: np.ndarray = np.asarray(player_effects["g"], dtype=float)
            self._player_ad_mean: float = float(player_effects["ad_mean"])
            self._player_ad_std: float = float(player_effects["ad_std"])
        else:
            self._player_alpha = None

        spray = balls["spray_deg"].to_numpy(float)
        spray_rad = np.radians(spray)
        self._spray: np.ndarray = spray
        self._sin_spray, self._cos_spray = np.sin(spray_rad), np.cos(spray_rad)
        self._launch_speed_ft_s: np.ndarray = balls["launch_speed"].to_numpy(float) * MPH_TO_FTS
        self._row_indices: np.ndarray = np.arange(len(balls))

        def strip(spline):
            spline = copy.deepcopy(spline)
            if hasattr(spline, "feature_names_in_"):
                del spline.feature_names_in_
            return spline

        # ── 階段1（On1bOutFeatures = FielderGeometryFeatures + throw_dist_2b）──
        stage1_features = out_model.named_steps["features"]
        stage1_lr = out_model.named_steps["lr"]
        self._stage1_ad_min_spline = strip(stage1_features.splines_["ad_min"])
        self._stage1_ball_time_spline = strip(stage1_features.splines_["ball_time"])
        stage1_scaler_mean, stage1_scaler_scale = stage1_features.scaler_.mean_, stage1_features.scaler_.scale_
        stage1_launch_speed_z = (balls["launch_speed"].to_numpy(float) - stage1_scaler_mean[0]) / stage1_scaler_scale[0]
        stage1_hp_to_1b_z = (balls["hp_to_1b"].to_numpy(float) - stage1_scaler_mean[2]) / stage1_scaler_scale[2]
        self._stage1_throw_dist_norm = (stage1_scaler_mean[1], stage1_scaler_scale[1])
        self._stage1_throw_dist_2b_norm = (float(stage1_features.scaler_2b_.mean_[0]),
                                           float(stage1_features.scaler_2b_.scale_[0]))
        self._stage1_launch_speed_z, self._stage1_hp_to_1b_z = stage1_launch_speed_z, stage1_hp_to_1b_z
        stage1_launch_angle_basis = stage1_features.splines_["launch_angle"].transform(balls[["launch_angle"]])
        n_stage1_ad_basis = self._stage1_ad_min_spline.n_features_out_
        n_stage1_ball_time_basis = self._stage1_ball_time_spline.n_features_out_

        active_coef, coef_offset = stage1_lr.coef_[0], 0

        def take(width: int) -> np.ndarray:
            nonlocal coef_offset
            segment = active_coef[coef_offset:coef_offset + width]
            coef_offset += width
            return segment

        self._stage1_coef_ad_min_spline, self._stage1_coef_ball_time_spline, stage1_coef_launch_angle = (
            take(n_stage1_ad_basis), take(n_stage1_ball_time_basis), take(stage1_launch_angle_basis.shape[1])
        )
        stage1_coef_launch_speed, self._stage1_coef_throw_dist, stage1_coef_hp_to_1b = take(3)
        self._stage1_coef_ad_ball_tensor = take(n_stage1_ad_basis * n_stage1_ball_time_basis).reshape(
            n_stage1_ad_basis, n_stage1_ball_time_basis)
        self._stage1_coef_ad_ev_interaction = take(n_stage1_ad_basis)
        self._stage1_coef_hp_throw_interaction = take(1)[0]
        self._stage1_coef_hp_ball_interaction = take(n_stage1_ball_time_basis)
        self._stage1_coef_throw_dist_2b = take(1)[0]
        assert coef_offset == len(active_coef), "階段1 係數切段與 On1bOutFeatures 欄位順序不符"
        self._stage1_intercept_term = (stage1_launch_angle_basis @ stage1_coef_launch_angle
                                       + stage1_launch_speed_z * stage1_coef_launch_speed
                                       + stage1_hp_to_1b_z * stage1_coef_hp_to_1b
                                       + stage1_lr.intercept_[0])

        # ── 階段2（On1bDPFeatures）──────────────────────────────────
        stage2_features = dp_model.named_steps["features"]
        stage2_lr = dp_model.named_steps["lr"]
        self._stage2_ad_min_spline = strip(stage2_features.splines_["ad_min"])
        self._stage2_ball_time_spline = strip(stage2_features.splines_["ball_time"])
        stage2_scaler_mean, stage2_scaler_scale = stage2_features.scaler_.mean_, stage2_features.scaler_.scale_
        # _LINEAR = [launch_speed, hp_to_1b, runner_hp_to_1b, pivot_dist, throw_dist_2b]
        stage2_launch_speed_z = (balls["launch_speed"].to_numpy(float) - stage2_scaler_mean[0]) / stage2_scaler_scale[0]
        stage2_hp_to_1b_z = (balls["hp_to_1b"].to_numpy(float) - stage2_scaler_mean[1]) / stage2_scaler_scale[1]
        stage2_runner_hp_to_1b_z = (balls["runner_hp_to_1b"].to_numpy(float) - stage2_scaler_mean[2]) / stage2_scaler_scale[2]
        self._stage2_pivot_dist_norm = (stage2_scaler_mean[3], stage2_scaler_scale[3])
        self._stage2_throw_dist_2b_norm = (stage2_scaler_mean[4], stage2_scaler_scale[4])
        self._stage2_hp_to_1b_z = stage2_hp_to_1b_z
        n_stage2_ad_basis = self._stage2_ad_min_spline.n_features_out_
        n_stage2_ball_time_basis = self._stage2_ball_time_spline.n_features_out_

        active_coef, coef_offset = stage2_lr.coef_[0], 0
        self._stage2_coef_ad_min_spline, self._stage2_coef_ball_time_spline = (
            take(n_stage2_ad_basis), take(n_stage2_ball_time_basis)
        )
        (stage2_coef_launch_speed, stage2_coef_hp_to_1b, stage2_coef_runner_hp_to_1b,
         self._stage2_coef_pivot_dist, self._stage2_coef_throw_dist_2b) = take(5)
        self._stage2_has_interactions = stage2_features.interactions
        self._stage2_coef_hp_ball_interaction = take(n_stage2_ball_time_basis) if stage2_features.interactions else None
        assert coef_offset == len(active_coef), "階段2 係數切段與 On1bDPFeatures 欄位順序不符"
        self._stage2_intercept_term = (stage2_launch_speed_z * stage2_coef_launch_speed
                                       + stage2_hp_to_1b_z * stage2_coef_hp_to_1b
                                       + stage2_runner_hp_to_1b_z * stage2_coef_runner_hp_to_1b
                                       + stage2_lr.intercept_[0])

    def _probs(self, angles3: np.ndarray, depths3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        angles4 = np.concatenate([[self._pinned_1b[0]], np.asarray(angles3, dtype=float)])
        depths4 = np.concatenate([[self._pinned_1b[1]], np.asarray(depths3, dtype=float)])
        dtheta = np.abs(angles4[None, :] - self._spray[:, None])
        nearest = dtheta.argmin(axis=1)
        ad_min = dtheta[self._row_indices, nearest]
        near_depth = depths4[nearest]
        ball_time = near_depth / self._launch_speed_ft_s
        ix = near_depth * self._sin_spray
        iy = near_depth * self._cos_spray
        throw_dist = np.hypot(ix - _FIRST_BASE_XY[0], iy - _FIRST_BASE_XY[1])
        throw_dist_2b = np.hypot(ix - SECOND_BASE_X, iy - SECOND_BASE_Y)

        px = depths4 * np.sin(np.radians(angles4))
        py = depths4 * np.cos(np.radians(angles4))
        pivot_dist = min(np.hypot(px[1] - SECOND_BASE_X, py[1] - SECOND_BASE_Y),
                         np.hypot(px[3] - SECOND_BASE_X, py[3] - SECOND_BASE_Y))

        stage1_ad_basis = self._stage1_ad_min_spline.transform(ad_min[:, None])
        stage1_ball_time_basis = self._stage1_ball_time_spline.transform(ball_time[:, None])
        stage1_throw_dist_z = (throw_dist - self._stage1_throw_dist_norm[0]) / self._stage1_throw_dist_norm[1]
        stage1_throw_dist_2b_z = (throw_dist_2b - self._stage1_throw_dist_2b_norm[0]) / self._stage1_throw_dist_2b_norm[1]
        logit1 = (self._stage1_intercept_term
                 + stage1_ad_basis @ self._stage1_coef_ad_min_spline
                 + stage1_ball_time_basis @ self._stage1_coef_ball_time_spline
                 + stage1_throw_dist_z * self._stage1_coef_throw_dist
                 + ((stage1_ad_basis @ self._stage1_coef_ad_ball_tensor) * stage1_ball_time_basis).sum(axis=1)
                 + (stage1_ad_basis @ self._stage1_coef_ad_ev_interaction) * self._stage1_launch_speed_z
                 + self._stage1_coef_hp_throw_interaction * self._stage1_hp_to_1b_z * stage1_throw_dist_z
                 + (stage1_ball_time_basis @ self._stage1_coef_hp_ball_interaction) * self._stage1_hp_to_1b_z
                 + stage1_throw_dist_2b_z * self._stage1_coef_throw_dist_2b)
        if self._player_alpha is not None:
            ad_z = (ad_min - self._player_ad_mean) / self._player_ad_std
            logit1 = (logit1 + self._player_alpha[nearest]
                      + self._player_g[nearest] * ad_z)
        p1 = 1.0 / (1.0 + np.exp(-logit1))

        stage2_ad_basis = self._stage2_ad_min_spline.transform(ad_min[:, None])
        stage2_ball_time_basis = self._stage2_ball_time_spline.transform(ball_time[:, None])
        stage2_pivot_dist_z = (pivot_dist - self._stage2_pivot_dist_norm[0]) / self._stage2_pivot_dist_norm[1]
        stage2_throw_dist_2b_z = (throw_dist_2b - self._stage2_throw_dist_2b_norm[0]) / self._stage2_throw_dist_2b_norm[1]
        logit2 = (self._stage2_intercept_term
                 + stage2_ad_basis @ self._stage2_coef_ad_min_spline
                 + stage2_ball_time_basis @ self._stage2_coef_ball_time_spline
                 + stage2_pivot_dist_z * self._stage2_coef_pivot_dist
                 + stage2_throw_dist_2b_z * self._stage2_coef_throw_dist_2b)
        if self._stage2_has_interactions:
            logit2 = logit2 + (stage2_ball_time_basis @ self._stage2_coef_hp_ball_interaction) * self._stage2_hp_to_1b_z
        p2 = 1.0 / (1.0 + np.exp(-logit2))
        return p1, p2

    def expected_re(self, angles3: np.ndarray, depths3: np.ndarray) -> float:
        p1, p2 = self._probs(angles3, depths3)
        return float(np.mean((1 - p1) * self._miss_cost
                             + p1 * (1 - p2) * self._single_out_delta_re
                             + p1 * p2 * self._double_play_delta_re))

    def expected_outs(self, angles3: np.ndarray, depths3: np.ndarray) -> float:
        p1, p2 = self._probs(angles3, depths3)
        return float(np.mean(p1 * (1 + p2)))

    def expected_p1(self, angles3: np.ndarray, depths3: np.ndarray) -> float:
        """P(≥1 出局) 平均——「懂釘死 1B 但不懂雙殺」的分解基準用。"""
        p1, _ = self._probs(angles3, depths3)
        return float(np.mean(p1))

    def per_ball_p1(self, angles3: np.ndarray, depths3: np.ndarray) -> np.ndarray:
        """逐球 P(≥1 出局)——web 端點的逐球上色用（UI 不顯示雙殺機率）。"""
        p1, _ = self._probs(angles3, depths3)
        return p1


def params_to_positions_dp(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6 維參數（2B/3B/SS 的角度+深度比例）→ (angles3, depths3)。"""
    angles = np.asarray(x[:3], dtype=float)
    fracs = np.asarray(x[3:], dtype=float)
    depths = MIN_DEPTH + fracs * (dirt_max_depth(angles) - MIN_DEPTH)
    return angles, depths


def positions_to_params_dp(angles3: np.ndarray, depths3: np.ndarray) -> np.ndarray:
    angles3 = np.asarray(angles3, dtype=float)
    depths3 = np.asarray(depths3, dtype=float)
    fracs = (depths3 - MIN_DEPTH) / (dirt_max_depth(angles3) - MIN_DEPTH)
    return np.concatenate([angles3, np.clip(fracs, 0.0, 1.0)])


def anchored_starts(anchor: np.ndarray, n_jitter: int = 8,
                    seed: int = 42) -> list[np.ndarray]:
    """錨定精修的起點：錨點本身＋周圍小抖動。

    零效應最佳解常同時壓在多個 bound 與最近野手 argmin 交界的 kink 上，
    L-BFGS-B 的數值梯度在該點會 line search 失敗（ABNORMAL）原地不動，
    連旁邊的小改善都撿不到；從錨點附近起步就正常。抖動幅度小
    （角度 ±0.75°、深度比例 ±0.05）維持錨定語意——位移只反映球員效應
    的拉力，不做全域重找。
    """
    bounds = [ANGLE_BOUNDS[1], ANGLE_BOUNDS[2], ANGLE_BOUNDS[3]] + FRAC_BOUNDS[:3]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(seed)
    scale = np.array([0.75, 0.75, 0.75, 0.05, 0.05, 0.05])
    anchor = np.asarray(anchor, dtype=float)
    return [anchor] + [np.clip(anchor + rng.uniform(-1, 1, 6) * scale, lo, hi)
                       for _ in range(n_jitter)]


def optimize_infield_dp(balls: pd.DataFrame, out_model: Pipeline, dp_model: Pipeline,
                        pinned_1b: tuple[float, float], miss_cost: np.ndarray,
                        single_out_delta_re: float, double_play_delta_re: float,
                        n_restarts: int = 16,
                        seed: int = 42,
                        extra_starts: list[np.ndarray] | None = None,
                        objective: str = "re",
                        player_effects: PlayerEffects | None = None) -> DPOptimizeResult:
    """LHS 多起點 + L-BFGS-B，最小化 E[ΔRE]（objective="re"，預設）。

    objective="p1"：改最大化 P(≥1 出局)——「懂釘死 1B 的幾何但不懂雙殺計價」，
    只作為分解實驗的中間基準（隔離補洞效果 vs 雙殺感知，勿用於生產）。
    bounds：2B 角度 [1°,44°]（右側，不與釘死的 1B 換邊）、3B/SS [-44°,-1°]，
    深度重參數化同 if_optimize（內野土約束）。
    player_effects：見 DPScorer docstring；個人化時 3B/SS 槽位綁定特定野手，
    不做標籤正規化（同 if_optimize 慣例）。
    """
    bounds = [ANGLE_BOUNDS[1], ANGLE_BOUNDS[2], ANGLE_BOUNDS[3]] + FRAC_BOUNDS[:3]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    scorer = DPScorer(out_model, dp_model, balls, pinned_1b, miss_cost,
                      single_out_delta_re, double_play_delta_re, player_effects)

    if objective == "re":
        def obj(x: np.ndarray) -> float:
            angles, depths = params_to_positions_dp(x)
            return scorer.expected_re(angles, depths)
    elif objective == "p1":
        def obj(x: np.ndarray) -> float:
            angles, depths = params_to_positions_dp(x)
            return -scorer.expected_p1(angles, depths)
    else:
        raise ValueError(f"unknown objective: {objective}")

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
    # 3B/SS 標籤可互換，正規化：角度最負者為 3B（2B 單獨在右側無此問題）。
    # 個人化時槽位綁定特定野手，不可重排。
    if player_effects is None and angles[1] > angles[2]:
        angles[[1, 2]] = angles[[2, 1]]
        depths[[1, 2]] = depths[[2, 1]]
    return {"angles": angles, "depths": depths, "exp_re": float(best_val),
            "exp_outs": scorer.expected_outs(angles, depths),
            "positions": dict(zip(DP_POSITIONS, zip(angles.round(1), depths.round(1))))}
