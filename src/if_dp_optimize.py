"""Stage B: infield positioning optimization with a runner on first (<2 outs).

Differences from if_optimize:
- **1B is pinned to the hold-runner position** (the league-average -26 to -35 ft shift with
  a runner on first is a rule-driven reaction to the baserunner, not a free variable for
  positioning optimization); only 2B/3B/SS are optimized (6 dimensions).
- The out model is two-stage (P(>=1 out) x P(double play | >=1 out), Stage B model in
  src/if_model.py), with geometric features including throw_dist_2b and pivot_dist
  (recomputed for each candidate positioning).
- Objective = expected run value E[deltaRE] (there is no "out-rate target" version --
  double plays make the out count no longer 0/1, so the run-value framing is needed to
  correctly price trading one ball for two outs):
      E[deltaRE] = (1-p1)*w + p1*(1-p2)*d1 + p1*p2*d2
  w  = miss cost (priced via the XB model, same as src/if_runvalue.gb_miss_costs)
  d1 = single-out deltaRE = empirical mix of force-at-2nd and out-at-1st (see FORCE_SHARE)
  d2 = double-play deltaRE (runner erased, outs+2; if o=1 the half-inning ends)

Runner speed: the runner in an optimization scenario is unknown, so we use that year's
league median (the runner_hp_to_1b constant column).
Batter distribution: uses the same historical ground balls as the existing optimizer
(fetch_batter_gbs) -- a batter's ground-ball profile doesn't change with the base state,
and the per-batter runner-on-first subsample is too small anyway.
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


DP_POSITIONS: tuple[str, ...] = ("2B", "3B", "SS")          # optimization variables; 1B is pinned
# Share of single-out events that are a force at 2nd (2023-24 runner-on-first main scope:
# force_out 4,254 / (force_out + field_out 3,092 + fielders_choice_out 4))
FORCE_SHARE: float = 0.579


def league_average_positions_on1b(years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """League-average positioning split by runner-on-first (PA-weighted), ordered as ("1B",)+DP_POSITIONS."""
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
    """(single_out_delta_re, double_play_delta_re): deltaRE for a single out vs. a double play, with a runner on first and `outs` outs already recorded.

    single_out_delta_re = FORCE_SHARE x [force at 2nd: batter reaches first, runner is out -> (1,0,0,o+1)]
       + (1-FORCE_SHARE) x [out at 1st: runner advances to second -> (0,1,0,o+1)]
    double_play_delta_re = double play: o=0 -> (0,0,0,2); o=1 -> half-inning ends (-RE)
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
    """Given the four fielders' positioning (including the pinned 1B), recompute every feature needed by the Stage B two-stage model.

    Ordering convention: angles4/depths4 follow ("1B",)+DP_POSITIONS. Uses the same
    formulas as if_dataset.attach_features/attach_dp_features.
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

    # pivot_dist: the smaller of 2B's (index 1) and SS's (index 3) distances to the second base bag
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
    """(angles3, depths3) -> (E[deltaRE], E[outs]), with 1B pinned.

    Fast path: unrolls the two-stage GLM pipeline into pure numpy (same pattern as
    if_optimize._FastGLMObjective), precomputing the parts that don't depend on
    positioning so each evaluation only recomputes the geometry-related terms.
    Numerically equivalent to predict_proba (verified to 1e-10 in
    tests/test_if_dp_optimize.py). The coefficient-slicing order must match the
    transform order of On1bOutFeatures / On1bDPFeatures.

    player_effects: same format and semantics as if_optimize ({"alpha","g","ad_mean","ad_std"},
    ordered as ("1B","2B","3B","SS"), added per ball to the nearest fielder:
    logit += alpha_j + g_j*ad_z). The effects come from the bases-empty Bayesian player
    layer (scripts/train_if_bayes.py) -- a fielder's conversion skill is an intercept-like
    property that doesn't change with the base state, so it's carried over to Stage 1's
    P(>=1 out); Stage 2 (double-play conversion) has no player-level data, so it isn't
    attached there.
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

        # -- Stage 1 (On1bOutFeatures = FielderGeometryFeatures + throw_dist_2b) --
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
        assert coef_offset == len(active_coef), "Stage 1 coefficient slicing does not match On1bOutFeatures column order"
        self._stage1_intercept_term = (stage1_launch_angle_basis @ stage1_coef_launch_angle
                                       + stage1_launch_speed_z * stage1_coef_launch_speed
                                       + stage1_hp_to_1b_z * stage1_coef_hp_to_1b
                                       + stage1_lr.intercept_[0])

        # -- Stage 2 (On1bDPFeatures) ------------------------------------
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
        assert coef_offset == len(active_coef), "Stage 2 coefficient slicing does not match On1bDPFeatures column order"
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
        """Average P(>=1 out) -- used as the decomposition baseline that "knows about pinning 1B but not about double plays"."""
        p1, _ = self._probs(angles3, depths3)
        return float(np.mean(p1))

    def per_ball_p1(self, angles3: np.ndarray, depths3: np.ndarray) -> np.ndarray:
        """Per-ball P(>=1 out) -- used by the web endpoint to color each ball (the UI does not display double-play probability)."""
        p1, _ = self._probs(angles3, depths3)
        return p1


def params_to_positions_dp(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6-dimensional parameters (angle + depth fraction for 2B/3B/SS) -> (angles3, depths3)."""
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
    """Starting points for anchored refinement: the anchor itself plus small surrounding jitter.

    The zero-effect optimum often sits pinned on a kink where multiple bounds and the
    nearest-fielder argmin boundary meet; at that point L-BFGS-B's numerical gradient
    fails the line search (ABNORMAL) and gets stuck, missing even small nearby
    improvements -- starting near the anchor avoids this. The jitter is kept small
    (angle +/-0.75 degrees, depth fraction +/-0.05) to preserve the anchoring semantics --
    the displacement should only reflect the pull of the player effects, not a global
    re-search.
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
    """LHS multi-start + L-BFGS-B, minimizing E[deltaRE] (objective="re", the default).

    objective="p1": instead maximizes P(>=1 out) -- "knows the geometry of pinning 1B but
    not double-play pricing" -- used only as an intermediate baseline for decomposition
    experiments (isolating the gap-coverage effect vs. double-play awareness; do not use
    in production).
    bounds: 2B angle [1,44] degrees (right side, never swaps sides with the pinned 1B),
    3B/SS [-44,-1] degrees; depth is reparameterized the same way as if_optimize (the
    infield dirt constraint).
    player_effects: see the DPScorer docstring; when personalizing, the 3B/SS slots are
    bound to specific fielders, so no label normalization is applied (same convention as
    if_optimize).
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
        raise ValueError("extra_starts must be provided when n_restarts=0")

    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-8, "gtol": 1e-6})
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    angles, depths = params_to_positions_dp(best_x)
    # The 3B/SS labels are interchangeable, so normalize: whichever has the more negative
    # angle is labeled 3B (2B is alone on the right side, so it has no such issue).
    # When personalizing, slots are bound to specific fielders and must not be reordered.
    if player_effects is None and angles[1] > angles[2]:
        angles[[1, 2]] = angles[[2, 1]]
        depths[[1, 2]] = depths[[2, 1]]
    return {"angles": angles, "depths": depths, "exp_re": float(best_val),
            "exp_outs": scorer.expected_outs(angles, depths),
            "positions": dict(zip(DP_POSITIONS, zip(angles.round(1), depths.round(1))))}
