"""Infield positioning optimization (constrained optimization under the 2023 shift ban).

How rule constraints are handled (all folded into box bounds so L-BFGS-B can consume them directly):
- At least two infielders on each side of second base, no side-swapping -> 1B/2B angle limited to [1°, 44°],
  3B/SS limited to [-44°, -1°]
- Must stand on the infield dirt -> depth is reparameterized as "fraction f in [0,1] from 60 ft to the dirt's
  outer edge", where the dirt's outer edge is approximated as an arc centered on the pitcher's mound
  (60.5 ft from home plate) with radius 95 ft:
  r_max(θ) = 60.5·cosθ + sqrt(95² − (60.5·sinθ)²)
  (the dirt actually extends farther toward the corners, so this approximation is conservative on the
  foul-line side)

Objective: maximize mean P(out) over the batter's historical ground-ball sample, where P(out) comes from the
optimization GLM (models/if_gb/if_gb_optimizer_glm.joblib; contains only fielder-relative geometry, so it
supports counterfactuals). The batter distribution is taken directly from historical balls (spray angle isn't
contaminated by positioning, see the src/if_dataset.py docstring), so there's no need to rebuild the KDE.

Player personalization (Bayesian player layer, see scripts/train_if_bayes.py): `player_effects` gives the
posterior mean (alpha_j, g_j) for the four positional fielders, added per-ball to the nearest fielder:
logit += alpha_j + g_j × ad_z. With personalization, same-side labels are no longer interchangeable (each
slot is tied to a specific fielder), so no corner normalization is applied. Format:
{"alpha": ndarray(4), "g": ndarray(4), "ad_mean": float, "ad_std": float}
(in POSITIONS order; league-average fielder = alpha=g=0).

Known limitation: the GLM is trained on data near actual (season-average) positioning, so candidate positions
far from the norm are extrapolation -- interpret them conservatively.
"""
import copy
from typing import Callable, Protocol, TypedDict

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.pipeline import Pipeline

from src.config import DSN
from src.if_dataset import MPH_TO_FTS, OUT_EVENTS, NONOUT_EVENTS, HOME_X, HOME_Y


class ProbabilisticClassifier(Protocol):
    """Minimal model interface required by the `optimize_infield` family of functions (a GLM pipeline or any other classifier works)."""
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class PlayerEffects(TypedDict):
    """Bayesian player-layer posterior means, in POSITIONS order (see the docstring at the top of this file)."""
    alpha: np.ndarray
    g: np.ndarray
    ad_mean: float
    ad_std: float


class InfieldOptimizeResult(TypedDict):
    angles: np.ndarray
    depths: np.ndarray
    exp_outs: float
    positions: dict[str, tuple[float, float]]


MOUND_DIST: float = 60.5
DIRT_RADIUS: float = 95.0
# The depth lower bound is set at the edge of the training data's support (season-average positioning is
# roughly 100-155 ft): shallower than that is GLM extrapolation, and empirically it lets the optimizer park
# the "idle fielder" (on the side of the batter's cold zone) at an artificially optimistic extrapolated position
MIN_DEPTH: float = 75.0
POSITIONS: tuple[str, ...] = ("1B", "2B", "3B", "SS")
# box bounds: the first 4 dims are angles (degrees), the last 4 are depth fraction f
ANGLE_BOUNDS: list[tuple[float, float]] = [(1.0, 44.0), (1.0, 44.0), (-44.0, -1.0), (-44.0, -1.0)]
FRAC_BOUNDS: list[tuple[float, float]] = [(0.0, 1.0)] * 4
_FIRST_BASE_XY: tuple[float, float] = (90.0 * np.sin(np.radians(45.0)), 90.0 * np.cos(np.radians(45.0)))


def dirt_max_depth(angle_deg: np.ndarray | float) -> np.ndarray | float:
    """Depth (ft) of the infield dirt's outer edge at a given angle."""
    angle_rad = np.radians(angle_deg)
    return MOUND_DIST * np.cos(angle_rad) + np.sqrt(
        DIRT_RADIUS ** 2 - (MOUND_DIST * np.sin(angle_rad)) ** 2)


def params_to_positions(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8-dim parameter vector -> (angles[4], depths[4]), in POSITIONS order."""
    angles = np.asarray(x[:4], dtype=float)
    fracs = np.asarray(x[4:], dtype=float)
    depths = MIN_DEPTH + fracs * (dirt_max_depth(angles) - MIN_DEPTH)
    return angles, depths


def positions_to_params(angles: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """(angles, depths) -> 8-dim parameter vector (inverse of params_to_positions)."""
    angles = np.asarray(angles, dtype=float)
    depths = np.asarray(depths, dtype=float)
    fracs = (depths - MIN_DEPTH) / (dirt_max_depth(angles) - MIN_DEPTH)
    return np.concatenate([angles, np.clip(fracs, 0.0, 1.0)])


def geometry_features(balls: pd.DataFrame, angles: np.ndarray, depths: np.ndarray) -> pd.DataFrame:
    """Given positioning, recompute fielder-relative features for each ball (same formulas as if_dataset.attach_features)."""
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


def predict_p_out(model: ProbabilisticClassifier, balls: pd.DataFrame,
                  angles: np.ndarray, depths: np.ndarray,
                  player_effects: PlayerEffects | None = None) -> np.ndarray:
    """Per-ball P(out). When player_effects is given, alpha/g are added to the nearest fielder (in logit space)."""
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


def expected_outs(model: ProbabilisticClassifier, balls: pd.DataFrame,
                  angles: np.ndarray, depths: np.ndarray,
                  player_effects: PlayerEffects | None = None,
                  ball_weights: np.ndarray | None = None) -> float:
    """ball_weights=None -> expected out rate (current default); with weights -> mean(p×w).

    The run-value objective (Phase A) uses the weighted version: w_j = miss_cost_j − ΔRE(out) (both > 0),
    E[ΔRE] = mean(miss_cost) − mean(p×w), so maximizing mean(p×w) is equivalent to minimizing expected run cost.
    See src/if_runvalue.py."""
    p = predict_p_out(model, balls, angles, depths, player_effects)
    if ball_weights is None:
        return float(p.mean())
    return float((p * np.asarray(ball_weights, dtype=float)).mean())


def league_average_positions(years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """League-average positioning (PA-weighted), returns (angles, depths) in POSITIONS order."""
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
    """A batter's historical ground balls (the sample distribution used for optimization) plus their own hp_to_1b."""
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
    """Unrolls the optimization GLM pipeline into a pure-numpy expected-out-rate evaluator.

    The bottleneck in multi-start optimization is that every objective function evaluation has to rebuild a
    DataFrame and rerun the entire sklearn pipeline. Here we precompute the parts that don't change with
    positioning (the launch_angle spline, the z-scores and coefficient contributions for launch_speed/hp_to_1b,
    and the intercept); each evaluation then only recomputes the positioning-dependent ad_min / ball_time /
    throw_dist terms, and the logit is assembled directly with numpy. This is numerically equivalent to
    model.predict_proba (verified to 1e-10 in tests/test_if_optimize.py). The coefficient slicing order must
    match the column order in FielderGeometryFeatures.transform.
    """

    def __init__(self, model: Pipeline, balls: pd.DataFrame,
                 player_effects: PlayerEffects | None = None,
                 ball_weights: np.ndarray | None = None) -> None:
        self._ball_weights: np.ndarray | None = (
            np.asarray(ball_weights, dtype=float) if ball_weights is not None else None
        )
        if player_effects is not None:
            self._player_alpha: np.ndarray | None = np.asarray(player_effects["alpha"], dtype=float)
            self._player_g: np.ndarray = np.asarray(player_effects["g"], dtype=float)
            self._player_ad_mean: float = float(player_effects["ad_mean"])
            self._player_ad_std: float = float(player_effects["ad_std"])
        else:
            self._player_alpha = None
        feature_transformer = model.named_steps["features"]
        logistic_regression = model.named_steps["lr"]
        # The spline was fit on a DataFrame; make a copy with the feature-name check removed so
        # transform can take an ndarray directly (otherwise every evaluation triggers name validation and a warning)
        self._ad_min_spline = copy.deepcopy(feature_transformer.splines_["ad_min"])
        self._ball_time_spline = copy.deepcopy(feature_transformer.splines_["ball_time"])
        for spline in (self._ad_min_spline, self._ball_time_spline):
            if hasattr(spline, "feature_names_in_"):
                del spline.feature_names_in_

        spray = balls["spray_deg"].to_numpy(float)
        spray_rad = np.radians(spray)
        self._spray: np.ndarray = spray
        self._sin_spray, self._cos_spray = np.sin(spray_rad), np.cos(spray_rad)
        self._launch_speed_ft_s: np.ndarray = balls["launch_speed"].to_numpy(float) * MPH_TO_FTS
        self._row_indices: np.ndarray = np.arange(len(balls))

        scaler_mean = feature_transformer.scaler_.mean_
        scaler_scale = feature_transformer.scaler_.scale_
        launch_speed_z = (balls["launch_speed"].to_numpy(float) - scaler_mean[0]) / scaler_scale[0]
        hp_to_1b_z = (balls["hp_to_1b"].to_numpy(float) - scaler_mean[2]) / scaler_scale[2]
        self._throw_dist_mean, self._throw_dist_scale = scaler_mean[1], scaler_scale[1]
        self._launch_speed_z, self._hp_to_1b_z = launch_speed_z, hp_to_1b_z

        launch_angle_basis = feature_transformer.splines_["launch_angle"].transform(balls[["launch_angle"]])
        n_ad_basis = self._ad_min_spline.n_features_out_
        n_ball_time_basis = self._ball_time_spline.n_features_out_
        n_launch_angle_basis = launch_angle_basis.shape[1]

        lr_coef = logistic_regression.coef_[0]
        coef_offset = 0

        def take(width: int) -> np.ndarray:
            nonlocal coef_offset
            segment = lr_coef[coef_offset:coef_offset + width]
            coef_offset += width
            return segment

        self._coef_ad_min_spline, self._coef_ball_time_spline, coef_launch_angle = (
            take(n_ad_basis), take(n_ball_time_basis), take(n_launch_angle_basis)
        )
        coef_launch_speed, self._coef_throw_dist, coef_hp_to_1b = take(3)
        self._coef_ad_ball_tensor = take(n_ad_basis * n_ball_time_basis).reshape(n_ad_basis, n_ball_time_basis)
        self._coef_ad_ev_interaction = take(n_ad_basis)
        self._coef_hp_throw_interaction = take(1)[0]
        self._coef_hp_ball_interaction = take(n_ball_time_basis)
        assert coef_offset == len(lr_coef), "係數切段與 FielderGeometryFeatures 欄位順序不符"

        self._intercept_term = (launch_angle_basis @ coef_launch_angle
                                + launch_speed_z * coef_launch_speed
                                + hp_to_1b_z * coef_hp_to_1b
                                + logistic_regression.intercept_[0])

    def expected_outs(self, angles: np.ndarray, depths: np.ndarray) -> float:
        dtheta = np.abs(np.asarray(angles)[None, :] - self._spray[:, None])
        nearest = dtheta.argmin(axis=1)
        ad_min = dtheta[self._row_indices, nearest]
        near_depth = np.asarray(depths)[nearest]
        ball_time = near_depth / self._launch_speed_ft_s
        ix = near_depth * self._sin_spray
        iy = near_depth * self._cos_spray
        throw_z = (np.hypot(ix - _FIRST_BASE_XY[0], iy - _FIRST_BASE_XY[1])
                   - self._throw_dist_mean) / self._throw_dist_scale
        ad_basis = self._ad_min_spline.transform(ad_min[:, None])
        ball_time_basis = self._ball_time_spline.transform(ball_time[:, None])
        logit = (self._intercept_term
                 + ad_basis @ self._coef_ad_min_spline
                 + ball_time_basis @ self._coef_ball_time_spline
                 + throw_z * self._coef_throw_dist
                 + ((ad_basis @ self._coef_ad_ball_tensor) * ball_time_basis).sum(axis=1)
                 + (ad_basis @ self._coef_ad_ev_interaction) * self._launch_speed_z
                 + self._coef_hp_throw_interaction * self._hp_to_1b_z * throw_z
                 + (ball_time_basis @ self._coef_hp_ball_interaction) * self._hp_to_1b_z)
        if self._player_alpha is not None:
            ad_z = (ad_min - self._player_ad_mean) / self._player_ad_std
            logit = logit + self._player_alpha[nearest] + self._player_g[nearest] * ad_z
        p = 1.0 / (1.0 + np.exp(-logit))
        if self._ball_weights is not None:
            return float(np.mean(p * self._ball_weights))
        return float(np.mean(p))


def _make_scorer(model: ProbabilisticClassifier, balls: pd.DataFrame,
                 player_effects: PlayerEffects | None = None,
                 ball_weights: np.ndarray | None = None) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns (angles, depths) -> expected out rate (or weighted score, see the expected_outs docstring).
    GLM pipelines take the fast path; other models fall back to the generic path."""
    if hasattr(model, "named_steps") and {"features", "lr"} <= set(model.named_steps):
        return _FastGLMObjective(model, balls, player_effects,
                                 ball_weights).expected_outs
    return lambda angles, depths: expected_outs(model, balls, angles, depths,
                                                player_effects, ball_weights)


def optimize_infield(balls: pd.DataFrame, model: ProbabilisticClassifier, n_restarts: int = 20,
                     seed: int = 42, extra_starts: list[np.ndarray] | None = None,
                     player_effects: PlayerEffects | None = None,
                     ball_weights: np.ndarray | None = None) -> InfieldOptimizeResult:
    """LHS multi-start + L-BFGS-B. Returns the best positioning and expected out rate.

    When ball_weights is given, the objective becomes mean(p×w) (the run-value objective; see src/if_runvalue.py
    for how weights convert to run cost), and the exp_outs field of the returned dict is that weighted score."""
    bounds = ANGLE_BOUNDS + FRAC_BOUNDS
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    score = _make_scorer(model, balls, player_effects, ball_weights)

    def neg_exp_outs(x: np.ndarray) -> float:
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
        # The two labels on the same side are interchangeable in the model; normalize to the convention that
        # the corner position (1B/3B) is the one closer to the foul line.
        # With personalization, slots are bound to specific fielders and can't be reordered.
        right = np.argsort(-angles[:2])          # larger angle = 1B
        left = 2 + np.argsort(angles[2:])        # most negative angle = 3B
        order = np.concatenate([right, left])
        angles, depths = angles[order], depths[order]
    return {"angles": angles, "depths": depths, "exp_outs": -best_val,
            "positions": dict(zip(POSITIONS, zip(angles.round(1), depths.round(1))))}
