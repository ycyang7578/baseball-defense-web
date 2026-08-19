"""Two model definitions for infield ground ball out probability (different roles -- do not mix them up).

1. make_optimizer_glm() -- for positioning optimization (counterfactual).
   Uses only "fielder-relative geometry" features (ad_min/ball_time/throw_dist...), and
   deliberately **excludes** the raw spray_deg term: spray's position-specific out-rate
   pattern reflects "where the league currently stands" (endogeneity) -- those patterns
   won't hold after a fielder is moved; including it would make the optimizer double-count
   (e.g. after moving the SS into the 5.5 hole, the spray term would still predict a low
   out rate there).
2. make_difficulty_glm() -- for player evaluation (Stage 2's p_hat).
   A league-average difficulty model (xBA-style) under fixed positioning; spray's
   positional pattern is valid and should be used here -- evaluation doesn't move
   fielders and has no counterfactual requirement, so the endogeneity ban (raw
   spray / flexible shapes) doesn't apply.
   **As of 2026-07-12, an interpretable spline GLM replaces the GBM** (a user decision:
   don't use a model that can't be explained, accept the accuracy cost). Measured cost
   (scripts/experiments/exp_if_difficulty_glm.py, full ground ball set 2023-24 -> 2025):
   per-ball AUC 0.770 vs GBM 0.824, but evaluation aggregates hundreds of balls, so
   qualified R only drops 0.525 -> 0.514, and the scale stays similarly healthy
   (SD 8.3 vs. official 6.9). Feature engineering: spray mirrored for left-handed
   batters (same as Melville) + spray/LA/EV/hp splines + spray x EV, spray x hp
   interactions (spray x EV is necessary for calibration: without the interaction,
   cal 0.043 -> 0.026).
3. make_difficulty_gbm() -- GBM benchmark (kept for comparison, not used in production).
   2023 -> 2024 validation: AUC 0.807, using only ball quality + runner + spray; adding
   fielder features gives no gain (+0.001) -- evidence that under Standard alignment,
   season-average positioning is approximately a deterministic function of spray.

Structure selection process (2026-07-06): train on 2023 -> validate on 2024 to pick the
interaction configuration, touching 2025 only once for the final evaluation. GLM
interaction configuration validation result: base 0.7460 -> +tensor(ad x bt)+ad x EV+
runner interaction 0.7536.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

# Input columns for the optimization GLM (fielder-relative geometry + ball quality + runner)
# stand_R was removed on 2026-07-12: hp_to_1b is the measured home-to-first time, which
# already captures the left-handed batter's positioning advantage; an ablation showed
# stand_R is redundant (2025 AUC -0.0006, calibration slightly better,
# scripts/experiments/exp_if_drop_features.py)
OPTIMIZER_FEATURES: list[str] = ["ad_min", "ball_time", "launch_angle", "launch_speed",
                                  "throw_dist", "hp_to_1b"]
# Input columns for the evaluation GBM (no fielder information at all)
DIFFICULTY_FEATURES: list[str] = ["spray_deg", "launch_angle", "launch_speed",
                                   "hp_to_1b", "stand_R"]


class FielderGeometryFeatures(BaseEstimator, TransformerMixin):
    """Design matrix for the optimization GLM.

    Spline basis: ad_min, ball_time, launch_angle (nonlinear effects).
    Linear terms: launch_speed, throw_dist, hp_to_1b (the linear structure of the timing-race gap).
    throw_dist/hp_to_1b are **deliberately kept linear**: throw_dist is a deterministic
    function of interception geometry (spray x depth), and giving it a flexible shape
    would smuggle back in the raw-spray positional signal that was supposed to be excluded
    (2026-07-12 experiment: a throw_dist spline alone gives +0.012 AUC, but the partial
    effect shows a wave shaped like positional identity, with a trough at 105 ft
    corresponding to the 2B/SS boundary zone -- the optimizer would chase this fake bump;
    see scripts/experiments/exp_if_drop_features.py).
    Interactions (chosen via 2024 validation):
    - tensor spline(ad_min) x spline(ball_time): a hard-hit ball straight at the fielder
      vs. a slow roller far away are different plays
    - spline(ad_min) x launch_speed
    - hp_to_1b x throw_dist, hp_to_1b x spline(ball_time): runner speed only matters on
      close plays
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FielderGeometryFeatures":
        self.splines_: dict[str, SplineTransformer] = {
            c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
            for c in ("ad_min", "ball_time", "launch_angle")
        }
        self.scaler_: StandardScaler = StandardScaler().fit(
            X[["launch_speed", "throw_dist", "hp_to_1b"]])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        a = self.splines_["ad_min"].transform(X[["ad_min"]])
        b = self.splines_["ball_time"].transform(X[["ball_time"]])
        la = self.splines_["launch_angle"].transform(X[["launch_angle"]])
        z = self.scaler_.transform(X[["launch_speed", "throw_dist", "hp_to_1b"]])
        ev, throw, hp = z[:, [0]], z[:, [1]], z[:, [2]]
        n_rows = len(X)
        return np.hstack([
            a, b, la, z,
            (a[:, :, None] * b[:, None, :]).reshape(n_rows, -1),  # tensor ad x bt
            a * ev,                                               # ad x EV
            hp * throw, hp * b,                                   # runner interactions
        ])


def make_optimizer_glm() -> Pipeline:
    return Pipeline([("features", FielderGeometryFeatures()),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


class DifficultyGLMFeatures(BaseEstimator, TransformerMixin):
    """Design matrix for the evaluation difficulty GLM (no fielder information).

    spray is first mirrored for left-handed batters (negative = pull side, so left- and
    right-handed batters share the same shape, same as Melville §2), then fit with an
    8-knot spline (needs to accommodate the lane structure of all four positions);
    LA/EV/hp each get 6 knots.
    Interactions: spray x EV (hard-hit balls finding gaps), spray x hp (directionality
    of infield hits on slow rollers).
    """

    def __init__(self, spray_ev: bool = True, spray_hp: bool = True) -> None:
        self.spray_ev = spray_ev
        self.spray_hp = spray_hp

    @staticmethod
    def _spray_rel(X: pd.DataFrame) -> np.ndarray:
        sign = np.where(X["stand_R"].to_numpy(float) == 1, 1.0, -1.0)
        return (X["spray_deg"].to_numpy(float) * sign)[:, None]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DifficultyGLMFeatures":
        self.spl_spray_: SplineTransformer = SplineTransformer(n_knots=8, degree=3).fit(
            self._spray_rel(X))
        self.spl_: dict[str, SplineTransformer] = {
            c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
            for c in ("launch_angle", "launch_speed", "hp_to_1b")
        }
        self.scaler_: StandardScaler = StandardScaler().fit(X[["launch_speed", "hp_to_1b"]])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
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


def make_difficulty_glm(**kw: bool) -> Pipeline:
    return Pipeline([("features", DifficultyGLMFeatures(**kw)),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


def make_difficulty_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                          early_stopping=True, random_state=42)


# -- Stage B: two-stage optimization GLM for runner-on-first (<2 outs) -----------
# E[outs] = P(>=1 out) x (1 + P(double play | >=1 out)), both stages counterfactual
# (using only fielder-relative geometry + ball quality + runner speed, excluding raw
# spray, following the same discipline as make_optimizer_glm).

ON1B_OUT_FEATURES: list[str] = OPTIMIZER_FEATURES + ["throw_dist_2b"]
ON1B_DP_FEATURES: list[str] = ["ad_min", "ball_time", "launch_speed", "hp_to_1b",
                                "runner_hp_to_1b", "pivot_dist", "throw_dist_2b"]


class On1bOutFeatures(FielderGeometryFeatures):
    """Stage 1 P(>=1 out): the baseline geometry design matrix plus a linear throw_dist_2b term.

    With a runner on first, the nearest force target is second base, so we add the
    distance to second on top of throw_dist (to first); for the same reason as
    throw_dist, this is **kept linear** (a deterministic function of interception
    geometry; a flexible shape would be a backdoor for the excluded raw spray signal).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "On1bOutFeatures":
        super().fit(X, y)
        self.scaler_2b_: StandardScaler = StandardScaler().fit(X[["throw_dist_2b"]])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.hstack([super().transform(X),
                          self.scaler_2b_.transform(X[["throw_dist_2b"]])])


class On1bDPFeatures(BaseEstimator, TransformerMixin):
    """Design matrix for Stage 2's P(double play | >=1 out).

    Spline: ad_min, ball_time (only a ball fielded cleanly and quickly enough gives time
    to turn the double play).
    Linear: launch_speed, hp_to_1b (batter's time to first = the race for the double
    play's second out), runner_hp_to_1b (runner's time to second = the race for the
    force), pivot_dist (pivot geometry), throw_dist_2b -- geometric terms are always
    kept linear (the functional-form endogeneity discipline).
    interactions=True adds hp_to_1b x spline(ball_time) (runner speed only matters on
    close plays, following Stage 1's conclusion; whether to include it was decided by
    2023 -> 2024 validation).
    """

    def __init__(self, interactions: bool = True) -> None:
        self.interactions = interactions

    _LINEAR: list[str] = ["launch_speed", "hp_to_1b", "runner_hp_to_1b",
                           "pivot_dist", "throw_dist_2b"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "On1bDPFeatures":
        self.splines_: dict[str, SplineTransformer] = {
            c: SplineTransformer(n_knots=6, degree=3).fit(X[[c]])
            for c in ("ad_min", "ball_time")
        }
        self.scaler_: StandardScaler = StandardScaler().fit(X[self._LINEAR])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
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


def make_on1b_dp_glm(**kw: bool) -> Pipeline:
    return Pipeline([("features", On1bDPFeatures(**kw)),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])
