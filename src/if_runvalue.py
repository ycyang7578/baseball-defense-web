"""Building blocks for the infield run-value objective (Phase A): ground-ball hit-type model + run-cost pricing.

Aligned with the outfield objective_re24 convention (src/optimization.py compute_w_j):
- w_j (miss cost) = Σ_k P(k | ball j) × ΔRE(k, base/out state)
- For outfield, P(k|j) comes from a landing-spot KDE; for ground balls the landing spot is endogenous (hc is
  the position after fielding), so instead we model P(extra-base hit | hit) from launch parameters: balls down
  the line carry double risk, high exit velocity that gets through carries deeper, and fast runners turn
  singles into extra bases. Triples are extremely rare on ground balls, so they're folded into extra-base hits
  and priced at ΔRE(2B) (a negligible underestimate).
- Out value ΔRE(out): a ground-ball out is approximated as "no runner advances, outs +1" (exact with the
  bases empty).

This model is the same kind as the difficulty GLM: used for valuation/pricing, doesn't move fielders, has no
counterfactual requirement, so spray angle is valid to use directly.
How this feeds the objective function (see the if_optimize.expected_outs docstring):
    E[ΔRE] = mean(w_j) − mean(p_j × (w_j − ΔRE_out))
optimize_infield(ball_weights = w_j − ΔRE_out) maximizes the second term, which is equivalent to minimizing
expected run cost.
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
from src.re24 import BaseOutState, HitDeltaKey

XB_FEATURES: list[str] = ["spray_deg", "launch_speed", "hp_to_1b"]


class XBRateFeatures(BaseEstimator, TransformerMixin):
    """Design matrix for P(extra-base hit | ground-ball hit).

    spray uses an **absolute-angle** spline (both foul lines are double-risk zones, and field geometry doesn't
    mirror with the batter's handedness); launch_speed / hp_to_1b are linear.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "XBRateFeatures":
        self.spl_spray_: SplineTransformer = SplineTransformer(n_knots=8, degree=3).fit(
            X[["spray_deg"]])
        self.scaler_: StandardScaler = StandardScaler().fit(X[["launch_speed", "hp_to_1b"]])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        s = self.spl_spray_.transform(X[["spray_deg"]])
        z = self.scaler_.transform(X[["launch_speed", "hp_to_1b"]])
        return np.hstack([s, z])


def make_gb_xb_model() -> Pipeline:
    return Pipeline([("features", XBRateFeatures()),
                     ("lr", LogisticRegression(max_iter=8000, C=1.0))])


def fetch_gb_hits(years: list[int]) -> pd.DataFrame:
    """League-wide ground-ball hits (singles/doubles/triples), including the batter's hp_to_1b for that season (missing values filled with the season median)."""
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


def delta_re_out(re24: dict[BaseOutState, float], state: BaseOutState) -> float:
    """deltaRE for a ground-ball out: no runner advances, outs +1 (exact with the bases empty)."""
    b1, b2, b3, outs = state
    if outs >= 2:
        return -re24.get(state, 0.0)          # third out: half-inning ends
    return re24.get((b1, b2, b3, outs + 1), 0.0) - re24.get(state, 0.0)


def gb_miss_costs(balls: pd.DataFrame, xb_model: Pipeline,
                  delta_re: dict[HitDeltaKey, float], state: BaseOutState) -> np.ndarray:
    """Per-ball miss cost w_j (aligned with the outfield compute_w_j key convention)."""
    p_xb = xb_model.predict_proba(balls[XB_FEATURES])[:, 1]
    dre_1b = delta_re.get(("1B", *state), 0.0)
    dre_2b = delta_re.get(("2B", *state), 0.0)
    return (1.0 - p_xb) * dre_1b + p_xb * dre_2b


def runvalue_ball_weights(balls: pd.DataFrame, xb_model: Pipeline,
                          re24: dict[BaseOutState, float],
                          delta_re: dict[HitDeltaKey, float],
                          state: BaseOutState
                          ) -> tuple[np.ndarray, float]:
    """Returns (ball_weights for optimize_infield, mean_miss_cost).

    E[ΔRE](angles, depths) = mean_miss_cost − score(the ball_weights-based scorer).
    """
    w = gb_miss_costs(balls, xb_model, delta_re, state)
    dre_o = delta_re_out(re24, state)
    return w - dre_o, float(w.mean())
