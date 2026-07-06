"""
KDE model for hit-type probabilities P(1B|j), P(2B|j), P(3B|j).

Given a batted ball j with features (launch_speed, launch_angle, spray_angle, stand),
predict the probability that j becomes each hit type IF it falls for a hit.

Method: Bayesian KDE classification
  P(k|x) ∝ prior_k × KDE_k(x)

- Features are standardized before fitting (sklearn StandardScaler).
- KDE bandwidth is applied in the standardized space (bandwidth=0.5).
- Separate models for L/R batters (spray angle distributions differ).
- Trained on outfield fly ball/line drive hits (hit_location ∈ {7,8,9}).

Usage:
  from src.hit_prob import fit_hit_type_kde, predict_hit_probs, save_hit_prob, load_hit_prob

  models = fit_hit_type_kde([2021, 2022, 2023, 2024])
  save_hit_prob(models, Path("data/precomputed"))

  models = load_hit_prob(Path("data/precomputed"))
  p = predict_hit_probs(models, launch_speed=90, launch_angle=25,
                        spray_angle=15.0, stand='R')
  # p = {'1B': 0.62, '2B': 0.31, '3B': 0.07}
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from .config import DSN
from .physics import _X0, _Y0

_HIT_TYPES = ("1B", "2B", "3B")
_EVENT_MAP = {"single": "1B", "double": "2B", "triple": "3B"}
_FEATURES = ["launch_speed", "launch_angle", "spray_angle"]

_QUERY = """
    SELECT launch_speed, launch_angle, hc_x, hc_y, stand, events
    FROM statcast
    WHERE game_year = ANY(%(years)s)
      AND game_type = 'R'
      AND events IN ('single', 'double', 'triple')
      AND bb_type IN ('fly_ball', 'line_drive')
      AND hit_location IN (7, 8, 9)
      AND launch_speed IS NOT NULL
      AND launch_angle IS NOT NULL
      AND hc_x IS NOT NULL
      AND hc_y IS NOT NULL
      AND stand IN ('L', 'R')
"""

def _spray_angle(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    """Spray angle in degrees (-90°=left foul, 0°=center, +90°=right foul)."""
    return np.degrees(np.arctan2(hc_x - _X0, _Y0 - hc_y))


def fit_hit_type_kde(
    years: list[int],
    dsn: str = DSN,
    bandwidth: float = 0.5,
) -> dict:
    """
    Fit KDE models and return a bundle containing:
        {
          'models': {(stand, hit_type): KernelDensity},
          'scalers': {stand: StandardScaler},
          'priors': {(stand, hit_type): float},
        }

    Models are fitted in standardized feature space for each batter hand.
    Features: [launch_speed, launch_angle, spray_angle].
    """
    with psycopg2.connect(dsn) as conn:
        df = pd.read_sql(_QUERY, conn, params={"years": years})

    df["spray_angle"] = _spray_angle(df["hc_x"], df["hc_y"])
    df["hit_type"] = df["events"].map(_EVENT_MAP)
    df = df.dropna(subset=["launch_speed", "launch_angle", "spray_angle", "hit_type"])

    models: dict = {}
    scalers: dict = {}
    priors: dict = {}

    for stand in ("L", "R"):
        sub = df[df["stand"] == stand].copy()
        if len(sub) < 30:
            continue

        scaler = StandardScaler()
        scaler.fit(sub[_FEATURES].values)
        scalers[stand] = scaler

        total = len(sub)
        for ht in _HIT_TYPES:
            mask = sub["hit_type"] == ht
            X = sub.loc[mask, _FEATURES].values
            prior = len(X) / total
            priors[(stand, ht)] = prior

            if len(X) < 10:
                continue
            X_std = scaler.transform(X)
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
            kde.fit(X_std)
            models[(stand, ht)] = kde

    return {"models": models, "scalers": scalers, "priors": priors}


def predict_hit_probs(
    bundle: dict,
    launch_speed: float,
    launch_angle: float,
    spray_angle: float,
    stand: str,
) -> dict:
    """
    Return {'1B': p1, '2B': p2, '3B': p3} for a single batted ball.
    Probabilities sum to 1.

    If stand not in bundle (rare), falls back to uniform.
    """
    x_raw = np.array([[launch_speed, launch_angle, spray_angle]])

    scaler = bundle["scalers"].get(stand)
    if scaler is None:
        return {"1B": 1 / 3, "2B": 1 / 3, "3B": 1 / 3}

    x_std = scaler.transform(x_raw)

    # P(k|x) ∝ prior_k × density_k(x)
    weighted: dict = {}
    for ht in _HIT_TYPES:
        prior = bundle["priors"].get((stand, ht), 0.0)
        kde = bundle["models"].get((stand, ht))
        if kde is None or prior == 0:
            weighted[ht] = 0.0
        else:
            log_density = kde.score_samples(x_std)[0]
            weighted[ht] = prior * np.exp(log_density)

    total = sum(weighted.values())
    if total == 0:
        return {"1B": 1 / 3, "2B": 1 / 3, "3B": 1 / 3}
    return {ht: weighted[ht] / total for ht in _HIT_TYPES}


def predict_hit_probs_batch(
    bundle: dict,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Vectorized prediction for a DataFrame with columns:
      launch_speed, launch_angle, spray_angle, stand

    Returns array of shape (N, 3): columns = [P(1B), P(2B), P(3B)].
    """
    N = len(df)
    probs = np.full((N, 3), 1 / 3)

    for stand in ("L", "R"):
        scaler = bundle["scalers"].get(stand)
        if scaler is None:
            continue
        idx = df.index[df["stand"] == stand]
        if len(idx) == 0:
            continue

        pos = df.index.get_indexer(idx)  # integer positions in the original array
        x_std = scaler.transform(df.loc[idx, _FEATURES].values)

        log_dens = np.zeros((len(idx), 3))
        for i, ht in enumerate(_HIT_TYPES):
            kde = bundle["models"].get((stand, ht))
            prior = bundle["priors"].get((stand, ht), 0.0)
            if kde is None or prior == 0:
                log_dens[:, i] = -np.inf
            else:
                log_dens[:, i] = kde.score_samples(x_std) + np.log(prior)

        # Stable softmax-style normalization
        log_dens -= log_dens.max(axis=1, keepdims=True)
        w = np.exp(log_dens)
        row_sum = w.sum(axis=1, keepdims=True)
        valid = (row_sum > 0).ravel()
        w[valid] /= row_sum[valid]
        w[~valid] = 1 / 3

        probs[pos] = w

    return probs


def save_hit_prob(bundle: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "hit_type_kde.joblib")


def load_hit_prob(data_dir: Path) -> dict:
    return joblib.load(Path(data_dir) / "hit_type_kde.joblib")
