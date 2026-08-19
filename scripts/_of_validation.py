"""Shared: data computation for validating outfield Model OAA against official OAA.

make_validation_plot.py and make_validation_plot_v2.py render validation scatter
plots with different layouts (v2 deliberately leaves v1 untouched and only changes
the layout — see the v2 docstring), but the data fetching / model OAA computation /
official OAA lookup / name normalization steps are identical, so they're factored
out here to keep the two definitions from drifting apart.
"""
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler

from src.defender_features import get_defender_opportunities, mark_official

POSITIONS: list[str] = ["LF", "CF", "RF"]
FEATURE_COLS: list[str] = ["speed", "cos_angle", "sin_angle", "fielder_dist"]


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def load_of_model(models_dir: Path) -> tuple[StandardScaler, pd.Series]:
    """Return the unified OF model's (scaler, mu), where mu is the population-level posterior mean (Series)."""
    of_dir = models_dir / "OF"
    scaler = joblib.load(of_dir / "OF_scaler.joblib")
    mu = pd.read_csv(of_dir / "OF_summary_group.csv", encoding="utf-8-sig", index_col=0)["mean"]
    return scaler, mu


def compute_model_oaa(pos: str, year: int, scaler: StandardScaler, mu: pd.Series) -> pd.DataFrame:
    """Per-play model OAA for a single position (is_official subset, population-level mu, never player-level)."""
    df = get_defender_opportunities(pos, year)
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=FEATURE_COLS + ["caught", "name_fielder"])
    df = mark_official(df)
    df = df[df["is_official"]].copy()

    X = scaler.transform(df[FEATURE_COLS])
    logit = (mu["mu_alpha"]
             + mu["mu_beta_speed"] * X[:, 0]
             + mu["mu_beta_cos"]   * X[:, 1]
             + mu["mu_beta_sin"]   * X[:, 2]
             + mu["mu_beta_dist"]  * X[:, 3])
    df["catch_prob"] = sigmoid(logit)
    df["oaa_play"] = df["caught"].astype(float) - df["catch_prob"]
    return df[["name_fielder", "caught", "catch_prob", "oaa_play"]]


def load_model_oaa(models_dir: Path, year: int) -> pd.DataFrame:
    """Sum per-play oaa_play across LF+CF+RF -> one row per player, with a normalized-name key."""
    scaler, mu = load_of_model(models_dir)
    all_plays = pd.concat(
        [compute_model_oaa(p, year, scaler, mu) for p in POSITIONS], ignore_index=True)
    model_oaa = all_plays.groupby("name_fielder", as_index=False)["oaa_play"].sum()
    model_oaa["key"] = model_oaa["name_fielder"].apply(normalize_name)
    return model_oaa


def load_official_oaa(dsn: str, year: int) -> pd.DataFrame:
    """Official OAA (is_qualified=True; the same player may qualify at multiple positions, summed to a total outfield OAA)."""
    with psycopg2.connect(dsn) as conn:
        raw_off = pd.read_sql(
            "SELECT player_name, oaa FROM oaa_leaderboard "
            "WHERE year = %(y)s AND is_qualified = TRUE",
            conn, params={"y": year})
    official = raw_off.groupby("player_name", as_index=False)["oaa"].sum()
    official["key"] = official["player_name"].apply(normalize_name)
    return official
