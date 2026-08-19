"""Ablation: speed + cos_angle + sin_angle + fielder_dist (no flight_time).

Note: speed = fielder_dist / flight_time, so including both speed and fielder_dist means the model can
implicitly recover flight_time as well (time = dist/speed).
This is the same structural collinearity as pass18's dist+time+speed combination -- be sure to check
r_hat/ess_bulk after convergence.
"""
import os

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=constant_folding")

import joblib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .defender_features import get_defender_opportunities

FEATURE_COLS: list[str] = ["speed", "cos_angle", "sin_angle", "fielder_dist"]
ALL_COLS: list[str] = FEATURE_COLS

MCMC_KWARGS: dict[str, int | float | str] = dict(
    draws=2000, tune=2000, chains=4, cores=4,
    target_accept=0.95, nuts_sampler="pymc", random_seed=42,
)


def load_training_data(position: str, years: list[int]) -> pd.DataFrame:
    frames = [get_defender_opportunities(position, y) for y in years]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=ALL_COLS + ["caught", "name_fielder"])
    return df


def build_model(df: pd.DataFrame, scaler: StandardScaler) -> pm.Model:
    X = scaler.transform(df[ALL_COLS])
    player_idx, players = pd.factorize(df["name_fielder"])
    y_obs = df["caught"].values

    coords = {"player": players, "obs": np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        idx_p = pm.Data("idx_p", player_idx, dims="obs")
        data_speed = pm.Data("data_speed", X[:, 0], dims="obs")
        data_cos = pm.Data("data_cos", X[:, 1], dims="obs")
        data_sin = pm.Data("data_sin", X[:, 2], dims="obs")
        data_dist = pm.Data("data_dist", X[:, 3], dims="obs")

        mu_alpha = pm.Normal("mu_alpha", mu=0.5, sigma=3.0)
        mu_beta_speed = pm.Normal("mu_beta_speed", mu=-1.0, sigma=2.0)
        mu_beta_cos = pm.Normal("mu_beta_cos", mu=0.0, sigma=1.0)
        mu_beta_sin = pm.Normal("mu_beta_sin", mu=0.0, sigma=1.0)
        # No sign prior on dist (speed already encodes dist information, so the sign of dist's marginal effect is uncertain)
        mu_beta_dist = pm.Normal("mu_beta_dist", mu=0.0, sigma=2.0)

        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        sigma_beta_speed = pm.HalfNormal("sigma_beta_speed", sigma=0.5)
        sigma_beta_cos = pm.HalfNormal("sigma_beta_cos", sigma=0.5)
        sigma_beta_sin = pm.HalfNormal("sigma_beta_sin", sigma=0.5)
        sigma_beta_dist = pm.HalfNormal("sigma_beta_dist", sigma=0.5)

        z_alpha = pm.Normal("z_alpha", 0, 1, dims="player")
        z_speed = pm.Normal("z_speed", 0, 1, dims="player")
        z_cos = pm.Normal("z_cos", 0, 1, dims="player")
        z_sin = pm.Normal("z_sin", 0, 1, dims="player")
        z_dist = pm.Normal("z_dist", 0, 1, dims="player")

        alpha = pm.Deterministic("alpha", mu_alpha + z_alpha * sigma_alpha, dims="player")
        beta_speed = pm.Deterministic("beta_speed", mu_beta_speed + z_speed * sigma_beta_speed, dims="player")
        beta_cos = pm.Deterministic("beta_cos", mu_beta_cos + z_cos * sigma_beta_cos, dims="player")
        beta_sin = pm.Deterministic("beta_sin", mu_beta_sin + z_sin * sigma_beta_sin, dims="player")
        beta_dist = pm.Deterministic("beta_dist", mu_beta_dist + z_dist * sigma_beta_dist, dims="player")

        logit_p = (alpha[idx_p]
                   + beta_speed[idx_p] * data_speed
                   + beta_cos[idx_p] * data_cos
                   + beta_sin[idx_p] * data_sin
                   + beta_dist[idx_p] * data_dist)

        pm.Bernoulli("y", logit_p=logit_p, observed=y_obs, dims="obs")

    return model


def train_position(position: str, years: list[int], output_dir: Path) -> None:
    output_dir = Path(output_dir) / position
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{position}] 載入 {years} 年資料...")
    df = load_training_data(position, years)
    print(f"[{position}] 訓練資料筆數: {len(df):,}, 球員數: {df['name_fielder'].nunique()}")

    scaler = StandardScaler()
    scaler.fit(df[ALL_COLS])
    joblib.dump(scaler, output_dir / f"{position}_scaler.joblib")

    model = build_model(df, scaler)

    print(f"[{position}] 開始 MCMC 採樣（{MCMC_KWARGS}）...")
    with model:
        trace = pm.sample(**MCMC_KWARGS)

    az.to_netcdf(trace, output_dir / f"{position}_trace.nc")

    group_vars = ["mu_alpha", "mu_beta_speed", "mu_beta_cos", "mu_beta_sin", "mu_beta_dist"]
    summary_group = az.summary(trace, var_names=group_vars)
    summary_group.to_csv(output_dir / f"{position}_summary_group.csv", encoding="utf-8-sig")

    player_vars = ["alpha", "beta_speed", "beta_cos", "beta_sin", "beta_dist"]
    summary_players = az.summary(trace, var_names=player_vars)
    summary_players.to_csv(output_dir / f"{position}_summary_players.csv", encoding="utf-8-sig")

    print(f"\n[{position}] === 群體層後驗摘要 ===")
    print(summary_group.to_string())
    print(f"\n[{position}] === 收斂診斷 ===")
    print(f"群體層 r_hat: {summary_group['r_hat'].min():.3f} ~ {summary_group['r_hat'].max():.3f}")
    print(f"群體層 ess_bulk: {summary_group['ess_bulk'].min():.0f} ~ {summary_group['ess_bulk'].max():.0f}")
    print(f"球員層 r_hat: {summary_players['r_hat'].min():.3f} ~ {summary_players['r_hat'].max():.3f}")
    print(f"球員層 ess_bulk: {summary_players['ess_bulk'].min():.0f} ~ {summary_players['ess_bulk'].max():.0f}")
    if summary_players['r_hat'].max() > 1.05 or summary_players['ess_bulk'].min() < 100:
        print(f"[{position}] [警告] 收斂可能有問題，需進一步診斷！")
    else:
        print(f"[{position}] [OK] 收斂正常")
