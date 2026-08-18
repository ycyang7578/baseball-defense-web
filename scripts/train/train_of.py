"""Train unified outfield model (LF+CF+RF combined).

Usage:
    python -m scripts.train.train_of [target_year]   # default 2025
    python -m scripts.train.train_of 2024             # trains on 2020-2023, saves to models/2024/OF/

訓練資料：target_year 前 4 年全量球（LF+CF+RF，無 is_official 過濾）。
"""
import os
os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=constant_folding")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

from src.defender_features import get_defender_opportunities

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("target_year", type=int, nargs="?", default=2025)
_args = _parser.parse_args()

TARGET_YEAR = _args.target_year
POSITIONS   = ["LF", "CF", "RF"]
YEARS       = [TARGET_YEAR - 4, TARGET_YEAR - 3, TARGET_YEAR - 2, TARGET_YEAR - 1]
FEATURE_COLS = ["speed", "cos_angle", "sin_angle", "fielder_dist"]
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "models" / str(TARGET_YEAR) / "OF"

MCMC_KWARGS = dict(draws=2000, tune=2000, chains=4, cores=4,
                   target_accept=0.95, nuts_sampler="pymc", random_seed=42)


def load_training_data(years: list[int]) -> pd.DataFrame:
    frames = []
    for pos in POSITIONS:
        for y in years:
            df = get_defender_opportunities(pos, y)
            df["position"] = pos
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"required_speed": "speed"})
    df = df.dropna(subset=FEATURE_COLS + ["caught", "name_fielder"])
    return df


def build_model(df: pd.DataFrame, scaler: StandardScaler) -> pm.Model:
    X = scaler.transform(df[FEATURE_COLS])
    player_idx, players = pd.factorize(df["name_fielder"])
    y_obs = df["caught"].values

    coords = {"player": players, "obs": np.arange(len(df))}

    with pm.Model(coords=coords) as model:
        idx_p = pm.Data("idx_p", player_idx, dims="obs")
        data_speed = pm.Data("data_speed", X[:, 0], dims="obs")
        data_cos   = pm.Data("data_cos",   X[:, 1], dims="obs")
        data_sin   = pm.Data("data_sin",   X[:, 2], dims="obs")
        data_dist  = pm.Data("data_dist",  X[:, 3], dims="obs")

        mu_alpha       = pm.Normal("mu_alpha",       mu=0.5,  sigma=3.0)
        mu_beta_speed  = pm.Normal("mu_beta_speed",  mu=-1.0, sigma=2.0)
        mu_beta_cos    = pm.Normal("mu_beta_cos",    mu=0.0,  sigma=1.0)
        mu_beta_sin    = pm.Normal("mu_beta_sin",    mu=0.0,  sigma=1.0)
        mu_beta_dist   = pm.Normal("mu_beta_dist",   mu=0.0,  sigma=2.0)

        sigma_alpha      = pm.HalfNormal("sigma_alpha",      sigma=1.0)
        sigma_beta_speed = pm.HalfNormal("sigma_beta_speed", sigma=0.5)
        sigma_beta_cos   = pm.HalfNormal("sigma_beta_cos",   sigma=0.5)
        sigma_beta_sin   = pm.HalfNormal("sigma_beta_sin",   sigma=0.5)
        sigma_beta_dist  = pm.HalfNormal("sigma_beta_dist",  sigma=0.5)

        z_alpha = pm.Normal("z_alpha", 0, 1, dims="player")
        z_speed = pm.Normal("z_speed", 0, 1, dims="player")
        z_cos   = pm.Normal("z_cos",   0, 1, dims="player")
        z_sin   = pm.Normal("z_sin",   0, 1, dims="player")
        z_dist  = pm.Normal("z_dist",  0, 1, dims="player")

        alpha      = pm.Deterministic("alpha",      mu_alpha      + z_alpha * sigma_alpha,      dims="player")
        beta_speed = pm.Deterministic("beta_speed", mu_beta_speed + z_speed * sigma_beta_speed, dims="player")
        beta_cos   = pm.Deterministic("beta_cos",   mu_beta_cos   + z_cos   * sigma_beta_cos,   dims="player")
        beta_sin   = pm.Deterministic("beta_sin",   mu_beta_sin   + z_sin   * sigma_beta_sin,   dims="player")
        beta_dist  = pm.Deterministic("beta_dist",  mu_beta_dist  + z_dist  * sigma_beta_dist,  dims="player")

        logit_p = (alpha[idx_p]
                   + beta_speed[idx_p] * data_speed
                   + beta_cos[idx_p]   * data_cos
                   + beta_sin[idx_p]   * data_sin
                   + beta_dist[idx_p]  * data_dist)

        pm.Bernoulli("y", logit_p=logit_p, observed=y_obs, dims="obs")

    return model


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"目標年份: {TARGET_YEAR}，訓練資料: {YEARS}")
    print(f"載入 {YEARS} 年 LF+CF+RF 合併資料...")
    df = load_training_data(YEARS)
    print(f"訓練資料: {len(df):,} 筆，球員數: {df['name_fielder'].nunique()}")
    print(f"各位置: {df.groupby('position').size().to_dict()}")

    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLS])
    joblib.dump(scaler, OUTPUT_DIR / "OF_scaler.joblib")
    print("Scaler 已儲存")

    model = build_model(df, scaler)

    print(f"\n開始 MCMC 採樣（{MCMC_KWARGS}）...")
    with model:
        trace = pm.sample(**MCMC_KWARGS)

    az.to_netcdf(trace, OUTPUT_DIR / "OF_trace.nc")
    print("Trace 已儲存")

    group_vars   = ["mu_alpha", "mu_beta_speed", "mu_beta_cos", "mu_beta_sin", "mu_beta_dist"]
    player_vars  = ["alpha", "beta_speed", "beta_cos", "beta_sin", "beta_dist"]

    summary_group   = az.summary(trace, var_names=group_vars)
    summary_players = az.summary(trace, var_names=player_vars)

    summary_group.to_csv(OUTPUT_DIR   / "OF_summary_group.csv",   encoding="utf-8-sig")
    summary_players.to_csv(OUTPUT_DIR / "OF_summary_players.csv", encoding="utf-8-sig")

    print("\n=== 群體層後驗摘要 ===")
    print(summary_group.to_string())
    print("\n=== 收斂診斷 ===")
    print(f"群體層 r_hat:    {summary_group['r_hat'].min():.3f} ~ {summary_group['r_hat'].max():.3f}")
    print(f"群體層 ess_bulk: {summary_group['ess_bulk'].min():.0f} ~ {summary_group['ess_bulk'].max():.0f}")
    print(f"球員層 r_hat:    {summary_players['r_hat'].min():.3f} ~ {summary_players['r_hat'].max():.3f}")
    print(f"球員層 ess_bulk: {summary_players['ess_bulk'].min():.0f} ~ {summary_players['ess_bulk'].max():.0f}")
    if summary_players["r_hat"].max() > 1.05 or summary_players["ess_bulk"].min() < 100:
        print("[警告] 收斂可能有問題，需進一步診斷！")
    else:
        print("[OK] 收斂正常")
