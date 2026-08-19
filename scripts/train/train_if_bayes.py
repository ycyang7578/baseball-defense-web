"""Infield hierarchical Bayesian out model: a player level (random intercept +
random slope on ad) aligned with the outfield architecture.

Structure (compare with train_of.py):
- Attribution: each ball is credited to the "nearest-angular-distance fielder"
  (the infield analogue of the outfield's "chaser")
- Population level: reuses the optimization GLM's full design matrix
  (FielderGeometryFeatures: splines + tensor interactions), matching GLM
  quality (2025 AUC 0.753) without sacrificing nonlinearity from
  Bayesianizing it
- Player level (non-centered): alpha_j (random intercept = overall conversion
  ability) + g_j x ad_z (random slope = range shape). Known going in: a
  gamma sweep showed the range signal from the external proxy isn't
  detectable (scripts/experiments/exp_if_speed_scaling /
  scripts/experiments/exp_if_oaa_scaling), and the outfield's own slope
  shrinkage ratio is also <1 (beta_dist 0.65) -- g_j is expected to be
  shrunk to near zero, relying on shrinkage as a safety net. This is a
  direction decision the user made with full knowledge of the tradeoff.

Scope and years match train_if_gb.py: bases empty + Standard alignment,
trained on 2023-2024, 2025 held out as out-of-sample.
Produces models/if_gb/bayes/: trace, population/player summary CSVs, feature
transformer, metadata.

Run: python -m scripts.train.train_if_bayes [--smoke] (smoke = quick sanity
check on a small sample)
"""
import os

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=constant_folding")

import argparse
import json
import time
from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import log_loss, roc_auc_score

from src.if_dataset import INFIELD_COLS, build_gb_dataset
from src.if_model import OPTIMIZER_FEATURES, FielderGeometryFeatures

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "if_gb" / "bayes"

MCMC_KWARGS = dict(draws=2000, tune=2000, chains=4, cores=4,
                   target_accept=0.95, nuts_sampler="pymc", random_seed=42)
SMOKE_KWARGS = dict(draws=200, tune=200, chains=2, cores=2,
                    target_accept=0.9, nuts_sampler="pymc", random_seed=42)
PROGRESS_LOG = OUT_DIR / "sampling_progress.log"


def make_progress_logger(total_per_chain: int, every: int = 50):
    """Callback for pm.sample: the progress bar doesn't work under redirected
    output, so write to a log file per chain instead. The callback runs in
    the main process (called once per draw returned by each worker), so
    writing to the file is safe."""
    counts: dict[int, int] = {}
    t0 = time.time()

    def cb(trace=None, draw=None, **kwargs):
        chain = getattr(draw, "chain", 0)
        counts[chain] = counts.get(chain, 0) + 1
        n = counts[chain]
        if n % every == 0 or n == total_per_chain:
            done = sum(counts.values())
            rate = done / max(time.time() - t0, 1)
            with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')}  chain {chain}: "
                        f"{n}/{total_per_chain}  全體 {done} draws  "
                        f"{rate:.2f} draws/s\n")

    return cb


def nearest_fielder_ids(df: pd.DataFrame) -> np.ndarray:
    pos_to_col = {v: k for k, v in INFIELD_COLS.items()}
    return np.select([df["nearest_pos"] == p for p in pos_to_col],
                     [df[c] for c in (pos_to_col[p] for p in pos_to_col)]).astype(int)


def build_model(X: np.ndarray, ad_z: np.ndarray, player_idx: np.ndarray,
                players: np.ndarray, y: np.ndarray) -> pm.Model:
    coords = {"player": players, "col": np.arange(X.shape[1]), "obs": np.arange(len(y))}
    with pm.Model(coords=coords) as model:
        beta0 = pm.Normal("beta0", mu=1.0, sigma=3.0)          # out rate ~0.73 -> logit ~1
        betas = pm.Normal("betas", mu=0.0, sigma=1.0, dims="col")  # ridge-style shrinkage

        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        sigma_g = pm.HalfNormal("sigma_g", sigma=0.5)
        z_alpha = pm.Normal("z_alpha", 0, 1, dims="player")
        z_g = pm.Normal("z_g", 0, 1, dims="player")
        alpha = pm.Deterministic("alpha", z_alpha * sigma_alpha, dims="player")
        g = pm.Deterministic("g", z_g * sigma_g, dims="player")

        logit_p = (beta0 + pm.math.dot(X, betas)
                   + alpha[player_idx] + g[player_idx] * ad_z)
        pm.Bernoulli("y", logit_p=logit_p, observed=y, dims="obs")
    return model


def posterior_logit(trace, X, ad_z, player_idx_or_none) -> np.ndarray:
    """Logit from posterior mean coefficients. player_idx_or_none=None -> population level only."""
    post = trace.posterior
    beta0 = float(post["beta0"].mean())
    betas = post["betas"].mean(dim=("chain", "draw")).to_numpy()
    logit = beta0 + X @ betas
    if player_idx_or_none is not None:
        alpha = post["alpha"].mean(dim=("chain", "draw")).to_numpy()
        g = post["g"].mean(dim=("chain", "draw")).to_numpy()
        known = player_idx_or_none >= 0
        idx = player_idx_or_none[known]
        logit[known] += alpha[idx] + g[idx] * ad_z[known]
    return logit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="小抽樣快速驗證（不存正式產出）")
    args = parser.parse_args()

    train = build_gb_dataset(TRAIN_YEARS, bases_empty=True, alignment="Standard")
    test = build_gb_dataset([TEST_YEAR], bases_empty=True, alignment="Standard")
    if args.smoke:
        train = train.sample(5000, random_state=42).reset_index(drop=True)
    print(f"train n={len(train):,}, test n={len(test):,}")

    feat = FielderGeometryFeatures().fit(train[OPTIMIZER_FEATURES])
    X_tr = feat.transform(train[OPTIMIZER_FEATURES])
    X_te = feat.transform(test[OPTIMIZER_FEATURES])
    ad_mean, ad_std = train["ad_min"].mean(), train["ad_min"].std()
    adz_tr = ((train["ad_min"] - ad_mean) / ad_std).to_numpy()
    adz_te = ((test["ad_min"] - ad_mean) / ad_std).to_numpy()

    ids_tr = nearest_fielder_ids(train)
    ids_te = nearest_fielder_ids(test)
    player_idx, players = pd.factorize(ids_tr)
    te_map = pd.Series(np.arange(len(players)), index=players)
    te_idx = te_map.reindex(ids_te).fillna(-1).to_numpy(int)
    print(f"野手數（train）: {len(players)}，"
          f"test 球的野手已見比例: {(te_idx >= 0).mean():.1%}，"
          f"設計矩陣: {X_tr.shape[1]} 欄")

    model = build_model(X_tr, adz_tr, player_idx, players,
                        train["is_out"].to_numpy())
    kwargs = SMOKE_KWARGS if args.smoke else MCMC_KWARGS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.unlink(missing_ok=True)
    print(f"MCMC 採樣 {kwargs} ...（進度寫在 {PROGRESS_LOG}）", flush=True)
    with model:
        trace = pm.sample(
            **kwargs, progressbar=False,
            callback=make_progress_logger(kwargs["draws"] + kwargs["tune"]))

    # Save to disk **immediately** after sampling, before evaluation/printing --
    # a long-running job's output must not be destroyed by any later error
    # (2026-07-13 incident: the "≈" in a print statement triggered a
    #  UnicodeEncodeError under stdout redirected to cp950, and 3.1 hours of
    #  trace was lost entirely because saving happened after printing)
    sum_grp = az.summary(trace, var_names=["beta0", "sigma_alpha", "sigma_g"])
    sum_ply = az.summary(trace, var_names=["alpha", "g"])
    if not args.smoke:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        az.to_netcdf(trace, OUT_DIR / "IF_trace.nc")
        joblib.dump(feat, OUT_DIR / "IF_features.joblib")
        sum_grp.to_csv(OUT_DIR / "IF_summary_group.csv", encoding="utf-8-sig")
        sum_ply.to_csv(OUT_DIR / "IF_summary_players.csv", encoding="utf-8-sig")
        (OUT_DIR / "IF_meta.json").write_text(json.dumps({
            "train_years": TRAIN_YEARS, "test_year": TEST_YEAR,
            "n_train": len(train), "n_players": len(players),
            "ad_mean": ad_mean, "ad_std": ad_std,
            "players": [int(p) for p in players]}, indent=2), encoding="utf-8")
        print(f"[saved] {OUT_DIR}", flush=True)

    y_te = test["is_out"].to_numpy()
    p_grp = 1 / (1 + np.exp(-posterior_logit(trace, X_te, adz_te, None)))
    p_ply = 1 / (1 + np.exp(-posterior_logit(trace, X_te, adz_te, te_idx)))
    print("\n2025 樣本外（GLM 基準 AUC=0.7531 / logloss~0.4967）：")
    print(f"  群體層 only   : AUC={roc_auc_score(y_te, p_grp):.4f}  "
          f"logloss={log_loss(y_te, p_grp):.5f}")
    print(f"  ＋球員層      : AUC={roc_auc_score(y_te, p_ply):.4f}  "
          f"logloss={log_loss(y_te, p_ply):.5f}")

    print("\n=== 超參數 ===")
    print(sum_grp.to_string())
    for p in ("alpha", "g"):
        rows = sum_ply[sum_ply.index.str.startswith(p + "[")]
        print(f"{p:<6} 跨球員SD={rows['mean'].std():.4f}  "
              f"平均後驗SD={rows['sd'].mean():.4f}  "
              f"ratio={rows['mean'].std() / rows['sd'].mean():.2f}")

    diag = az.summary(trace, var_names=["beta0", "betas", "sigma_alpha", "sigma_g"])
    print(f"\n收斂：r_hat max={diag['r_hat'].max():.3f}, "
          f"ess_bulk min={diag['ess_bulk'].min():.0f}")

    if args.smoke:
        print("[smoke] 不儲存產出")


if __name__ == "__main__":
    main()
