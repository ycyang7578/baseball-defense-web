"""Export the infield Bayesian model's posterior means into artifacts the optimizer can use.

Two outputs (models/if_gb/bayes/):
- if_bayes_group_pipeline.joblib: population-level coefficients packed into an
  sklearn Pipeline (a FielderGeometryFeatures + LogisticRegression shell). Its
  interface is identical to the existing optimization GLM -> _FastGLMObjective,
  precompute, and the tests all reuse it directly.
- IF_player_effects.csv: player_id, alpha, g (posterior means) plus the ad
  standardization params from meta. The optimizer personalizes with
  alpha_j + g_j*ad_z (see src/if_optimize.py).

Verified after export: pipeline predict_proba matches the trace's posterior-mean
logit (within 1e-10).

Run: python -m scripts.export_if_bayes
"""
import json
from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.if_dataset import build_gb_dataset
from src.if_model import OPTIMIZER_FEATURES

BAYES_DIR = Path(__file__).resolve().parent.parent / "models" / "if_gb" / "bayes"


def main() -> None:
    meta = json.loads((BAYES_DIR / "IF_meta.json").read_text(encoding="utf-8"))
    feat = joblib.load(BAYES_DIR / "IF_features.joblib")
    post = az.from_netcdf(BAYES_DIR / "IF_trace.nc").posterior

    beta0 = float(post["beta0"].mean())
    betas = post["betas"].mean(dim=("chain", "draw")).to_numpy()
    lr = LogisticRegression()
    lr.coef_ = betas[None, :]
    lr.intercept_ = np.array([beta0])
    lr.classes_ = np.array([0, 1])
    lr.n_features_in_ = len(betas)
    pipe = Pipeline([("features", feat), ("lr", lr)])
    joblib.dump(pipe, BAYES_DIR / "if_bayes_group_pipeline.joblib")

    alpha = post["alpha"].mean(dim=("chain", "draw")).to_numpy()
    g = post["g"].mean(dim=("chain", "draw")).to_numpy()
    eff = pd.DataFrame({"player_id": meta["players"], "alpha": alpha.round(6),
                        "g": g.round(6)})
    eff.to_csv(BAYES_DIR / "IF_player_effects.csv", index=False)

    # Verify: pipeline matches the posterior-mean logit (sampled from the 2025 main range)
    test = build_gb_dataset([meta["test_year"]], bases_empty=True,
                            alignment="Standard").head(2000)
    X = feat.transform(test[OPTIMIZER_FEATURES])
    logit_ref = beta0 + X @ betas
    p_ref = 1 / (1 + np.exp(-logit_ref))
    p_pipe = pipe.predict_proba(test[OPTIMIZER_FEATURES])[:, 1]
    err = np.abs(p_pipe - p_ref).max()
    assert err < 1e-10, f"pipeline 與後驗平均不等價: {err}"
    print(f"[OK] pipeline 等價驗證 max_err={err:.2e}")
    print(f"[saved] if_bayes_group_pipeline.joblib（{len(betas)} 係數）、"
          f"IF_player_effects.csv（{len(eff)} 位野手）")


if __name__ == "__main__":
    main()
