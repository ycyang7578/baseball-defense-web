"""Infield player-level route A prototype: scale the nearest fielder's effective
angular distance by sprint speed.

Hypothesis: fielder speed v scales lateral coverage -> ad_eff = ad_min *
(v_ref / v_i)^gamma. gamma=0 degenerates to the current model; only if gamma is
significantly >0 and out-of-sample performance improves is there value in going
to the player level.

Two v_ref variants (separating "positional effect" from "individual effect"):
- league: v_ref = league average -- gamma would also absorb between-position
  differences (SS is fast, 1B is slow)
- within: v_ref = same-position same-year average -- keeps only the within-
  position individual differences, a clean test of the player-level signal

Methodology (follows the same process as the interaction-configuration
experiment): gamma is selected on train 2021-23 -> validate 2024; 2025 is
touched only once for the final evaluation after selection. Placebo control:
shuffle v randomly within the same (position, year) group -- if the gamma gain
doesn't disappear, that indicates a spurious signal.

Run: python scripts/experiments/exp_if_speed_scaling.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DSN
from src.if_dataset import INFIELD_COLS, build_gb_dataset
from src.if_model import OPTIMIZER_FEATURES, make_optimizer_glm

SELECT_TRAIN = [2021, 2022, 2023]   # for gamma selection
SELECT_VAL = 2024
FINAL_TRAIN = [2021, 2022, 2023, 2024]
FINAL_TEST = 2025
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SEED = 42


def attach_nearest_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest fielder's sprint_speed (v) and the same-position-year
    average (v_pos)."""
    pos_to_col = {v: k for k, v in INFIELD_COLS.items()}
    ids = np.select([df["nearest_pos"] == p for p in pos_to_col],
                    [df[c] for c in (pos_to_col[p] for p in pos_to_col)])
    df = df.copy()
    df["nearest_id"] = ids.astype(int)
    years = [int(y) for y in sorted(df["game_year"].unique())]
    with psycopg2.connect(DSN) as conn:
        ss = pd.read_sql(
            "SELECT player_id, season, sprint_speed FROM sprint_speed "
            "WHERE season = ANY(%(y)s)", conn, params={"y": years})
    ss_map = ss.set_index(["player_id", "season"])["sprint_speed"]
    idx = pd.MultiIndex.from_arrays([df["nearest_id"], df["game_year"]])
    df["v"] = ss_map.reindex(idx).to_numpy()
    # same-position-year average computed over distinct fielders, to avoid weighting by regular starters' ball counts
    uniq = (df[["nearest_id", "nearest_pos", "game_year", "v"]]
            .drop_duplicates(["nearest_id", "nearest_pos", "game_year"]))
    pos_mean = (uniq.groupby(["nearest_pos", "game_year"])["v"].mean()
                .rename("v_pos"))
    df = df.merge(pos_mean, left_on=["nearest_pos", "game_year"],
                  right_index=True, how="left")
    df["v"] = df["v"].fillna(df["v_pos"])   # missing values (<0.3%) -> ratio of 1, equivalent to no scaling
    return df


def scaled(df: pd.DataFrame, gamma: float, v_ref) -> pd.DataFrame:
    out = df.copy()
    out["ad_min"] = df["ad_min"] * (v_ref / df["v"]) ** gamma
    return out


def fit_eval(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    glm = make_optimizer_glm()
    glm.fit(train[OPTIMIZER_FEATURES], train["is_out"])
    p = glm.predict_proba(test[OPTIMIZER_FEATURES])[:, 1]
    return log_loss(test["is_out"], p), roc_auc_score(test["is_out"], p)


def shuffle_within(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Placebo: randomly shuffle the fielder-to-speed mapping within the same
    (position, year) group (per fielder, not per ball)."""
    rng = np.random.default_rng(seed)
    uniq = (df[["nearest_id", "nearest_pos", "game_year", "v"]]
            .drop_duplicates(["nearest_id", "nearest_pos", "game_year"]))
    parts = []
    for _, g in uniq.groupby(["nearest_pos", "game_year"]):
        g = g.copy()
        g["v_sh"] = rng.permutation(g["v"].to_numpy())
        parts.append(g)
    sh = pd.concat(parts)[["nearest_id", "nearest_pos", "game_year", "v_sh"]]
    out = df.merge(sh, on=["nearest_id", "nearest_pos", "game_year"], how="left")
    out["v"] = out["v_sh"]
    return out.drop(columns=["v_sh"])


def main() -> None:
    pool = attach_nearest_speed(
        build_gb_dataset(FINAL_TRAIN + [FINAL_TEST],
                         bases_empty=True, alignment="Standard"))
    sel_tr = pool[pool["game_year"].isin(SELECT_TRAIN)]
    sel_va = pool[pool["game_year"] == SELECT_VAL]
    print(f"γ 選擇：train {SELECT_TRAIN} n={len(sel_tr):,} → "
          f"validate {SELECT_VAL} n={len(sel_va):,}")

    v_league = pool["v"].median()
    results = []
    for variant, ref_tr, ref_va in [
            ("league", v_league, v_league),
            ("within", sel_tr["v_pos"], sel_va["v_pos"])]:
        for g in GAMMAS:
            ll, auc = fit_eval(scaled(sel_tr, g, ref_tr), scaled(sel_va, g, ref_va))
            results.append({"variant": variant, "gamma": g,
                            "val_logloss": ll, "val_auc": auc})
            print(f"  {variant:<7} γ={g:<5} val logloss={ll:.5f}  AUC={auc:.4f}",
                  flush=True)
    res = pd.DataFrame(results)
    base_ll = res[res["gamma"] == 0].iloc[0]["val_logloss"]

    print("\n=== 2024 驗證總表（Δlogloss<0 = 比 γ=0 好）===")
    res["d_logloss"] = res["val_logloss"] - base_ll
    print(res.round(5).to_string(index=False))

    best = res.loc[res["val_logloss"].idxmin()]
    print(f"\n最佳：variant={best['variant']} γ={best['gamma']} "
          f"Δlogloss={best['d_logloss']:+.5f}")
    if best["gamma"] == 0:
        print("γ=0 最佳 → 個人化訊號不存在，路線 A 到此為止")
        return

    ref = v_league if best["variant"] == "league" else None
    print("\n=== 安慰劑（同位置-年內重排 v，5 個 seed）===")
    for s in range(5):
        sh_tr = shuffle_within(sel_tr, SEED + s)
        sh_va = shuffle_within(sel_va, SEED + s)
        rt = ref if ref is not None else sh_tr["v_pos"]
        rv = ref if ref is not None else sh_va["v_pos"]
        ll, auc = fit_eval(scaled(sh_tr, best["gamma"], rt),
                           scaled(sh_va, best["gamma"], rv))
        print(f"  seed {SEED + s}: Δlogloss={ll - base_ll:+.5f}  AUC={auc:.4f}",
              flush=True)

    print(f"\n=== 最終評估（train {FINAL_TRAIN} → test {FINAL_TEST}，只跑一次）===")
    fin_tr = pool[pool["game_year"].isin(FINAL_TRAIN)]
    fin_te = pool[pool["game_year"] == FINAL_TEST]
    ll0, auc0 = fit_eval(fin_tr, fin_te)
    print(f"  baseline γ=0     : logloss={ll0:.5f}  AUC={auc0:.4f}")
    rt = ref if ref is not None else fin_tr["v_pos"]
    rv = ref if ref is not None else fin_te["v_pos"]
    ll1, auc1 = fit_eval(scaled(fin_tr, best["gamma"], rt),
                         scaled(fin_te, best["gamma"], rv))
    print(f"  {best['variant']} γ={best['gamma']}: logloss={ll1:.5f}  "
          f"AUC={auc1:.4f}  Δlogloss={ll1 - ll0:+.5f}")


if __name__ == "__main__":
    main()
