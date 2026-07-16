"""內野球員層路線 A 變體 2：前一年官方 OAA rate 縮放最近野手有效角距。

sprint speed（跑壘量測）縮放無訊號（exp_if_speed_scaling.py，γ=0 最佳），
換守備結果量測再試最後一次：ad_eff = ad_min × exp(−γ·z)，z = 野手前一年
OAA rate 在同 primary_pos-年內的 z 分數（proxy 永遠取球年份的前一年，無洩漏；
無前一年資料者 z=0 = 不縮放）。

兩種 proxy：overall = oaa/n_opp；lateral = (oaa_toward3b+oaa_toward1b)/n_opp
（橫向 range 才是縮放 ad 的機制，overall 混入 charge/hands）。

官方 leaderboard 只有 2023–25 → 球 2024 訓練（proxy 2023）→ 球 2025 驗證
（proxy 2024）。此掃描即最終答案（事前註冊 γ 網格，不再調），若有改善加跑
安慰劑（z 同位置-年內重排）。

執行：python scripts/experiments/exp_if_oaa_scaling.py
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

TRAIN_BALLS, TRAIN_PROXY = 2024, 2023
VAL_BALLS, VAL_PROXY = 2025, 2024
GAMMAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25]   # z=±2 時 γ=0.05 → ad ±10%
SEED = 42


def load_proxy(year: int) -> pd.DataFrame:
    """該年 leaderboard → player_id、z_overall、z_lateral（同 primary_pos 內標準化）。"""
    with psycopg2.connect(DSN) as conn:
        lb = pd.read_sql(
            "SELECT player_id, primary_pos, oaa, oaa_toward3b, oaa_toward1b, n_opp "
            "FROM if_oaa_leaderboard WHERE year = %(y)s AND n_opp >= 50",
            conn, params={"y": year})
    lb["r_overall"] = lb["oaa"] / lb["n_opp"]
    lb["r_lateral"] = (lb["oaa_toward3b"] + lb["oaa_toward1b"]) / lb["n_opp"]
    for c in ("r_overall", "r_lateral"):
        g = lb.groupby("primary_pos")[c]
        lb["z" + c[1:]] = (lb[c] - g.transform("mean")) / g.transform("std")
    return lb[["player_id", "z_overall", "z_lateral"]]


def attach(df: pd.DataFrame, proxy_year: int) -> pd.DataFrame:
    pos_to_col = {v: k for k, v in INFIELD_COLS.items()}
    ids = np.select([df["nearest_pos"] == p for p in pos_to_col],
                    [df[c] for c in (pos_to_col[p] for p in pos_to_col)])
    df = df.copy()
    df["nearest_id"] = ids.astype(int)
    df = df.merge(load_proxy(proxy_year), left_on="nearest_id",
                  right_on="player_id", how="left")
    n_hit = df["z_overall"].notna().mean()
    df[["z_overall", "z_lateral"]] = df[["z_overall", "z_lateral"]].fillna(0.0)
    print(f"  球 {int(df['game_year'].iloc[0])}（proxy {proxy_year}）："
          f"n={len(df):,}，最近野手有前一年 OAA 的比例={n_hit:.1%}")
    return df


def scaled(df: pd.DataFrame, gamma: float, zcol: str) -> pd.DataFrame:
    out = df.copy()
    out["ad_min"] = df["ad_min"] * np.exp(-gamma * df[zcol])
    return out


def fit_eval(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    glm = make_optimizer_glm()
    glm.fit(train[OPTIMIZER_FEATURES], train["is_out"])
    p = glm.predict_proba(test[OPTIMIZER_FEATURES])[:, 1]
    return log_loss(test["is_out"], p), roc_auc_score(test["is_out"], p)


def shuffle_z(df: pd.DataFrame, zcol: str, seed: int) -> pd.DataFrame:
    """安慰劑：z 在同 (最近位置, 年) 內逐野手重排。"""
    rng = np.random.default_rng(seed)
    uniq = df[["nearest_id", "nearest_pos", zcol]].drop_duplicates("nearest_id")
    parts = []
    for _, g in uniq.groupby("nearest_pos"):
        g = g.copy()
        g["z_sh"] = rng.permutation(g[zcol].to_numpy())
        parts.append(g)
    sh = pd.concat(parts)[["nearest_id", "z_sh"]]
    out = df.merge(sh, on="nearest_id", how="left")
    out[zcol] = out["z_sh"]
    return out.drop(columns=["z_sh"])


def main() -> None:
    train = attach(build_gb_dataset([TRAIN_BALLS], bases_empty=True,
                                    alignment="Standard"), TRAIN_PROXY)
    val = attach(build_gb_dataset([VAL_BALLS], bases_empty=True,
                                  alignment="Standard"), VAL_PROXY)

    results = []
    for zcol in ("z_overall", "z_lateral"):
        for g in GAMMAS:
            ll, auc = fit_eval(scaled(train, g, zcol), scaled(val, g, zcol))
            results.append({"proxy": zcol, "gamma": g,
                            "val_logloss": ll, "val_auc": auc})
            print(f"  {zcol:<10} γ={g:<5} val logloss={ll:.5f}  AUC={auc:.4f}",
                  flush=True)
    res = pd.DataFrame(results)
    base_ll = res[res["gamma"] == 0].iloc[0]["val_logloss"]
    res["d_logloss"] = res["val_logloss"] - base_ll
    print("\n=== 2025 驗證總表（Δlogloss<0 = 比 γ=0 好）===")
    print(res.round(5).to_string(index=False))

    best = res.loc[res["val_logloss"].idxmin()]
    print(f"\n最佳：proxy={best['proxy']} γ={best['gamma']} "
          f"Δlogloss={best['d_logloss']:+.5f}")
    if best["gamma"] == 0:
        print("γ=0 最佳 → OAA proxy 也無訊號，路線 A 收案")
        return

    print("\n=== 安慰劑（z 同位置內重排，5 個 seed）===")
    for s in range(5):
        ll, auc = fit_eval(
            scaled(shuffle_z(train, best["proxy"], SEED + s), best["gamma"], best["proxy"]),
            scaled(shuffle_z(val, best["proxy"], SEED + s), best["gamma"], best["proxy"]))
        print(f"  seed {SEED + s}: Δlogloss={ll - base_ll:+.5f}  AUC={auc:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
