"""共用：外野 Model OAA vs 官方 OAA 驗證用的資料計算。

make_validation_plot.py 與 make_validation_plot_v2.py 畫不同版面的驗證散點圖
（v2 刻意不改 v1、只是換版面設計，見 v2 docstring），但抓資料／算 model OAA／
查官方 OAA／名字標準化這段完全相同，抽出來避免兩份定義互相漂移。
"""
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import psycopg2

from src.defender_features import get_defender_opportunities, mark_official

POSITIONS = ["LF", "CF", "RF"]
FEATURE_COLS = ["speed", "cos_angle", "sin_angle", "fielder_dist"]


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_of_model(models_dir):
    """回傳統一 OF 模型的 (scaler, mu)，mu 為群體層後驗平均（Series）。"""
    of_dir = models_dir / "OF"
    scaler = joblib.load(of_dir / "OF_scaler.joblib")
    mu = pd.read_csv(of_dir / "OF_summary_group.csv", encoding="utf-8-sig", index_col=0)["mean"]
    return scaler, mu


def compute_model_oaa(pos: str, year: int, scaler, mu) -> pd.DataFrame:
    """單一位置逐球 model OAA（is_official 子集，群體層 mu，絕不用 player-level）。"""
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


def load_model_oaa(models_dir, year: int) -> pd.DataFrame:
    """跨 LF+CF+RF 加總逐球 oaa_play → 每位球員一筆，含標準化姓名 key。"""
    scaler, mu = load_of_model(models_dir)
    all_plays = pd.concat(
        [compute_model_oaa(p, year, scaler, mu) for p in POSITIONS], ignore_index=True)
    model_oaa = all_plays.groupby("name_fielder", as_index=False)["oaa_play"].sum()
    model_oaa["key"] = model_oaa["name_fielder"].apply(normalize_name)
    return model_oaa


def load_official_oaa(dsn: str, year: int) -> pd.DataFrame:
    """官方 OAA（is_qualified=True，同一球員可能多位置達標，加總視為整體外野 OAA）。"""
    with psycopg2.connect(dsn) as conn:
        raw_off = pd.read_sql(
            "SELECT player_name, oaa FROM oaa_leaderboard "
            "WHERE year = %(y)s AND is_qualified = TRUE",
            conn, params={"y": year})
    official = raw_off.groupby("player_name", as_index=False)["oaa"].sum()
    official["key"] = official["player_name"].apply(normalize_name)
    return official
