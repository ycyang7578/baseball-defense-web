"""共用：分類機率預測的校準表。

train_if_gb.py 與 train_if_on1b.py 都要看「預測機率十分位 vs 實際發生率」
偏差多大，原本各自複製一份同樣的 qcut + groupby 邏輯，抽成這裡的
calibration_table。
"""
import pandas as pd


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
