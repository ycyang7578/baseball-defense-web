"""Shared: calibration table for classification probability predictions.

Both train_if_gb.py and train_if_on1b.py need to check how much "predicted
probability decile vs. actual occurrence rate" deviates. Each script used to
duplicate the same qcut + groupby logic; it has been factored out here into
calibration_table.
"""
import numpy as np
import pandas as pd


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
