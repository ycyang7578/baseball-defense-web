"""共用：解析 Baseball Savant leaderboard 頁面內嵌的 `var data = [...]` JSON。

fetch_savant_fielding.py／fetch_oaa_leaderboard.py（outfield_directional_outs_
average）與 fetch_if_oaa_leaderboard.py（outs_above_average，pos=if）打的是不同
端點、不同參數，但頁面格式一樣：都是非官方端點，回傳內嵌在 `var data = [...]`
的 JSON。抓取＋解析這段共用，端點/參數與欄位整理留在各自檔案。
"""
import json
import re

import pandas as pd
import requests

BASE_URL: str = "https://baseballsavant.mlb.com"
HEADERS: dict[str, str] = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_leaderboard_raw(endpoint: str, params: dict[str, str | int]) -> pd.DataFrame:
    """打指定 leaderboard 端點，回傳未整理欄位的原始 JSON。"""
    r = requests.get(
        f"{BASE_URL}/leaderboard/{endpoint}",
        params=params, headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    match = re.search(r"var data\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not match:
        raise ValueError("找不到 leaderboard var data，頁面結構可能已更新")
    return pd.DataFrame(json.loads(match.group(1)))
