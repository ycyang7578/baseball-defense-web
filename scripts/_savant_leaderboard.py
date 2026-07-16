"""共用：解析 Baseball Savant leaderboard 頁面內嵌的 `var data = [...]` JSON。

fetch_savant_fielding.py 與 fetch_oaa_leaderboard.py 都打
outfield_directional_outs_above_average 這個非官方端點、格式相同，
只是各自取用/整理的欄位不同，因此共用抓取＋解析這段，欄位整理留在各自檔案。
"""
import json
import re

import pandas as pd
import requests

BASE_URL = "https://baseballsavant.mlb.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_outfield_leaderboard_raw(year: int) -> pd.DataFrame:
    """打 outfield_directional_outs_above_average 端點，回傳未整理欄位的原始 JSON。"""
    r = requests.get(
        f"{BASE_URL}/leaderboard/outfield_directional_outs_above_average",
        params={"year": year}, headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    match = re.search(r"var data\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not match:
        raise ValueError("找不到 leaderboard var data，頁面結構可能已更新")
    return pd.DataFrame(json.loads(match.group(1)))
