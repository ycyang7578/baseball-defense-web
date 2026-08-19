"""Shared: parses the `var data = [...]` JSON embedded in Baseball Savant
leaderboard pages.

fetch_savant_fielding.py / fetch_oaa_leaderboard.py (outfield_directional_outs_
average) and fetch_if_oaa_leaderboard.py (outs_above_average, pos=if) hit
different endpoints with different params, but the page format is the same: all
are unofficial endpoints that return JSON embedded in `var data = [...]`. The
fetch + parse step is shared here; endpoint/params and column cleanup stay in
each individual file.
"""
import json
import re

import pandas as pd
import requests

BASE_URL: str = "https://baseballsavant.mlb.com"
HEADERS: dict[str, str] = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_leaderboard_raw(endpoint: str, params: dict[str, str | int]) -> pd.DataFrame:
    """Hit the given leaderboard endpoint and return the raw JSON with columns unprocessed."""
    r = requests.get(
        f"{BASE_URL}/leaderboard/{endpoint}",
        params=params, headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    match = re.search(r"var data\s*=\s*(\[.*?\]);", r.text, re.DOTALL)
    if not match:
        raise ValueError("找不到 leaderboard var data，頁面結構可能已更新")
    return pd.DataFrame(json.loads(match.group(1)))
