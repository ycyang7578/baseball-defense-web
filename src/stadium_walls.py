"""
球場圍牆多邊形載入與打牆球判定。

從 MLBStadiaPathData（GeomMLBStadiums R 套件）載入各球場外野圍牆邊界，
判斷球的落點是否在該球場 polygon 外（= 打牆球，外野手無法接殺）。

座標系：以本壘為原點，y 軸朝中外野，單位 feet。
  x_ft = (pixel_x - 125.42) * 2.484
  y_ft = (198.27  - pixel_y) * 2.484
"""
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyreadr
from shapely.geometry import Point, Polygon

from .physics import _STATCAST_ORIGIN_X, _STATCAST_ORIGIN_Y


class ParkBoundaryPoint(TypedDict):
    """球場外野圍牆多邊形的單一頂點座標（feet），供前端 SVG 繪製。"""
    x: float
    y: float


_SCALE: float = 2.484

_RDA_PATH: Path = Path(__file__).parent.parent / "data" / "reference" / "MLBStadiaPathData.rda"

_TEAM_MAP: dict[str, str] = {
    'LAA': 'angels',    'HOU': 'astros',     'OAK': 'athletics',
    'ATH': 'athletics', 'TOR': 'blue_jays',  'ATL': 'braves',
    'MIL': 'brewers',   'STL': 'cardinals',  'CHC': 'cubs',
    'AZ':  'diamondbacks', 'ARI': 'diamondbacks', 'LAD': 'dodgers',
    'SF':  'giants',    'CLE': 'guardians',  'SEA': 'mariners',
    'MIA': 'marlins',   'NYM': 'mets',       'WSH': 'nationals',
    'BAL': 'orioles',   'SD':  'padres',     'PHI': 'phillies',
    'PIT': 'pirates',   'TEX': 'rangers',    'TB':  'rays',
    'BOS': 'red_sox',   'CIN': 'reds',       'COL': 'rockies',
    'KC':  'royals',    'DET': 'tigers',     'MIN': 'twins',
    'CWS': 'white_sox', 'NYY': 'yankees',
}


@lru_cache(maxsize=1)
def _load_wall_polygons() -> dict[str, Polygon]:
    """載入 .rda，為每個球場建 Shapely Polygon。回傳 {team_name: Polygon}。"""
    if not _RDA_PATH.exists():
        raise FileNotFoundError(f"MLBStadiaPathData.rda 找不到：{_RDA_PATH}")

    raw = pyreadr.read_r(str(_RDA_PATH))
    df  = list(raw.values())[0]

    wall_df = df[df['segment'] == 'outfield_outer'].copy()
    wall_df['x_ft'] = (wall_df['x'].values - _STATCAST_ORIGIN_X) * _SCALE
    wall_df['y_ft'] = (_STATCAST_ORIGIN_Y - wall_df['y'].values) * _SCALE

    polygons: dict[str, Polygon] = {}
    for team, grp in wall_df.groupby('team'):
        coords = list(zip(grp['x_ft'], grp['y_ft']))
        if len(coords) >= 3:
            polygons[team] = Polygon(coords)
    return polygons


def _get_polygon(home_team: str) -> Polygon | None:
    polys = _load_wall_polygons()
    name  = _TEAM_MAP.get(home_team)
    return polys.get(name) if name else None


def is_wall_ball(x_coord: np.ndarray, y_coord: np.ndarray,
                 home_team: str) -> np.ndarray:
    """
    判斷每顆球是否在 home_team 球場圍牆外（打牆球）。

    Returns bool array, shape (N,).
    True = 落點在圍牆多邊形外，外野手無法接殺。
    找不到球場資料時全部回傳 False（不排除任何球）。
    """
    poly = _get_polygon(home_team)
    if poly is None:
        return np.zeros(len(x_coord), dtype=bool)

    flags = np.zeros(len(x_coord), dtype=bool)
    for i, (x, y) in enumerate(zip(x_coord, y_coord)):
        if y > 0:
            flags[i] = not poly.contains(Point(x, y))
    return flags


def get_park_boundary_coords(home_team: str) -> list[ParkBoundaryPoint] | None:
    """
    回傳球場外野圍牆多邊形的頂點座標（feet），供前端 SVG 繪製。
    找不到球場時回傳 None。
    格式：[{"x": float, "y": float}, ...]
    """
    poly = _get_polygon(home_team)
    if poly is None:
        return None
    coords = poly.exterior.coords
    return [ParkBoundaryPoint(x=round(float(x), 1), y=round(float(y), 1)) for x, y in coords]


SUPPORTED_TEAMS: list[str] = sorted(_TEAM_MAP.keys())
