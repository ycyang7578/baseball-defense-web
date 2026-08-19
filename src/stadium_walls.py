"""
Loading of park outfield-wall polygons and wall-ball detection.

Loads each park's outfield wall boundary from MLBStadiaPathData (the GeomMLBStadiums R package), and
determines whether a ball's landing spot is outside that park's polygon (= a wall ball, one the outfielders
cannot catch).

Coordinate system: home plate as the origin, y-axis toward center field, units in feet.
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
    """A single vertex coordinate (feet) of a park's outfield wall polygon, for frontend SVG rendering."""
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
    """Loads the .rda file and builds a Shapely Polygon for each park. Returns {team_name: Polygon}."""
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
    Determines whether each ball lands outside home_team's park wall (a wall ball).

    Returns bool array, shape (N,).
    True = landing spot is outside the wall polygon, outfielders cannot catch it.
    Returns all False when park data isn't found (no balls excluded).
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
    Returns the vertex coordinates (feet) of a park's outfield wall polygon, for frontend SVG rendering.
    Returns None if the park isn't found.
    Format: [{"x": float, "y": float}, ...]
    """
    poly = _get_polygon(home_team)
    if poly is None:
        return None
    coords = poly.exterior.coords
    return [ParkBoundaryPoint(x=round(float(x), 1), y=round(float(y), 1)) for x, y in coords]


SUPPORTED_TEAMS: list[str] = sorted(_TEAM_MAP.keys())
