from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    batter_id: int
    year: int = 2025
    on_1b: int = Field(0, ge=0, le=1)
    on_2b: int = Field(0, ge=0, le=1)
    on_3b: int = Field(0, ge=0, le=1)
    outs:  int = Field(0, ge=0, le=2)
    home_team: str | None = None
    # 指定外野手（player-level 能力）。key 為 "LF"/"CF"/"RF"，value 為球員名；
    # 未指定的位置用聯盟平均（group mu）。任一位置有指定 → 圖只顯示這組站位。
    fielders: dict[str, str] | None = None


class FielderInfo(BaseModel):
    name:      str
    oaa:       float | None = None
    n_opp:     int   | None = None
    player_id: int   | None = None
    team_id:   int   | None = None


class PositionXY(BaseModel):
    x: float
    y: float


class PositionSet(BaseModel):
    LF: PositionXY
    CF: PositionXY
    RF: PositionXY
    objective: float
    catch_pct: float     # 平均接殺率 mean(p̂_j)


class BallPoint(BaseModel):
    x:            float
    y:            float
    catch_prob:   float
    is_wall_ball: bool
    responsible:  str | None = None   # 'LF'/'CF'/'RF'，None = 接殺機率不足 5%


class ParkCoord(BaseModel):
    x: float
    y: float


class OptimizeStats(BaseModel):
    n_balls:      int
    n_wall_balls: int
    re_state:     float
    home_team:    str | None = None


class OptimizeResponse(BaseModel):
    title:         str
    situation:     str
    positions:     dict[str, PositionSet]   # keys: "no_park", "with_park"(optional), "league_avg"
    balls:         list[BallPoint]
    park_boundary: list[ParkCoord] | None
    stats:         OptimizeStats



class BatterInfo(BaseModel):
    batter_id: int
    name:      str
    n_balls:   int
