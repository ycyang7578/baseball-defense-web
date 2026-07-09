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


# ── 內野（結果全部離線預算，見 scripts/precompute_if_optimize.py）─────

class IFBatterInfo(BaseModel):
    batter_id: int
    name:      str
    n_gb:      int
    stand:     str


class IFPosition(BaseModel):
    x:     float   # 呎，本壘原點，+x 朝一壘側
    y:     float
    angle: float   # 度，0=正對中外野，+朝一壘側
    depth: float   # 呎


class IFPositionSet(BaseModel):
    positions: dict[str, IFPosition]   # keys: "1B"/"2B"/"3B"/"SS"
    exp_outs:  float                   # 期望出局率（打者歷史滾地球平均 P(out)）


class IFBallPoint(BaseModel):
    spray_deg:    float
    x:            float   # 呎，Statcast 記錄的處理/撿球位置（展示用，非落點）
    y:            float
    launch_speed: float
    is_out:       bool
    p_out_league: float
    p_out_opt:    float


class IFStats(BaseModel):
    n_gb:         int
    gain:         float   # exp_outs_opt − exp_outs_league
    outs_per_450: float   # gain × 450（一季規模的滾地球數）
    hp_to_1b:     float


class IFResultResponse(BaseModel):
    batter_id: int
    name:      str
    year:      int
    stand:     str
    league:    IFPositionSet
    optimized: IFPositionSet
    balls:     list[IFBallPoint]
    stats:     IFStats


class IFFielderInfo(BaseModel):
    name:           str
    player_id:      int
    team_id:        int | None = None
    oaa:            float          # model OAA（分位置中心化）
    n_balls:        int
    official_oaa:   int | None = None
    official_n_opp: int | None = None
