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


class IFFielderOption(BaseModel):
    """野手選單項（個人化站位用）。has_effects=False 表示無球員層估計，
    選了等同聯盟平均。oaa/n_balls 來自 if_model_oaa（該年該位置的評價），
    無紀錄則為 None（前端標籤與最低守備次數滑桿用）。"""
    player_id:   int
    name:        str
    has_effects: bool
    oaa:         float | None = None
    n_balls:     int   | None = None


# ── 內野線上優化（外野 /api/optimize 的內野鏡像：壘況+野手，run-value 計價）──

class IFOptimizeRequest(BaseModel):
    batter_id: int
    year: int = 2025
    on_1b: int = Field(0, ge=0, le=1)
    on_2b: int = Field(0, ge=0, le=1)
    on_3b: int = Field(0, ge=0, le=1)
    outs:  int = Field(0, ge=0, le=2)
    # 指定內野手（球員層效應）。key 為 "1B"/"2B"/"3B"/"SS"，value 為 player_id；
    # 未指定的位置視為聯盟平均野手（效應 0）。
    fielders: dict[str, int] | None = None


class IFOptimizeSet(BaseModel):
    positions: dict[str, IFPosition]   # keys: "1B"/"2B"/"3B"/"SS"
    exp_outs:  float                   # 期望出局率（打者滾地球平均 P(out)）
    runs:      float                   # 期望失分 E[ΔRE]×n_gb（該壘況，越低越好）


class IFOptimizeStats(BaseModel):
    n_gb:          int
    re_state:      float
    hp_to_1b:      float
    gain_outs:     float   # 出局率增益（最佳化 − 聯盟平均）
    outs_per_450:  float
    runs_saved:    float   # 省分（聯盟平均 runs − 最佳化 runs，正=較省）
    runs_per_450:  float   # 每 450 顆滾地球的省分


class IFOptimizeResponse(BaseModel):
    batter_id: int
    name:      str
    year:      int
    stand:     str
    situation: str
    fielders:  dict[str, str | None]   # pos → 野手名（None=聯盟平均）
    league:    IFOptimizeSet
    optimized: IFOptimizeSet
    balls:     list[IFBallPoint]
    stats:     IFOptimizeStats


# ── 內外野整合（統一計價=期望失分，見 ARCHITECTURE.md「內外野整合路線」）──

class IntegratedRequest(BaseModel):
    batter_id: int
    year: int = 2025
    on_1b: int = Field(0, ge=0, le=1)
    on_2b: int = Field(0, ge=0, le=1)
    on_3b: int = Field(0, ge=0, le=1)
    outs:  int = Field(0, ge=0, le=2)


class IntegratedSet(BaseModel):
    """一組七人站位與其期望失分（打者該季擊球加總，越低越好）。

    runs_of = Σ(1−p̂)×w_j（外野球）；runs_if = E[ΔRE]×n_gb（滾地球）。
    優化可分離（滾地歸內野/飛球歸外野），runs_total = 兩側相加即聯合口徑。"""
    positions:  dict[str, PositionXY]   # keys: LF/CF/RF/1B/2B/3B/SS
    runs_of:    float
    runs_if:    float
    runs_total: float


class PopupBall(BaseModel):
    """展示用內野高飛（不參與優化——站位無槓桿，出局率 98.6%）。"""
    x:      float
    y:      float
    is_out: bool


class IntegratedStats(BaseModel):
    n_of_balls:       int
    n_gb:             int
    n_popups:         int = 0  # 展示用，不在任何 runs 計算內
    re_state:         float
    runs_saved_of:    float   # league − optimized（正=最佳化較省分）
    runs_saved_if:    float
    runs_saved_total: float


class IntegratedResponse(BaseModel):
    batter_id: int
    name:      str
    year:      int
    stand:     str
    situation: str
    league:    IntegratedSet
    optimized: IntegratedSet
    of_balls:  list[BallPoint]     # catch_prob 為最佳化站位下的接殺機率
    if_balls:  list[IFBallPoint]
    popup_balls: list[PopupBall] = []  # 展示用（雲端表未 sync 時為空）
    stats:     IntegratedStats


class IFCustomResultResponse(BaseModel):
    """指定野手陣容的個人化結果。optimized 為錨定式解（從零效應最佳解
    warm start），balls 的 p_out_* 皆在該陣容效應下評估。"""
    batter_id:         int
    name:              str
    year:              int
    stand:             str
    fielders:          dict[str, str | None]   # pos → 野手名（None=聯盟平均）
    league:            IFPositionSet           # 聯盟平均站位（陣容效應下評估）
    optimized:         IFPositionSet           # 錨定式個人化最佳解
    baseline_exp_outs: float                   # 零效應最佳站位在陣容效應下的期望出局率
    balls:             list[IFBallPoint]
    stats:             IFStats
