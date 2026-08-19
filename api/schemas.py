from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    batter_id: int
    year: int = 2025
    on_1b: int = Field(0, ge=0, le=1)
    on_2b: int = Field(0, ge=0, le=1)
    on_3b: int = Field(0, ge=0, le=1)
    outs:  int = Field(0, ge=0, le=2)
    home_team: str | None = None
    # Specified outfielders (player-level ability). Key is "LF"/"CF"/"RF", value is the player name;
    # unspecified positions use the league average (group mu). If any position is specified → the chart only shows this position set.
    fielders: dict[str, str] | None = None


class FielderInfo(BaseModel):
    name:      str
    oaa:       float | None = None
    n_opp:     int   | None = None
    player_id: int   | None = None
    team_id:   int   | None = None
    official_oaa:   int | None = None   # Official Statcast OAA (for cross-referencing on the unified rankings page)
    official_n_opp: int | None = None


class PositionXY(BaseModel):
    x: float
    y: float


class PositionSet(BaseModel):
    LF: PositionXY
    CF: PositionXY
    RF: PositionXY
    objective: float
    catch_pct: float     # average catch probability, mean(p̂_j)


class BallPoint(BaseModel):
    x:            float
    y:            float
    catch_prob:   float
    is_wall_ball: bool
    responsible:  str | None = None   # 'LF'/'CF'/'RF', None = catch probability below 5%
    bb_type:      str | None = None   # 'fly_ball'/'line_drive' (for batted-ball-type filtering on the integrated page)


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


# ── Infield (results are all precomputed offline, see scripts/precompute_if_optimize.py) ─────

class IFBatterInfo(BaseModel):
    batter_id: int
    name:      str
    n_gb:      int
    stand:     str


class IFPosition(BaseModel):
    x:     float   # feet, home plate as origin, +x toward the 1B side
    y:     float
    angle: float   # degrees, 0=facing straight to center field, + toward the 1B side
    depth: float   # feet


class IFPositionSet(BaseModel):
    positions: dict[str, IFPosition]   # keys: "1B"/"2B"/"3B"/"SS"
    exp_outs:  float                   # expected out rate (average P(out) over the batter's historical ground balls)


class IFBallPoint(BaseModel):
    spray_deg:    float
    x:            float   # feet, the fielding/pickup location recorded by Statcast (for display, not the landing spot)
    y:            float
    launch_speed: float
    is_out:       bool
    p_out_league: float
    p_out_opt:    float


class IFStats(BaseModel):
    n_gb:         int
    gain:         float   # exp_outs_opt − exp_outs_league
    outs_per_450: float   # gain × 450 (a season-scale ground ball count)
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
    oaa:            float          # model OAA (centered by position)
    n_balls:        int
    official_oaa:   int | None = None
    official_n_opp: int | None = None


class IFFielderOption(BaseModel):
    """A fielder menu entry (for personalized positioning). has_effects=False means there's
    no player-level estimate, so selecting them is equivalent to the league average. oaa/n_balls
    come from if_model_oaa (that year's/position's rating); no record means None
    (used for the frontend label and the minimum-opportunities slider)."""
    player_id:   int
    name:        str
    has_effects: bool
    oaa:         float | None = None
    n_balls:     int   | None = None


# ── Infield live optimization (infield mirror of outfield /api/optimize: base state + fielders, run-value pricing) ──

class IFOptimizeRequest(BaseModel):
    batter_id: int
    year: int = 2025
    on_1b: int = Field(0, ge=0, le=1)
    on_2b: int = Field(0, ge=0, le=1)
    on_3b: int = Field(0, ge=0, le=1)
    outs:  int = Field(0, ge=0, le=2)
    # Specified infielders (player-level effects). Key is "1B"/"2B"/"3B"/"SS", value is player_id;
    # unspecified positions are treated as a league-average fielder (effect 0).
    fielders: dict[str, int] | None = None


class IFOptimizeSet(BaseModel):
    positions: dict[str, IFPosition]   # keys: "1B"/"2B"/"3B"/"SS"
    exp_outs:  float                   # expected out rate (average P(out) over the batter's ground balls)
    runs:      float                   # expected runs, E[ΔRE]×n_gb (for this base state, lower is better)


class IFOptimizeStats(BaseModel):
    n_gb:          int
    re_state:      float
    hp_to_1b:      float
    gain_outs:     float   # out-rate gain (optimized − league average)
    outs_per_450:  float
    runs_saved:    float   # runs saved (league-average runs − optimized runs, positive = better)
    runs_per_450:  float   # runs saved per 450 ground balls


class IFOptimizeResponse(BaseModel):
    batter_id: int
    name:      str
    year:      int
    stand:     str
    situation: str
    fielders:  dict[str, str | None]   # pos → fielder name (None = league average)
    league:    IFOptimizeSet
    optimized: IFOptimizeSet
    balls:     list[IFBallPoint]
    stats:     IFOptimizeStats


# ── Infield/outfield integration (unified pricing = expected runs, see ARCHITECTURE.md "Infield/outfield integration route") ──

class IntegratedRequest(BaseModel):
    batter_id: int
    year: int = 2025
    on_1b: int = Field(0, ge=0, le=1)
    on_2b: int = Field(0, ge=0, le=1)
    on_3b: int = Field(0, ge=0, le=1)
    outs:  int = Field(0, ge=0, le=2)
    # Park (outfield wall factor, same as /api/optimize: wall-ball catch probability forced to 0 +
    # a second warm-start optimization pass). None = generic park. Infield has no wall, so it's unaffected.
    home_team: str | None = None
    # Specified fielders (unspecified positions = league average). Outfield uses player names (same as
    # /api/optimize, the key used by load_player_params); infield uses player_id (same as the Bayesian
    # player-level effects used by /api/if_optimize).
    of_fielders: dict[str, str] | None = None   # "LF"/"CF"/"RF" → player name
    if_fielders: dict[str, int] | None = None   # "1B"/"2B"/"3B"/"SS" → player_id


class IntegratedSet(BaseModel):
    """A seven-fielder position set and its expected runs (summed over the batter's batted balls that season, lower is better).

    Full ΔRE accounting (same scale for infield and outfield): runs_of = Σ[(1−p̂)×w_j＋p̂×ΔRE(out)],
    runs_if = E[ΔRE]×n_gb. An out drives ΔRE negative, so the infield side is usually negative
    (a gain for the defense). The optimization is separable (ground balls go to the infield, fly balls
    to the outfield), so runs_total = summing both sides gives the joint accounting.
    The league set = average positions + average parameters; the optimized set carries the specified fielders' parameters."""
    positions:  dict[str, PositionXY]   # keys: LF/CF/RF/1B/2B/3B/SS
    catch_pct:  float = 0.0             # outfield average catch probability % (wall balls counted as 0)
    exp_outs_if: float = 0.0            # infield ground-ball average P(out)
    runs_of:    float
    runs_if:    float
    runs_total: float


class PopupBall(BaseModel):
    """Infield popup for display only (not part of the optimization — positioning has no leverage, out rate is 98.6%)."""
    x:      float
    y:      float
    is_out: bool


class IntegratedStats(BaseModel):
    n_of_balls:       int
    n_gb:             int
    n_popups:         int = 0  # for display only, not included in any runs calculation
    n_wall_balls:     int = 0  # wall-ball count when a park is specified (catch probability forced to 0)
    home_team:        str | None = None
    re_state:         float
    runs_saved_of:    float   # league − optimized (positive = optimized saves more runs)
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
    of_balls:  list[BallPoint]     # catch_prob is the catch probability under the optimized positioning
    if_balls:  list[IFBallPoint]
    popup_balls: list[PopupBall] = []  # for display only (empty if the cloud table isn't synced)
    park_boundary: list[ParkCoord] | None = None  # wall outline when a park is specified
    fielders:  dict[str, str | None] = {}  # seven positions → fielder name (None = league average)
    stats:     IntegratedStats


class IntegratedBatterInfo(BaseModel):
    """A batter menu entry for the integrated page. n_total = ground balls + outfield fly balls/line drives + popups (all balls that appear in the chart)."""
    batter_id: int
    name:      str
    n_total:   int
    n_gb:      int


class IFCustomResultResponse(BaseModel):
    """Personalized result for a specified fielder lineup. optimized is the anchored solution (warm-started
    from the zero-effect optimum). Comparison baseline = average positions + average parameters: p_out_league / league
    is evaluated under zero effect, while p_out_opt / optimized carries the lineup's effects."""
    batter_id:         int
    name:              str
    year:              int
    stand:             str
    fielders:          dict[str, str | None]   # pos → fielder name (None = league average)
    league:            IFPositionSet           # league-average positions (evaluated under the lineup's effects)
    optimized:         IFPositionSet           # anchored personalized optimum
    baseline_exp_outs: float                   # expected out rate of the zero-effect optimal positioning, evaluated under the lineup's effects
    balls:             list[IFBallPoint]
    stats:             IFStats
