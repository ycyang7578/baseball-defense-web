"""
Outfield position optimizer using the RE24 objective function.

Objective: θ*(s) = argmin_θ  Σ_j (1 - p̂_j(θ)) × w_j(s)

  p̂_j(θ)  = 1 - ∏_i (1 - p_ij(θ))      joint catch probability (LF × CF × RF)
  w_j(s)  = Σ_k P(k|j) × ΔRE(k, s)     expected RE24 cost of missing ball j

Minimizing this objective = minimizing expected runs allowed.

Usage (high-level):
  from src.optimization import optimize_positions

  result = optimize_positions(
      batter_id=660271,
      on_1b=1, on_2b=0, on_3b=0, outs=0,
      years=[2021, 2022, 2023, 2024],
      models_dir=Path("models/2025"),
      re24_dir=Path("data/precomputed"),
  )
  # result = {'LF': (x, y), 'CF': (x, y), 'RF': (x, y), 'objective': float}
"""
import warnings
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from scipy.special import expit as _expit
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler

# The model scaler was fit on a DataFrame but is called with numpy arrays; suppress the known-harmless warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

from .config import DSN
from .hit_prob import HitProbBundle, load_hit_prob, predict_hit_probs_batch
from .re24 import BaseOutState, HitDeltaKey, load_re24

POSITIONS: tuple[str, ...] = ("LF", "CF", "RF")
FEATURE_COLS: list[str] = ["speed", "cos_angle", "sin_angle", "fielder_dist"]

# {"LF": (x, y), "CF": (x, y), "RF": (x, y)} -- the standard positioning coordinate format used in this file
OutfieldXY = dict[str, tuple[float, float]]


class GroupMu(TypedDict):
    """Group-level (or player-level override) posterior mean coefficients for the outfield catch-probability model."""
    mu_alpha: float
    mu_beta_speed: float
    mu_beta_cos: float
    mu_beta_sin: float
    mu_beta_dist: float


class ObjectiveContext(TypedDict):
    """Objective-function context for objective_re24: precomputed data that doesn't change with candidate positioning."""
    ball_x: np.ndarray
    ball_y: np.ndarray
    flight_time: np.ndarray
    w_j: np.ndarray
    sc_means: dict[str, np.ndarray]
    sc_scales: dict[str, np.ndarray]
    mus: dict[str, GroupMu]


class OptimizeResult(TypedDict):
    LF: tuple[float, float]
    CF: tuple[float, float]
    RF: tuple[float, float]
    objective: float
    n_balls: int
    n_wall_balls: int


class QualifyingBatter(TypedDict):
    batter_id: int
    n_balls: int


# ── Polar-coordinate sector search bounds ─────────────────────────────────────────────
# Variable layout: [r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF]
# θ is measured from the y-axis (center-field direction = 0°), positive toward right field,
# negative toward left field, units: degrees
# The sector angles ensure fielders stay in fair territory (foul line ≈ ±45°)
_POLAR_BOUNDS: list[tuple[float, float]] = [
    (150, 400), (-45.0,   0.0),   # LF: r(ft), θ(deg)
    (150, 400), (-22.5, +22.5),   # CF: r(ft), θ(deg)
    (150, 400),  (0.0,  +45.0),   # RF: r(ft), θ(deg)
]


def _polar_to_xy(params: np.ndarray) -> np.ndarray:
    """[r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF] -> [x_LF, y_LF, x_CF, y_CF, x_RF, y_RF]

    x = r·sin(θ),  y = r·cos(θ)   (same convention as physics.polar_to_fielder_xy)
    """
    out = np.empty(6)
    for i in range(3):
        radius_ft = params[2 * i]
        theta_rad = np.radians(params[2 * i + 1])
        out[2 * i]     = radius_ft * np.sin(theta_rad)   # x
        out[2 * i + 1] = radius_ft * np.cos(theta_rad)   # y
    return out


def _xy_to_polar_params(xy: OutfieldXY) -> np.ndarray:
    """{'LF':(x,y),'CF':(x,y),'RF':(x,y)} -> [r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF] (inverse of _polar_to_xy, used for warm start)"""
    out = np.empty(6)
    for i, pos in enumerate(POSITIONS):
        x, y = xy[pos]
        out[2 * i]     = np.hypot(x, y)
        out[2 * i + 1] = np.degrees(np.arctan2(x, y))
    return out

# Queries the lean precomputed table (produced by scripts/precompute_batter_balls.py) instead of hitting the
# 5GB statcast table directly.
# The old version that queried statcast directly and computed the physics formulas can be found in git
# history (prepare_batter_balls, prior to 2026-07).
_BATTER_QUERY = """
    SELECT ball_x, ball_y, flight_time, launch_speed, launch_angle, spray_angle, stand, bb_type
    FROM precomputed_batter_balls
    WHERE batter = %(batter_id)s AND game_year = ANY(%(years)s)
"""


# ── Model parameter loading ────────────────────────────────────────────────────

def _resolve_model_dir(pos: str, models_dir: Path) -> tuple[Path, str]:
    """Return (dir, prefix) for the model files. Falls back to unified OF if pos-specific missing."""
    pos_dir = Path(models_dir) / pos
    if (pos_dir / f"{pos}_scaler.joblib").exists():
        return pos_dir, pos
    # unified OF model fallback (2021-2024 only have OF/)
    return Path(models_dir) / "OF", "OF"


def load_model_params(pos: str, models_dir: Path) -> tuple[StandardScaler, GroupMu]:
    """
    Returns (scaler, mu_dict) for position pos.
    mu_dict keys: mu_alpha, mu_beta_speed, mu_beta_cos, mu_beta_sin, mu_beta_dist
    """
    pos_dir, prefix = _resolve_model_dir(pos, models_dir)
    scaler = joblib.load(pos_dir / f"{prefix}_scaler.joblib")
    group = pd.read_csv(pos_dir / f"{prefix}_summary_group.csv", encoding="utf-8-sig", index_col=0)
    mu = group["mean"]
    mu_dict = GroupMu(
        mu_alpha=float(mu["mu_alpha"]),
        mu_beta_speed=float(mu["mu_beta_speed"]),
        mu_beta_cos=float(mu["mu_beta_cos"]),
        mu_beta_sin=float(mu["mu_beta_sin"]),
        mu_beta_dist=float(mu["mu_beta_dist"]),
    )
    return scaler, mu_dict


def load_player_params(pos: str, player_name: str, models_dir: Path) -> GroupMu:
    """
    Loads player-level parameters for the given player, returning a dict with the same key format as group mu
    (reusing the shared scaler for that position). Used for positioning optimization with a specified outfielder.
    """
    pos_dir, prefix = _resolve_model_dir(pos, models_dir)
    players = pd.read_csv(pos_dir / f"{prefix}_summary_players.csv", index_col=0, encoding="utf-8-sig")

    def player_param(param: str) -> float:
        return float(players.loc[f"{param}[{player_name}]", "mean"])

    return GroupMu(
        mu_alpha=player_param("alpha"),
        mu_beta_speed=player_param("beta_speed"),
        mu_beta_cos=player_param("beta_cos"),
        mu_beta_sin=player_param("beta_sin"),
        mu_beta_dist=player_param("beta_dist"),
    )


# ── Batter historical ball data preparation ─────────────────────────────────────────────

def prepare_batter_balls(
    batter_id: int,
    years: list[int],
    dsn: str = DSN,
) -> pd.DataFrame:
    """
    Query a batter's historical fly balls / line drives (precomputed physics features).

    Returns DataFrame with columns:
      ball_x, ball_y, flight_time, launch_speed, launch_angle, spray_angle, stand
    """
    with psycopg2.connect(dsn) as conn:
        df = pd.read_sql(
            _BATTER_QUERY, conn,
            params={"batter_id": batter_id, "years": years},
        )

    if df.empty:
        return df

    return df[["ball_x", "ball_y", "flight_time",
               "launch_speed", "launch_angle", "spray_angle", "stand",
               "bb_type"]].reset_index(drop=True)


# ── w_j computation ──────────────────────────────────────────────────────

def compute_w_j(
    balls: pd.DataFrame,
    hit_bundle: HitProbBundle,
    delta_re: dict[HitDeltaKey, float],
    on_1b: int,
    on_2b: int,
    on_3b: int,
    outs: int,
    hit_probs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute w_j = Σ_k P(k|j) × ΔRE(k, s) for each ball j.

    hit_probs: pre-computed (N,3) array from predict_hit_probs_batch; pass to skip KDE.
    Returns array of shape (N,).
    """
    # P(k|j): shape (N, 3), column order = 1B, 2B, 3B
    if hit_probs is None:
        hit_probs = predict_hit_probs_batch(hit_bundle, balls)

    state: BaseOutState = (on_1b, on_2b, on_3b, outs)
    dre_1b = delta_re.get(("1B", *state), 0.0)
    dre_2b = delta_re.get(("2B", *state), 0.0)
    dre_3b = delta_re.get(("3B", *state), 0.0)

    w_j = (hit_probs[:, 0] * dre_1b
           + hit_probs[:, 1] * dre_2b
           + hit_probs[:, 2] * dre_3b)
    return w_j


# ── Core objective function ───────────────────────────────────────────────────

def _catch_prob_single_fielder(
    fielder_x: float, fielder_y: float,
    ball_x: np.ndarray, ball_y: np.ndarray, flight_time: np.ndarray,
    sc_mean: np.ndarray, sc_scale: np.ndarray, mu: GroupMu,
) -> np.ndarray:
    """
    Vectorized catch probability for one fielder at (fielder_x, fielder_y) for N balls.
    Replicates physics.compute_relative_angle inline for speed.
    Returns array of shape (N,).
    sc_mean / sc_scale are pre-extracted from StandardScaler to avoid sklearn overhead.
    """
    dx = ball_x - fielder_x
    dy = ball_y - fielder_y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    speed = dist / flight_time                          # required speed ft/s

    run_angle = np.arctan2(dx, dy)                      # angle to ball (from +y axis)
    pos_angle = np.arctan2(fielder_x, fielder_y)        # fielder's radial direction
    rel = run_angle - (pos_angle + np.pi)               # relative to "away from home"
    rel = (rel + np.pi) % (2 * np.pi) - np.pi

    raw_features = np.column_stack([speed, np.cos(rel), np.sin(rel), dist])
    standardized_features = (raw_features - sc_mean) / sc_scale

    logit = (mu["mu_alpha"]
             + mu["mu_beta_speed"] * standardized_features[:, 0]
             + mu["mu_beta_cos"]   * standardized_features[:, 1]
             + mu["mu_beta_sin"]   * standardized_features[:, 2]
             + mu["mu_beta_dist"]  * standardized_features[:, 3])

    return 1.0 / (1.0 + np.exp(-logit))


def objective_re24(positions_flat: np.ndarray, ctx: ObjectiveContext) -> float:
    """
    RE24 objective: Σ_j (1 - p̂_j(θ)) × w_j

    positions_flat: [lf_x, lf_y, cf_x, cf_y, rf_x, rf_y]
    ctx keys: ball_x, ball_y, flight_time, w_j, sc_means, sc_scales, mus
    """
    lf_x, lf_y, cf_x, cf_y, rf_x, rf_y = positions_flat

    ball_x = ctx["ball_x"]
    ball_y = ctx["ball_y"]
    flight_time = ctx["flight_time"]

    # p_not_caught = ∏_i (1 - p_i), multiplied in one at a time
    p_not_caught = np.ones(len(ball_x))
    for pos, (fielder_x, fielder_y) in zip(POSITIONS, [(lf_x, lf_y), (cf_x, cf_y), (rf_x, rf_y)]):
        p_catch = _catch_prob_single_fielder(
            fielder_x, fielder_y, ball_x, ball_y, flight_time,
            ctx["sc_means"][pos], ctx["sc_scales"][pos], ctx["mus"][pos],
        )
        p_not_caught *= (1.0 - p_catch)

    return float(np.sum(p_not_caught * ctx["w_j"]))


# ── Main entry point ───────────────────────────────────────────────────────

def optimize_positions(
    batter_id: int,
    on_1b: int,
    on_2b: int,
    on_3b: int,
    outs: int,
    years: list[int],
    models_dir: Path,
    re24_dir: Path,
    home_team: str | None = None,
    hit_prob_dir: Path | None = None,
    dsn: str = DSN,
    seed: int = 42,
    n_restarts: int = 20,
    fielder_mus: dict[str, GroupMu] | None = None,
    balls: pd.DataFrame | None = None,
    hit_probs: np.ndarray | None = None,
    warm_start_xy: OutfieldXY | None = None,
    delta_re: dict[HitDeltaKey, float] | None = None,
    hit_bundle: HitProbBundle | None = None,
) -> OptimizeResult:
    """
    Compute optimal outfield positions for a given batter and game state.
    Uses L-BFGS-B with multiple random restarts.

    home_team: MLB team abbreviation (e.g. 'BOS', 'LAD').
               When provided, balls that would clear the wall at that park
               are excluded from the objective (they cannot be caught).

    warm_start_xy: optional {"LF":(x,y),"CF":(x,y),"RF":(x,y)}, typically the solution from another
        optimize_positions call (with a similar objective function), used to replace one of the random
        starting points (not an extra evaluation on top -- total evaluate calls still equal n_restarts,
        so there's no added compute cost), to speed up convergence.
        Typical usage: with_park warm-starts from the no_park solution, since the two differ only in
        whether wall balls are excluded, and the objective function is nearly identical.

    delta_re / hit_bundle: optional; if the caller already has cached versions (e.g. the globals loaded
        at api/main.py startup), pass them in directly to skip re-reading delta_re.json / re-running
        joblib.load(hit_type_kde.joblib). A single request often calls this function twice
        (no_park + with_park); if not passed in, each call re-reads the files independently.

    Returns:
        {
          'LF': (x, y), 'CF': (x, y), 'RF': (x, y),
          'objective': float,
          'n_balls': int,
          'n_wall_balls': int,
        }
    """
    from .stadium_walls import is_wall_ball

    if hit_prob_dir is None:
        hit_prob_dir = re24_dir

    # Load precomputed data (only reads files if the caller didn't supply cached versions)
    if delta_re is None:
        _, delta_re = load_re24(re24_dir)
    if hit_bundle is None:
        hit_bundle = load_hit_prob(hit_prob_dir)

    # Load model parameters: unified OF model -- LF/CF/RF share the same scaler and group-level parameters,
    # consistent with precompute_model_oaa and the API display layer (finalized 2026-07-13).
    # If position-specific directories (LF/CF/RF/) exist under models/{year}/, they're leftovers from an older
    # model generation -- deliberately not used here: previously _resolve_model_dir preferred the
    # position-specific directories, which caused the optimizer and the display layer to use two different
    # surfaces in 2025, and the ranking of the two positioning sets could come out reversed.
    # (When a specific outfielder is given, override that position with their player-level parameters)
    of_scaler, of_mu = load_model_params("OF", models_dir)
    scalers = {pos: of_scaler for pos in POSITIONS}
    mus = {pos: of_mu for pos in POSITIONS}
    if fielder_mus:
        for pos, m in fielder_mus.items():
            if m is not None:
                mus[pos] = m

    # Prepare batter ball data (skip the DB query if already supplied externally)
    if balls is None:
        balls = prepare_batter_balls(batter_id, years, dsn)
    if balls.empty:
        raise ValueError(f"Batter {batter_id} has no qualifying balls in years {years}")

    # Wall-ball filtering: balls beyond the target park's wall can't be caught by outfielders, exclude them from optimization
    n_wall_balls = 0
    if home_team:
        wall_mask = is_wall_ball(
            balls["ball_x"].values, balls["ball_y"].values, home_team
        )
        n_wall_balls = int(wall_mask.sum())
        balls = balls[~wall_mask].reset_index(drop=True)
        if hit_probs is not None:
            hit_probs = hit_probs[~wall_mask]

    if balls.empty:
        raise ValueError(f"No catchable balls remaining after wall filtering for {home_team}")

    # Compute w_j (skip the KDE if hit_probs is already supplied externally)
    w_j = compute_w_j(balls, hit_bundle, delta_re, on_1b, on_2b, on_3b, outs, hit_probs=hit_probs)

    # Exclude balls with w_j=0
    mask = w_j > 0
    balls_f = balls[mask].reset_index(drop=True)
    w_j_f = w_j[mask]

    if len(balls_f) == 0:
        raise ValueError("No balls with positive w_j for this game state")

    sc_means  = {pos: scalers[pos].mean_  for pos in POSITIONS}
    sc_scales = {pos: scalers[pos].scale_ for pos in POSITIONS}

    ctx: ObjectiveContext = {
        "ball_x":     balls_f["ball_x"].values,
        "ball_y":     balls_f["ball_y"].values,
        "flight_time": balls_f["flight_time"].values,
        "w_j":        w_j_f,
        "sc_means":   sc_means,
        "sc_scales":  sc_scales,
        "mus":        mus,
    }

    # Polar-coordinate normalization: r ∈ [150,400] vs θ ∈ [-45,45], mapped to [0,1] to unify gradient scaling
    _lows  = np.array([b[0] for b in _POLAR_BOUNDS])
    _highs = np.array([b[1] for b in _POLAR_BOUNDS])
    _range = _highs - _lows

    def _obj_normalized(params_norm: np.ndarray) -> float:
        params = _lows + params_norm * _range
        return objective_re24(_polar_to_xy(params), ctx)

    unit_bounds = [(0.0, 1.0)] * 6
    best: dict[str, np.ndarray | float] | None = None

    # The warm start takes the slot of one random starting point (not an extra addition), keeping the total
    # evaluate count equal to n_restarts -- no added compute cost, and no slowdown from an extra evaluation.
    #
    # Random-start sampling method: Latin Hypercube Sampling when there's no warm start, uniform random when
    # there is. This isn't an arbitrary choice -- a 30-sample test (2026-07-05, see ARCHITECTURE.md) found the
    # two behave oppositely:
    #   - no_park (no warm start): LHS has a lower miss rate than uniform random (n_restarts=20: 2/30 vs 4/30)
    #   - with_park (warm-started from the no_park solution): LHS is actually worse than uniform random
    #     (n_restarts=8, same sample batch: 6/30 vs 4/30) -- the hypothesis is that LHS's stratification is
    #     computed over "this batch of starting points" as a whole, and forcibly inserting an external
    #     warm-start point disrupts the uniform-coverage assumption behind the stratification, a problem
    #     uniform random sampling doesn't have.
    n_random = n_restarts - 1 if warm_start_xy is not None else n_restarts
    n_random = max(n_random, 0)
    if warm_start_xy is not None:
        starts = list(np.random.default_rng(seed).uniform(0.0, 1.0, size=(n_random, 6)))
    else:
        starts = list(qmc.LatinHypercube(d=6, seed=seed).random(n=n_random))
    if warm_start_xy is not None:
        warm_params = _xy_to_polar_params(warm_start_xy)
        starts.append(np.clip((warm_params - _lows) / _range, 0.0, 1.0))

    # ftol/gtol were loosened (originally 1e-10/1e-6): diagnostics showed maxiter=500 was never actually hit
    # (the empirical median only ran 15 iterations), so convergence tolerance didn't need to be that strict.
    # A 30-sample validation (2026-07-05, see ARCHITECTURE.md) showed that 1e-6/1e-4 gives the same miss rate
    # as the original, 14-17% faster; loosening further (1e-4/1e-3 or beyond) noticeably worsens the miss rate,
    # so don't relax it any further.
    for x0_norm in starts:
        res = minimize(
            _obj_normalized,
            x0_norm,
            method="L-BFGS-B",
            bounds=unit_bounds,
            options={"maxiter": 500, "ftol": 1e-6, "gtol": 1e-4},
        )
        if best is None or res.fun < best["fun"]:
            best = {"x": res.x, "fun": res.fun}

    polar_best = _lows + best["x"] * _range
    xy = _polar_to_xy(polar_best)
    lf_x, lf_y, cf_x, cf_y, rf_x, rf_y = xy
    return {
        "LF":          (float(lf_x), float(lf_y)),
        "CF":          (float(cf_x), float(cf_y)),
        "RF":          (float(rf_x), float(rf_y)),
        "objective":   float(best["fun"]),
        "n_balls":     len(balls_f),
        "n_wall_balls": n_wall_balls,
    }


def compute_per_fielder_probs(
    positions: OutfieldXY,
    balls: pd.DataFrame,
    scalers: dict[str, StandardScaler],
    mus: dict[str, GroupMu],
) -> dict[str, np.ndarray]:
    """
    Individual catch probability for each fielder, for each ball.
    Returns {"LF": ndarray(N), "CF": ndarray(N), "RF": ndarray(N)}
    """
    ball_x = balls["ball_x"].values
    ball_y = balls["ball_y"].values
    flight_time = balls["flight_time"].values
    return {
        pos: _catch_prob_single_fielder(
            positions[pos][0], positions[pos][1], ball_x, ball_y, flight_time,
            scalers[pos].mean_, scalers[pos].scale_, mus[pos],
        )
        for pos in POSITIONS
    }


def compute_ball_catch_probs(
    positions: OutfieldXY,
    balls: pd.DataFrame,
    scalers: dict[str, StandardScaler],
    mus: dict[str, GroupMu],
) -> np.ndarray:
    """
    Given the three fielders' positions, compute the joint catch probability p̂_j for each ball.
    positions: {"LF": (x,y), "CF": (x,y), "RF": (x,y)}
    balls: DataFrame with ball_x, ball_y, flight_time
    Returns array of shape (N,): p̂_j for each ball.
    """
    ball_x = balls["ball_x"].values
    ball_y = balls["ball_y"].values
    flight_time = balls["flight_time"].values
    p_not_caught = np.ones(len(ball_x))
    for pos in POSITIONS:
        fielder_x, fielder_y = positions[pos]
        p_catch = _catch_prob_single_fielder(
            fielder_x, fielder_y, ball_x, ball_y, flight_time,
            scalers[pos].mean_, scalers[pos].scale_, mus[pos],
        )
        p_not_caught *= (1.0 - p_catch)
    return 1.0 - p_not_caught


def get_league_avg_positions(year: int, dsn: str = DSN) -> OutfieldXY:
    """
    Fetches league-average positioning for each position in the given year, from fielder_positioning.
    Returns {"LF": (x,y), "CF": (x,y), "RF": (x,y)}
    """
    query = """
        SELECT position,
               AVG(avg_norm_start_distance * sin(radians(avg_norm_start_angle))) AS x,
               AVG(avg_norm_start_distance * cos(radians(avg_norm_start_angle))) AS y
        FROM fielder_positioning
        WHERE season = %(year)s
          AND position IN ('LF', 'CF', 'RF')
        GROUP BY position
    """
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"year": year})
            rows = cur.fetchall()
    return {pos: (float(x), float(y)) for pos, x, y in rows}


def get_batter_stand(batter_id: int, year: int, dsn: str = DSN) -> str:
    """Fetches the batter's most common batting stance ('L' / 'R' / 'S') for the given year. Queries the lean
    precomputed table (see scripts/precompute_batter_balls.py); the most-common value is already computed at
    the precompute stage."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT stand FROM precomputed_batter_stand
                WHERE batter = %(bid)s AND game_year = %(year)s
            """, {"bid": batter_id, "year": year})
            row = cur.fetchone()
    return row[0] if row else "R"


def load_qualifying_batters(year: int, dsn: str = DSN, min_balls: int = 30) -> list[QualifyingBatter]:
    """List of batters with ball count >= min_balls for the given year, used by /api/batters in api/main.py.
    Queries the lean precomputed table (see scripts/precompute_batter_balls.py)."""
    query = """
        SELECT batter, COUNT(*) AS n_balls
        FROM precomputed_batter_balls
        WHERE game_year = %(year)s
        GROUP BY batter
        HAVING COUNT(*) >= %(min_balls)s
        ORDER BY n_balls DESC
    """
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"year": year, "min_balls": min_balls})
            rows = cur.fetchall()
    return [{"batter_id": r[0], "n_balls": r[1]} for r in rows]
