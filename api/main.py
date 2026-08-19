"""
Baseball Defense Optimizer — FastAPI backend

See the ARCHITECTURE.md "API (FastAPI)" section for the full endpoint list.
The POST /api/optimize family's latency depends on n_restarts/concurrency; see the
ARCHITECTURE.md "Known performance limits" section for measured numbers — not duplicated
here to avoid the two places drifting out of sync.
"""
import json
import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, TypedDict

import numpy as np
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.optimization import (
    optimize_positions, prepare_batter_balls, compute_w_j,
    compute_ball_catch_probs,
    get_league_avg_positions, get_batter_stand, load_qualifying_batters,
    load_model_params, load_player_params,
    GroupMu, OptimizeResult, OutfieldXY, QualifyingBatter, POSITIONS,
)
from src.config import DSN
from src.hit_prob import HitProbBundle, predict_hit_probs_batch, load_hit_prob
from src.re24 import BaseOutState, HitDeltaKey, load_re24
from src.stadium_walls import SUPPORTED_TEAMS, get_park_boundary_coords, is_wall_ball
from src.if_dp_optimize import (DP_POSITIONS, DPScorer, anchored_starts,
                                dp_delta_re, optimize_infield_dp,
                                positions_to_params_dp)
from src.if_optimize import (PlayerEffects, expected_outs as if_expected_outs,
                             optimize_infield, positions_to_params,
                             predict_p_out)
from src.if_runvalue import delta_re_out, gb_miss_costs, runvalue_ball_weights
from .schemas import (
    BatterInfo, OptimizeRequest, OptimizeResponse,
    BallPoint, ParkCoord, PositionSet, PositionXY, OptimizeStats, FielderInfo,
    IFBallPoint, IFBatterInfo, IFCustomResultResponse, IFFielderInfo,
    IFFielderOption, IFOptimizeRequest, IFOptimizeResponse, IFOptimizeSet,
    IFOptimizeStats, IFPosition, IFPositionSet, IFResultResponse, IFStats,
    IntegratedBatterInfo, IntegratedRequest, IntegratedResponse, IntegratedSet,
    IntegratedStats, PopupBall,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE:    Path = Path(__file__).parent.parent
PRE_DIR: Path = BASE / "data" / "precomputed"
_MIN_BALLS: int = 30


class BatterEntry(TypedDict):
    batter_id: int
    name: str
    n_balls: int


class IFBatterEntry(TypedDict):
    batter_id: int
    name: str
    n_gb: int
    stand: str


class FielderCacheEntry(TypedDict):
    """The lightweight cache shape returned by _load_fielders() (only used for the startup
    existence check/count; see FielderEntry, which get_fielders() queries directly, for the full field set)."""
    name: str
    oaa: float
    n_opp: int
    player_id: int | None


class FielderEntry(TypedDict):
    name: str
    oaa: float
    n_opp: int
    player_id: int | None
    team_id: int | None
    official_oaa: int | None
    official_n_opp: int | None


class IFFielderOptionEntry(TypedDict):
    player_id: int
    name: str
    has_effects: bool
    oaa: float | None
    n_balls: int | None


# Render's free tier only has 0.1 CPU: multiple optimize_positions runs at the same time
# fight over CPU and slow each other down (measured: 50s running alone, ~70-80s each when
# two run concurrently). Use a semaphore to serialize the CPU-intensive optimization
# calls, so concurrent requests (e.g. compare mode A/B sent together) don't drag each other down.
_optimize_semaphore = threading.Semaphore(1)

# Scan which years have a model summary
_AVAILABLE_YEARS: list[int] = sorted(
    y for y in range(2020, 2030)
    if (BASE / "models" / str(y) / "OF" / "OF_summary_players.csv").exists()
)
_DEFAULT_YEAR: int = _AVAILABLE_YEARS[-1] if _AVAILABLE_YEARS else 2025

# ── Startup caches ──────────────────────────────────────────────────────
_name_map:    dict[int, str]  = {}
_re24_table: dict[BaseOutState, float] | None = None
_delta_re:   dict[HitDeltaKey, float] | None = None
_hit_bundle: HitProbBundle | None = None

# year-keyed caches
_scalers:       dict[int, dict[str, StandardScaler]] = {}   # year → pos → scaler
_mus:           dict[int, dict[str, GroupMu]] = {}           # year → pos → mus
_batters_cache: dict[int, list[BatterEntry]] = {}             # year → list[{batter_id, name, n_balls}]

# ── Batter data cache (skip the DB query and KDE when the same batter switches base states) ───────────────
_batter_balls_cache:    dict[int, dict[int, pd.DataFrame]] = {}  # year → batter_id → DataFrame
_batter_hitprobs_cache: dict[int, dict[int, np.ndarray]] = {}    # year → batter_id → ndarray

# ── Rankings multi-year cache (year → ...) ────────────────────────────
_fielders_cache: dict[int, dict[str, list[FielderCacheEntry]]] = {}  # year → pos → list
_model_names:    dict[int, dict[str, set[str]]]           = {}  # year → pos → name set
_team_map:       dict[int, dict[int, int]]                = {}  # year → player_id → team_id

# ── Infield caches (results are all precomputed offline, see scripts/precompute_if_optimize.py) ──
IF_POSITIONS: tuple[str, ...] = ("1B", "2B", "3B", "SS")
_if_years:          list[int] = []                       # years present in precomputed_if_positions
_if_ranking_years:  list[int] = []                       # years present in if_model_oaa
_if_batters_cache:  dict[int, list[IFBatterEntry]] = {}      # year → [{batter_id, name, n_gb, stand}]
_if_league:         dict[int, dict[str, list[float]]] = {}  # year → pos → [angle, depth]
_if_team_map:       dict[int, dict[int, int]] = {}       # year → player_id → team_id
# Personalized positioning (Bayesian player-level layer, see scripts/train_if_bayes.py / export_if_bayes.py)
_if_bayes_model: Pipeline | None = None                   # group-level pipeline (joblib)
_if_effects:        dict[int, tuple[float, float]] = {}  # player_id → (alpha, g)
_if_ad_norm:        tuple[float, float] | None = None    # (ad_mean, ad_std)
_if_fielder_opts:   dict[int, dict[str, list[IFFielderOptionEntry]]] = {}  # year → pos → options
# Batted-ball-type model (run-value pricing, used by the infield/outfield integration page, see src/if_runvalue.py)
_if_xb_model: Pipeline | None = None
# Phase B: runner-on-1B DP optimization assets (models/if_gb/on1b/, see src/if_dp_optimize.py)
_if_dp_out_model: Pipeline | None = None                  # two-stage GLM: P(≥1 out)
_if_dp_model: Pipeline | None = None                      # two-stage GLM: P(DP|≥1 out)
_if_on1b_league: dict[str, tuple[float, float]] = {}      # pos → (angle, depth)
_if_on1b_runner_hp: float | None = None                   # league median runner hp_to_1b


def _load_infield_caches() -> None:
    """Infield caches. The tables may not exist yet / may not have synced to this DB;
    a missing table doesn't stop startup, and the infield endpoints will return empty lists/404."""
    league_json = PRE_DIR / "if_league_positions.json"
    if league_json.exists():
        raw = json.loads(league_json.read_text(encoding="utf-8"))
        _if_league.update({int(y): v for y, v in raw.items()})
    try:
        with psycopg2.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('precomputed_if_positions')")
                if cur.fetchone()[0] is None:
                    logger.warning("precomputed_if_positions 不存在，內野端點停用")
                    return
                cur.execute("SELECT DISTINCT game_year FROM precomputed_if_positions ORDER BY 1")
                _if_years.extend(y for (y,) in cur.fetchall())
                for yr in _if_years:
                    cur.execute(
                        "SELECT batter, n_gb, stand FROM precomputed_if_positions "
                        "WHERE game_year = %s ORDER BY n_gb DESC", (yr,))
                    _if_batters_cache[yr] = [
                        {"batter_id": b, "name": _name_map.get(b, f"#{b}"),
                         "n_gb": n, "stand": s}
                        for b, n, s in cur.fetchall()]
                cur.execute("SELECT to_regclass('if_model_oaa')")
                if cur.fetchone()[0] is not None:
                    cur.execute("SELECT DISTINCT year FROM if_model_oaa ORDER BY 1")
                    _if_ranking_years.extend(y for (y,) in cur.fetchall())
                    for yr in _if_ranking_years:
                        cur.execute(
                            "SELECT DISTINCT player_id FROM if_model_oaa "
                            "WHERE year = %s AND player_name IS NOT NULL", (yr,))
                        pids = [p for (p,) in cur.fetchall()]
                        _if_team_map[yr] = _load_team_info(pids, season=yr)
    except Exception as e:
        logger.warning(f"內野快取載入失敗: {e}")
    _load_if_bayes()
    _load_if_xb()
    _load_if_dp()
    logger.info(f"內野: years={_if_years}, ranking years={_if_ranking_years}, "
                f"bayes={'on' if _if_bayes_model is not None else 'off'}, "
                f"xb={'on' if _if_xb_model is not None else 'off'}, "
                f"dp={'on' if _if_dp_out_model is not None else 'off'}, "
                + ", ".join(f"{y}={len(_if_batters_cache[y])} batters" for y in _if_years))


def _load_if_xb() -> None:
    """Batted-ball-type model (P(extra-base hit | ground ball)). Missing doesn't stop startup; the integration endpoint returns 503."""
    global _if_xb_model
    import joblib
    try:
        _if_xb_model = joblib.load(BASE / "models" / "if_gb" / "if_gb_xb_model.joblib")
    except Exception as e:
        logger.warning(f"安打類型模型載入失敗，整合端點停用: {e}")
        _if_xb_model = None


def _load_if_dp() -> None:
    """Phase B (runner-on-1B DP optimization) assets: two-stage GLMs + offline constants
    (league runner-on-1B positioning, runner median speed, produced by
    scripts/precompute_if_on1b_constants.py). Missing doesn't stop startup;
    /api/if_optimize falls back to the current no-runner-state run-value refinement for that base state."""
    global _if_dp_out_model, _if_dp_model, _if_on1b_runner_hp
    import joblib
    try:
        on1b_dir = BASE / "models" / "if_gb" / "on1b"
        const = json.loads((PRE_DIR / "if_on1b_constants.json")
                           .read_text(encoding="utf-8"))
        out_model = joblib.load(on1b_dir / "if_on1b_out_glm.joblib")
        dp_model = joblib.load(on1b_dir / "if_on1b_dp_glm.joblib")
    except Exception as e:
        logger.warning(f"階段B DP 資產載入失敗，一壘有人壘況退回無壘況精修: {e}")
        return
    _if_on1b_league.update({p: (float(a), float(d))
                            for p, (a, d) in const["positions"].items()})
    _if_on1b_runner_hp = float(const["runner_hp_to_1b"])
    _if_dp_out_model, _if_dp_model = out_model, dp_model


def _load_if_bayes() -> None:
    """Personalized positioning assets: Bayesian group-level pipeline + player effects + fielder menu.
    Missing doesn't stop startup; /api/if_fielder_options and /api/if_result_custom return 404/503."""
    global _if_bayes_model, _if_ad_norm
    import joblib

    bayes_dir = BASE / "models" / "if_gb" / "bayes"
    try:
        _if_bayes_model = joblib.load(bayes_dir / "if_bayes_group_pipeline.joblib")
        meta = json.loads((bayes_dir / "IF_meta.json").read_text(encoding="utf-8"))
        _if_ad_norm = (meta["ad_mean"], meta["ad_std"])
        eff = pd.read_csv(bayes_dir / "IF_player_effects.csv")
        _if_effects.update({int(r.player_id): (float(r.alpha), float(r.g))
                            for r in eff.itertuples()})
    except Exception as e:
        logger.warning(f"貝葉斯個人化資產載入失敗，個人化端點停用: {e}")
        _if_bayes_model = None
        return
    _load_if_fielder_menu()


def _load_if_fielder_menu() -> None:
    """Fielder menu cache. Split out into its own function: a transient DB failure at startup
    shouldn't kill the feature until the next restart (this is exactly what caused the menu
    to 404 on the live instance on 2026-07-13) — if_fielder_options retries lazily."""
    try:
        opts: dict[int, dict[str, list[IFFielderOptionEntry]]] = {}
        with psycopg2.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('fielder_positioning')")
                if cur.fetchone()[0] is None:
                    logger.warning("fielder_positioning 不存在，野手選單停用")
                    return
                # This year/position's model rating (used for the OAA/100 label and the minimum-opportunities slider)
                oaa_map: dict[tuple[int, str, int], tuple[float, int]] = {}
                cur.execute("SELECT to_regclass('if_model_oaa')")
                if cur.fetchone()[0] is not None:
                    cur.execute(
                        "SELECT year, position, player_id, model_oaa, n_balls "
                        "FROM if_model_oaa")
                    oaa_map = {(int(y), p, int(pid)): (float(oaa), int(n))
                               for y, p, pid, oaa, n in cur.fetchall()}
                cur.execute(
                    "SELECT DISTINCT season, position, fielder_id "
                    "FROM fielder_positioning WHERE position IN %s "
                    "ORDER BY season, position", (IF_POSITIONS,))
                for season, pos, fid in cur.fetchall():
                    oaa_n = oaa_map.get((int(season), pos, int(fid)))
                    (opts.setdefault(int(season), {})
                     .setdefault(pos, []).append({
                         "player_id": int(fid),
                         "name": _name_map.get(int(fid), f"#{fid}"),
                         "has_effects": int(fid) in _if_effects,
                         "oaa": oaa_n[0] if oaa_n else None,
                         "n_balls": oaa_n[1] if oaa_n else None}))
        def _oaa_rate(option: IFFielderOptionEntry) -> float | None:
            if option["oaa"] is None or not option["n_balls"]:
                return None
            return option["oaa"] / option["n_balls"]

        def _sort_key(option: IFFielderOptionEntry) -> tuple[float, str]:
            rate = _oaa_rate(option)
            return (-rate if rate is not None else float("inf"), option["name"])

        # Pure descending OAA/100 (same as the outfield menu); those without a rating sort to the
        # bottom by name. Don't use has_effects as a sort key — it would kick "has a label but no
        # effect" players to the bottom, which looks like the descending order got broken
        # (reported by the user on 2026-07-14)
        for year_options in opts.values():
            for position_options in year_options.values():
                position_options.sort(key=_sort_key)
        _if_fielder_opts.clear()
        _if_fielder_opts.update(opts)
        logger.info(f"野手選單: {sorted(_if_fielder_opts)} 年份已載入")
    except Exception as e:
        logger.warning(f"野手選單載入失敗: {e}")


def _load_fielders(year: int) -> dict[str, list[FielderCacheEntry]]:
    """Per-position outfielder list for a given year (requires that year's models/{year}/OF/OF_summary_players.csv)."""
    import re

    models_dir  = BASE / "models" / str(year) / "OF"
    players_csv = models_dir / "OF_summary_players.csv"
    if not players_csv.exists():
        logger.warning(f"No model summary for {year}, skipping")
        return {pos: [] for pos in POSITIONS}

    df_players = pd.read_csv(players_csv, index_col=0, encoding="utf-8-sig")
    of_names: set[str] = set()
    for i in df_players.index:
        match = re.match(r"alpha\[(.+)\]", str(i))
        if match is not None:
            of_names.add(match.group(1))
    _model_names[year] = {pos: of_names for pos in POSITIONS}

    _SQL = """
        SELECT m.name_fielder, m.model_oaa, m.n_opp, MAX(o.player_id) AS player_id
        FROM model_oaa m
        LEFT JOIN oaa_leaderboard o
               ON o.player_name = m.name_fielder AND o.year = %(year)s
        WHERE m.year = %(year)s
          AND m.position = %(pos)s
          AND m.n_opp >= 100
        GROUP BY m.name_fielder, m.model_oaa, m.n_opp
        ORDER BY m.model_oaa / m.n_opp DESC
    """
    out: dict[str, list[FielderCacheEntry]] = {}
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            for pos in POSITIONS:
                cur.execute(_SQL, {"year": year, "pos": pos})
                rows = cur.fetchall()
                out[pos] = [
                    {"name": name, "oaa": float(oaa), "n_opp": n_opp, "player_id": pid}
                    for name, oaa, n_opp, pid in rows
                    if name in _model_names[year][pos]
                ]
    return out


_MLB_TEAM_IDS = {
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 158,
}

def _load_team_info(player_ids: list[int], season: int) -> dict[int, int]:
    """Batch-query the MLB Stats API for a season's fielding splits to get the MLB team
    (player_id → team_id). For traded players, takes the MLB team with the most games played.
    Failure doesn't stop startup."""
    import requests
    result: dict[int, int] = {}
    for i in range(0, len(player_ids), 500):
        chunk = player_ids[i : i + 500]
        try:
            r = requests.get(
                "https://statsapi.mlb.com/api/v1/people",
                params={
                    "personIds": ",".join(str(p) for p in chunk),
                    "hydrate": f"stats(group=fielding,type=season,season={season})",
                },
                timeout=20,
            )
            r.raise_for_status()
            for person in r.json().get("people", []):
                best_tid, best_g = None, -1
                for grp in person.get("stats", []):
                    for split in grp.get("splits", []):
                        tid = split.get("team", {}).get("id")
                        if tid not in _MLB_TEAM_IDS:
                            continue
                        g = (split.get("stat") or {}).get("gamesPlayed") or 0
                        if g > best_g:
                            best_g, best_tid = g, tid
                if best_tid:
                    result[person["id"]] = best_tid
        except Exception as e:
            logger.warning(f"MLB Stats API team lookup failed (season={season}): {e}")
    return result


def _load_batters(year: int) -> list[QualifyingBatter]:
    return load_qualifying_batters(year, DSN, _MIN_BALLS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _re24_table, _delta_re, _hit_bundle

    # Name cache
    name_path = BASE / "data" / "reference" / "batter_names.json"
    if name_path.exists():
        raw = json.loads(name_path.read_text(encoding="utf-8"))
        _name_map.update({int(k): v for k, v in raw.items()})
    logger.info(f"Loaded {len(_name_map)} batter names")

    # Precomputed data (RE24 / hit prob KDE — year-independent)
    _re24_table, _delta_re = load_re24(PRE_DIR)
    _hit_bundle  = load_hit_prob(PRE_DIR)
    logger.info("Preloaded RE24, KDE")

    # Per year: batter list + model parameters
    for yr in _AVAILABLE_YEARS:
        rows = _load_batters(yr)
        _batters_cache[yr] = [
            {"batter_id": r["batter_id"], "name": _name_map.get(r["batter_id"], f"#{r['batter_id']}"), "n_balls": r["n_balls"]}
            for r in rows
        ]
        logger.info(f"Loaded {len(_batters_cache[yr])} batters for {yr}")
        models_dir = BASE / "models" / str(yr)
        try:
            of_scaler, of_mus = load_model_params("OF", models_dir)
            _scalers[yr] = {pos: of_scaler for pos in POSITIONS}
            _mus[yr]     = {pos: of_mus    for pos in POSITIONS}
            logger.info(f"Loaded model params for {yr}")
        except Exception as e:
            logger.warning(f"Could not load model params for {yr}: {e}")

    # Per-year outfielder list (also builds _model_names for dynamic lookups)
    logger.info(f"Available ranking years: {_AVAILABLE_YEARS}")
    for yr in _AVAILABLE_YEARS:
        _fielders_cache[yr] = _load_fielders(yr)
        logger.info(f"  {yr}: " + ", ".join(f"{p}={len(_fielders_cache[yr][p])}" for p in POSITIONS))
        # Look up player_id for all players (not limited by n_opp), so low-opportunity players still have team info
        with psycopg2.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT o.player_id
                    FROM model_oaa m
                    JOIN oaa_leaderboard o
                      ON o.player_name = m.name_fielder AND o.year = %(yr)s
                    WHERE m.year = %(yr)s AND o.player_id IS NOT NULL
                """, {"yr": yr})
                all_pids = [row[0] for row in cur.fetchall()]
        _team_map[yr] = _load_team_info(all_pids, season=yr)
        logger.info(f"  {yr}: team info for {len(_team_map[yr])} players")

    _load_infield_caches()

    yield


app = FastAPI(title="Baseball Defense Optimizer", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/api/batters", response_model=list[BatterInfo])
def get_batters(year: int = _DEFAULT_YEAR) -> list[BatterEntry]:
    return _batters_cache.get(year, _batters_cache.get(_DEFAULT_YEAR, []))


@app.get("/api/teams", response_model=list[str])
def get_teams() -> list[str]:
    return SUPPORTED_TEAMS


@app.get("/api/years")
def get_years() -> list[int]:
    return sorted(_AVAILABLE_YEARS)


def _compute_avg_oaa_per_ball(rows: list[tuple], yr_model_names: dict[str, set[str]]) -> float:
    """League average used for unified centering across LF+CF+RF: computed only from players who have model parameters.

    rows: an iterable of (name, position, model_oaa, n_opp, ...); only the first four columns are used.
    """
    visible = [(float(oaa), int(n))
               for name, pos, oaa, n, *_ in rows
               if name in yr_model_names.get(pos, set())]
    total_oaa = sum(r[0] for r in visible)
    total_opp = sum(r[1] for r in visible)
    return total_oaa / total_opp if total_opp else 0.0


class PlayerTrendEntry(TypedDict):
    year: int
    position: str
    oaa: float
    n_opp: int
    rate: float | None


@app.get("/api/player_trend")
def player_trend(name: str) -> list[PlayerTrendEntry]:
    """Return year-by-year centered OAA/100 matching the Rankings table."""
    result: list[PlayerTrendEntry] = []
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            for yr in _AVAILABLE_YEARS:
                yr_model_names = _model_names.get(yr, {})
                cur.execute(
                    "SELECT name_fielder, position, model_oaa, n_opp FROM model_oaa WHERE year = %s",
                    (yr,),
                )
                all_rows = cur.fetchall()

                avg_per_ball = _compute_avg_oaa_per_ball(all_rows, yr_model_names)

                for fielder_name, pos, oaa, n in all_rows:
                    if fielder_name == name and fielder_name in yr_model_names.get(pos, set()):
                        centered_oaa = float(oaa) - avg_per_ball * int(n)
                        result.append({
                            "year": yr, "position": pos,
                            "oaa": round(centered_oaa, 2), "n_opp": int(n),
                            "rate": round(centered_oaa / int(n) * 100, 2) if n else None,
                        })
    return sorted(result, key=lambda entry: entry["year"])


@app.get("/api/fielders", response_model=dict[str, list[FielderInfo]])
def get_fielders(year: int = 2025, min_opp: int = 100) -> dict[str, list[FielderEntry]]:
    if year not in _fielders_cache:
        raise HTTPException(404, f"No ranking data for year {year}. Available: {sorted(_AVAILABLE_YEARS)}")

    yr_model_names = _model_names.get(year, {})
    yr_team_map    = _team_map.get(year, {})

    _SQL_ALL = """
        SELECT m.name_fielder, m.position, m.model_oaa, m.n_opp,
               MAX(o.player_id) AS player_id,
               MAX(o.oaa) AS official_oaa, MAX(o.n_opp) AS official_n_opp
        FROM model_oaa m
        LEFT JOIN oaa_leaderboard o
               ON o.player_name = m.name_fielder AND o.year = %(year)s
        WHERE m.year = %(year)s
        GROUP BY m.name_fielder, m.position, m.model_oaa, m.n_opp
    """
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL_ALL, {"year": year})
            all_rows = cur.fetchall()

    avg_oaa_per_ball = _compute_avg_oaa_per_ball(all_rows, yr_model_names)

    result: dict[str, list[FielderEntry]] = {}
    for pos in POSITIONS:
        rows_pos = [
            (name, float(oaa) - avg_oaa_per_ball * int(n), int(n), pid, ooaa, onopp)
            for name, p, oaa, n, pid, ooaa, onopp in all_rows
            if p == pos and name in yr_model_names.get(pos, set())
        ]
        filtered = [r for r in rows_pos if r[2] >= min_opp]
        filtered.sort(key=lambda x: x[1] / x[2] if x[2] else 0, reverse=True)
        result[pos] = [{"name": name, "oaa": round(c, 2), "n_opp": n,
                        "player_id": pid,
                        "team_id": yr_team_map.get(pid) if pid else None,
                        "official_oaa": ooaa, "official_n_opp": onopp}
                       for name, c, n, pid, ooaa, onopp in filtered]
    return result


class StarBucket(TypedDict):
    opp: int
    outs: int


class StarStatsEntry(TypedDict):
    stars: list[StarBucket]
    all: StarBucket


@app.get("/api/star_stats")
def get_star_stats(year: int = _DEFAULT_YEAR) -> dict[str, StarStatsEntry]:
    # Read the star-rating distribution computed by our model (model_star_stats), already merged across positions
    _SQL = """
        SELECT name_fielder,
               n_opp_0stars, n_fieldout_0stars,
               n_opp_1stars, n_fieldout_1stars,
               n_opp_2stars, n_fieldout_2stars,
               n_opp_3stars, n_fieldout_3stars,
               n_opp_4stars, n_fieldout_4stars,
               n_opp_5stars, n_fieldout_5stars
        FROM model_star_stats
        WHERE year = %(year)s
    """
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL, {"year": year})
            rows = cur.fetchall()

    result: dict[str, StarStatsEntry] = {}
    for row in rows:
        name = row[0]
        stars: list[StarBucket] = []
        total_opp = total_out = 0
        for i in range(6):
            opp = int(row[1 + i * 2] or 0)
            out = int(row[2 + i * 2] or 0)
            total_opp += opp
            total_out += out
            stars.append({"opp": opp, "outs": out})
        result[name] = {
            "stars": stars,
            "all": {"opp": total_opp, "outs": total_out},
        }
    return result


# ── Infield endpoints (results are all precomputed offline, just a table lookup, no computation) ───────────────────

@app.get("/api/if_years")
def if_years() -> list[int]:
    return _if_years


@app.get("/api/if_batters", response_model=list[IFBatterInfo])
def if_batters(year: int | None = None) -> list[IFBatterEntry]:
    if year is None and _if_years:
        year = _if_years[-1]
    if year is None:
        return []
    return _if_batters_cache.get(year, [])


class IntegratedBatterEntry(TypedDict):
    batter_id: int
    name: str
    n_gb: int
    n_total: int


_integrated_batters_cache: dict[int, list[IntegratedBatterEntry]] = {}   # year → menu (includes total ball counts)


@app.get("/api/integrated_batters", response_model=list[IntegratedBatterInfo])
def integrated_batters(year: int | None = None) -> list[IntegratedBatterEntry]:
    """Batter menu for the integrated page: qualifying infield batters (the offline precomputed
    list of ground balls ≥ 50; OF-only batters were briefly included then removed — the degraded
    experience from insufficient samples wasn't good, decided by the user on 2026-07-14).
    The number shown in parentheses is the total ball count that appears in the chart
    (ground balls + outfield fly balls/line drives + popups)."""
    if year is None and _if_years:
        year = _if_years[-1]
    if year not in _if_batters_cache:
        return []
    if year not in _integrated_batters_cache:
        of_n: dict[int, int] = {}
        pu_n: dict[int, int] = {}
        try:
            with psycopg2.connect(DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT batter, count(*) FROM precomputed_batter_balls "
                        "WHERE game_year = %s GROUP BY batter", (year,))
                    of_n = dict(cur.fetchall())
                    cur.execute("SELECT to_regclass('precomputed_batter_popups')")
                    if cur.fetchone()[0] is not None:
                        cur.execute(
                            "SELECT batter, count(*) FROM precomputed_batter_popups "
                            "WHERE game_year = %s GROUP BY batter", (year,))
                        pu_n = dict(cur.fetchall())
        except Exception as e:
            logger.warning(f"integrated_batters 球數統計失敗，退回滾地球數: {e}")
        rows: list[IntegratedBatterEntry] = [
            {"batter_id": b["batter_id"], "name": b["name"], "n_gb": b["n_gb"],
             "n_total": b["n_gb"] + of_n.get(b["batter_id"], 0)
                        + pu_n.get(b["batter_id"], 0)}
            for b in _if_batters_cache[year]]
        _integrated_batters_cache[year] = sorted(
            rows, key=lambda entry: entry["n_total"], reverse=True)
    return _integrated_batters_cache[year]


IfBatterData = tuple[str, int, float, np.ndarray, np.ndarray, list[tuple], pd.DataFrame]


def _load_if_batter(batter_id: int, year: int) -> IfBatterData:
    """The precomputed out-rate optimum + the batter's ground balls (shared across the infield live endpoints).

    Returns (stand, n_gb, hp_to_1b, warm_angles, warm_depths, ball_rows, balls_df);
    balls_df already has launch_angle nulls filled in and carries the hp_to_1b / stand_R feature columns."""
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT stand, n_gb, hp_to_1b, "
                "       angle_1b, depth_1b, angle_2b, depth_2b, "
                "       angle_3b, depth_3b, angle_ss, depth_ss "
                "FROM precomputed_if_positions "
                "WHERE batter = %s AND game_year = %s", (batter_id, year))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(
                    404, f"Batter {batter_id} has no precomputed result in {year}")
            cur.execute(
                "SELECT spray_deg, ball_x, ball_y, launch_speed, launch_angle, is_out "
                "FROM precomputed_if_gbs "
                "WHERE batter = %s AND game_year = %s", (batter_id, year))
            ball_rows = cur.fetchall()

    stand, n_gb, hp_to_1b = row[:3]
    warm_angles = np.array([row[3], row[5], row[7], row[9]], dtype=float)
    warm_depths = np.array([row[4], row[6], row[8], row[10]], dtype=float)

    balls = pd.DataFrame(ball_rows, columns=[
        "spray_deg", "ball_x", "ball_y", "launch_speed", "launch_angle", "is_out"])
    balls["launch_angle"] = pd.to_numeric(balls["launch_angle"], errors="coerce")
    balls["launch_angle"] = balls["launch_angle"].fillna(balls["launch_angle"].median())
    balls["launch_speed"] = balls["launch_speed"].astype(float)
    balls["hp_to_1b"] = float(hp_to_1b)
    balls["stand_R"] = int(stand == "R")
    return stand, n_gb, float(hp_to_1b), warm_angles, warm_depths, ball_rows, balls


def _if_player_effects(fids: dict[str, int | None]) -> PlayerEffects:
    """Specified fielders → player-level effects (unspecified = league average, effect 0)."""
    assert _if_ad_norm is not None

    def _effect(player_id: int | None, component: int) -> float:
        if player_id is None:
            return 0.0
        return _if_effects.get(player_id, (0.0, 0.0))[component]

    alpha = np.array([_effect(fids[p], 0) for p in IF_POSITIONS])
    g = np.array([_effect(fids[p], 1) for p in IF_POSITIONS])
    return {"alpha": alpha, "g": g,
            "ad_mean": _if_ad_norm[0], "ad_std": _if_ad_norm[1]}


def _fielder_display_name(fids: dict[str, int | None], pos: str) -> str | None:
    """Display name of the specified fielder; returns None when unspecified (the frontend shows this as league average)."""
    player_id = fids.get(pos)
    if player_id is None:
        return None
    return _name_map.get(player_id, f"#{player_id}")


def _outfield_xy(result: OptimizeResult) -> OutfieldXY:
    """Extract only the LF/CF/RF position coordinates from the OptimizeResult returned by optimize_positions()."""
    return {"LF": result["LF"], "CF": result["CF"], "RF": result["RF"]}


def _if_position_set(pairs: dict[str, tuple[float, float]], exp_outs: float) -> IFPositionSet:
    positions: dict[str, IFPosition] = {}
    for pos, (angle, depth) in pairs.items():
        rad = math.radians(angle)
        positions[pos] = IFPosition(x=round(depth * math.sin(rad), 1),
                                    y=round(depth * math.cos(rad), 1),
                                    angle=angle, depth=depth)
    return IFPositionSet(positions=positions, exp_outs=exp_outs)


@app.get("/api/if_result", response_model=IFResultResponse)
def if_result(batter_id: int, year: int) -> IFResultResponse:
    if year not in _if_years:
        raise HTTPException(404, f"No infield data for year {year}. Available: {_if_years}")
    if year not in _if_league:
        raise HTTPException(500, f"if_league_positions.json 缺 {year}")
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT stand, n_gb, hp_to_1b, exp_outs_league, exp_outs_opt, "
                "       angle_1b, depth_1b, angle_2b, depth_2b, "
                "       angle_3b, depth_3b, angle_ss, depth_ss "
                "FROM precomputed_if_positions "
                "WHERE batter = %s AND game_year = %s", (batter_id, year))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(404, f"Batter {batter_id} has no precomputed result in {year}")
            cur.execute(
                "SELECT spray_deg, ball_x, ball_y, launch_speed, is_out, "
                "       p_out_league, p_out_opt "
                "FROM precomputed_if_gbs "
                "WHERE batter = %s AND game_year = %s", (batter_id, year))
            ball_rows = cur.fetchall()

    stand, n_gb, hp_to_1b, exp_league, exp_opt = row[:5]
    opt_pairs = {pos: (row[5 + i * 2], row[6 + i * 2])
                 for i, pos in enumerate(IF_POSITIONS)}
    league_pairs: dict[str, tuple[float, float]] = {
        pos: (_if_league[year][pos][0], _if_league[year][pos][1]) for pos in IF_POSITIONS
    }
    gain = exp_opt - exp_league
    return IFResultResponse(
        batter_id=batter_id,
        name=_name_map.get(batter_id, f"#{batter_id}"),
        year=year,
        stand=stand,
        league=_if_position_set(league_pairs, exp_league),
        optimized=_if_position_set(opt_pairs, exp_opt),
        balls=[IFBallPoint(spray_deg=s, x=bx, y=by, launch_speed=ls, is_out=o,
                           p_out_league=pl, p_out_opt=po)
               for s, bx, by, ls, o, pl, po in ball_rows],
        stats=IFStats(n_gb=n_gb, gain=round(gain, 4),
                      outs_per_450=round(gain * 450, 1), hp_to_1b=hp_to_1b),
    )


@app.get("/api/if_fielder_options", response_model=dict[str, list[IFFielderOption]])
def if_fielder_options(year: int = 2025) -> dict[str, list[IFFielderOptionEntry]]:
    """Fielder menu for personalized positioning (fielders with positioning data for that year, grouped by position)."""
    if _if_bayes_model is None:
        raise HTTPException(503, "個人化模型未載入")
    if year not in _if_fielder_opts:
        _load_if_fielder_menu()  # may be empty at startup due to a transient DB failure — lazily rebuild
    if year not in _if_fielder_opts:
        raise HTTPException(404, f"No fielder options for year {year}")
    return _if_fielder_opts[year]


@app.get("/api/if_result_custom", response_model=IFCustomResultResponse)
def if_result_custom(batter_id: int, year: int,
                     fielder_1b: int | None = None, fielder_2b: int | None = None,
                     fielder_3b: int | None = None, fielder_ss: int | None = None) -> IFCustomResultResponse:
    """Personalized positioning for a specified fielder lineup (anchored: local optimization
    warm-started from the zero-effect optimum, so the displacement reflects only the pull of
    player effects, not equivalent drift over flat terrain — see ARCHITECTURE.md
    "Infield Bayesian player layer"). Unspecified positions are treated as a league-average fielder (effect 0)."""
    if _if_bayes_model is None:
        raise HTTPException(503, "個人化模型未載入")
    if year not in _if_years:
        raise HTTPException(404, f"No infield data for year {year}. Available: {_if_years}")
    stand, n_gb, hp_to_1b, opt_angles, opt_depths, ball_rows, balls = \
        _load_if_batter(batter_id, year)

    fids = {"1B": fielder_1b, "2B": fielder_2b, "3B": fielder_3b, "SS": fielder_ss}
    pe = _if_player_effects(fids)

    warm = positions_to_params(opt_angles, opt_depths)
    with _optimize_semaphore:
        res = optimize_infield(balls, _if_bayes_model, n_restarts=0,
                               extra_starts=[warm], player_effects=pe)

    league_pairs: dict[str, tuple[float, float]] = {
        pos: (_if_league[year][pos][0], _if_league[year][pos][1]) for pos in IF_POSITIONS
    }
    lg_angles = np.array([league_pairs[p][0] for p in IF_POSITIONS])
    lg_depths = np.array([league_pairs[p][1] for p in IF_POSITIONS])
    # Baseline = average positions + average parameters (effect 0); only the optimized set carries the specified fielders' effects
    exp_league = if_expected_outs(_if_bayes_model, balls, lg_angles, lg_depths)
    baseline = if_expected_outs(_if_bayes_model, balls, opt_angles, opt_depths, pe)
    p_league = predict_p_out(_if_bayes_model, balls, lg_angles, lg_depths)
    p_custom = predict_p_out(_if_bayes_model, balls, res["angles"], res["depths"], pe)

    custom_pairs = {pos: (round(float(a), 2), round(float(d), 2))
                    for pos, a, d in zip(IF_POSITIONS, res["angles"], res["depths"])}
    gain = res["exp_outs"] - exp_league
    return IFCustomResultResponse(
        batter_id=batter_id,
        name=_name_map.get(batter_id, f"#{batter_id}"),
        year=year, stand=stand,
        fielders={p: _fielder_display_name(fids, p) for p in IF_POSITIONS},
        league=_if_position_set(league_pairs, round(exp_league, 6)),
        optimized=_if_position_set(custom_pairs, round(res["exp_outs"], 6)),
        baseline_exp_outs=round(baseline, 6),
        balls=[IFBallPoint(spray_deg=r[0], x=r[1], y=r[2], launch_speed=r[3],
                           is_out=r[5], p_out_league=round(float(pl), 4),
                           p_out_opt=round(float(pc), 4))
               for r, pl, pc in zip(ball_rows, p_league, p_custom)],
        stats=IFStats(n_gb=n_gb, gain=round(gain, 4),
                      outs_per_450=round(gain * 450, 1), hp_to_1b=hp_to_1b),
    )


class IFFielderEntry(TypedDict):
    name: str
    player_id: int
    team_id: int | None
    oaa: float
    n_balls: int
    official_oaa: int | None
    official_n_opp: int | None


@app.get("/api/if_fielders", response_model=dict[str, list[IFFielderInfo]])
def if_fielders(year: int = 2025, min_balls: int = 100) -> dict[str, list[IFFielderEntry]]:
    if year not in _if_ranking_years:
        raise HTTPException(404, f"No infield ranking data for year {year}. "
                                 f"Available: {_if_ranking_years}")
    _SQL = """
        SELECT m.player_id, m.player_name, m.position, m.model_oaa, m.n_balls,
               o.oaa, o.n_opp
        FROM if_model_oaa m
        LEFT JOIN if_oaa_leaderboard o
               ON o.player_id = m.player_id AND o.year = m.year
        WHERE m.year = %(year)s AND m.player_name IS NOT NULL
          AND m.n_balls >= %(min_balls)s
    """
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL, {"year": year, "min_balls": min_balls})
            rows = cur.fetchall()

    yr_team_map = _if_team_map.get(year, {})
    result: dict[str, list[IFFielderEntry]] = {}
    for pos in IF_POSITIONS:
        rows_pos = [r for r in rows if r[2] == pos]
        rows_pos.sort(key=lambda r: float(r[3]) / r[4] if r[4] else 0, reverse=True)
        result[pos] = [
            {"name": name, "player_id": pid, "team_id": yr_team_map.get(pid),
             "oaa": round(float(oaa), 2), "n_balls": n,
             "official_oaa": off_oaa, "official_n_opp": off_n}
            for pid, name, _, oaa, n, off_oaa, off_n in rows_pos
        ]
    return result


# ── Infield live optimization (infield mirror of outfield /api/optimize) ──────────────────

class IfDpSolution(TypedDict):
    angles3: np.ndarray
    depths3: np.ndarray
    lg3_angles: np.ndarray
    lg3_depths: np.ndarray
    pinned_1b: tuple[float, float]
    scorer: DPScorer
    scorer_base: DPScorer


def _solve_if_dp(balls: pd.DataFrame, state: BaseOutState, warm_angles: np.ndarray,
                 warm_depths: np.ndarray, pe: PlayerEffects | None) -> IfDpSolution:
    """Phase B DP solution (runner on 1B only, <2 outs): shared by /api/if_optimize and the integration endpoint.

    balls gets the runner_hp_to_1b constant column added in place; warm_angles/depths = the precomputed
    out-rate optimum (includes 1B, uses 2B/3B/SS as the starting point). The starting-point setup matches
    the cross-year validation (LHS 8 + no-runner-state solution + league positions); when fielders are
    specified, refine with the anchored approach (anchored_starts avoids kinks). scorer carries pe to
    evaluate the optimized set, scorer_base has effect 0 to evaluate the league baseline.
    """
    assert _re24_table is not None and _delta_re is not None
    balls["runner_hp_to_1b"] = _if_on1b_runner_hp
    miss_cost = gb_miss_costs(balls, _if_xb_model, _delta_re, state)
    single_out_delta_re, double_play_delta_re = dp_delta_re(_re24_table, _delta_re, state[3])
    pinned_1b = _if_on1b_league["1B"]
    lg3_angles = np.array([_if_on1b_league[p][0] for p in DP_POSITIONS])
    lg3_depths = np.array([_if_on1b_league[p][1] for p in DP_POSITIONS])

    starts = [positions_to_params_dp(warm_angles[1:], warm_depths[1:]),
              positions_to_params_dp(lg3_angles, lg3_depths)]
    with _optimize_semaphore:
        res = optimize_infield_dp(balls, _if_dp_out_model, _if_dp_model,
                                  pinned_1b, miss_cost, single_out_delta_re,
                                  double_play_delta_re, n_restarts=8, seed=42,
                                  extra_starts=starts)
        if pe is not None:
            # The anchor point often gets stuck on a kink (see the anchored_starts docstring), so add slightly jittered starting points
            anchor = positions_to_params_dp(res["angles"], res["depths"])
            res = optimize_infield_dp(balls, _if_dp_out_model, _if_dp_model,
                                      pinned_1b, miss_cost, single_out_delta_re,
                                      double_play_delta_re, n_restarts=0,
                                      extra_starts=anchored_starts(anchor),
                                      player_effects=pe)

    scorer = DPScorer(_if_dp_out_model, _if_dp_model, balls, pinned_1b, miss_cost,
                      single_out_delta_re, double_play_delta_re, pe)
    scorer_base = scorer if pe is None else DPScorer(
        _if_dp_out_model, _if_dp_model, balls, pinned_1b, miss_cost,
        single_out_delta_re, double_play_delta_re)
    return {"angles3": res["angles"], "depths3": res["depths"],
            "lg3_angles": lg3_angles, "lg3_depths": lg3_depths,
            "pinned_1b": pinned_1b, "scorer": scorer, "scorer_base": scorer_base}


def _if_optimize_dp(req: IFOptimizeRequest) -> IFOptimizeResponse:
    """Runner on 1B only (<2 outs): Phase B DP optimization (src/if_dp_optimize.py).

    1B is pinned at the league hold-runner position; only 2B/3B/SS are optimized. The comparison
    baseline = the league's runner-on-1B average positioning (an offline constant split out by
    fielder_positioning_on1b). UI decision (2026-07-13): only display positioning — exp_outs /
    p_out_* / gain_outs are always P(≥1 out) (same convention as "out rate"); double-play
    probability is not returned; runs is still the double-play-aware E[ΔRE]×n_gb.
    When fielders are specified, player-level effects are carried (the no-runner-on-base Bayesian
    layer is ported over to Phase 1, see the DPScorer docstring); anchored: refine via warm start
    from the zero-effect DP optimum, the same pattern as /api/if_result_custom. 1B's effect still
    participates in the per-ball evaluation (the balls near him are still his to field), but his
    pinned position doesn't move with the effect.
    The starting-point setup for the zero-effect solution matches the cross-year validation
    (scripts/validate_if_dp.py): LHS 8 + no-runner-state optimum + league positions — don't reduce
    this without redoing the convergence test.
    """
    assert _re24_table is not None
    t_start = time.perf_counter()
    year = req.year
    state: BaseOutState = (req.on_1b, req.on_2b, req.on_3b, req.outs)
    stand, _, hp_to_1b, warm_angles, warm_depths, ball_rows, balls = \
        _load_if_batter(req.batter_id, year)

    fids: dict[str, int | None] = {p: (req.fielders or {}).get(p) for p in IF_POSITIONS}
    pe = _if_player_effects(fids) if any(fids.values()) else None

    sol = _solve_if_dp(balls, state, warm_angles, warm_depths, pe)
    pinned_1b = sol["pinned_1b"]
    lg3_angles, lg3_depths = sol["lg3_angles"], sol["lg3_depths"]

    def eval_set(angles3: np.ndarray, depths3: np.ndarray,
                scorer: DPScorer) -> tuple[np.ndarray, float, float]:
        """(per-ball p1, average p1, E[ΔRE]×n_gb)."""
        p1 = scorer.per_ball_p1(angles3, depths3)
        runs = scorer.expected_re(angles3, depths3) * len(balls)
        return p1, float(p1.mean()), runs

    p1_opt, eo_opt, runs_opt = eval_set(sol["angles3"], sol["depths3"], sol["scorer"])
    p1_lg, eo_lg, runs_lg = eval_set(lg3_angles, lg3_depths, sol["scorer_base"])

    def positions_dict(angles4: np.ndarray, depths4: np.ndarray) -> dict[str, IFPosition]:
        out: dict[str, IFPosition] = {}
        for pos, a, d in zip(IF_POSITIONS, angles4, depths4):
            rad = math.radians(float(a))
            out[pos] = IFPosition(x=round(float(d) * math.sin(rad), 1),
                                  y=round(float(d) * math.cos(rad), 1),
                                  angle=round(float(a), 2), depth=round(float(d), 2))
        return out

    opt_angles4 = np.concatenate([[pinned_1b[0]], sol["angles3"]])
    opt_depths4 = np.concatenate([[pinned_1b[1]], sol["depths3"]])
    lg_angles4 = np.concatenate([[pinned_1b[0]], lg3_angles])
    lg_depths4 = np.concatenate([[pinned_1b[1]], lg3_depths])

    gain_outs = eo_opt - eo_lg
    runs_saved = runs_lg - runs_opt
    raw_name = _name_map.get(req.batter_id, f"#{req.batter_id}")
    logger.info(f"[timing] if_optimize(DP) TOTAL: "
                f"{time.perf_counter() - t_start:.2f}s "
                f"(batter={req.batter_id}, year={year}, outs={req.outs}, "
                f"fielders={any(fids.values())})")

    return IFOptimizeResponse(
        batter_id=req.batter_id,
        name=raw_name.replace(", ", " ") if ", " in raw_name else raw_name,
        year=year, stand=stand,
        situation=f"1--  {req.outs} out",
        fielders={p: _fielder_display_name(fids, p) for p in IF_POSITIONS},
        league=IFOptimizeSet(positions=positions_dict(lg_angles4, lg_depths4),
                             exp_outs=round(eo_lg, 6), runs=round(runs_lg, 3)),
        optimized=IFOptimizeSet(positions=positions_dict(opt_angles4, opt_depths4),
                                exp_outs=round(eo_opt, 6), runs=round(runs_opt, 3)),
        balls=[IFBallPoint(spray_deg=r[0], x=r[1], y=r[2], launch_speed=r[3],
                           is_out=r[5], p_out_league=round(float(pl), 4),
                           p_out_opt=round(float(po), 4))
               for r, pl, po in zip(ball_rows, p1_lg, p1_opt)],
        stats=IFOptimizeStats(
            n_gb=len(balls), re_state=round(float(_re24_table.get(state, 0.0)), 4),
            hp_to_1b=hp_to_1b,
            gain_outs=round(gain_outs, 4), outs_per_450=round(gain_outs * 450, 1),
            runs_saved=round(runs_saved, 3),
            runs_per_450=round(runs_saved / len(balls) * 450, 2)),
    )


@app.post("/api/if_optimize", response_model=IFOptimizeResponse)
def if_optimize(req: IFOptimizeRequest) -> IFOptimizeResponse:
    """Batter + base state (+ specified fielders) → infield four-fielder positioning and runs saved.

    Positioning is warm-started from the offline out-rate optimum and refined with run-value
    weights for that base state (the two objectives yield almost the same solution in 2025
    out-of-sample tests, see models/if_gb/runvalue_objective_rows.csv); when fielders are
    specified, player-level effects are also carried (anchored, same as /api/if_result_custom).
    Pricing matches the integration endpoint: runs = E[ΔRE]×n_gb, runs saved = league average − optimized.
    The out-probability model's primary scope is no runners on base; the base state only affects
    the pricing weights — the exception is runner on 1B only (<2 outs), which switches to Phase B
    DP optimization (_if_optimize_dp).
    """
    if _if_bayes_model is None or _if_xb_model is None:
        raise HTTPException(503, "內野模型未載入")
    if req.year not in _if_years:
        raise HTTPException(404, f"No infield data for year {req.year}. Available: {_if_years}")
    if req.year not in _if_league:
        raise HTTPException(500, f"if_league_positions.json 缺 {req.year}")
    if ((req.on_1b, req.on_2b, req.on_3b) == (1, 0, 0) and req.outs < 2
            and _if_dp_out_model is not None):
        return _if_optimize_dp(req)
    assert _re24_table is not None and _delta_re is not None

    t_start = time.perf_counter()
    year = req.year
    state: BaseOutState = (req.on_1b, req.on_2b, req.on_3b, req.outs)
    stand, _, hp_to_1b, warm_angles, warm_depths, ball_rows, balls = \
        _load_if_batter(req.batter_id, year)

    fids: dict[str, int | None] = {p: (req.fielders or {}).get(p) for p in IF_POSITIONS}
    pe = _if_player_effects(fids)

    ball_weights, mean_miss_cost = runvalue_ball_weights(balls, _if_xb_model, _re24_table, _delta_re, state)
    warm = positions_to_params(warm_angles, warm_depths)
    with _optimize_semaphore:
        res = optimize_infield(balls, _if_bayes_model, n_restarts=0,
                               extra_starts=[warm], ball_weights=ball_weights,
                               player_effects=pe)

    lg_angles = np.array([_if_league[year][p][0] for p in IF_POSITIONS])
    lg_depths = np.array([_if_league[year][p][1] for p in IF_POSITIONS])

    def eval_set(angles: np.ndarray, depths: np.ndarray,
                pe_use: PlayerEffects | None) -> tuple[float, float]:
        """(expected out rate, expected runs×n_gb). The baseline set passes pe_use=None
        (average positions + average parameters); only the optimized set carries the specified fielders' effects."""
        eo = if_expected_outs(_if_bayes_model, balls, angles, depths, pe_use)
        runs = (mean_miss_cost - if_expected_outs(_if_bayes_model, balls, angles, depths,
                                                   pe_use, ball_weights=ball_weights)) * len(balls)
        return eo, runs

    eo_opt, runs_opt = eval_set(res["angles"], res["depths"], pe)
    eo_lg, runs_lg = eval_set(lg_angles, lg_depths, None)
    p_league = predict_p_out(_if_bayes_model, balls, lg_angles, lg_depths)
    p_opt = predict_p_out(_if_bayes_model, balls, res["angles"], res["depths"], pe)

    def positions_dict(angles: np.ndarray, depths: np.ndarray) -> dict[str, IFPosition]:
        out: dict[str, IFPosition] = {}
        for pos, a, d in zip(IF_POSITIONS, angles, depths):
            rad = math.radians(float(a))
            out[pos] = IFPosition(x=round(float(d) * math.sin(rad), 1),
                                  y=round(float(d) * math.cos(rad), 1),
                                  angle=round(float(a), 2), depth=round(float(d), 2))
        return out

    gain_outs = eo_opt - eo_lg
    runs_saved = runs_lg - runs_opt
    raw_name = _name_map.get(req.batter_id, f"#{req.batter_id}")
    bases = (("1" if req.on_1b else "-") + ("2" if req.on_2b else "-")
             + ("3" if req.on_3b else "-"))
    logger.info(f"[timing] if_optimize TOTAL: {time.perf_counter() - t_start:.2f}s "
                f"(batter={req.batter_id}, year={year}, state={state}, "
                f"fielders={any(fids.values())})")

    return IFOptimizeResponse(
        batter_id=req.batter_id,
        name=raw_name.replace(", ", " ") if ", " in raw_name else raw_name,
        year=year, stand=stand,
        situation=f"{bases}  {req.outs} out",
        fielders={p: _fielder_display_name(fids, p) for p in IF_POSITIONS},
        league=IFOptimizeSet(positions=positions_dict(lg_angles, lg_depths),
                             exp_outs=round(eo_lg, 6), runs=round(runs_lg, 3)),
        optimized=IFOptimizeSet(positions=positions_dict(res["angles"], res["depths"]),
                                exp_outs=round(eo_opt, 6), runs=round(runs_opt, 3)),
        balls=[IFBallPoint(spray_deg=r[0], x=r[1], y=r[2], launch_speed=r[3],
                           is_out=r[5], p_out_league=round(float(pl), 4),
                           p_out_opt=round(float(po), 4))
               for r, pl, po in zip(ball_rows, p_league, p_opt)],
        stats=IFOptimizeStats(
            n_gb=len(balls), re_state=round(float(_re24_table.get(state, 0.0)), 4),
            hp_to_1b=hp_to_1b,
            gain_outs=round(gain_outs, 4), outs_per_450=round(gain_outs * 450, 1),
            runs_saved=round(runs_saved, 3),
            runs_per_450=round(runs_saved / len(balls) * 450, 2)),
    )


# ── Infield/outfield integration (unified pricing = expected runs, see ARCHITECTURE.md "Infield/outfield integration route") ──

def _load_batter_popups(batter_id: int, year: int) -> list[PopupBall]:
    """Infield popups for display on the integrated page (from the table produced by
    scripts/precompute_batter_popups.py). Popups are not part of the optimization; returns an
    empty list without blocking the main flow if the table doesn't exist yet / hasn't synced to this DB."""
    try:
        with psycopg2.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ball_x, ball_y, is_out FROM precomputed_batter_popups "
                    "WHERE batter = %s AND game_year = %s", (batter_id, year))
                return [PopupBall(x=x, y=y, is_out=o) for x, y, o in cur.fetchall()]
    except Exception as e:
        logger.warning(f"popup 展示球載入失敗（不影響優化）: {e}")
        return []


@app.post("/api/optimize_integrated", response_model=IntegratedResponse)
def optimize_integrated(req: IntegratedRequest) -> IntegratedResponse:
    """Batter + base state → seven-fielder positioning and total runs saved.

    Pricing is unified as full ΔRE expected runs (same scale across infield and outfield):
    outfield = Σ[(1−p̂)×w_j ＋ p̂×ΔRE(out)], infield = E[ΔRE]×n_gb (run-value weights,
    src/if_runvalue.py). An out drives ΔRE down, so the ground-ball-dominated infield side is
    usually negative (a gain for the defense). The optimization is separable (ground balls go
    to the infield, fly balls to the outfield), and each side is compared against its own
    league-average positioning for the same base state.
    When runner on 1B only (<2 outs), the infield side switches to Phase B DP optimization
    (pin 1B + double-play awareness, _solve_if_dp, the same switch as /api/if_optimize); the
    league baseline switches to the runner-on-1B split league positioning, and the infield out
    rate becomes P(≥1 out). Batters with insufficient ground-ball samples (no precomputed
    infield data) degrade to positioning only the outfield trio: positions only has LF/CF/RF,
    n_gb=0, runs_if=0.
    The infield is refined via warm start from the offline out-rate optimum — the two objectives
    yield almost the same solution in 2025 out-of-sample tests (models/if_gb/runvalue_objective_rows.csv),
    the same anchored pattern as the personalization endpoint.
    When home_team is specified, the outfield follows the same general mode as /api/optimize:
    wall-ball catch probability forced to 0 and counted into RE24, plus a second warm-start
    with_park optimization pass; the infield has no wall so it's unaffected.
    When fielders are specified: the outfield carries player-level mu (of_fielders, player names),
    the infield carries Bayesian effects (if_fielders, player_id); unspecified positions = league average.
    """
    if _if_bayes_model is None or _if_xb_model is None:
        raise HTTPException(503, "整合模型未載入")
    if req.home_team and req.home_team.upper() not in SUPPORTED_TEAMS:
        raise HTTPException(422, f"Unsupported team '{req.home_team}'. Use GET /api/teams.")
    if req.year not in _AVAILABLE_YEARS:
        raise HTTPException(422, f"No OF model for year {req.year}. Available: {_AVAILABLE_YEARS}")
    if req.year not in _if_years:
        raise HTTPException(404, f"No infield data for year {req.year}. Available: {_if_years}")
    if req.year not in _if_league:
        raise HTTPException(500, f"if_league_positions.json 缺 {req.year}")
    assert _re24_table is not None and _delta_re is not None and _hit_bundle is not None

    t_start = time.perf_counter()
    year = req.year
    home_team = req.home_team.upper() if req.home_team else None
    state: BaseOutState = (req.on_1b, req.on_2b, req.on_3b, req.outs)

    # ── Outfield side (same general mode as /api/optimize; shares the batter-balls cache) ──────────
    yr_balls_cache = _batter_balls_cache.setdefault(year, {})
    yr_hprob_cache = _batter_hitprobs_cache.setdefault(year, {})
    if req.batter_id not in yr_balls_cache:
        try:
            balls_of = prepare_batter_balls(req.batter_id, [year], DSN)
        except Exception as e:
            raise HTTPException(422, str(e))
        if balls_of.empty:
            raise HTTPException(422, f"Batter {req.batter_id} has no qualifying OF balls in {year}")
        yr_balls_cache[req.batter_id] = balls_of
        yr_hprob_cache[req.batter_id] = predict_hit_probs_batch(_hit_bundle, balls_of)
    balls_of = yr_balls_cache[req.batter_id]
    hit_probs = yr_hprob_cache[req.batter_id]

    w_j = compute_w_j(balls_of, _hit_bundle, _delta_re,
                      req.on_1b, req.on_2b, req.on_3b, req.outs, hit_probs=hit_probs)
    of_mask = w_j > 0
    if not of_mask.any():
        raise HTTPException(422, "No balls with positive w_j for this game state")

    # Wall balls (when a park is specified): catch probability forced to 0, counted into RE24 (same convention as _run_optimize)
    wall_flags = (np.array(is_wall_ball(balls_of["ball_x"].values,
                                        balls_of["ball_y"].values, home_team), dtype=bool)
                  if home_team else np.zeros(len(balls_of), dtype=bool))

    models_dir = BASE / "models" / str(year)

    # Specified outfielders (player-level mu, same as _run_optimize); unspecified positions use the group mu
    fielder_mus: dict[str, GroupMu] | None = None
    if req.of_fielders:
        fielder_mu_overrides: dict[str, GroupMu] = {}
        for pos in POSITIONS:
            player_name = req.of_fielders.get(pos)
            if player_name:
                try:
                    fielder_mu_overrides[pos] = load_player_params("OF", player_name, models_dir)
                except (KeyError, FileNotFoundError):
                    raise HTTPException(422, f"{pos} 找不到球員 '{player_name}' 的模型參數")
        fielder_mus = fielder_mu_overrides or None
    mus_eff = dict(_mus.get(year, {}))
    if fielder_mus:
        mus_eff.update(fielder_mus)

    with _optimize_semaphore:
        opt_of = optimize_positions(
            batter_id=req.batter_id,
            on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
            years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
            home_team=None, dsn=DSN, balls=balls_of, hit_probs=hit_probs,
            fielder_mus=fielder_mus, n_restarts=10,
            delta_re=_delta_re, hit_bundle=_hit_bundle,
        )
        pos_of_opt = _outfield_xy(opt_of)
        if home_team:
            # Same as _run_optimize: with_park refinement warm-started from the no_park solution
            opt_of_park = optimize_positions(
                batter_id=req.batter_id,
                on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                home_team=home_team, dsn=DSN, balls=balls_of, hit_probs=hit_probs,
                fielder_mus=fielder_mus, warm_start_xy=pos_of_opt, n_restarts=8,
                delta_re=_delta_re, hit_bundle=_hit_bundle,
            )
            pos_of_opt = _outfield_xy(opt_of_park)
    try:
        pos_of_league = get_league_avg_positions(year, DSN)
    except Exception:
        pos_of_league = {"LF": (-130.0, 250.0), "CF": (0.0, 310.0), "RF": (130.0, 250.0)}

    # Outfield runs = full ΔRE accounting (same scale as infield, per the user's request on
    # 2026-07-14): a miss is charged w_j, a catch is charged the out's ΔRE (runners don't
    # advance, approximated as out+1, the same pricing as a ground-ball out; sac-fly runner
    # advancement is ignored). Only the displayed pricing changes — the optimization target is
    # unchanged, since the run-value vs. out-rate objectives have already been verified to give
    # almost the same solution (runvalue_objective_rows.csv)
    dre_out_of = delta_re_out(_re24_table, state)

    def of_runs(pos_dict: OutfieldXY, mus_use: dict[str, GroupMu]) -> tuple[np.ndarray, float]:
        probs = np.asarray(compute_ball_catch_probs(
            pos_dict, balls_of, _scalers.get(year, {}), mus_use),
            dtype=float).copy()
        probs[wall_flags] = 0.0                      # a wall ball can't be caught no matter where the fielder stands
        runs = float(np.sum((1.0 - probs[of_mask]) * w_j[of_mask])
                     + dre_out_of * probs.sum())
        return probs, runs

    # Baseline = average positions + average parameters (group mu); only the optimized set carries the specified fielders' player mu
    probs_of_opt, runs_of_opt = of_runs(pos_of_opt, mus_eff)
    probs_of_league, runs_of_league = of_runs(pos_of_league, _mus.get(year, {}))

    # ── Infield side (warm-started from the precomputed out-rate optimum + refined with
    #    run-value weights; runner on 1B only <2 outs switches to Phase B DP optimization,
    #    same as /api/if_optimize). Batters with insufficient ground-ball samples have no
    #    precomputed data: degrade to positioning only the outfield trio ──
    try:
        stand, _, hp_to_1b, warm_angles, warm_depths, ball_rows, balls_if = \
            _load_if_batter(req.batter_id, year)
        has_if = True
    except HTTPException:
        has_if = False
        stand = str(balls_of["stand"].mode().iloc[0]) if len(balls_of) else "R"
        ball_rows, balls_if = [], pd.DataFrame()

    # Specified infielders (Bayesian player-level effects, same as /api/if_optimize)
    if_fids: dict[str, int | None] = {p: (req.if_fielders or {}).get(p) for p in IF_POSITIONS}
    pe = _if_player_effects(if_fids) if any(if_fids.values()) else None

    dp_state = (has_if
                and (req.on_1b, req.on_2b, req.on_3b) == (1, 0, 0) and req.outs < 2
                and _if_dp_out_model is not None)
    if not has_if:
        if_opt_angles = np.array([])
        if_opt_depths = np.array([])
        lg_angles = np.array([])
        lg_depths = np.array([])
        runs_if_opt = runs_if_league = 0.0
        p_if_league = np.array([])
        p_if_opt = np.array([])
    elif dp_state:
        # Phase B: pin 1B hold-runner + double-play-aware pricing (E[ΔRE] itself is already the full accounting);
        # the displayed infield out rate = P(≥1 out), league baseline = the runner-on-1B split league positioning
        sol = _solve_if_dp(balls_if, state, warm_angles, warm_depths, pe)
        if_opt_angles = np.concatenate([[sol["pinned_1b"][0]], sol["angles3"]])
        if_opt_depths = np.concatenate([[sol["pinned_1b"][1]], sol["depths3"]])
        lg_angles = np.concatenate([[sol["pinned_1b"][0]], sol["lg3_angles"]])
        lg_depths = np.concatenate([[sol["pinned_1b"][1]], sol["lg3_depths"]])
        runs_if_opt = sol["scorer"].expected_re(
            sol["angles3"], sol["depths3"]) * len(balls_if)
        runs_if_league = sol["scorer_base"].expected_re(
            sol["lg3_angles"], sol["lg3_depths"]) * len(balls_if)
        p_if_opt = sol["scorer"].per_ball_p1(sol["angles3"], sol["depths3"])
        p_if_league = sol["scorer_base"].per_ball_p1(sol["lg3_angles"], sol["lg3_depths"])
    else:
        ball_weights, mean_miss_cost = runvalue_ball_weights(balls_if, _if_xb_model, _re24_table,
                                                              _delta_re, state)
        warm = positions_to_params(warm_angles, warm_depths)
        with _optimize_semaphore:
            res = optimize_infield(balls_if, _if_bayes_model, n_restarts=0,
                                   extra_starts=[warm], ball_weights=ball_weights,
                                   player_effects=pe)
        if_opt_angles, if_opt_depths = res["angles"], res["depths"]

        lg_angles = np.array([_if_league[year][p][0] for p in IF_POSITIONS])
        lg_depths = np.array([_if_league[year][p][1] for p in IF_POSITIONS])
        # E[ΔRE] (per ground ball) = mean(miss_cost) − mean(p×ball_weights).
        # Baseline = average positions + average parameters (effect 0); only the optimized set carries the specified fielders' effects
        e_if_opt = mean_miss_cost - res["exp_outs"]
        e_if_league = mean_miss_cost - if_expected_outs(
            _if_bayes_model, balls_if, lg_angles, lg_depths, ball_weights=ball_weights)
        runs_if_opt = e_if_opt * len(balls_if)
        runs_if_league = e_if_league * len(balls_if)

        p_if_league = predict_p_out(_if_bayes_model, balls_if, lg_angles, lg_depths)
        p_if_opt = predict_p_out(_if_bayes_model, balls_if,
                                 res["angles"], res["depths"], pe)

    # ── Assembly (seven-fielder positions share PositionXY; infield angle/depth → x/y) ──────
    def if_xy(angle: float, depth: float) -> PositionXY:
        rad = math.radians(angle)
        return PositionXY(x=round(depth * math.sin(rad), 1),
                          y=round(depth * math.cos(rad), 1))

    def pack(pos_of: OutfieldXY, if_angles: np.ndarray, if_depths: np.ndarray,
             runs_of: float, runs_if: float,
             probs_of: np.ndarray, p_if: np.ndarray) -> IntegratedSet:
        positions: dict[str, PositionXY] = {
            p: PositionXY(x=round(float(pos_of[p][0]), 1), y=round(float(pos_of[p][1]), 1))
            for p in POSITIONS
        }
        positions.update({p: if_xy(float(a), float(d))
                          for p, a, d in zip(IF_POSITIONS, if_angles, if_depths)})
        return IntegratedSet(positions=positions,
                             catch_pct=round(float(np.mean(probs_of)) * 100, 1),
                             exp_outs_if=(round(float(np.mean(p_if)), 4)
                                          if len(p_if) else 0.0),
                             runs_of=round(runs_of, 3), runs_if=round(runs_if, 3),
                             runs_total=round(runs_of + runs_if, 3))

    popups = _load_batter_popups(req.batter_id, year)
    raw_name = _name_map.get(req.batter_id, f"#{req.batter_id}")
    bases = (("1" if req.on_1b else "-") + ("2" if req.on_2b else "-")
             + ("3" if req.on_3b else "-"))
    logger.info(f"[timing] optimize_integrated TOTAL: {time.perf_counter() - t_start:.2f}s "
                f"(batter={req.batter_id}, year={year}, state={state})")

    boundary_coords = get_park_boundary_coords(home_team) if home_team else None
    park_boundary = ([ParkCoord(x=c["x"], y=c["y"]) for c in boundary_coords]
                     if boundary_coords else None)

    return IntegratedResponse(
        batter_id=req.batter_id,
        name=raw_name.replace(", ", " ") if ", " in raw_name else raw_name,
        year=year, stand=stand,
        situation=f"{bases}  {req.outs} out",
        league=pack(pos_of_league, lg_angles, lg_depths, runs_of_league, runs_if_league,
                    probs_of_league, p_if_league),
        optimized=pack(pos_of_opt, if_opt_angles, if_opt_depths, runs_of_opt, runs_if_opt,
                       probs_of_opt, p_if_opt),
        of_balls=[BallPoint(x=float(balls_of.iloc[i]["ball_x"]),
                            y=float(balls_of.iloc[i]["ball_y"]),
                            catch_prob=float(probs_of_opt[i]),
                            is_wall_ball=bool(wall_flags[i]),
                            bb_type=(str(balls_of.iloc[i]["bb_type"])
                                     if "bb_type" in balls_of.columns else None))
                  for i in range(len(balls_of))],
        if_balls=[IFBallPoint(spray_deg=r[0], x=r[1], y=r[2], launch_speed=r[3],
                              is_out=r[5], p_out_league=round(float(pl), 4),
                              p_out_opt=round(float(po), 4))
                  for r, pl, po in zip(ball_rows, p_if_league, p_if_opt)],
        popup_balls=popups,
        park_boundary=park_boundary,
        fielders={**{p: (req.of_fielders or {}).get(p) or None for p in POSITIONS},
                  **{p: _fielder_display_name(if_fids, p) for p in IF_POSITIONS}},
        stats=IntegratedStats(
            n_of_balls=len(balls_of), n_gb=len(balls_if), n_popups=len(popups),
            n_wall_balls=int(wall_flags.sum()), home_team=home_team,
            re_state=round(float(_re24_table.get(state, 0.0)), 4),
            runs_saved_of=round(runs_of_league - runs_of_opt, 3),
            runs_saved_if=round(runs_if_league - runs_if_opt, 3),
            runs_saved_total=round((runs_of_league - runs_of_opt)
                                   + (runs_if_league - runs_if_opt), 3)),
    )


@app.get("/api/park_boundary/{team}", response_model=list[ParkCoord] | None)
def park_boundary_endpoint(team: str) -> list[ParkCoord]:
    coords = get_park_boundary_coords(team.upper())
    if coords is None:
        raise HTTPException(status_code=404, detail=f"Park boundary not found for {team}")
    return [ParkCoord(x=c["x"], y=c["y"]) for c in coords]


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    return _run_optimize(req)


class OptimizePlotResponse(TypedDict):
    image_b64: str
    title: str
    situation: str
    positions: dict[str, dict]
    stats: dict
    balls: list[dict]
    park_boundary: list[dict] | None


@app.post("/api/optimize_plot")
def optimize_plot(req: OptimizeRequest) -> OptimizePlotResponse:
    import base64
    from .plot import render_plot
    resp = _run_optimize(req)
    png = render_plot(resp)
    return {
        "image_b64": base64.b64encode(png).decode(),
        "title": resp.title,
        "situation": resp.situation,
        "positions": {k: v.model_dump() for k, v in resp.positions.items()},
        "stats": resp.stats.model_dump(),
        "balls": [b.model_dump() for b in resp.balls],
        "park_boundary": [c.model_dump() for c in resp.park_boundary] if resp.park_boundary else None,
    }



def _run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    t_start = time.perf_counter()
    if req.home_team and req.home_team.upper() not in SUPPORTED_TEAMS:
        raise HTTPException(422, f"Unsupported team '{req.home_team}'. Use GET /api/teams.")
    if req.year not in _AVAILABLE_YEARS:
        raise HTTPException(422, f"No model for year {req.year}. Available: {_AVAILABLE_YEARS}")
    assert _re24_table is not None and _delta_re is not None and _hit_bundle is not None

    year      = req.year
    models_dir = BASE / "models" / str(year)
    home_team = req.home_team.upper() if req.home_team else None

    # ── Prepare ball data (cached: skip the DB query and KDE for the same batter/year) ──────────
    yr_balls_cache  = _batter_balls_cache.setdefault(year, {})
    yr_hprob_cache  = _batter_hitprobs_cache.setdefault(year, {})

    if req.batter_id not in yr_balls_cache:
        t_db = time.perf_counter()
        try:
            balls_all = prepare_batter_balls(req.batter_id, [year], DSN)
        except Exception as e:
            raise HTTPException(422, str(e))
        if balls_all.empty:
            raise HTTPException(422, f"Batter {req.batter_id} has no qualifying balls in {year}")
        yr_balls_cache[req.batter_id]  = balls_all
        yr_hprob_cache[req.batter_id]  = predict_hit_probs_batch(_hit_bundle, balls_all)
        logger.info(f"[timing] prepare_batter_balls+hit_probs (DB, cache miss): {time.perf_counter() - t_db:.2f}s")
    else:
        balls_all = yr_balls_cache[req.batter_id]

    hit_probs_all = yr_hprob_cache[req.batter_id]

    # Wall-ball flag (for the target park). Wall balls are kept in the data; during evaluation
    # the catch probability is forced to 0 and counted into RE24 (matching the thesis convention),
    # so they're no longer excluded from the data.
    wall_flags = (
        np.array(
            is_wall_ball(balls_all["ball_x"].values, balls_all["ball_y"].values, home_team),
            dtype=bool,
        )
        if home_team else np.zeros(len(balls_all), dtype=bool)
    )
    n_wall_balls = int(wall_flags.sum())

    # ── w_j (all balls, including wall balls) ──────────────────────────────────
    w_j = compute_w_j(
        balls_all, _hit_bundle, _delta_re,
        req.on_1b, req.on_2b, req.on_3b, req.outs,
        hit_probs=hit_probs_all,
    )
    mask = w_j > 0
    if not mask.any():
        raise HTTPException(422, "No balls with positive w_j for this game state")

    # ── RE24 state expectation (uses the cache loaded at startup, no re-reading the file) ─────────
    re_state = float(_re24_table.get((req.on_1b, req.on_2b, req.on_3b, req.outs), 0.0))

    # ── Specified outfielders (player-level ability); unspecified positions use the league-average group mu ──
    fielder_mus: dict[str, GroupMu] | None = None
    if req.fielders:
        fielder_mu_overrides: dict[str, GroupMu] = {}
        for pos in POSITIONS:
            player_name = req.fielders.get(pos)
            if player_name:
                try:
                    fielder_mu_overrides[pos] = load_player_params("OF", player_name, models_dir)
                except (KeyError, FileNotFoundError):
                    raise HTTPException(422, f"{pos} 找不到球員 '{player_name}' 的模型參數")
        fielder_mus = fielder_mu_overrides or None
    mus_eff = dict(_mus.get(year, {}))
    if fielder_mus:
        mus_eff.update(fielder_mus)

    # ── Positioning evaluation: wall-ball catch probability forced to 0, counted into RE24 (unified convention) ──────
    def eval_positions(pos_dict: OutfieldXY) -> tuple[np.ndarray, float, float]:
        probs = np.asarray(
            compute_ball_catch_probs(pos_dict, balls_all, _scalers.get(year, {}), mus_eff), dtype=float
        ).copy()
        probs[wall_flags] = 0.0                       # a wall ball can't be caught no matter where the fielder stands
        re24 = float(np.sum((1.0 - probs[mask]) * w_j[mask]))
        catch_pct = float(probs.mean() * 100)         # average over all balls (including wall balls)
        return probs, re24, catch_pct

    # ── League average positioning ─────────────────────────────────────────────
    try:
        league_avg_pos = get_league_avg_positions(year, DSN)
    except Exception:
        league_avg_pos = {"LF": (-130.0, 250.0), "CF": (0.0, 310.0), "RF": (130.0, 250.0)}

    def make_pos_set(pos_dict: OutfieldXY, obj: float, catch: float) -> PositionSet:
        return PositionSet(
            LF=PositionXY(x=pos_dict["LF"][0], y=pos_dict["LF"][1]),
            CF=PositionXY(x=pos_dict["CF"][0], y=pos_dict["CF"][1]),
            RF=PositionXY(x=pos_dict["RF"][0], y=pos_dict["RF"][1]),
            objective=obj,
            catch_pct=catch,
        )

    if fielder_mus:
        # ── Specified outfielders: compute only one custom position set (using the selected players' abilities) ──────
        t_opt = time.perf_counter()
        with _optimize_semaphore:
            opt_custom = optimize_positions(
                batter_id=req.batter_id,
                on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                home_team=home_team, dsn=DSN, fielder_mus=fielder_mus,
                balls=balls_all, hit_probs=hit_probs_all,
                delta_re=_delta_re, hit_bundle=_hit_bundle,
            )
        logger.info(f"[timing] optimize_positions(custom): {time.perf_counter() - t_opt:.2f}s")
        pos_custom = _outfield_xy(opt_custom)
        probs_custom, re_custom, catch_custom = eval_positions(pos_custom)
        positions_out: dict[str, PositionSet] = {"custom": make_pos_set(pos_custom, re_custom, catch_custom)}
        scatter_probs = probs_custom
    else:
        # ── General mode: league_avg + no_park (+ with_park) ─────────
        t_opt = time.perf_counter()
        with _optimize_semaphore:
            opt_no_park = optimize_positions(
                batter_id=req.batter_id,
                on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                home_team=None, dsn=DSN,
                balls=balls_all, hit_probs=hit_probs_all,
                n_restarts=10,
                delta_re=_delta_re, hit_bundle=_hit_bundle,
            )
        logger.info(f"[timing] optimize_positions(no_park): {time.perf_counter() - t_opt:.2f}s")
        pos_no_park = _outfield_xy(opt_no_park)
        probs_no_park, re_no_park, catch_no_park = eval_positions(pos_no_park)
        _, re_league, catch_league = eval_positions(league_avg_pos)
        positions_out = {
            "league_avg": make_pos_set(league_avg_pos, re_league, catch_league),
            "no_park":    make_pos_set(pos_no_park,    re_no_park, catch_no_park),
        }
        scatter_probs = probs_no_park
        if home_team:
            t_opt = time.perf_counter()
            with _optimize_semaphore:
                opt_with_park_res = optimize_positions(
                    batter_id=req.batter_id,
                    on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                    years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                    home_team=home_team, dsn=DSN,
                    balls=balls_all, hit_probs=hit_probs_all,
                    warm_start_xy=pos_no_park, n_restarts=8,
                    delta_re=_delta_re, hit_bundle=_hit_bundle,
                )
            logger.info(f"[timing] optimize_positions(with_park): {time.perf_counter() - t_opt:.2f}s")
            pos_with_park = _outfield_xy(opt_with_park_res)
            probs_with_park, re_with_park, catch_with_park = eval_positions(pos_with_park)
            positions_out["with_park"] = make_pos_set(pos_with_park, re_with_park, catch_with_park)
            scatter_probs = probs_with_park

    # ── Ball scatter points (all balls; wall balls have catch_prob=0, marked with an orange star on the frontend) ─────
    # responsible_fielder: the fielder with the highest catch probability for this ball (the model decides by distance)
    # catch_prob < 5% → not assigned to anyone
    primary_pos = (
        pos_custom if fielder_mus
        else (pos_with_park if home_team else pos_no_park)
    )
    # Responsibility assignment: nearest fielder (_catch_prob_single_fielder includes a directional-
    # angle feature; when comparing across positions, the angle term can make a farther fielder's
    # probability come out higher, so it's not suitable for dividing responsibility ranges)
    bx_arr = balls_all["ball_x"].values
    by_arr = balls_all["ball_y"].values
    dists = {
        code: np.hypot(bx_arr - primary_pos[code][0], by_arr - primary_pos[code][1])
        for code in POSITIONS
    }

    def _nearest_fielder(i: int) -> str:
        return min(POSITIONS, key=lambda position_code: dists[position_code][i])

    nearest = [_nearest_fielder(i) for i in range(len(balls_all))]

    balls_out = [
        BallPoint(
            x=float(balls_all.iloc[i]["ball_x"]),
            y=float(balls_all.iloc[i]["ball_y"]),
            catch_prob=float(scatter_probs[i]),
            is_wall_ball=bool(wall_flags[i]),
            responsible=nearest[i] if scatter_probs[i] >= 0.05 else None,
        )
        for i in range(len(balls_all))
    ]

    # ── Park boundary ─────────────────────────────────────────────────
    park_boundary = None
    if home_team:
        coords = get_park_boundary_coords(home_team)
        park_boundary = [ParkCoord(x=c["x"], y=c["y"]) for c in coords] if coords else None

    # ── Title ────────────────────────────────────────────────────
    raw_name = _name_map.get(req.batter_id, f"#{req.batter_id}")
    display_name = raw_name.replace(", ", " ") if ", " in raw_name else raw_name
    stand = get_batter_stand(req.batter_id, year, DSN)
    title = f"{display_name} ({year}, {stand}HB)"
    if home_team:
        title += f" @ {home_team}"
    if fielder_mus:
        assert req.fielders is not None
        tags = [
            f"{p}:{fielder_name.split(',')[0]}" if (fielder_name := req.fielders.get(p)) else f"{p}:avg"
            for p in POSITIONS
        ]
        title += " | " + " ".join(tags)

    bases = ("1" if req.on_1b else "-") + ("2" if req.on_2b else "-") + ("3" if req.on_3b else "-")
    situation = f"{bases}  {req.outs} out"

    logger.info(f"[timing] _run_optimize TOTAL: {time.perf_counter() - t_start:.2f}s "
                f"(batter={req.batter_id}, year={year}, home_team={home_team}, fielders={bool(fielder_mus)})")

    return OptimizeResponse(
        title=title,
        situation=situation,
        positions=positions_out,
        balls=balls_out,
        park_boundary=park_boundary,
        stats=OptimizeStats(
            n_balls=len(balls_all),
            n_wall_balls=n_wall_balls,
            re_state=re_state,
            home_team=home_team,
        ),
    )


# ── Frontend static files (placed last so all /api/* routes are already registered and won't conflict) ─────────
# At deployment, a single service serves both the API and the frontend build output (frontend/dist);
# since they're same-origin, the frontend's frontend/src/api.js relative path '/api' needs no extra base URL or CORS setup.
_FRONTEND_DIST: Path = BASE / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str) -> FileResponse:
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")
