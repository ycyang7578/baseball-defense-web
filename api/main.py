"""
Baseball Defense Optimizer — FastAPI backend

完整端點清單見 ARCHITECTURE.md「API（FastAPI）」章節。
POST /api/optimize 系列耗時受 n_restarts/併發影響，實測數字見 ARCHITECTURE.md「已知效能限制」章節，
不在這裡重複維護避免兩處數字不同步。
"""
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.optimization import (
    optimize_positions, prepare_batter_balls, compute_w_j,
    compute_ball_catch_probs,
    get_league_avg_positions, get_batter_stand, load_qualifying_batters,
    load_model_params, load_player_params, POSITIONS,
)
from src.config import DSN
from src.hit_prob import predict_hit_probs_batch, load_hit_prob
from src.re24 import load_re24
from src.stadium_walls import SUPPORTED_TEAMS, get_park_boundary_coords, is_wall_ball
from src.if_optimize import (expected_outs as if_expected_outs,
                             optimize_infield, positions_to_params,
                             predict_p_out)
from .schemas import (
    BatterInfo, OptimizeRequest, OptimizeResponse,
    BallPoint, ParkCoord, PositionSet, PositionXY, OptimizeStats, FielderInfo,
    IFBallPoint, IFBatterInfo, IFCustomResultResponse, IFFielderInfo,
    IFFielderOption, IFPosition, IFPositionSet, IFResultResponse, IFStats,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE    = Path(__file__).parent.parent
PRE_DIR = BASE / "data" / "precomputed"
_MIN_BALLS = 30

# Render 免費方案只有 0.1 CPU：多個 optimize_positions 同時跑會互搶 CPU、
# 讓每一個都變慢（實測：單獨跑 50s，兩個同時跑各要 70~80s）。用 semaphore
# 把 CPU 密集的優化計算序列化，避免併發請求（如比較模式 A/B 同時送出）互相拖慢。
_optimize_semaphore = threading.Semaphore(1)

# 掃描哪些年份有模型 summary
_AVAILABLE_YEARS: list[int] = sorted(
    y for y in range(2020, 2030)
    if (BASE / "models" / str(y) / "OF" / "OF_summary_players.csv").exists()
)
_DEFAULT_YEAR = _AVAILABLE_YEARS[-1] if _AVAILABLE_YEARS else 2025

# ── 啟動快取 ──────────────────────────────────────────────────────
_name_map:    dict[int, str]  = {}
_delta_re   = None
_hit_bundle = None

# year-keyed caches
_scalers:       dict[int, dict] = {}          # year → pos → scaler
_mus:           dict[int, dict] = {}          # year → pos → mus
_batters_cache: dict[int, list[dict]] = {}    # year → list[{batter_id, name, n_balls}]

# ── 打者資料快取（同打者換壘況時跳過 DB 查詢與 KDE）───────────────
_batter_balls_cache:    dict[int, dict[int, object]] = {}  # year → batter_id → DataFrame
_batter_hitprobs_cache: dict[int, dict[int, object]] = {}  # year → batter_id → ndarray

# ── Rankings 多年份快取（year → ...）────────────────────────────
_fielders_cache: dict[int, dict[str, list[dict]]] = {}  # year → pos → list
_model_names:    dict[int, dict[str, set]]        = {}  # year → pos → name set
_team_map:       dict[int, dict[int, int]]        = {}  # year → player_id → team_id

# ── 內野快取（結果全部離線預算，見 scripts/precompute_if_optimize.py）──
IF_POSITIONS = ("1B", "2B", "3B", "SS")
_if_years:          list[int] = []                       # precomputed_if_positions 有的年份
_if_ranking_years:  list[int] = []                       # if_model_oaa 有的年份
_if_batters_cache:  dict[int, list[dict]] = {}           # year → [{batter_id, name, n_gb, stand}]
_if_league:         dict[int, dict[str, list[float]]] = {}  # year → pos → [angle, depth]
_if_team_map:       dict[int, dict[int, int]] = {}       # year → player_id → team_id
# 個人化站位（貝葉斯球員層，見 scripts/train_if_bayes.py / export_if_bayes.py）
_if_bayes_model = None                                    # 群體層 pipeline（joblib）
_if_effects:        dict[int, tuple[float, float]] = {}  # player_id → (alpha, g)
_if_ad_norm:        tuple[float, float] | None = None    # (ad_mean, ad_std)
_if_fielder_opts:   dict[int, dict[str, list[dict]]] = {}  # year → pos → options


def _load_infield_caches() -> None:
    """內野快取。資料表可能還沒建立/還沒 sync 到這顆 DB，缺了不中斷啟動，
    內野端點會回空清單/404。"""
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
    logger.info(f"內野: years={_if_years}, ranking years={_if_ranking_years}, "
                f"bayes={'on' if _if_bayes_model is not None else 'off'}, "
                + ", ".join(f"{y}={len(_if_batters_cache[y])} batters" for y in _if_years))


def _load_if_bayes() -> None:
    """個人化站位資產：貝葉斯群體層 pipeline＋球員效應＋野手選單。缺了不中斷
    啟動，/api/if_fielder_options 與 /api/if_result_custom 回 404/503。"""
    global _if_bayes_model, _if_ad_norm
    import joblib
    import pandas as pd

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
    """野手選單快取。獨立成函式：啟動時對 DB 的暫時性失敗不該讓功能死到重啟
    （2026-07-13 線上實例即因此選單 404），if_fielder_options 會惰性重試。"""
    try:
        opts: dict[int, dict[str, list[dict]]] = {}
        with psycopg2.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('fielder_positioning')")
                if cur.fetchone()[0] is None:
                    logger.warning("fielder_positioning 不存在，野手選單停用")
                    return
                cur.execute(
                    "SELECT DISTINCT season, position, fielder_id "
                    "FROM fielder_positioning WHERE position IN %s "
                    "ORDER BY season, position", (IF_POSITIONS,))
                for season, pos, fid in cur.fetchall():
                    (opts.setdefault(int(season), {})
                     .setdefault(pos, []).append({
                         "player_id": int(fid),
                         "name": _name_map.get(int(fid), f"#{fid}"),
                         "has_effects": int(fid) in _if_effects}))
        for year in opts.values():
            for lst in year.values():
                lst.sort(key=lambda o: (not o["has_effects"], o["name"]))
        _if_fielder_opts.clear()
        _if_fielder_opts.update(opts)
        logger.info(f"野手選單: {sorted(_if_fielder_opts)} 年份已載入")
    except Exception as e:
        logger.warning(f"野手選單載入失敗: {e}")


def _load_fielders(year: int) -> dict[str, list[dict]]:
    """指定年度每位置外野手清單（需要該年 models/{year}/OF/OF_summary_players.csv）。"""
    import re
    import pandas as pd

    models_dir  = BASE / "models" / str(year) / "OF"
    players_csv = models_dir / "OF_summary_players.csv"
    if not players_csv.exists():
        logger.warning(f"No model summary for {year}, skipping")
        return {pos: [] for pos in POSITIONS}

    df_players = pd.read_csv(players_csv, index_col=0, encoding="utf-8-sig")
    of_names = {
        re.match(r"alpha\[(.+)\]", str(i)).group(1)
        for i in df_players.index if str(i).startswith("alpha[")
    }
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
    out: dict[str, list[dict]] = {}
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
    """MLB Stats API 批次查指定賽季守備 splits 取 MLB 球隊（player_id → team_id）。
    traded players 取出賽數最多的 MLB 球隊。失敗不中斷啟動。"""
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


def _load_batters(year: int) -> list[dict]:
    return load_qualifying_batters(year, DSN, _MIN_BALLS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _delta_re, _hit_bundle

    # 名稱快取
    name_path = BASE / "data" / "reference" / "batter_names.json"
    if name_path.exists():
        raw = json.loads(name_path.read_text(encoding="utf-8"))
        _name_map.update({int(k): v for k, v in raw.items()})
    logger.info(f"Loaded {len(_name_map)} batter names")

    # 預計算資料（RE24 / hit prob KDE — 年份無關）
    _, _delta_re = load_re24(PRE_DIR)
    _hit_bundle  = load_hit_prob(PRE_DIR)
    logger.info("Preloaded RE24, KDE")

    # 各年度：打者清單 + 模型參數
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

    # 各年度外野手清單（同時建立 _model_names 供動態查詢用）
    logger.info(f"Available ranking years: {_AVAILABLE_YEARS}")
    for yr in _AVAILABLE_YEARS:
        _fielders_cache[yr] = _load_fielders(yr)
        logger.info(f"  {yr}: " + ", ".join(f"{p}={len(_fielders_cache[yr][p])}" for p in POSITIONS))
        # 查全部球員的 player_id（不限 n_opp），確保低機會球員也有球隊資訊
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
def get_batters(year: int = _DEFAULT_YEAR):
    return _batters_cache.get(year, _batters_cache.get(_DEFAULT_YEAR, []))


@app.get("/api/teams", response_model=list[str])
def get_teams():
    return SUPPORTED_TEAMS


@app.get("/api/years")
def get_years():
    return sorted(_AVAILABLE_YEARS)


def _compute_avg_oaa_per_ball(rows, yr_model_names: dict) -> float:
    """跨 LF+CF+RF 統一中心化用的聯盟平均：只用有模型參數的球員計算。

    rows: (name, position, model_oaa, n_opp, ...) 的可迭代物件，只用前四欄。
    """
    visible = [(float(oaa), int(n))
               for name, pos, oaa, n, *_ in rows
               if name in yr_model_names.get(pos, set())]
    total_oaa = sum(r[0] for r in visible)
    total_opp = sum(r[1] for r in visible)
    return total_oaa / total_opp if total_opp else 0.0


@app.get("/api/player_trend")
def player_trend(name: str):
    """Return year-by-year centered OAA/100 matching the Rankings table."""
    result = []
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

                for nm, pos, oaa, n in all_rows:
                    if nm == name and nm in yr_model_names.get(pos, set()):
                        c = float(oaa) - avg_per_ball * int(n)
                        result.append({
                            "year": yr, "position": pos,
                            "oaa": round(c, 2), "n_opp": int(n),
                            "rate": round(c / int(n) * 100, 2) if n else None,
                        })
    return sorted(result, key=lambda x: x["year"])


@app.get("/api/fielders", response_model=dict[str, list[FielderInfo]])
def get_fielders(year: int = 2025, min_opp: int = 100):
    if year not in _fielders_cache:
        raise HTTPException(404, f"No ranking data for year {year}. Available: {sorted(_AVAILABLE_YEARS)}")

    yr_model_names = _model_names.get(year, {})
    yr_team_map    = _team_map.get(year, {})

    _SQL_ALL = """
        SELECT m.name_fielder, m.position, m.model_oaa, m.n_opp,
               MAX(o.player_id) AS player_id
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

    result: dict[str, list[dict]] = {}
    for pos in POSITIONS:
        rows_pos = [
            (name, float(oaa) - avg_oaa_per_ball * int(n), int(n), pid)
            for name, p, oaa, n, pid in all_rows
            if p == pos and name in yr_model_names.get(pos, set())
        ]
        filtered = [(name, c, n, pid) for name, c, n, pid in rows_pos if n >= min_opp]
        filtered.sort(key=lambda x: x[1] / x[2] if x[2] else 0, reverse=True)
        result[pos] = [{"name": name, "oaa": round(c, 2), "n_opp": n,
                        "player_id": pid,
                        "team_id": yr_team_map.get(pid) if pid else None}
                       for name, c, n, pid in filtered]
    return result


@app.get("/api/star_stats")
def get_star_stats(year: int = _DEFAULT_YEAR):
    # 讀我方模型算出的星級分布（model_star_stats），跨位置已合併
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

    result = {}
    for row in rows:
        name = row[0]
        stars = []
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


# ── 內野端點（結果全部離線預算，查表即回，無運算）───────────────────

@app.get("/api/if_years")
def if_years():
    return _if_years


@app.get("/api/if_batters", response_model=list[IFBatterInfo])
def if_batters(year: int | None = None):
    if year is None and _if_years:
        year = _if_years[-1]
    return _if_batters_cache.get(year, [])


def _if_position_set(pairs: dict[str, tuple[float, float]], exp_outs: float) -> IFPositionSet:
    import math
    positions = {}
    for pos, (angle, depth) in pairs.items():
        rad = math.radians(angle)
        positions[pos] = IFPosition(x=round(depth * math.sin(rad), 1),
                                    y=round(depth * math.cos(rad), 1),
                                    angle=angle, depth=depth)
    return IFPositionSet(positions=positions, exp_outs=exp_outs)


@app.get("/api/if_result", response_model=IFResultResponse)
def if_result(batter_id: int, year: int):
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
    league_pairs = {pos: tuple(_if_league[year][pos]) for pos in IF_POSITIONS}
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
def if_fielder_options(year: int = 2025):
    """個人化站位的野手選單（該年有站位資料的野手，依位置分組）。"""
    if _if_bayes_model is None:
        raise HTTPException(503, "個人化模型未載入")
    if year not in _if_fielder_opts:
        _load_if_fielder_menu()  # 啟動時可能因暫時性 DB 失敗而空，惰性重建
    if year not in _if_fielder_opts:
        raise HTTPException(404, f"No fielder options for year {year}")
    return _if_fielder_opts[year]


@app.get("/api/if_result_custom", response_model=IFCustomResultResponse)
def if_result_custom(batter_id: int, year: int,
                     fielder_1b: int | None = None, fielder_2b: int | None = None,
                     fielder_3b: int | None = None, fielder_ss: int | None = None):
    """指定野手陣容的個人化站位（錨定式：從零效應最佳解 warm start 局部優化，
    位移只反映球員效應的拉力，不是平坦地形的等值漂移——見 ARCHITECTURE.md
    「內野貝葉斯球員層」）。未指定的位置視為聯盟平均野手（效應 0）。"""
    import numpy as np
    import pandas as pd

    if _if_bayes_model is None:
        raise HTTPException(503, "個人化模型未載入")
    if year not in _if_years:
        raise HTTPException(404, f"No infield data for year {year}. Available: {_if_years}")
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
                raise HTTPException(404, f"Batter {batter_id} has no precomputed result in {year}")
            cur.execute(
                "SELECT spray_deg, ball_x, ball_y, launch_speed, launch_angle, is_out "
                "FROM precomputed_if_gbs "
                "WHERE batter = %s AND game_year = %s", (batter_id, year))
            ball_rows = cur.fetchall()

    stand, n_gb, hp_to_1b = row[:3]
    opt_angles = np.array([row[3], row[5], row[7], row[9]], dtype=float)
    opt_depths = np.array([row[4], row[6], row[8], row[10]], dtype=float)

    balls = pd.DataFrame(ball_rows, columns=[
        "spray_deg", "ball_x", "ball_y", "launch_speed", "launch_angle", "is_out"])
    balls["launch_angle"] = pd.to_numeric(balls["launch_angle"], errors="coerce")
    balls["launch_angle"] = balls["launch_angle"].fillna(balls["launch_angle"].median())
    balls["launch_speed"] = balls["launch_speed"].astype(float)
    balls["hp_to_1b"] = float(hp_to_1b)
    balls["stand_R"] = int(stand == "R")

    fids = {"1B": fielder_1b, "2B": fielder_2b, "3B": fielder_3b, "SS": fielder_ss}
    alpha = np.array([_if_effects.get(fids[p], (0.0, 0.0))[0] if fids[p] else 0.0
                      for p in IF_POSITIONS])
    g = np.array([_if_effects.get(fids[p], (0.0, 0.0))[1] if fids[p] else 0.0
                  for p in IF_POSITIONS])
    pe = {"alpha": alpha, "g": g,
          "ad_mean": _if_ad_norm[0], "ad_std": _if_ad_norm[1]}

    warm = positions_to_params(opt_angles, opt_depths)
    with _optimize_semaphore:
        res = optimize_infield(balls, _if_bayes_model, n_restarts=0,
                               extra_starts=[warm], player_effects=pe)

    league_pairs = {pos: tuple(_if_league[year][pos]) for pos in IF_POSITIONS}
    lg_angles = np.array([league_pairs[p][0] for p in IF_POSITIONS])
    lg_depths = np.array([league_pairs[p][1] for p in IF_POSITIONS])
    exp_league = if_expected_outs(_if_bayes_model, balls, lg_angles, lg_depths, pe)
    baseline = if_expected_outs(_if_bayes_model, balls, opt_angles, opt_depths, pe)
    p_league = predict_p_out(_if_bayes_model, balls, lg_angles, lg_depths, pe)
    p_custom = predict_p_out(_if_bayes_model, balls, res["angles"], res["depths"], pe)

    custom_pairs = {pos: (round(float(a), 2), round(float(d), 2))
                    for pos, a, d in zip(IF_POSITIONS, res["angles"], res["depths"])}
    gain = res["exp_outs"] - exp_league
    return IFCustomResultResponse(
        batter_id=batter_id,
        name=_name_map.get(batter_id, f"#{batter_id}"),
        year=year, stand=stand,
        fielders={p: (_name_map.get(fids[p], f"#{fids[p]}") if fids[p] else None)
                  for p in IF_POSITIONS},
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


@app.get("/api/if_fielders", response_model=dict[str, list[IFFielderInfo]])
def if_fielders(year: int = 2025, min_balls: int = 100):
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
    result: dict[str, list[dict]] = {}
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


@app.get("/api/park_boundary/{team}", response_model=list[ParkCoord] | None)
def park_boundary(team: str):
    coords = get_park_boundary_coords(team.upper())
    if coords is None:
        raise HTTPException(status_code=404, detail=f"Park boundary not found for {team}")
    return coords


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    return _run_optimize(req)


@app.post("/api/optimize_plot")
def optimize_plot(req: OptimizeRequest):
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

    year      = req.year
    models_dir = BASE / "models" / str(year)
    home_team = req.home_team.upper() if req.home_team else None

    # ── 準備球資料（快取：同打者同年份跳過 DB 查詢與 KDE）──────────
    import numpy as np

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

    # 打牆球旗標（以目標球場）。打牆球保留在資料中，評估時強制接殺機率 0、
    # 計入 RE24（對齊論文口徑），不再從資料中排除。
    wall_flags = (
        np.array(
            is_wall_ball(balls_all["ball_x"].values, balls_all["ball_y"].values, home_team),
            dtype=bool,
        )
        if home_team else np.zeros(len(balls_all), dtype=bool)
    )
    n_wall_balls = int(wall_flags.sum())

    # ── w_j（全部球，含打牆球）──────────────────────────────────
    w_j = compute_w_j(
        balls_all, _hit_bundle, _delta_re,
        req.on_1b, req.on_2b, req.on_3b, req.outs,
        hit_probs=hit_probs_all,
    )
    mask = w_j > 0
    if not mask.any():
        raise HTTPException(422, "No balls with positive w_j for this game state")

    # ── RE24 狀態期望值 ──────────────────────────────────────────
    re_table, _ = load_re24(PRE_DIR)
    re_state    = float(re_table.get((req.on_1b, req.on_2b, req.on_3b, req.outs), 0.0))

    # ── 指定外野手（player-level 能力）；未指定的位置用聯盟平均 group mu ──
    fielder_mus = None
    if req.fielders:
        fm = {}
        for pos in POSITIONS:
            nm = req.fielders.get(pos)
            if nm:
                try:
                    fm[pos] = load_player_params("OF", nm, models_dir)
                except (KeyError, FileNotFoundError):
                    raise HTTPException(422, f"{pos} 找不到球員 '{nm}' 的模型參數")
        fielder_mus = fm or None
    mus_eff = dict(_mus.get(year, {}))
    if fielder_mus:
        mus_eff.update(fielder_mus)

    # ── 站位評估：打牆球強制接殺機率 0、計入 RE24（統一口徑）──────
    def eval_positions(pos_dict):
        probs = np.asarray(
            compute_ball_catch_probs(pos_dict, balls_all, _scalers.get(year, {}), mus_eff), dtype=float
        ).copy()
        probs[wall_flags] = 0.0                       # 打牆球無論站哪都接不到
        re24 = float(np.sum((1.0 - probs[mask]) * w_j[mask]))
        catch_pct = float(probs.mean() * 100)         # 全部球（含打牆球）平均
        return probs, re24, catch_pct

    # ── 聯盟平均站位 ─────────────────────────────────────────────
    try:
        league_avg_pos = get_league_avg_positions(year, DSN)
    except Exception:
        league_avg_pos = {"LF": (-130.0, 250.0), "CF": (0.0, 310.0), "RF": (130.0, 250.0)}

    def make_pos_set(pos_dict, obj, catch):
        return PositionSet(
            LF=PositionXY(x=pos_dict["LF"][0], y=pos_dict["LF"][1]),
            CF=PositionXY(x=pos_dict["CF"][0], y=pos_dict["CF"][1]),
            RF=PositionXY(x=pos_dict["RF"][0], y=pos_dict["RF"][1]),
            objective=obj,
            catch_pct=catch,
        )

    if fielder_mus:
        # ── 指定外野手：只算一組 custom 站位（用選定球員能力）──────
        t_opt = time.perf_counter()
        with _optimize_semaphore:
            opt_custom = optimize_positions(
                batter_id=req.batter_id,
                on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                home_team=home_team, dsn=DSN, fielder_mus=fielder_mus,
                balls=balls_all, hit_probs=hit_probs_all,
            )
        logger.info(f"[timing] optimize_positions(custom): {time.perf_counter() - t_opt:.2f}s")
        pos_custom = {p: opt_custom[p] for p in POSITIONS}
        probs_custom, re_custom, catch_custom = eval_positions(pos_custom)
        positions_out = {"custom": make_pos_set(pos_custom, re_custom, catch_custom)}
        scatter_probs = probs_custom
    else:
        # ── 一般模式：league_avg + no_park (+ with_park) ─────────
        t_opt = time.perf_counter()
        with _optimize_semaphore:
            opt_no_park = optimize_positions(
                batter_id=req.batter_id,
                on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                home_team=None, dsn=DSN,
                balls=balls_all, hit_probs=hit_probs_all,
                n_restarts=10,
            )
        logger.info(f"[timing] optimize_positions(no_park): {time.perf_counter() - t_opt:.2f}s")
        pos_no_park = {p: opt_no_park[p] for p in POSITIONS}
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
                )
            logger.info(f"[timing] optimize_positions(with_park): {time.perf_counter() - t_opt:.2f}s")
            pos_with_park = {p: opt_with_park_res[p] for p in POSITIONS}
            probs_with_park, re_with_park, catch_with_park = eval_positions(pos_with_park)
            positions_out["with_park"] = make_pos_set(pos_with_park, re_with_park, catch_with_park)
            scatter_probs = probs_with_park

    # ── 球散點（全部球；打牆球 catch_prob=0、前端以橘星標示）─────
    # responsible_fielder：對這顆球接殺機率最高的守備員（模型以距離決定）
    # catch_prob < 5% → 不歸任何人管
    primary_pos = (
        pos_custom if fielder_mus
        else (pos_with_park if home_team else pos_no_park)
    )
    # 責任分配：最近守備員（_catch_prob_single_fielder 含方向角特徵，
    # 跨位置比較時角度項可能讓較遠守備員機率反而更高，不適合用來切割責任範圍）
    bx_arr = balls_all["ball_x"].values
    by_arr = balls_all["ball_y"].values
    dists = {
        code: np.hypot(bx_arr - primary_pos[code][0], by_arr - primary_pos[code][1])
        for code in POSITIONS
    }
    nearest = [min(POSITIONS, key=lambda c, i=i: dists[c][i]) for i in range(len(balls_all))]

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

    # ── 球場邊界 ─────────────────────────────────────────────────
    park_boundary = None
    if home_team:
        coords = get_park_boundary_coords(home_team)
        park_boundary = [ParkCoord(x=c["x"], y=c["y"]) for c in coords] if coords else None

    # ── 標題 ────────────────────────────────────────────────────
    raw_name = _name_map.get(req.batter_id, f"#{req.batter_id}")
    display_name = raw_name.replace(", ", " ") if ", " in raw_name else raw_name
    stand = get_batter_stand(req.batter_id, year, DSN)
    title = f"{display_name} ({year}, {stand}HB)"
    if home_team:
        title += f" @ {home_team}"
    if fielder_mus:
        tags = [
            f"{p}:{req.fielders.get(p).split(',')[0]}" if req.fielders.get(p) else f"{p}:avg"
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


# ── 前端靜態檔案（放最後，所有 /api/* 路由都已註冊完畢，不會衝突）─────────
# 部署時單一服務同時提供 API 與前端建置產物（frontend/dist），兩者同源，
# 前端 frontend/src/api.js 的相對路徑 '/api' 不用額外設定 base URL 或 CORS。
_FRONTEND_DIST = BASE / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")
