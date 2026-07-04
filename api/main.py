"""
Baseball Defense Optimizer — FastAPI backend

Endpoints:
  GET  /api/batters              列出 2025 可查詢打者（含姓名）
  GET  /api/teams                列出支援球場縮寫
  GET  /api/park_boundary/{team} 回傳球場圍牆多邊形座標
  POST /api/optimize             計算最佳外野站位（同步，約 10-20s）
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.optimization import (
    optimize_positions, prepare_batter_balls, compute_w_j,
    compute_ball_catch_probs, compute_per_fielder_probs,
    get_league_avg_positions, get_batter_stand,
    load_model_params, load_player_params, POSITIONS,
)
from src.config import DSN
from src.hit_prob import predict_hit_probs_batch
from src.re24 import load_re24
from src.hit_prob import load_hit_prob
from src.stadium_walls import SUPPORTED_TEAMS, get_park_boundary_coords, is_wall_ball
from .schemas import (
    BatterInfo, OptimizeRequest, OptimizeResponse,
    BallPoint, ParkCoord, PositionSet, PositionXY, OptimizeStats, FielderInfo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE    = Path(__file__).parent.parent
PRE_DIR = BASE / "data" / "precomputed"
_MIN_BALLS = 30

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
    query = """
        SELECT batter, COUNT(*) AS n_balls
        FROM statcast
        WHERE game_year  = %(year)s
          AND game_type  = 'R'
          AND type       = 'X'
          AND bb_type    IN ('fly_ball', 'line_drive')
          AND events     != 'home_run'
          AND hit_distance_sc IS NOT NULL
          AND launch_speed    IS NOT NULL
          AND launch_angle    IS NOT NULL
          AND hc_x            IS NOT NULL
          AND hc_y            IS NOT NULL
        GROUP BY batter
        HAVING COUNT(*) >= %(min_balls)s
        ORDER BY n_balls DESC
    """
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"year": year, "min_balls": _MIN_BALLS})
            rows = cur.fetchall()
    return [{"batter_id": r[0], "n_balls": r[1]} for r in rows]


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
        try:
            balls_all = prepare_batter_balls(req.batter_id, [year], DSN)
        except Exception as e:
            raise HTTPException(422, str(e))
        if balls_all.empty:
            raise HTTPException(422, f"Batter {req.batter_id} has no qualifying balls in {year}")
        yr_balls_cache[req.batter_id]  = balls_all
        yr_hprob_cache[req.batter_id]  = predict_hit_probs_batch(_hit_bundle, balls_all)
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
        opt_custom = optimize_positions(
            batter_id=req.batter_id,
            on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
            years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
            home_team=home_team, dsn=DSN, fielder_mus=fielder_mus,
            balls=balls_all, hit_probs=hit_probs_all,
        )
        pos_custom = {p: opt_custom[p] for p in POSITIONS}
        probs_custom, re_custom, catch_custom = eval_positions(pos_custom)
        positions_out = {"custom": make_pos_set(pos_custom, re_custom, catch_custom)}
        scatter_probs = probs_custom
    else:
        # ── 一般模式：league_avg + no_park (+ with_park) ─────────
        opt_no_park = optimize_positions(
            batter_id=req.batter_id,
            on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
            years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
            home_team=None, dsn=DSN,
            balls=balls_all, hit_probs=hit_probs_all,
        )
        pos_no_park = {p: opt_no_park[p] for p in POSITIONS}
        probs_no_park, re_no_park, catch_no_park = eval_positions(pos_no_park)
        _, re_league, catch_league = eval_positions(league_avg_pos)
        positions_out = {
            "league_avg": make_pos_set(league_avg_pos, re_league, catch_league),
            "no_park":    make_pos_set(pos_no_park,    re_no_park, catch_no_park),
        }
        scatter_probs = probs_no_park
        if home_team:
            opt_with_park_res = optimize_positions(
                batter_id=req.batter_id,
                on_1b=req.on_1b, on_2b=req.on_2b, on_3b=req.on_3b, outs=req.outs,
                years=[year], models_dir=models_dir, re24_dir=PRE_DIR,
                home_team=home_team, dsn=DSN,
                balls=balls_all, hit_probs=hit_probs_all,
            )
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
