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
    compute_ball_catch_probs, get_league_avg_positions, get_batter_stand,
    load_model_params, load_player_params, POSITIONS,
)
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

BASE       = Path(__file__).parent.parent
MODELS_DIR = BASE / "models" / "2025"
PRE_DIR    = BASE / "data" / "precomputed"
YEAR       = 2025
TRAIN_YEARS = [2021, 2022, 2023, 2024]
DSN        = "host=localhost dbname=baseball user=postgres password=postgres"
_MIN_BALLS = 30

# ── 啟動快取 ──────────────────────────────────────────────────────
_batters_cache:  list[dict]   = []
_name_map:       dict[int, str] = {}
_fielders_cache: dict[str, list[dict]] = {}
_delta_re      = None
_hit_bundle    = None
_scalers       = {}
_mus           = {}

# ── 打者資料快取（同打者換壘況時跳過 DB 查詢與 KDE）───────────────
_batter_balls_cache:    dict[int, object] = {}   # batter_id → DataFrame
_batter_hitprobs_cache: dict[int, object] = {}   # batter_id → ndarray (N,3)
_model_names:           dict[str, set]    = {}   # pos → set of player names with model params
_team_map:              dict[int, int]    = {}   # player_id → MLB team_id


def _load_fielders() -> dict[str, list[dict]]:
    """每位置：清單來源為「合併 OF 模型有 player-level 參數的球員」。
    LF/CF/RF 共用同一組 OF 球員集合（574 人），再依 model_oaa 位置欄分類顯示。"""
    import re
    import pandas as pd

    # 從合併 OF 模型讀出全部有球員層參數的球員名稱集合，三個位置共用
    global _model_names
    players_csv = MODELS_DIR / "OF" / "OF_summary_players.csv"
    df_players  = pd.read_csv(players_csv, index_col=0, encoding="utf-8-sig")
    of_names = {
        re.match(r"alpha\[(.+)\]", str(i)).group(1)
        for i in df_players.index if str(i).startswith("alpha[")
    }
    for pos in POSITIONS:
        _model_names[pos] = of_names

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
                cur.execute(_SQL, {"year": YEAR, "pos": pos})
                rows = cur.fetchall()
                out[pos] = [
                    {"name": name, "oaa": float(oaa), "n_opp": n_opp, "player_id": pid}
                    for name, oaa, n_opp, pid in rows
                    if name in _model_names[pos]
                ]
    return out


_MLB_TEAM_IDS = {
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 158,
}

def _load_team_info(player_ids: list[int]) -> dict[int, int]:
    """MLB Stats API 批次查 2025 守備成績 splits 取 MLB 球隊（player_id → team_id）。
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
                    "hydrate": "stats(group=fielding,type=season,season=2025)",
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
            logger.warning(f"MLB Stats API team lookup failed: {e}")
    return result


def _load_batters() -> list[dict]:
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
            cur.execute(query, {"year": YEAR, "min_balls": _MIN_BALLS})
            rows = cur.fetchall()
    return [{"batter_id": r[0], "n_balls": r[1]} for r in rows]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _delta_re, _hit_bundle, _scalers, _mus

    # 名稱快取
    name_path = BASE / "data" / "reference" / "batter_names.json"
    if name_path.exists():
        raw = json.loads(name_path.read_text(encoding="utf-8"))
        _name_map.update({int(k): v for k, v in raw.items()})
    logger.info(f"Loaded {len(_name_map)} batter names")

    # 打者清單
    rows = _load_batters()
    for r in rows:
        bid  = r["batter_id"]
        name = _name_map.get(bid, f"#{bid}")
        _batters_cache.append({"batter_id": bid, "name": name, "n_balls": r["n_balls"]})
    logger.info(f"Loaded {len(_batters_cache)} batters")

    # 預計算資料 + 模型參數（避免每次請求重載）
    _, _delta_re = load_re24(PRE_DIR)
    _hit_bundle  = load_hit_prob(PRE_DIR)
    of_scaler, of_mus = load_model_params("OF", MODELS_DIR)
    for pos in POSITIONS:
        _scalers[pos] = of_scaler
        _mus[pos]     = of_mus
    logger.info("Preloaded RE24, KDE, model params (unified OF model)")

    # 各位置可選外野手清單（同時建立 _model_names 供動態查詢用）
    _fielders_cache.update(_load_fielders())
    logger.info("Loaded fielders: " + ", ".join(f"{p}={len(_fielders_cache[p])}" for p in POSITIONS))

    # MLB Stats API：查各 player 所屬球隊（currentTeam）
    all_pids = list({f["player_id"] for pos_list in _fielders_cache.values()
                     for f in pos_list if f.get("player_id")})
    _team_map.update(_load_team_info(all_pids))
    logger.info(f"Loaded team info for {len(_team_map)} players")

    yield


app = FastAPI(title="Baseball Defense Optimizer", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/api/batters", response_model=list[BatterInfo])
def get_batters():
    return _batters_cache


@app.get("/api/teams", response_model=list[str])
def get_teams():
    return SUPPORTED_TEAMS


@app.get("/api/fielders", response_model=dict[str, list[FielderInfo]])
def get_fielders(min_opp: int = 100):
    # 跨 LF+CF+RF 統一中心化：avg_oaa_per_ball 從有 OF 模型參數的球員計算
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
            cur.execute(_SQL_ALL, {"year": YEAR})
            all_rows = cur.fetchall()

    visible = [(float(oaa), int(n))
               for name, pos, oaa, n, pid in all_rows
               if name in _model_names.get(pos, set())]
    total_oaa = sum(r[0] for r in visible)
    total_opp = sum(r[1] for r in visible)
    avg_oaa_per_ball = total_oaa / total_opp if total_opp else 0.0

    result: dict[str, list[dict]] = {}
    for pos in POSITIONS:
        rows_pos = [
            (name, float(oaa) - avg_oaa_per_ball * int(n), int(n), pid)
            for name, p, oaa, n, pid in all_rows
            if p == pos and name in _model_names.get(pos, set())
        ]
        filtered = [(name, c, n, pid) for name, c, n, pid in rows_pos if n >= min_opp]
        filtered.sort(key=lambda x: x[1] / x[2] if x[2] else 0, reverse=True)
        result[pos] = [{"name": name, "oaa": round(c, 2), "n_opp": n,
                        "player_id": pid,
                        "team_id": _team_map.get(pid) if pid else None}
                       for name, c, n, pid in filtered]
    return result


@app.get("/api/star_stats")
def get_star_stats(year: int = YEAR):
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
    }


def _run_optimize(req: OptimizeRequest) -> OptimizeResponse:
    if req.home_team and req.home_team.upper() not in SUPPORTED_TEAMS:
        raise HTTPException(422, f"Unsupported team '{req.home_team}'. Use GET /api/teams.")

    home_team = req.home_team.upper() if req.home_team else None

    # ── 準備球資料（快取：同打者跳過 DB 查詢與 KDE）────────────────
    import numpy as np

    if req.batter_id not in _batter_balls_cache:
        try:
            balls_all = prepare_batter_balls(req.batter_id, [YEAR], DSN)
        except Exception as e:
            raise HTTPException(422, str(e))
        if balls_all.empty:
            raise HTTPException(422, f"Batter {req.batter_id} has no qualifying balls in {YEAR}")
        _batter_balls_cache[req.batter_id]    = balls_all
        _batter_hitprobs_cache[req.batter_id] = predict_hit_probs_batch(_hit_bundle, balls_all)
    else:
        balls_all = _batter_balls_cache[req.batter_id]

    hit_probs_all = _batter_hitprobs_cache[req.batter_id]

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
                    fm[pos] = load_player_params("OF", nm, MODELS_DIR)
                except (KeyError, FileNotFoundError):
                    raise HTTPException(422, f"{pos} 找不到球員 '{nm}' 的模型參數")
        fielder_mus = fm or None
    mus_eff = dict(_mus)
    if fielder_mus:
        mus_eff.update(fielder_mus)

    # ── 站位評估：打牆球強制接殺機率 0、計入 RE24（統一口徑）──────
    def eval_positions(pos_dict):
        probs = np.asarray(
            compute_ball_catch_probs(pos_dict, balls_all, _scalers, mus_eff), dtype=float
        ).copy()
        probs[wall_flags] = 0.0                       # 打牆球無論站哪都接不到
        re24 = float(np.sum((1.0 - probs[mask]) * w_j[mask]))
        catch_pct = float(probs.mean() * 100)         # 全部球（含打牆球）平均
        return probs, re24, catch_pct

    # ── 聯盟平均站位 ─────────────────────────────────────────────
    try:
        league_avg_pos = get_league_avg_positions(YEAR, DSN)
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
            years=[YEAR], models_dir=MODELS_DIR, re24_dir=PRE_DIR,
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
            years=[YEAR], models_dir=MODELS_DIR, re24_dir=PRE_DIR,
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
                years=[YEAR], models_dir=MODELS_DIR, re24_dir=PRE_DIR,
                home_team=home_team, dsn=DSN,
                balls=balls_all, hit_probs=hit_probs_all,
            )
            pos_with_park = {p: opt_with_park_res[p] for p in POSITIONS}
            probs_with_park, re_with_park, catch_with_park = eval_positions(pos_with_park)
            positions_out["with_park"] = make_pos_set(pos_with_park, re_with_park, catch_with_park)
            scatter_probs = probs_with_park

    # ── 球散點（全部球；打牆球 catch_prob=0、前端以橘星標示）─────
    balls_out = [
        BallPoint(
            x=float(balls_all.iloc[i]["ball_x"]),
            y=float(balls_all.iloc[i]["ball_y"]),
            catch_prob=float(scatter_probs[i]),
            is_wall_ball=bool(wall_flags[i]),
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
    stand = get_batter_stand(req.batter_id, YEAR, DSN)
    title = f"{display_name} ({YEAR}, {stand}HB)"
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
