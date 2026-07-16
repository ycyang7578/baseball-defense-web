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

import joblib
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize
from scipy.special import expit as _expit
from scipy.stats import qmc

# 模型 scaler 以 DataFrame 訓練但以 numpy array 呼叫，suppress 已知無害警告
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

from .config import DSN
from .hit_prob import load_hit_prob, predict_hit_probs_batch
from .re24 import load_re24

POSITIONS = ("LF", "CF", "RF")
FEATURE_COLS = ["speed", "cos_angle", "sin_angle", "fielder_dist"]

# ── 極座標扇形搜尋範圍 ─────────────────────────────────────────────
# 變數排列：[r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF]
# θ 從 y 軸量起（中外野方向=0°），正值朝右外野，負值朝左外野，單位：度
# 扇形角度確保守備員留在 fair territory（foul line ≈ ±45°）
_POLAR_BOUNDS = [
    (150, 400), (-45.0,   0.0),   # LF: r(ft), θ(deg)
    (150, 400), (-22.5, +22.5),   # CF: r(ft), θ(deg)
    (150, 400),  (0.0,  +45.0),   # RF: r(ft), θ(deg)
]


def _polar_to_xy(params: np.ndarray) -> np.ndarray:
    """[r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF] → [x_LF, y_LF, x_CF, y_CF, x_RF, y_RF]

    x = r·sin(θ),  y = r·cos(θ)   （與 physics.polar_to_fielder_xy 相同慣例）
    """
    out = np.empty(6)
    for i in range(3):
        r = params[2 * i]
        t = np.radians(params[2 * i + 1])
        out[2 * i]     = r * np.sin(t)   # x
        out[2 * i + 1] = r * np.cos(t)   # y
    return out


def _xy_to_polar_params(xy: dict) -> np.ndarray:
    """{'LF':(x,y),'CF':(x,y),'RF':(x,y)} → [r_LF, θ_LF, r_CF, θ_CF, r_RF, θ_RF]（_polar_to_xy 的反函式，供 warm start 用）"""
    out = np.empty(6)
    for i, pos in enumerate(POSITIONS):
        x, y = xy[pos]
        out[2 * i]     = np.hypot(x, y)
        out[2 * i + 1] = np.degrees(np.arctan2(x, y))
    return out

# 查精簡預計算表（scripts/precompute_batter_balls.py 產生），不直查 5GB 的 statcast。
# 舊版直查 statcast 再算物理公式的版本見 git 歷史（2026-07 前的 prepare_batter_balls）。
_BATTER_QUERY = """
    SELECT ball_x, ball_y, flight_time, launch_speed, launch_angle, spray_angle, stand, bb_type
    FROM precomputed_batter_balls
    WHERE batter = %(batter_id)s AND game_year = ANY(%(years)s)
"""


# ── 模型參數載入 ────────────────────────────────────────────────────

def _resolve_model_dir(pos: str, models_dir: Path) -> tuple[Path, str]:
    """Return (dir, prefix) for the model files. Falls back to unified OF if pos-specific missing."""
    d = Path(models_dir) / pos
    if (d / f"{pos}_scaler.joblib").exists():
        return d, pos
    # unified OF model fallback (2021-2024 only have OF/)
    return Path(models_dir) / "OF", "OF"


def load_model_params(pos: str, models_dir: Path) -> tuple:
    """
    Returns (scaler, mu_dict) for position pos.
    mu_dict keys: mu_alpha, mu_beta_speed, mu_beta_cos, mu_beta_sin, mu_beta_dist
    """
    d, prefix = _resolve_model_dir(pos, models_dir)
    scaler = joblib.load(d / f"{prefix}_scaler.joblib")
    group = pd.read_csv(d / f"{prefix}_summary_group.csv", encoding="utf-8-sig", index_col=0)
    mu = group["mean"]
    mu_dict = {
        "mu_alpha":      float(mu["mu_alpha"]),
        "mu_beta_speed": float(mu["mu_beta_speed"]),
        "mu_beta_cos":   float(mu["mu_beta_cos"]),
        "mu_beta_sin":   float(mu["mu_beta_sin"]),
        "mu_beta_dist":  float(mu["mu_beta_dist"]),
    }
    return scaler, mu_dict


def load_player_params(pos: str, player_name: str, models_dir: Path) -> dict:
    """
    讀取指定球員的 player-level 參數，回傳與 group mu 相同 key 格式的 dict
    （沿用該位置共用的 scaler）。供「指定特定外野手」的站位最佳化使用。
    """
    d, prefix = _resolve_model_dir(pos, models_dir)
    players = pd.read_csv(d / f"{prefix}_summary_players.csv", index_col=0, encoding="utf-8-sig")

    def g(param: str) -> float:
        return float(players.loc[f"{param}[{player_name}]", "mean"])

    return {
        "mu_alpha":      g("alpha"),
        "mu_beta_speed": g("beta_speed"),
        "mu_beta_cos":   g("beta_cos"),
        "mu_beta_sin":   g("beta_sin"),
        "mu_beta_dist":  g("beta_dist"),
    }


# ── 打者歷史球資料準備 ─────────────────────────────────────────────

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


# ── w_j 計算 ──────────────────────────────────────────────────────

def compute_w_j(
    balls: pd.DataFrame,
    hit_bundle: dict,
    delta_re: dict,
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
    # P(k|j): shape (N, 3) 欄位順序 = 1B, 2B, 3B
    if hit_probs is None:
        hit_probs = predict_hit_probs_batch(hit_bundle, balls)

    s = (on_1b, on_2b, on_3b, outs)
    dre_1b = delta_re.get(("1B", *s), 0.0)
    dre_2b = delta_re.get(("2B", *s), 0.0)
    dre_3b = delta_re.get(("3B", *s), 0.0)

    w_j = (hit_probs[:, 0] * dre_1b
           + hit_probs[:, 1] * dre_2b
           + hit_probs[:, 2] * dre_3b)
    return w_j


# ── 核心目標函數 ───────────────────────────────────────────────────

def _catch_prob_single_fielder(
    fx: float, fy: float,
    ball_x: np.ndarray, ball_y: np.ndarray, flight_time: np.ndarray,
    sc_mean: np.ndarray, sc_scale: np.ndarray, mu: dict,
) -> np.ndarray:
    """
    Vectorized catch probability for one fielder at (fx, fy) for N balls.
    Replicates physics.compute_relative_angle inline for speed.
    Returns array of shape (N,).
    sc_mean / sc_scale are pre-extracted from StandardScaler to avoid sklearn overhead.
    """
    dx = ball_x - fx
    dy = ball_y - fy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    speed = dist / flight_time                          # required speed ft/s

    run_angle = np.arctan2(dx, dy)                      # angle to ball (from +y axis)
    pos_angle = np.arctan2(fx, fy)                      # fielder's radial direction
    rel = run_angle - (pos_angle + np.pi)               # relative to "away from home"
    rel = (rel + np.pi) % (2 * np.pi) - np.pi

    X_raw = np.column_stack([speed, np.cos(rel), np.sin(rel), dist])
    X = (X_raw - sc_mean) / sc_scale

    logit = (mu["mu_alpha"]
             + mu["mu_beta_speed"] * X[:, 0]
             + mu["mu_beta_cos"]   * X[:, 1]
             + mu["mu_beta_sin"]   * X[:, 2]
             + mu["mu_beta_dist"]  * X[:, 3])

    return 1.0 / (1.0 + np.exp(-logit))


def objective_re24(positions_flat: np.ndarray, ctx: dict) -> float:
    """
    RE24 objective: Σ_j (1 - p̂_j(θ)) × w_j

    positions_flat: [lf_x, lf_y, cf_x, cf_y, rf_x, rf_y]
    ctx keys: ball_x, ball_y, flight_time, w_j, sc_means, sc_scales, mus
    """
    lf_x, lf_y, cf_x, cf_y, rf_x, rf_y = positions_flat

    bx = ctx["ball_x"]
    by = ctx["ball_y"]
    ft = ctx["flight_time"]

    # p_not_caught = ∏_i (1 - p_i) 逐一乘
    p_nc = np.ones(len(bx))
    for pos, (fx, fy) in zip(POSITIONS, [(lf_x, lf_y), (cf_x, cf_y), (rf_x, rf_y)]):
        p = _catch_prob_single_fielder(
            fx, fy, bx, by, ft,
            ctx["sc_means"][pos], ctx["sc_scales"][pos], ctx["mus"][pos],
        )
        p_nc *= (1.0 - p)

    return float(np.sum(p_nc * ctx["w_j"]))


# ── 主進入點 ───────────────────────────────────────────────────────

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
    fielder_mus: dict | None = None,
    balls: pd.DataFrame | None = None,
    hit_probs: np.ndarray | None = None,
    warm_start_xy: dict | None = None,
    delta_re: dict | None = None,
    hit_bundle: dict | None = None,
) -> dict:
    """
    Compute optimal outfield positions for a given batter and game state.
    Uses L-BFGS-B with multiple random restarts.

    home_team: MLB team abbreviation (e.g. 'BOS', 'LAD').
               When provided, balls that would clear the wall at that park
               are excluded from the objective (they cannot be caught).

    warm_start_xy: 選填 {"LF":(x,y),"CF":(x,y),"RF":(x,y)}，通常是另一次
        （目標函數相近的）optimize_positions 呼叫的解，拿來頂替其中一個
        隨機起點（不是額外多加一次評估，總 evaluate 次數仍等於 n_restarts，
        不增加算力成本），用來加速收斂。
        典型用法：with_park 用 no_park 的解 warm start，因為兩者只差在
        是否排除打牆球，目標函數幾乎一樣。

    delta_re / hit_bundle: 選填，呼叫端若已有快取好的版本（如 api/main.py
        startup 時載入的全域）可直接傳入，跳過重新讀 delta_re.json /
        重新 joblib.load(hit_type_kde.joblib)。單一請求常呼叫本函式 2 次
        （no_park + with_park），未傳入時每次都會各自重新讀檔。

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

    # 載入預計算資料（呼叫端未提供快取版本時才讀檔）
    if delta_re is None:
        _, delta_re = load_re24(re24_dir)
    if hit_bundle is None:
        hit_bundle = load_hit_prob(hit_prob_dir)

    # 載入模型參數：統一 OF 模型——LF/CF/RF 共用同一 scaler 與群體層參數，
    # 與 precompute_model_oaa 及 API 顯示層同一口徑（2026-07-13 定案）。
    # models/{year}/ 底下若存在分位置目錄（LF/CF/RF/）是舊模型時代的遺留產物，
    # 這裡刻意不用：曾因 _resolve_model_dir 偏好分位置目錄，使優化器與顯示層
    # 在 2025 用了兩套曲面，兩組站位的優劣排序可以相反。
    # （指定外野手時，以其 player-level 參數覆寫該位置）
    of_scaler, of_mu = load_model_params("OF", models_dir)
    scalers = {pos: of_scaler for pos in POSITIONS}
    mus = {pos: of_mu for pos in POSITIONS}
    if fielder_mus:
        for pos, m in fielder_mus.items():
            if m is not None:
                mus[pos] = m

    # 準備打者球資料（若外部已備妥則跳過 DB 查詢）
    if balls is None:
        balls = prepare_batter_balls(batter_id, years, dsn)
    if balls.empty:
        raise ValueError(f"Batter {batter_id} has no qualifying balls in years {years}")

    # 打牆球過濾：在目標球場圍牆外的球外野手接不到，排除出最佳化
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

    # 計算 w_j（若外部已備妥 hit_probs 則跳過 KDE）
    w_j = compute_w_j(balls, hit_bundle, delta_re, on_1b, on_2b, on_3b, outs, hit_probs=hit_probs)

    # 排除 w_j=0 的球
    mask = w_j > 0
    balls_f = balls[mask].reset_index(drop=True)
    w_j_f = w_j[mask]

    if len(balls_f) == 0:
        raise ValueError("No balls with positive w_j for this game state")

    sc_means  = {pos: scalers[pos].mean_  for pos in POSITIONS}
    sc_scales = {pos: scalers[pos].scale_ for pos in POSITIONS}

    ctx = {
        "ball_x":     balls_f["ball_x"].values,
        "ball_y":     balls_f["ball_y"].values,
        "flight_time": balls_f["flight_time"].values,
        "w_j":        w_j_f,
        "sc_means":   sc_means,
        "sc_scales":  sc_scales,
        "mus":        mus,
    }

    # 極座標正規化：r ∈ [150,400] vs θ ∈ [-45,45]，映射到 [0,1] 統一梯度縮放
    _lows  = np.array([b[0] for b in _POLAR_BOUNDS])
    _highs = np.array([b[1] for b in _POLAR_BOUNDS])
    _range = _highs - _lows

    def _obj_normalized(params_norm):
        params = _lows + params_norm * _range
        return objective_re24(_polar_to_xy(params), ctx)

    unit_bounds = [(0.0, 1.0)] * 6
    best: dict | None = None

    # warm start 頂替一個隨機起點的名額（不是額外加一次），維持總 evaluate 次數等於
    # n_restarts 不變 —— 不增加算力成本，也不會因為多做一次評估而變慢。
    #
    # 隨機起點取樣法：沒有 warm start 時用 Latin Hypercube Sampling，有 warm start 時用均勻隨機。
    # 這不是隨意選擇——30 樣本實測（2026-07-05，見 ARCHITECTURE.md）發現兩者效果相反：
    #   - no_park（無 warm start）：LHS 比均勻隨機 miss rate 更低（n_restarts=20 時 2/30 vs 4/30）
    #   - with_park（用 no_park 解 warm start）：LHS 反而比均勻隨機更差（n_restarts=8 時同一批樣本
    #     6/30 vs 4/30）——推測 LHS 的分層設計是針對「這批起點」整體算的，硬插入一個外部的 warm
    #     start 點會打亂分層假設的均勻覆蓋，均勻隨機沒有這個分層依賴問題。
    n_random = n_restarts - 1 if warm_start_xy is not None else n_restarts
    n_random = max(n_random, 0)
    if warm_start_xy is not None:
        starts = list(np.random.default_rng(seed).uniform(0.0, 1.0, size=(n_random, 6)))
    else:
        starts = list(qmc.LatinHypercube(d=6, seed=seed).random(n=n_random))
    if warm_start_xy is not None:
        warm_params = _xy_to_polar_params(warm_start_xy)
        starts.append(np.clip((warm_params - _lows) / _range, 0.0, 1.0))

    # ftol/gtol 放寬過（原本 1e-10/1e-6）：診斷發現 maxiter=500 從未被打到（實測中位數只跑 15 次
    # 迭代），真正的收斂容忍度沒有那麼嚴格的必要。30 樣本驗證（2026-07-05，見 ARCHITECTURE.md）顯示
    # 1e-6/1e-4 這組跟原本一樣的 miss rate，快 14~17%；再放寬（1e-4/1e-3 以上）miss rate 會明顯
    # 惡化，不要再往下調。
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
    positions: dict,
    balls: pd.DataFrame,
    scalers: dict,
    mus: dict,
) -> dict:
    """
    各守備員對每顆球的個別接殺機率。
    Returns {"LF": ndarray(N), "CF": ndarray(N), "RF": ndarray(N)}
    """
    bx = balls["ball_x"].values
    by = balls["ball_y"].values
    ft = balls["flight_time"].values
    return {
        pos: _catch_prob_single_fielder(
            positions[pos][0], positions[pos][1], bx, by, ft,
            scalers[pos].mean_, scalers[pos].scale_, mus[pos],
        )
        for pos in POSITIONS
    }


def compute_ball_catch_probs(
    positions: dict,
    balls: pd.DataFrame,
    scalers: dict,
    mus: dict,
) -> np.ndarray:
    """
    給定三個守備員站位，計算每顆球的聯合接殺機率 p̂_j。
    positions: {"LF": (x,y), "CF": (x,y), "RF": (x,y)}
    balls: DataFrame with ball_x, ball_y, flight_time
    Returns array of shape (N,): p̂_j for each ball.
    """
    bx = balls["ball_x"].values
    by = balls["ball_y"].values
    ft = balls["flight_time"].values
    p_nc = np.ones(len(bx))
    for pos in POSITIONS:
        fx, fy = positions[pos]
        p = _catch_prob_single_fielder(
            fx, fy, bx, by, ft,
            scalers[pos].mean_, scalers[pos].scale_, mus[pos],
        )
        p_nc *= (1.0 - p)
    return 1.0 - p_nc



def get_league_avg_positions(year: int, dsn: str = DSN) -> dict:
    """
    從 fielder_positioning 取指定年度各位置聯盟平均站位。
    回傳 {"LF": (x,y), "CF": (x,y), "RF": (x,y)}
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
    """取打者在指定年度最常見的打擊姿勢（'L' / 'R' / 'S'）。查精簡預計算表（見
    scripts/precompute_batter_balls.py），最常見值已在 precompute 階段算好。"""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT stand FROM precomputed_batter_stand
                WHERE batter = %(bid)s AND game_year = %(year)s
            """, {"bid": batter_id, "year": year})
            row = cur.fetchone()
    return row[0] if row else "R"


def load_qualifying_batters(year: int, dsn: str = DSN, min_balls: int = 30) -> list[dict]:
    """指定年度、球數 >= min_balls 的打者清單，供 api/main.py 的 /api/batters 用。
    查精簡預計算表（見 scripts/precompute_batter_balls.py）。"""
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
