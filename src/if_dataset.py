"""滾地球資料集建構：內野 out probability 模型的特徵工程。

設計依據：
- Melville (2024) §3.1 — a_d（球與最近內野手的最小角差）、b_t（球到野手時間 =
  野手深度 ÷ 擊球初速，不考慮減速）
- Tango, Introducing Infield OAA (MLB Technology Blog, 2020) — 出局是三段競速：
  攔截 + 傳球 vs 跑者到一壘時間（跑者用賽季平均速度，非該球實際速度）

座標慣例（與 Savant fielder_positioning 一致）：角度 -45°=三壘線、0°=中外野方向、
+45°=一壘線；深度為離本壘距離（呎）。

已知限制：滾地球的 hc_x/hc_y 是「球被處理的位置」不是落點（出局球在內野被攔、
安打球滾到外野才被撿），因此位置資訊只取 spray angle（直線滾動下角度近似不變），
不使用深度。
"""
import numpy as np
import pandas as pd
import psycopg2

from src.config import DSN

# 滾地球結果標籤：out=防守方把球轉換成至少一個出局（Tango：只看 play 的第一個 out）
OUT_EVENTS: tuple[str, ...] = ("field_out", "force_out", "grounded_into_double_play",
                                "double_play", "fielders_choice_out")
NONOUT_EVENTS: tuple[str, ...] = ("single", "double", "triple", "field_error")

# 階段B（一壘有人、<2 出局）：結果升級為「拿到幾個出局」。
# fielders_choice 經 des 逐筆抽驗（2026-07-13）＝打者上壘、跑者全推進，沒有出局；
# fielders_choice_out 才有記出局。
ON1B_EVENT_OUTS: dict[str, int] = {
    "grounded_into_double_play": 2, "double_play": 2,
    "force_out": 1, "field_out": 1, "fielders_choice_out": 1,
    "single": 0, "double": 0, "triple": 0, "field_error": 0, "fielders_choice": 0,
}

# Savant hc_x/hc_y 座標系的本壘位置
HOME_X: float = 125.42
HOME_Y: float = 198.27
MPH_TO_FTS: float = 1.46667          # 擊球初速 mph → ft/s
FIRST_BASE_R: float = 90.0           # 一壘壘包距本壘的距離（呎）
FIRST_BASE_DEG: float = 45.0         # 一壘壘包相對本壘的角度

INFIELD_COLS: dict[str, str] = {"fielder_3": "1B", "fielder_4": "2B",
                                 "fielder_5": "3B", "fielder_6": "SS"}
INFIELD_POSITIONS: tuple[str, ...] = tuple(INFIELD_COLS.values())


def _polar_to_xy(
    radius_ft: np.ndarray | float,
    angle_deg: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """(深度, 角度) → 平面座標；x 往一壘線側為正、y 往中外野為正。"""
    angle_rad = np.radians(angle_deg)
    return radius_ft * np.sin(angle_rad), radius_ft * np.cos(angle_rad)


def fetch_raw_gb(years: list[int]) -> pd.DataFrame:
    """撈指定年份的非觸擊、有明確結果標籤的滾地球（含壘況與佈陣欄位）。"""
    events = OUT_EVENTS + NONOUT_EVENTS
    # ORDER BY 確保回傳順序確定（無序時 GBM early-stopping 的內部驗證切分不可重現）
    sql = f"""
        SELECT game_year, batter, stand, hc_x, hc_y, launch_speed, launch_angle,
               events, if_fielding_alignment, hit_location,
               (on_1b IS NULL AND on_2b IS NULL AND on_3b IS NULL) AS bases_empty,
               fielder_3, fielder_4, fielder_5, fielder_6
        FROM statcast
        WHERE bb_type = 'ground_ball'
          AND game_year = ANY(%(years)s)
          AND hc_x IS NOT NULL AND launch_speed IS NOT NULL
          AND events IN {events}
          AND des NOT ILIKE '%%bunt%%'
        ORDER BY game_year, batter, hc_x, hc_y, launch_speed
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn, params={"years": list(years)})


def fetch_positioning(years: list[int]) -> pd.DataFrame:
    """內野四位置的球員賽季平均站位（載入時已按 (player, season, position) 去重）。"""
    sql = """
        SELECT fielder_id, season, position,
               avg_norm_start_distance AS depth, avg_norm_start_angle AS angle
        FROM fielder_positioning
        WHERE position IN %(pos)s AND season = ANY(%(years)s)
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn,
                           params={"pos": INFIELD_POSITIONS, "years": list(years)})


def fetch_positioning_on1b(years: list[int]) -> pd.DataFrame:
    """「一壘有人」切分的賽季平均站位（fielder_positioning_on1b，2023 起）。

    階段B 的幾何代理要用這份：一壘有人時站位系統性位移（1B hold runner
    −26~−35 呎、2B/SS 雙殺深度 −3~−4.4 呎），用全情境平均會把幾何特徵算錯。"""
    sql = """
        SELECT fielder_id, season, position,
               avg_norm_start_distance AS depth, avg_norm_start_angle AS angle
        FROM fielder_positioning_on1b
        WHERE position IN %(pos)s AND season = ANY(%(years)s)
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn,
                           params={"pos": INFIELD_POSITIONS, "years": list(years)})


def fetch_raw_gb_on1b(years: list[int]) -> pd.DataFrame:
    """一壘有人（僅一壘）、<2 出局的非觸擊滾地球（階段B 雙殺情境主範圍）。

    比 fetch_raw_gb 多帶 on_1b（跑者 id，接跑者速度）與 outs_when_up，
    事件集合含 fielders_choice（0 出局，無人在壘時不存在此事件）。"""
    events = tuple(ON1B_EVENT_OUTS)
    sql = f"""
        SELECT game_year, batter, stand, hc_x, hc_y, launch_speed, launch_angle,
               events, if_fielding_alignment, hit_location,
               on_1b, outs_when_up,
               (on_1b IS NULL AND on_2b IS NULL AND on_3b IS NULL) AS bases_empty,
               fielder_3, fielder_4, fielder_5, fielder_6
        FROM statcast
        WHERE bb_type = 'ground_ball'
          AND game_year = ANY(%(years)s)
          AND hc_x IS NOT NULL AND launch_speed IS NOT NULL
          AND on_1b IS NOT NULL AND on_2b IS NULL AND on_3b IS NULL
          AND outs_when_up < 2
          AND events IN {events}
          AND des NOT ILIKE '%%bunt%%'
        ORDER BY game_year, batter, hc_x, hc_y, launch_speed
    """
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(sql, conn, params={"years": list(years)})


def fetch_run_speed(years: list[int]) -> pd.DataFrame:
    with psycopg2.connect(DSN) as conn:
        return pd.read_sql(
            "SELECT player_id, season, hp_to_1b FROM sprint_speed WHERE season = ANY(%(years)s)",
            conn, params={"years": list(years)})


def attach_features(gb: pd.DataFrame, positioning: pd.DataFrame,
                    run_speed: pd.DataFrame) -> pd.DataFrame:
    """把站位/跑速接上滾地球並算出模型特徵。純函式（不碰 DB），方便測試。

    產出的特徵欄位：
    - spray_deg      擊球水平角度（-45=三壘線）
    - ad_min         與最近內野手的最小絕對角差（度）
    - near_depth     最近內野手深度（呎）
    - lat_ft         最近內野手到球路徑的橫向距離（呎）= depth × sin(ad_min)
    - ball_time      b_t：球滾到最近內野手深度的時間（秒，不考慮減速）
    - throw_dist     最近內野手深度處的攔截點到一壘的距離（呎）
    - hp_to_1b       打者本壘到一壘秒數（賽季平均；缺值以同年中位數填補）
    - has_run_speed  hp_to_1b 是否為實際值（False=中位數填補）
    """
    df = gb.copy()
    df["is_out"] = df["events"].isin(OUT_EVENTS).astype(int)
    df["spray_deg"] = np.degrees(
        np.arctan2(df["hc_x"] - HOME_X, HOME_Y - df["hc_y"]))
    # 界外線外 ±10 度以上多為記錄雜訊（Melville 用 [-55,55] 同一理由）
    df = df[df["spray_deg"].abs() <= 55].copy()

    pos_map = positioning.set_index(["fielder_id", "season", "position"])[["depth", "angle"]]
    for col, posname in INFIELD_COLS.items():
        idx = pd.MultiIndex.from_arrays(
            [df[col], df["game_year"], np.repeat(posname, len(df))])
        joined = pos_map.reindex(idx).to_numpy()
        df[f"{posname}_depth"] = joined[:, 0]
        df[f"{posname}_angle"] = joined[:, 1]

    have_all = df[[f"{p}_angle" for p in INFIELD_POSITIONS]].notna().all(axis=1)
    df = df[have_all].copy()

    angles = df[[f"{p}_angle" for p in INFIELD_POSITIONS]].to_numpy(float)
    depths = df[[f"{p}_depth" for p in INFIELD_POSITIONS]].to_numpy(float)
    dtheta = np.abs(angles - df["spray_deg"].to_numpy(float)[:, None])
    nearest = dtheta.argmin(axis=1)
    rows = np.arange(len(df))
    df["nearest_pos"] = np.array(INFIELD_POSITIONS)[nearest]
    df["ad_min"] = dtheta[rows, nearest]
    df["near_depth"] = depths[rows, nearest]
    df["lat_ft"] = df["near_depth"] * np.sin(np.radians(df["ad_min"]))
    df["ball_time"] = df["near_depth"] / (df["launch_speed"] * MPH_TO_FTS)

    # 攔截點（沿球路徑、取最近野手深度處）到一壘的距離
    ix, iy = _polar_to_xy(df["near_depth"].to_numpy(float),
                          df["spray_deg"].to_numpy(float))
    bx, by = _polar_to_xy(FIRST_BASE_R, FIRST_BASE_DEG)
    df["throw_dist"] = np.hypot(ix - bx, iy - by)

    rs_map = run_speed.set_index(["player_id", "season"])["hp_to_1b"]
    idx = pd.MultiIndex.from_arrays([df["batter"], df["game_year"]])
    df["hp_to_1b"] = rs_map.reindex(idx).to_numpy()
    df["has_run_speed"] = df["hp_to_1b"].notna()
    year_med = df.groupby("game_year")["hp_to_1b"].transform("median")
    df["hp_to_1b"] = df["hp_to_1b"].fillna(year_med)

    df["stand_R"] = (df["stand"] == "R").astype(int)
    # launch_angle 少數缺值（追蹤缺漏），以同年中位數填補；launch_speed 在 SQL 已強制非空
    la_med = df.groupby("game_year")["launch_angle"].transform("median")
    df["launch_angle"] = df["launch_angle"].fillna(la_med)
    return df


SECOND_BASE_X: float = 0.0
SECOND_BASE_Y: float = 90.0 * np.sqrt(2.0)  # 二壘壘包


def attach_dp_features(df: pd.DataFrame, run_speed: pd.DataFrame) -> pd.DataFrame:
    """階段B（雙殺情境）追加特徵。純函式，接在 attach_features 之後。

    - n_outs           該球拿到的出局數（0/1/2，ON1B_EVENT_OUTS）
    - throw_dist_2b    攔截點（沿球路徑、最近野手深度處）到二壘的距離（呎）
                       ——一壘有人時主要傳球目標是二壘 force，與 throw_dist（到一壘）
                       同性質的反事實合法幾何，僅線性使用（函數形式內生性教訓）
    - pivot_dist       2B/SS 野手到二壘壘包的較小距離（呎）＝雙殺軸心幾何，
                       純站位函數（搬野手預測會跟著變）
    - runner_hp_to_1b  一壘跑者的賽季 hp_to_1b（跑者速度代理；缺值同年中位數填補）
    - has_runner_speed 是否為實際值
    """
    df = df.copy()
    df["n_outs"] = df["events"].map(ON1B_EVENT_OUTS)

    ix, iy = _polar_to_xy(df["near_depth"].to_numpy(float),
                          df["spray_deg"].to_numpy(float))
    df["throw_dist_2b"] = np.hypot(ix - SECOND_BASE_X, iy - SECOND_BASE_Y)

    dists = []
    for pos in ("2B", "SS"):
        px, py = _polar_to_xy(df[f"{pos}_depth"].to_numpy(float),
                              df[f"{pos}_angle"].to_numpy(float))
        dists.append(np.hypot(px - SECOND_BASE_X, py - SECOND_BASE_Y))
    df["pivot_dist"] = np.minimum(*dists)

    rs_map = run_speed.set_index(["player_id", "season"])["hp_to_1b"]
    idx = pd.MultiIndex.from_arrays([df["on_1b"], df["game_year"]])
    df["runner_hp_to_1b"] = rs_map.reindex(idx).to_numpy()
    df["has_runner_speed"] = df["runner_hp_to_1b"].notna()
    year_med = df.groupby("game_year")["runner_hp_to_1b"].transform("median")
    df["runner_hp_to_1b"] = df["runner_hp_to_1b"].fillna(year_med)
    return df


def build_gb_on1b_dataset(years: list[int],
                          alignment: str | None = "Standard") -> pd.DataFrame:
    """一站式（階段B）：一壘有人（僅一壘）、<2 出局滾地球＋雙殺特徵。

    站位代理用 fielder_positioning_on1b（一壘有人切分）；標籤 n_outs ∈ {0,1,2}。
    """
    run_speed = fetch_run_speed(years)
    df = attach_features(fetch_raw_gb_on1b(years), fetch_positioning_on1b(years),
                         run_speed)
    df = attach_dp_features(df, run_speed)
    if alignment is not None:
        df = df[df["if_fielding_alignment"] == alignment]
    return df.reset_index(drop=True)


def build_gb_dataset(years: list[int], bases_empty: bool | None = None,
                     alignment: str | None = None) -> pd.DataFrame:
    """一站式：撈資料、接特徵、依範圍過濾。

    bases_empty=True 只留無人在壘（壘上有人會拉動站位，例如一壘手 hold runner）；
    alignment='Standard' 只留標準佈陣（賽季平均站位對非標準佈陣的誤差更大）。
    """
    df = attach_features(fetch_raw_gb(years), fetch_positioning(years),
                         fetch_run_speed(years))
    if bases_empty is not None:
        df = df[df["bases_empty"] == bases_empty]
    if alignment is not None:
        df = df[df["if_fielding_alignment"] == alignment]
    return df.reset_index(drop=True)
