-- 精簡衍生表，服務 src/optimization.py 的 prepare_batter_balls() 與 load_qualifying_batters()。
-- 每列 = 通過 _BATTER_QUERY 篩選（type='X', bb_type in fly_ball/line_drive, 排除全壘打等）
-- 且 flight_time > 0.5 的一顆球。欄位存已算好的衍生值（ball_x/ball_y/flight_time/spray_angle），
-- 不存 hc_x/hc_y 等原始欄位，物理公式邏輯只留在 physics.py 一處。
-- 由 scripts/precompute_batter_balls.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS precomputed_batter_balls (
    batter        INTEGER          NOT NULL,
    game_year     SMALLINT         NOT NULL,
    stand         CHAR(1)          NOT NULL,
    ball_x        DOUBLE PRECISION NOT NULL,
    ball_y        DOUBLE PRECISION NOT NULL,
    flight_time   DOUBLE PRECISION NOT NULL,
    launch_speed  DOUBLE PRECISION NOT NULL,
    launch_angle  DOUBLE PRECISION NOT NULL,
    spray_angle   DOUBLE PRECISION NOT NULL,
    bb_type       TEXT             NOT NULL   -- 'fly_ball' | 'line_drive'（整合頁球種篩選用）
);

CREATE INDEX IF NOT EXISTS idx_pbb_batter_year ON precomputed_batter_balls (batter, game_year);
CREATE INDEX IF NOT EXISTS idx_pbb_year         ON precomputed_batter_balls (game_year);
