-- 內野頁圖表用的逐球資料：precomputed_if_positions 每個 (打者, 年份) 對應的
-- 歷史滾地球（同 src/if_optimize.fetch_batter_gbs 的篩選），含兩組站位下的
-- 每球出局機率（優化用 GLM），供前端上色（比照外野 SprayChart 的 catch_prob 上色）。
-- 滾地球的 hc_x/hc_y 是被處理位置不是落點（內生性），所以不存 2D 座標，
-- 前端沿 spray_deg 畫弧點。由 scripts/precompute_if_optimize.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS precomputed_if_gbs (
    batter       INTEGER          NOT NULL,
    game_year    SMALLINT         NOT NULL,
    spray_deg    DOUBLE PRECISION NOT NULL,
    launch_speed DOUBLE PRECISION NOT NULL,
    launch_angle DOUBLE PRECISION NOT NULL,
    is_out       BOOLEAN          NOT NULL,
    p_out_league DOUBLE PRECISION NOT NULL,
    p_out_opt    DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pig_batter_year ON precomputed_if_gbs (batter, game_year);
