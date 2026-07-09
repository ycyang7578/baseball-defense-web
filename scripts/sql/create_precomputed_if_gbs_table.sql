-- 內野頁圖表用的逐球資料：precomputed_if_positions 每個 (打者, 年份) 對應的
-- 歷史滾地球（同 src/if_optimize.fetch_batter_gbs 的篩選），含兩組站位下的
-- 每球出局機率（優化用 GLM），供前端上色（比照外野 SprayChart 的 catch_prob 上色）。
-- ball_x/ball_y 是 hc_x/hc_y 換算成呎的座標（本壘原點，+x 朝一壘側）。
-- 注意語意：這是 Statcast 記錄的球被處理/撿起的位置，不是擊球落點——出局球
-- 系統性偏淺（中位 ~118 呎）、安打偏深（~224 呎），是結果與守備的函數，
-- 只能當展示用，不可拿來建打者分布（見 src/if_dataset.py 的內生性說明）。
-- 由 scripts/precompute_if_optimize.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS precomputed_if_gbs (
    batter       INTEGER          NOT NULL,
    game_year    SMALLINT         NOT NULL,
    spray_deg    DOUBLE PRECISION NOT NULL,
    ball_x       DOUBLE PRECISION NOT NULL,
    ball_y       DOUBLE PRECISION NOT NULL,
    launch_speed DOUBLE PRECISION NOT NULL,
    launch_angle DOUBLE PRECISION NOT NULL,
    is_out       BOOLEAN          NOT NULL,
    p_out_league DOUBLE PRECISION NOT NULL,
    p_out_opt    DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pig_batter_year ON precomputed_if_gbs (batter, game_year);
