-- 球員平均站位資料（baseballsavant.mlb.com/visuals/position_data API，2026-06-23 從瀏覽器網路請求找到的端點）
-- 每位球員每年每位置一筆，avg_norm_start_distance/angle 是計算 fielder_x/fielder_y 的依據

DROP TABLE IF EXISTS fielder_positioning;

CREATE TABLE fielder_positioning (
    name_fielder             TEXT NOT NULL,
    fielder_id               BIGINT NOT NULL,
    fld_name_display_club    TEXT,
    season                   INTEGER NOT NULL,
    position                 TEXT NOT NULL,        -- '1B' | '2B' | '3B' | 'SS' | 'LF' | 'CF' | 'RF'
    pa                       BIGINT,
    avg_norm_start_distance  BIGINT,
    avg_norm_start_angle     BIGINT,
    PRIMARY KEY (fielder_id, season, position)
);
