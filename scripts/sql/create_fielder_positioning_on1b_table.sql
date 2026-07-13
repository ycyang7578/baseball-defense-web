-- 「一壘有人（1B Only）」切分的內野球員平均站位（階段B 雙殺模型的幾何代理）
-- 來源同 fielder_positioning（Savant position_data），但帶 firstBase=1&shift=0&batSide 參數
-- 抓取、L/R 打者 PA 加權合併（見 scripts/fetch_positioning.py fetch_year_on1b docstring）。
-- 平均欄位是加權合併後的值，用 DOUBLE PRECISION（主表是 Savant 原始整數）。

DROP TABLE IF EXISTS fielder_positioning_on1b;

CREATE TABLE fielder_positioning_on1b (
    name_fielder             TEXT NOT NULL,
    fielder_id               BIGINT NOT NULL,
    fld_name_display_club    TEXT,
    season                   INTEGER NOT NULL,
    position                 TEXT NOT NULL,        -- '1B' | '2B' | '3B' | 'SS'
    pa                       BIGINT,
    avg_norm_start_distance  DOUBLE PRECISION,
    avg_norm_start_angle     DOUBLE PRECISION,
    PRIMARY KEY (fielder_id, season, position)
);
