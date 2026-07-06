-- 內野手 OAA leaderboard 摘要（每人每年一列，含分位置與方向分解）
-- 來源：baseballsavant.mlb.com/leaderboard/outs_above_average（pos=if，解析頁面 var data JSON）
-- 注意：頁面回傳的 year 欄位為空，由抓取腳本以請求年份填入

DROP TABLE IF EXISTS if_oaa_leaderboard;

CREATE TABLE if_oaa_leaderboard (
    player_id                 BIGINT  NOT NULL,
    year                      INTEGER NOT NULL,
    player_name               TEXT,
    primary_pos               TEXT,            -- '1B' | '2B' | '3B' | 'SS'（少數工具人可能是外野）
    oaa                       INTEGER,         -- 官方內野 OAA（全位置合計）
    fielding_runs_prevented   INTEGER,
    n_opp                     INTEGER,         -- 全部守備機會數（n）
    actual_success_rate       DOUBLE PRECISION,
    adj_estimated_success_rate DOUBLE PRECISION,
    n_pos3                    INTEGER,         -- 站 1B 的機會數
    n_pos4                    INTEGER,         -- 2B
    n_pos5                    INTEGER,         -- 3B
    n_pos6                    INTEGER,         -- SS
    oaa_role3                 INTEGER,         -- 站 1B 時的 OAA
    oaa_role4                 INTEGER,
    oaa_role5                 INTEGER,
    oaa_role6                 INTEGER,
    oaa_infront               INTEGER,         -- 方向分解：前進
    oaa_behind                INTEGER,         --           後退
    oaa_toward3b              INTEGER,         --           往三壘線側移
    oaa_toward1b              INTEGER,         --           往一壘線側移
    oaa_lhh                   INTEGER,         -- 對左打
    oaa_rhh                   INTEGER,         -- 對右打
    is_qualified              BOOLEAN,
    PRIMARY KEY (player_id, year)
);
