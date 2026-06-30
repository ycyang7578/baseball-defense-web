-- 外野手 OAA leaderboard 摘要（每人每年每守備位置一列）
-- 來源：baseballsavant.mlb.com/leaderboard/outfield_directional_outs_above_average
-- 包含官方 OAA、守備機會、1~5 星各難度的接殺數和機會數

DROP TABLE IF EXISTS oaa_leaderboard;

CREATE TABLE oaa_leaderboard (
    player_id           BIGINT      NOT NULL,
    year                INTEGER     NOT NULL,
    player_name         TEXT,
    position            TEXT,           -- LF / CF / RF
    oaa                 INTEGER,        -- 官方 OAA（n_outs_above_average）
    n_opp               INTEGER,        -- 全部守備機會（含 0 星）
    n_opp_0stars        INTEGER,
    n_opp_1stars        INTEGER,
    n_opp_2stars        INTEGER,
    n_opp_3stars        INTEGER,
    n_opp_4stars        INTEGER,
    n_opp_5stars        INTEGER,
    n_fieldout_0stars   INTEGER,
    n_fieldout_1stars   INTEGER,
    n_fieldout_2stars   INTEGER,
    n_fieldout_3stars   INTEGER,
    n_fieldout_4stars   INTEGER,
    n_fieldout_5stars   INTEGER,
    is_qualified        BOOLEAN,
    PRIMARY KEY (player_id, year, position)
);
