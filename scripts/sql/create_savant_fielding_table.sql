-- Baseball Savant 官方逐球守備記錄（baseballsavant.mlb.com/player-services/gamelogs API）
-- 用來判斷哪些球是官方計算OAA時真正算進去的（official_defenders的判斷依據）

DROP TABLE IF EXISTS savant_fielding;

CREATE TABLE savant_fielding (
    player_id       BIGINT NOT NULL,
    game_pk         BIGINT,
    play_id         TEXT,
    at_bat_number   BIGINT,
    game_date       DATE,
    ev              DOUBLE PRECISION,
    la              DOUBLE PRECISION,
    dist            DOUBLE PRECISION,
    catch_prob      DOUBLE PRECISION,
    star_savant     TEXT
);

CREATE INDEX idx_savant_fielding_gamepk_ab ON savant_fielding (game_pk, at_bat_number);
