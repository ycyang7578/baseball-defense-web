-- Baseball Savant's official per-play fielding records
-- (baseballsavant.mlb.com/player-services/gamelogs API)
-- Used to determine which balls are actually counted in the official OAA
-- calculation (the basis for the official_defenders determination)

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
