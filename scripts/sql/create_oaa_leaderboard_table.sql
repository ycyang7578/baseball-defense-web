-- Outfield OAA leaderboard summary (one row per player per year per fielding
-- position)
-- Source: baseballsavant.mlb.com/leaderboard/outfield_directional_outs_above_average
-- Includes official OAA, fielding opportunities, and putouts/opportunities for
-- each 1-5 star difficulty level

DROP TABLE IF EXISTS oaa_leaderboard;

CREATE TABLE oaa_leaderboard (
    player_id           BIGINT      NOT NULL,
    year                INTEGER     NOT NULL,
    player_name         TEXT,
    position            TEXT,           -- LF / CF / RF
    oaa                 INTEGER,        -- official OAA (n_outs_above_average)
    n_opp               INTEGER,        -- total fielding opportunities (including 0-star)
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
