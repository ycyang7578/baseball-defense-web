-- Infield OAA leaderboard summary (one row per player per year, including
-- per-position and directional breakdowns)
-- Source: baseballsavant.mlb.com/leaderboard/outs_above_average (pos=if, parsed
-- from the page's `var data` JSON)
-- Note: the page's year field comes back empty, so the fetch script fills it in
-- with the requested year

DROP TABLE IF EXISTS if_oaa_leaderboard;

CREATE TABLE if_oaa_leaderboard (
    player_id                 BIGINT  NOT NULL,
    year                      INTEGER NOT NULL,
    player_name               TEXT,
    primary_pos               TEXT,            -- '1B' | '2B' | '3B' | 'SS' (a few utility players may be OF)
    oaa                       INTEGER,         -- official infield OAA (summed across all positions)
    fielding_runs_prevented   INTEGER,
    n_opp                     INTEGER,         -- total number of fielding opportunities (n)
    actual_success_rate       DOUBLE PRECISION,
    adj_estimated_success_rate DOUBLE PRECISION,
    n_pos3                    INTEGER,         -- opportunities while playing 1B
    n_pos4                    INTEGER,         -- 2B
    n_pos5                    INTEGER,         -- 3B
    n_pos6                    INTEGER,         -- SS
    oaa_role3                 INTEGER,         -- OAA while playing 1B
    oaa_role4                 INTEGER,
    oaa_role5                 INTEGER,
    oaa_role6                 INTEGER,
    oaa_infront               INTEGER,         -- directional breakdown: charging in
    oaa_behind                INTEGER,         --                        going back
    oaa_toward3b              INTEGER,         --                        moving toward the 3B line
    oaa_toward1b              INTEGER,         --                        moving toward the 1B line
    oaa_lhh                   INTEGER,         -- vs. left-handed batters
    oaa_rhh                   INTEGER,         -- vs. right-handed batters
    is_qualified              BOOLEAN,
    PRIMARY KEY (player_id, year)
);
