-- Runner speed data (baseballsavant.mlb.com/leaderboard/sprint_speed CSV)
-- hp_to_1b (home-to-first time in seconds) is the runner-side input to the ground
-- ball out-rate model; sprint_speed is in ft/s
-- Column order must match data/raw/sprint_speed/{year}.parquet (COPY maps by
-- position)

DROP TABLE IF EXISTS sprint_speed;

CREATE TABLE sprint_speed (
    name_runner       TEXT NOT NULL,
    player_id         BIGINT NOT NULL,
    team_id           BIGINT,
    team              TEXT,
    position          TEXT,
    age               INTEGER,
    competitive_runs  INTEGER,
    bolts             DOUBLE PRECISION,
    hp_to_1b          DOUBLE PRECISION,
    sprint_speed      DOUBLE PRECISION,
    season            INTEGER NOT NULL,
    PRIMARY KEY (player_id, season)
);
