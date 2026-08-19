-- Average infield positioning split by "runner on 1B only" (geometric proxy for
-- the stage B double-play model)
-- Same source as fielder_positioning (Savant position_data), but fetched with the
-- firstBase=1&shift=0&batSide parameters, then PA-weighted merged across L/R
-- batters (see the fetch_year_on1b docstring in scripts/fetch_positioning.py).
-- The average columns are the post-weighted-merge values, hence DOUBLE PRECISION
-- (the main table keeps Savant's original integers).

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
