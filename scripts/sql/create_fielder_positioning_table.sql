-- Player average positioning data (baseballsavant.mlb.com/visuals/position_data
-- API, endpoint found from browser network requests on 2026-06-23)
-- One row per player per year per position; avg_norm_start_distance/angle are the
-- basis for computing fielder_x/fielder_y

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
