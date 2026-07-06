-- 跑者速度資料（baseballsavant.mlb.com/leaderboard/sprint_speed CSV）
-- hp_to_1b（本壘到一壘秒數）是滾地球出局率模型的跑者端輸入；sprint_speed 單位 ft/s
-- 欄位順序須與 data/raw/sprint_speed/{year}.parquet 一致（COPY 依序對應）

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
