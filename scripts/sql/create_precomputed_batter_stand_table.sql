-- 精簡衍生表，服務 src/optimization.py 的 get_batter_stand()。
-- 每個 (batter, game_year) 一列，存該打者該年「所有打席」（不限 bb_type，跟
-- precomputed_batter_balls 的篩選條件不同，不能共用同一張表）中最常見的 stand。
-- 由 scripts/precompute_batter_balls.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS precomputed_batter_stand (
    batter     INTEGER  NOT NULL,
    game_year  SMALLINT NOT NULL,
    stand      CHAR(1)  NOT NULL,
    PRIMARY KEY (batter, game_year)
);
