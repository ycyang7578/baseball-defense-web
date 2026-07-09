-- 我方模型算的內野手 OAA（web 排名頁用），對照表是 if_oaa_leaderboard（官方）。
-- model_oaa 是分位置中心化後的 Σ(is_out − p̂)，p̂ 來自評價用 GBM（無野手資訊）。
-- position 是歸責球數最多的位置。player_name 來自官方 leaderboard，對不上者為 NULL
-- （排名頁只顯示有名字的球員）。方法論見 scripts/evaluate_if_2025.py。
-- 由 scripts/precompute_if_model_oaa.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS if_model_oaa (
    player_id   INTEGER          NOT NULL,
    player_name TEXT,
    position    TEXT             NOT NULL,
    year        SMALLINT         NOT NULL,
    model_oaa   DOUBLE PRECISION NOT NULL,
    n_balls     INTEGER          NOT NULL,
    PRIMARY KEY (player_id, year)
);
