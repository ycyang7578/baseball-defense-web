-- 內野站位最佳化的離線預算結果，服務 web 的內野頁（線上即時算不可行：
-- n_restarts=20 在本機要 4 秒、Render 0.1 CPU 上要數十秒以上）。
-- 每列 = 一個 (打者, 年份)：該年滾地球 ≥ 門檻（預設 50）的打者，
-- 用該年歷史滾地球 + 高 n_restarts（預設 50，離線不必省）優化出的四內野手站位，
-- 以及對照的該年聯盟平均站位期望出局率。
-- 由 scripts/precompute_if_optimize.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS precomputed_if_positions (
    batter          INTEGER          NOT NULL,
    game_year       SMALLINT         NOT NULL,
    stand           CHAR(1)          NOT NULL,
    n_gb            INTEGER          NOT NULL,
    hp_to_1b        DOUBLE PRECISION NOT NULL,
    exp_outs_league DOUBLE PRECISION NOT NULL,
    exp_outs_opt    DOUBLE PRECISION NOT NULL,
    angle_1b        DOUBLE PRECISION NOT NULL,
    depth_1b        DOUBLE PRECISION NOT NULL,
    angle_2b        DOUBLE PRECISION NOT NULL,
    depth_2b        DOUBLE PRECISION NOT NULL,
    angle_3b        DOUBLE PRECISION NOT NULL,
    depth_3b        DOUBLE PRECISION NOT NULL,
    angle_ss        DOUBLE PRECISION NOT NULL,
    depth_ss        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (batter, game_year)
);
