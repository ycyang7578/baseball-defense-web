-- 精簡衍生表，服務 /api/optimize_integrated 的展示用內野高飛（popup）。
-- popup 站位無槓桿（出局率 98.6%，站哪都接得到）不參與任何優化，
-- 只為整合頁「打者所有場內球」的視覺完整性而存在。
-- ball_x/ball_y 用 hc × 2.5 呎換算（同 precomputed_if_gbs 的展示座標慣例；
-- hc 是記錄座標非落點，展示可、建分布不可——見 ARCHITECTURE.md 已知限制）。
-- is_out = events 不在 (single, double, triple, field_error) 之列。
-- 由 scripts/precompute_batter_popups.py 產生，不手動維護。
CREATE TABLE IF NOT EXISTS precomputed_batter_popups (
    batter        INTEGER          NOT NULL,
    game_year     SMALLINT         NOT NULL,
    ball_x        DOUBLE PRECISION NOT NULL,
    ball_y        DOUBLE PRECISION NOT NULL,
    launch_speed  DOUBLE PRECISION,
    is_out        BOOLEAN          NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pbp_batter_year ON precomputed_batter_popups (batter, game_year);
