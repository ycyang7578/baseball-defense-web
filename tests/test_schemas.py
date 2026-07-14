import pytest
from pydantic import ValidationError

from api.schemas import (BallPoint, IntegratedRequest, IntegratedSet,
                         OptimizeRequest, PositionXY)


def test_optimize_request_accepts_defaults():
    req = OptimizeRequest(batter_id=123)
    assert req.on_1b == 0
    assert req.outs == 0
    assert req.fielders is None


@pytest.mark.parametrize("field,value", [
    ("on_1b", 2),
    ("on_2b", -1),
    ("on_3b", 2),
])
def test_optimize_request_rejects_out_of_range_base_flags(field, value):
    with pytest.raises(ValidationError):
        OptimizeRequest(batter_id=123, **{field: value})


@pytest.mark.parametrize("outs", [-1, 3])
def test_optimize_request_rejects_out_of_range_outs(outs):
    with pytest.raises(ValidationError):
        OptimizeRequest(batter_id=123, outs=outs)


def test_optimize_request_accepts_boundary_outs():
    # outs 的合法範圍是 0~2（ge=0, le=2）
    OptimizeRequest(batter_id=123, outs=0)
    OptimizeRequest(batter_id=123, outs=2)


def test_ball_point_responsible_defaults_to_none():
    ball = BallPoint(x=1.0, y=2.0, catch_prob=0.5, is_wall_ball=False)
    assert ball.responsible is None


def test_integrated_request_accepts_defaults_and_bounds():
    req = IntegratedRequest(batter_id=123)
    assert (req.on_1b, req.on_2b, req.on_3b, req.outs) == (0, 0, 0, 0)
    with pytest.raises(ValidationError):
        IntegratedRequest(batter_id=123, outs=3)
    with pytest.raises(ValidationError):
        IntegratedRequest(batter_id=123, on_2b=2)


def test_integrated_set_holds_seven_positions():
    xy = PositionXY(x=0.0, y=100.0)
    s = IntegratedSet(
        positions={p: xy for p in ("LF", "CF", "RF", "1B", "2B", "3B", "SS")},
        runs_of=10.0, runs_if=5.0, runs_total=15.0)
    assert len(s.positions) == 7
    assert s.runs_total == s.runs_of + s.runs_if


def test_integrated_popup_fields_default_to_empty():
    """popup 是後加的展示欄位：舊呼叫端（不帶 popup）必須照樣能建構，
    雲端表未 sync 時 popup_balls 為空、n_popups 為 0。"""
    from api.schemas import IntegratedStats, PopupBall

    stats = IntegratedStats(n_of_balls=1, n_gb=1, re_state=0.5,
                            runs_saved_of=0.0, runs_saved_if=0.0,
                            runs_saved_total=0.0)
    assert stats.n_popups == 0
    ball = PopupBall(x=10.0, y=120.0, is_out=True)
    assert ball.is_out is True
