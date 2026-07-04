import pytest
from pydantic import ValidationError

from api.schemas import BallPoint, OptimizeRequest


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
