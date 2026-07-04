import pytest

from src.re24 import build_delta_re_deterministic


def _fake_re24():
    # 用容易手算的公式建構 24 種壘況的 RE 值：b1*1 + b2*2 + b3*4 + outs*10
    re24 = {}
    for b1 in (0, 1):
        for b2 in (0, 1):
            for b3 in (0, 1):
                for outs in (0, 1, 2):
                    re24[(b1, b2, b3, outs)] = b1 * 1 + b2 * 2 + b3 * 4 + outs * 10
    return re24


def test_delta_re_bases_empty_zero_outs():
    delta = build_delta_re_deterministic(_fake_re24())

    # 壘上無人、0 出局：old_re = 0
    assert delta[("1B", 0, 0, 0, 0)] == pytest.approx(1.0)   # 打者上一壘 -> re24[(1,0,0,0)]=1
    assert delta[("2B", 0, 0, 0, 0)] == pytest.approx(2.0)   # 打者上二壘 -> re24[(0,1,0,0)]=2
    assert delta[("3B", 0, 0, 0, 0)] == pytest.approx(4.0)   # 打者上三壘 -> re24[(0,0,1,0)]=4


def test_delta_re_bases_loaded_one_out():
    delta = build_delta_re_deterministic(_fake_re24())

    # 滿壘、1 出局：old_re = 1+2+4+10 = 17
    # 1B：三壘跑者回本壘得 1 分，壘況仍是滿壘 -> runs=1, new_re=17 -> delta=1+17-17=1
    assert delta[("1B", 1, 1, 1, 1)] == pytest.approx(1.0)
    # 2B：二、三壘跑者回本壘得 2 分，一壘跑者上三壘，一壘空出 -> new_re=re24[(0,1,1,1)]=16 -> delta=2+16-17=1
    assert delta[("2B", 1, 1, 1, 1)] == pytest.approx(1.0)
    # 3B：三個跑者全部回本壘得 3 分，壘上只剩打者在三壘 -> new_re=re24[(0,0,1,1)]=14 -> delta=3+14-17=0
    assert delta[("3B", 1, 1, 1, 1)] == pytest.approx(0.0)


def test_delta_re_covers_all_24_base_out_states_for_each_hit_type():
    delta = build_delta_re_deterministic(_fake_re24())

    # 3 種安打型別 x 24 種壘上/出局狀況 = 72 筆（ARCHITECTURE.md 記錄的 delta_re.json 筆數）
    assert len(delta) == 72
