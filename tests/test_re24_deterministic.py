import pytest

from src.re24 import build_delta_re_deterministic


def _fake_re24():
    # Build RE values for the 24 base-out states using a formula that's easy to hand-compute: b1*1 + b2*2 + b3*4 + outs*10
    re24 = {}
    for b1 in (0, 1):
        for b2 in (0, 1):
            for b3 in (0, 1):
                for outs in (0, 1, 2):
                    re24[(b1, b2, b3, outs)] = b1 * 1 + b2 * 2 + b3 * 4 + outs * 10
    return re24


def test_delta_re_bases_empty_zero_outs():
    delta = build_delta_re_deterministic(_fake_re24())

    # Bases empty, 0 outs: old_re = 0
    assert delta[("1B", 0, 0, 0, 0)] == pytest.approx(1.0)   # batter reaches first -> re24[(1,0,0,0)]=1
    assert delta[("2B", 0, 0, 0, 0)] == pytest.approx(2.0)   # batter reaches second -> re24[(0,1,0,0)]=2
    assert delta[("3B", 0, 0, 0, 0)] == pytest.approx(4.0)   # batter reaches third -> re24[(0,0,1,0)]=4


def test_delta_re_bases_loaded_one_out():
    delta = build_delta_re_deterministic(_fake_re24())

    # Bases loaded, 1 out: old_re = 1+2+4+10 = 17
    # 1B: runner on third scores for 1 run, bases stay loaded -> runs=1, new_re=17 -> delta=1+17-17=1
    assert delta[("1B", 1, 1, 1, 1)] == pytest.approx(1.0)
    # 2B: runners on second and third score for 2 runs, runner on first advances to third, first base empties -> new_re=re24[(0,1,1,1)]=16 -> delta=2+16-17=1
    assert delta[("2B", 1, 1, 1, 1)] == pytest.approx(1.0)
    # 3B: all three runners score for 3 runs, only the batter remains on third -> new_re=re24[(0,0,1,1)]=14 -> delta=3+14-17=0
    assert delta[("3B", 1, 1, 1, 1)] == pytest.approx(0.0)


def test_delta_re_covers_all_24_base_out_states_for_each_hit_type():
    delta = build_delta_re_deterministic(_fake_re24())

    # 3 hit types x 24 base/out states = 72 entries (matches the delta_re.json entry count documented in ARCHITECTURE.md)
    assert len(delta) == 72
