from pathlib import Path

import pytest

from src.optimization import _resolve_model_dir, load_model_params, load_player_params

MODELS_DIR = Path(__file__).parent.parent / "models" / "2025"

MU_KEYS = {"mu_alpha", "mu_beta_speed", "mu_beta_cos", "mu_beta_sin", "mu_beta_dist"}


@pytest.mark.parametrize("pos", ["LF", "CF", "RF"])
def test_load_model_params_returns_scaler_and_full_mu_dict(pos):
    scaler, mu_dict = load_model_params(pos, MODELS_DIR)

    assert set(mu_dict.keys()) == MU_KEYS
    assert all(isinstance(v, float) for v in mu_dict.values())
    assert scaler.mean_.shape == (4,)  # speed, cos_angle, sin_angle, fielder_dist


def test_resolve_model_dir_uses_position_specific_files_when_present():
    # This only tests _resolve_model_dir's own fallback priority (low-level function behavior).
    # Production callers (api/main.py, optimize_positions) always explicitly pass "OF" and
    # don't rely on this "prefer a position-specific directory if one exists" behavior --
    # that once caused the optimizer and the display layer to use two different surfaces
    # in 2025, see the "outfield optimizer surface bug" in ARCHITECTURE.md.
    d, prefix = _resolve_model_dir("CF", MODELS_DIR)

    assert prefix == "CF"
    assert d == MODELS_DIR / "CF"
    assert (d / "CF_scaler.joblib").exists()


def test_resolve_model_dir_falls_back_to_unified_of_when_position_missing(tmp_path):
    # Build a fake models_dir that only has OF/ with no LF/CF/RF-specific files
    of_dir = tmp_path / "OF"
    of_dir.mkdir()
    (of_dir / "OF_scaler.joblib").write_bytes(b"")  # content doesn't matter, _resolve_model_dir only checks existence

    d, prefix = _resolve_model_dir("LF", tmp_path)

    assert prefix == "OF"
    assert d == tmp_path / "OF"


def test_load_player_params_returns_same_key_structure_as_group_mu():
    # api/main.py always passes "OF" to load_player_params (unified model, see the
    # "model parameter convention" decision in optimize_positions), so this test follows
    # the actual production path instead of the old per-position models.
    # "Tucker, Kyle" actually exists in models/2025/OF/OF_summary_players.csv.
    player_dict = load_player_params("OF", "Tucker, Kyle", MODELS_DIR)

    assert set(player_dict.keys()) == MU_KEYS
    assert all(isinstance(v, float) for v in player_dict.values())


def test_load_player_params_raises_key_error_for_unknown_player():
    with pytest.raises(KeyError):
        load_player_params("OF", "Definitely Not A Real Player Name", MODELS_DIR)
