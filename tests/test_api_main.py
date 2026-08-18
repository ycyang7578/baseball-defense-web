"""api/main.py 的整合測試：起 FastAPI TestClient（觸發真正的 lifespan startup），
實際連本機 PostgreSQL + 讀本機 models/ 底下訓練好的模型檔。

跟 src/ 那些合成資料的單元測試不同，這支需要完整本機環境才能跑，預設不跑
（見 pytest.ini 的 integration marker）。手動執行：
    python -m pytest tests/test_api_main.py -m integration -v
"""
import base64

import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.integration

# 已知本機資料庫有完整資料的打者/年份（2026-08 手動驗證過）
_OF_BATTER_ID = 680757   # Kwan, Steven —— 外野打者
_IF_BATTER_ID = 650333   # Arráez, Luis —— 內野合格打者（滾地球足量）
_IF_FIELDER_ID = 592192  # Canha, Mark —— 有貝葉斯球員層效應的內野手
_YEAR = 2025


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── 基本查表端點 ─────────────────────────────────────────────────

def test_years_includes_target_year(client):
    resp = client.get("/api/years")
    assert resp.status_code == 200
    assert _YEAR in resp.json()


def test_teams_includes_known_team(client):
    resp = client.get("/api/teams")
    assert resp.status_code == 200
    assert "BOS" in resp.json()


def test_batters_returns_qualifying_batters(client):
    resp = client.get(f"/api/batters?year={_YEAR}")
    assert resp.status_code == 200
    batters = resp.json()
    assert len(batters) > 0
    assert {"batter_id", "name", "n_balls"} <= set(batters[0])


def test_if_batters_returns_qualifying_batters(client):
    resp = client.get(f"/api/if_batters?year={_YEAR}")
    assert resp.status_code == 200
    batters = resp.json()
    assert len(batters) > 0
    assert {"batter_id", "name", "n_gb", "stand"} <= set(batters[0])


def test_if_fielders_returns_all_four_positions(client):
    resp = client.get(f"/api/if_fielders?year={_YEAR}")
    assert resp.status_code == 200
    result = resp.json()
    assert set(result) == {"1B", "2B", "3B", "SS"}


# ── 外野站位優化（/api/optimize 系列）───────────────────────────

def test_optimize_no_park_returns_league_avg_and_no_park(client):
    resp = client.post("/api/optimize", json={
        "batter_id": _OF_BATTER_ID, "year": _YEAR,
        "on_1b": 0, "on_2b": 0, "on_3b": 0, "outs": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["positions"]) == {"league_avg", "no_park"}
    for position_set in body["positions"].values():
        for code in ("LF", "CF", "RF"):
            assert {"x", "y"} <= set(position_set[code])
    assert body["stats"]["n_balls"] > 0
    assert body["stats"]["n_wall_balls"] == 0


def test_optimize_with_park_adds_with_park_and_wall_boundary(client):
    resp = client.post("/api/optimize", json={
        "batter_id": _OF_BATTER_ID, "year": _YEAR,
        "on_1b": 0, "on_2b": 0, "on_3b": 0, "outs": 0,
        "home_team": "bos",  # 小寫也要能過（後端會 .upper()）
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "with_park" in body["positions"]
    assert body["park_boundary"] and len(body["park_boundary"]) > 10


def test_optimize_rejects_unsupported_team(client):
    resp = client.post("/api/optimize", json={
        "batter_id": _OF_BATTER_ID, "year": _YEAR,
        "on_1b": 0, "on_2b": 0, "on_3b": 0, "outs": 0,
        "home_team": "ZZZ",
    })
    assert resp.status_code == 422


def test_optimize_plot_returns_valid_png(client):
    resp = client.post("/api/optimize_plot", json={
        "batter_id": _OF_BATTER_ID, "year": _YEAR,
        "on_1b": 0, "on_2b": 0, "on_3b": 0, "outs": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    png_bytes = base64.b64decode(body["image_b64"])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


# ── 內野站位優化（/api/if_optimize，含階段B DP 分支）─────────────

def test_if_optimize_no_runner_uses_runvalue_refinement(client):
    resp = client.post("/api/if_optimize", json={
        "batter_id": _IF_BATTER_ID, "year": _YEAR,
        "on_1b": 0, "on_2b": 0, "on_3b": 0, "outs": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["situation"] == "---  0 out"
    assert set(body["optimized"]["positions"]) == {"1B", "2B", "3B", "SS"}


def test_if_optimize_runner_on_first_pins_1b_to_hold_runner_position(client):
    """一壘有人、<2 出局要切到階段B DP 分支：1B 釘死在聯盟 hold-runner 位置，
    且不管有沒有指定野手，league/optimized 兩組的 1B 站位都要完全一樣（釘死
    不隨效應動，見 ARCHITECTURE.md「內野貝葉斯球員層」/if_dp_optimize.py docstring）。
    """
    resp = client.post("/api/if_optimize", json={
        "batter_id": _IF_BATTER_ID, "year": _YEAR,
        "on_1b": 1, "on_2b": 0, "on_3b": 0, "outs": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["situation"] == "1--  0 out"
    pinned = body["optimized"]["positions"]["1B"]
    assert pinned["angle"] == pytest.approx(40.6, abs=0.05)
    assert pinned["depth"] == pytest.approx(88.3, abs=0.5)
    assert pinned == body["league"]["positions"]["1B"]


def test_if_result_custom_with_specified_fielder(client):
    resp = client.get(
        f"/api/if_result_custom?batter_id={_IF_BATTER_ID}&year={_YEAR}"
        f"&fielder_2b={_IF_FIELDER_ID}"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fielders"]["2B"] is not None
    assert body["fielders"]["1B"] is None  # 未指定的位置＝聯盟平均


# ── 內外野整合（/api/optimize_integrated）────────────────────────

def test_optimize_integrated_returns_seven_positions(client):
    resp = client.post("/api/optimize_integrated", json={
        "batter_id": _IF_BATTER_ID, "year": _YEAR,
        "on_1b": 0, "on_2b": 0, "on_3b": 0, "outs": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["optimized"]["positions"]) == {"LF", "CF", "RF", "1B", "2B", "3B", "SS"}
    stats = body["stats"]
    assert stats["runs_saved_total"] == pytest.approx(
        stats["runs_saved_of"] + stats["runs_saved_if"], abs=2e-3)


def test_optimize_integrated_runner_on_first_uses_dp_branch(client):
    """整合頁一壘有人時內野側也要切 DP 分支，1B 同樣釘死。"""
    resp = client.post("/api/optimize_integrated", json={
        "batter_id": _IF_BATTER_ID, "year": _YEAR,
        "on_1b": 1, "on_2b": 0, "on_3b": 0, "outs": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    pinned = body["optimized"]["positions"]["1B"]
    assert pinned == body["league"]["positions"]["1B"]


# ── 球場圍牆 ──────────────────────────────────────────────────────

def test_park_boundary_known_team_returns_polygon(client):
    resp = client.get("/api/park_boundary/BOS")
    assert resp.status_code == 200
    assert len(resp.json()) > 10


def test_park_boundary_unknown_team_returns_404(client):
    resp = client.get("/api/park_boundary/ZZZ")
    assert resp.status_code == 404
