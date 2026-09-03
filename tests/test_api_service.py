"""Tests for the FastAPI inference service."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mahjong_mind.api import service as service_module
from mahjong_mind.api.service import app


@pytest.fixture
def client():
    """FastAPI test client with loaded model."""
    # Prefer the trained weights when a local checkout has them. CI has none,
    # and skipping there left the request path untested -- which is exactly
    # where both /recommend bugs lived. A tiny untrained fixture stands in:
    # the point is that the path works, not what it predicts.
    trained = (
        Path(__file__).parent.parent / "data" / "checkpoints" / "transformer_model" / "epoch-10.pt"
    )
    checkpoint_path = (
        trained
        if trained.exists()
        else Path(__file__).parent / "fixtures" / "untrained-transformer.pt"
    )
    service_module._service = service_module.load_service(checkpoint_path)
    # Handed to the test so it can check the service reports what it loaded.
    app.state.checkpoint_path = checkpoint_path

    return TestClient(app)


def test_health_endpoint(client):
    """GET /health reports liveness and the weights actually loaded.

    The version is derived from the checkpoint's contents rather than written
    down, so serving a different file cannot keep reporting the old name.
    """
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == service_module.checkpoint_identity(
        app.state.checkpoint_path
    )


def test_recommend_endpoint_basic(client):
    """POST /recommend accepts a valid observation and returns a recommendation."""
    # Minimal valid observation for a discard decision.
    observation = {
        "match_id": "test-match-001",
        "observer": 0,
        "names": ["Player0", "Player1", "Player2", "Player3"],
        "aka_flag": True,
        "hand_index": 0,
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "dealer": 0,
        "scores": [30000, 25000, 25000, 20000],
        "dora_markers": ["1m"],
        "draws_remaining": 70,
        "hand_ended": False,
        "own_hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
        "own_last_draw": "S",
        "actor_turn_index": 0,
        "seat_wind": "E",
        "legal_discard_mask": [True] * 37,
        "players": [
            {
                "concealed_tile_count": 13,
                "discards": [],
                "melds": [],
                "riichi": "none",
            }
            for _ in range(4)
        ],
    }

    response = client.post("/recommend", json=observation)
    assert response.status_code == 200

    data = response.json()
    assert "model_version" in data
    assert "top_3_actions" in data
    assert "inference_ms" in data

    # Verify top-3 structure.
    assert len(data["top_3_actions"]) <= 3
    for action in data["top_3_actions"]:
        assert "tile" in action
        assert "probability" in action
        assert 0 <= action["probability"] <= 1


def test_recommend_with_no_1m_in_hand(client):
    """A hand without 1m must still get a recommendation.

    The encoder rejects a label_index that is not a legal action, so a fixed
    placeholder of 0 (tile "1m") made every such hand fail with a 500.
    """
    from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES

    own_hand = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "2s", "3s", "4s", "W", "N"]
    legal_mask = [tile in own_hand for tile in DISCARD_TILE_TYPES]
    assert not legal_mask[0], "index 0 must be illegal for this test to be meaningful"

    observation = {
        "match_id": "test-match-002",
        "observer": 0,
        "names": ["Player0", "Player1", "Player2", "Player3"],
        "aka_flag": True,
        "hand_index": 0,
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "dealer": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_markers": ["5m"],
        "draws_remaining": 60,
        "hand_ended": False,
        "own_hand": own_hand,
        "own_last_draw": "N",
        "actor_turn_index": 3,
        "seat_wind": "E",
        "legal_discard_mask": legal_mask,
        "players": [
            {
                "concealed_tile_count": 13,
                "discards": [],
                "melds": [],
                "riichi": "none",
            }
            for _ in range(4)
        ],
    }

    response = client.post("/recommend", json=observation)
    assert response.status_code == 200

    # Every recommended tile must be one the player can legally discard.
    for action in response.json()["top_3_actions"]:
        assert action["tile"] in own_hand
