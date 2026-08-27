"""Tests for the FastAPI inference service."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mahjong_mind.api import service as service_module
from mahjong_mind.api.service import app


@pytest.fixture
def client():
    """FastAPI test client with loaded model."""
    # Manually load the service for testing.
    checkpoint_path = (
        Path(__file__).parent.parent / "data" / "checkpoints" / "transformer_model" / "epoch-10.pt"
    )
    service_module._service = service_module.load_service(checkpoint_path, model_version="transformer-epoch-10")

    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
