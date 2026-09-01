"""Tests for WebSocket and event publishing endpoints."""

import json

from fastapi.testclient import TestClient

from mahjong_mind.api.service import app


def test_publish_event_endpoint() -> None:
    """Test that /api/publish-event endpoint accepts events."""
    client = TestClient(app)

    event_data = {
        "game_id": "test-game-001",
        "event_index": 0,
        "event": {"type": "start_game", "names": ["A", "B", "C", "D"], "kyoku_first": 0, "aka_flag": False},
        "observation": {
            "match_id": "test-game-001",
            "hand_index": 0,
            "kyoku": 1,
            "bakaze": "E",
            "dealer": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["A", "B", "C", "D"],
            "aka_flag": False,
        },
        "recommendations": {
            "model_version": "transformer-epoch-10",
            "top_3_actions": [
                {"tile": "1m", "probability": 0.5},
                {"tile": "2m", "probability": 0.3},
                {"tile": "3m", "probability": 0.2},
            ],
            "inference_ms": 10.0,
        },
    }

    response = client.post("/api/publish-event", json=event_data)
    assert response.status_code == 200
    assert response.json()["status"] == "published"


def test_websocket_connection() -> None:
    """Test basic WebSocket connection."""
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        # Send a ping message
        websocket.send_text("ping")
        # Just verify connection doesn't crash


def test_root_endpoint() -> None:
    """Test root endpoint redirects to viewer."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200


def test_list_games_endpoint() -> None:
    """GET /api/games reports live games, years, and recorded games."""
    client = TestClient(app)
    response = client.get("/api/games")
    assert response.status_code == 200

    data = response.json()
    assert set(data) == {"live", "years", "selected_year", "recorded", "replaying"}
    assert isinstance(data["live"], list)
    assert isinstance(data["recorded"], list)


def test_watch_unknown_game_returns_404() -> None:
    """POST /api/watch rejects a game id with no matching file."""
    client = TestClient(app)
    response = client.post(
        "/api/watch", json={"game_id": "2009999999gm-does-not-exist", "speed": 1.0}
    )
    assert response.status_code == 404


def test_playing_list_only_shows_running_replays() -> None:
    """A game leaves the playing list as soon as its replay stops.

    Events stay in flight for a moment after a replay is stopped, so the list
    is filtered to replays this service is actually running. Without that, a
    stopped game reappears when one of those late events arrives.
    """
    client = TestClient(app)
    game_id = "not-being-replayed"

    client.post(
        "/api/publish-event",
        json={
            "game_id": game_id,
            "event_index": 0,
            "event": {"type": "tsumo", "actor": 0, "pai": "1m"},
        },
    )

    live = client.get("/api/games").json()["live"]
    assert all(game["game_id"] != game_id for game in live)


def test_websocket_subscription_filters_by_game() -> None:
    """A subscribed client receives only its game's events."""
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps({"action": "subscribe", "game_id": "wanted-game"}))

        # An event for a different game must not reach this client.
        client.post(
            "/api/publish-event",
            json={
                "game_id": "other-game",
                "event_index": 0,
                "event": {"type": "tsumo", "actor": 0, "pai": "1m"},
            },
        )
        client.post(
            "/api/publish-event",
            json={
                "game_id": "wanted-game",
                "event_index": 1,
                "event": {"type": "dahai", "actor": 0, "pai": "2p"},
            },
        )

        received = websocket.receive_json()
        assert received["game_id"] == "wanted-game"
        assert received["event_index"] == 1


def test_control_endpoints_require_running_replay() -> None:
    """Controlling a game that is not replaying returns 404, not a crash."""
    client = TestClient(app)
    for path in ("/api/step", "/api/pause", "/api/resume", "/api/stop"):
        response = client.post(path, json={"game_id": "not-running"})
        assert response.status_code == 404, path


def test_starting_a_game_stops_any_other() -> None:
    """Only one game plays at a time, so a new one retires the rest."""
    from unittest.mock import MagicMock

    from mahjong_mind.api import service

    client = TestClient(app)
    running = MagicMock()
    service._active_replays["already-running"] = running

    try:
        # A real game id, so the replay actually starts.
        response = client.post(
            "/api/watch",
            json={"game_id": "2017011403gm-00a9-0000-ec687b95", "step_mode": True},
        )
        if response.status_code == 404:
            import pytest

            pytest.skip("game files not available")

        assert running.stop.called
        assert "already-running" not in service._active_replays
    finally:
        for game_id in list(service._active_replays):
            replayer = service._active_replays.pop(game_id)
            if hasattr(replayer, "stop"):
                replayer.stop()
