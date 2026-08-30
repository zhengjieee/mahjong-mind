"""Tests for game state consumer."""

from unittest.mock import MagicMock, patch

from mahjong_mind.kafka_events.consumer import GameStateConsumer


def test_consumer_reconstructs_event_from_dict() -> None:
    """Test that consumer can reconstruct events from Kafka dicts."""
    consumer = GameStateConsumer(api_base_url="http://localhost:8000")

    # Create a start_game event dict
    event_dict = {
        "type": "start_game",
        "names": ["A", "B", "C", "D"],
        "kyoku_first": 0,
        "aka_flag": False,
    }

    parsed = consumer._reconstruct_event("test-game-001", 0, event_dict)

    assert parsed.match_id == "test-game-001"
    assert parsed.event_index == 0
    assert parsed.event.type == "start_game"


def test_consumer_processes_tsumo_and_calls_api() -> None:
    """Test that consumer calls /recommend on TsumoEvent."""
    consumer = GameStateConsumer(api_base_url="http://localhost:8000")

    # Mock HTTP client
    mock_http_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model_version": "test",
        "top_3_actions": [
            {"tile": "1m", "probability": 0.5},
            {"tile": "2m", "probability": 0.3},
            {"tile": "3m", "probability": 0.2},
        ],
        "inference_ms": 10.0,
    }
    mock_http_client.post.return_value = mock_response
    consumer.http_client = mock_http_client

    # Create a game with start_game, start_kyoku, tsumo events
    game_id = "test-game-002"

    # Process start_game
    start_game_event = {
        "game_id": game_id,
        "event_index": 0,
        "event": {
            "type": "start_game",
            "names": ["A", "B", "C", "D"],
            "kyoku_first": 0,
            "aka_flag": False,
        },
    }
    consumer.process_event(start_game_event)

    # Process start_kyoku
    start_kyoku_event = {
        "game_id": game_id,
        "event_index": 1,
        "event": {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1m",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
            ],
        },
    }
    consumer.process_event(start_kyoku_event)

    # Process tsumo (draw event)
    tsumo_event = {
        "game_id": game_id,
        "event_index": 2,
        "event": {"type": "tsumo", "actor": 0, "pai": "1m"},
    }
    consumer.process_event(tsumo_event)

    # Verify that /recommend was called
    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    assert call_args[0][0] == "/recommend"

    # Verify that prediction was cached
    assert game_id in consumer.pending_predictions
    assert consumer.pending_predictions[game_id].predicted_tiles == ["1m", "2m", "3m"]


def test_consumer_logs_dahai_vs_prediction() -> None:
    """Test that consumer logs actual discard vs prediction."""
    consumer = GameStateConsumer(api_base_url="http://localhost:8000")

    # Setup cached prediction
    game_id = "test-game-003"
    from mahjong_mind.kafka_events.consumer import PredictionResult

    consumer.pending_predictions[game_id] = PredictionResult(
        game_id=game_id,
        event_index=2,
        actor=0,
        predicted_tiles=["1m", "2m", "3m"],
        predicted_probabilities=[0.5, 0.3, 0.2],
    )

    # Process dahai (discard event)
    dahai_event_dict: dict[str, object] = {
        "type": "dahai",
        "actor": 0,
        "pai": "1m",
        "tsumogiri": False,
    }

    # Mock logger to verify logging
    with patch("mahjong_mind.kafka_events.consumer.logger") as mock_logger:
        parsed_event = consumer._reconstruct_event(game_id, 3, dahai_event_dict)  # type: ignore[arg-type]
        from mahjong_mind.mjai.events import DahaiEvent
        assert isinstance(parsed_event.event, DahaiEvent)
        consumer._handle_dahai(game_id, 3, parsed_event.event)

        # Verify log message includes the actual tile and prediction
        assert mock_logger.info.called
        log_msg = mock_logger.info.call_args[0][0]
        assert "1m" in log_msg  # actual discard
        assert "1m" in log_msg  # in predicted list


def test_consumer_seat_wind_calculation() -> None:
    """Test seat wind calculation."""
    # Player 0 (dealer): seat wind E
    assert GameStateConsumer._seat_wind(0, 0) == "E"
    # Player 1 (discard player): seat wind S
    assert GameStateConsumer._seat_wind(1, 0) == "S"
    # Player 2: seat wind W
    assert GameStateConsumer._seat_wind(2, 0) == "W"
    # Player 3: seat wind N
    assert GameStateConsumer._seat_wind(3, 0) == "N"

    # With dealer = 1
    assert GameStateConsumer._seat_wind(1, 1) == "E"
    assert GameStateConsumer._seat_wind(2, 1) == "S"
    assert GameStateConsumer._seat_wind(3, 1) == "W"
    assert GameStateConsumer._seat_wind(0, 1) == "N"
