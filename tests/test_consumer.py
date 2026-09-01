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

    mock_producer = MagicMock()
    consumer.producer = mock_producer

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

    # /recommend is called once, on the tsumo
    posted_paths = [call[0][0] for call in mock_http_client.post.call_args_list]
    assert posted_paths.count("/recommend") == 1

    # Verify that prediction was cached
    assert game_id in consumer.pending_predictions
    assert consumer.pending_predictions[game_id].predicted_tiles == ["1m", "2m", "3m"]

    # The enriched tsumo is published to the predictions topic, not back to the API
    published = [
        call for call in mock_producer.send.call_args_list
        if call[0][0] == GameStateConsumer.TOPIC_PREDICTIONS
    ]
    tsumo_payload = published[-1][1]["value"]
    assert tsumo_payload["event"]["type"] == "tsumo"
    assert tsumo_payload["observation"]["own_last_draw"] == "1m"
    assert tsumo_payload["recommendations"]["top_3_actions"][0]["tile"] == "1m"


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


def test_consumer_sends_to_dlq_on_error() -> None:
    """A failed event goes to the dead-letter topic."""
    consumer = GameStateConsumer(api_base_url="http://localhost:8000")
    mock_producer = MagicMock()
    consumer.producer = mock_producer

    consumer._send_to_dlq("test-game-001", 5, {"type": "invalid"}, "Test error")

    assert mock_producer.send.called
    call_args = mock_producer.send.call_args
    assert call_args[0][0] == consumer.TOPIC_DLQ
    assert call_args[1]["key"] == b"test-game-001"
    assert call_args[1]["value"]["error"] == "Test error"


def test_dahai_outcome_rides_the_predictions_topic() -> None:
    """The resolved prediction is attached so other consumers can score it."""
    from mahjong_mind.kafka_events.consumer import PredictionResult
    from mahjong_mind.mjai.events import DahaiEvent

    consumer = GameStateConsumer(api_base_url="http://localhost:8000")
    consumer.pending_predictions["g"] = PredictionResult(
        game_id="g",
        event_index=2,
        actor=0,
        predicted_tiles=["9p", "1m", "3s"],
        predicted_probabilities=[0.5, 0.3, 0.2],
    )

    event = DahaiEvent(type="dahai", actor=0, pai="1m", tsumogiri=False)
    outcome = consumer._handle_dahai("g", 3, event)

    assert outcome is not None
    assert outcome["actual_tile"] == "1m"
    assert outcome["rank"] == 2
    assert outcome["top_1"] is False
    assert outcome["top_3"] is True
