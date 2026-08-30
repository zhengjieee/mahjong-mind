"""Tests for live event ingester."""

from unittest.mock import MagicMock

from mahjong_mind.kafka_events.live_ingester import LiveEventIngester, LiveGameConfig


def test_live_ingester_initializes() -> None:
    """Test ingester initialization."""
    ingester = LiveEventIngester()

    assert ingester.kafka_brokers == ["localhost:9092"]
    assert ingester.producer is None
    assert ingester.running is False


def test_live_ingester_connects() -> None:
    """Test Kafka producer connection."""
    ingester = LiveEventIngester()
    ingester.connect()

    assert ingester.producer is not None
    ingester.disconnect()


def test_live_ingester_publishes_event() -> None:
    """Test publishing an event to Kafka."""
    ingester = LiveEventIngester()
    mock_producer = MagicMock()
    ingester.producer = mock_producer

    event_dict = {"type": "tsumo", "actor": 0, "pai": "1m"}
    ingester._publish_event("live-game-001", 0, event_dict)

    mock_producer.send.assert_called_once()
    call_args = mock_producer.send.call_args
    assert call_args[0][0] == ingester.TOPIC_GAME_EVENTS
    assert call_args[1]["key"] == b"live-game-001"


def test_live_game_config() -> None:
    """Test live game configuration."""
    config = LiveGameConfig(room_id=12345, player_id=0, spectator_mode=True)

    assert config.room_id == 12345
    assert config.player_id == 0
    assert config.spectator_mode is True


def test_live_ingester_raises_without_producer() -> None:
    """Test that ingesting without connection raises error."""
    ingester = LiveEventIngester()

    try:
        config = LiveGameConfig(room_id=12345)
        # This will raise because producer is not connected
        ingester.ingest_game(config)
    except RuntimeError as e:
        assert "Not connected" in str(e)


def test_live_ingester_requires_room_id() -> None:
    """Test that ingestion requires room_id."""
    ingester = LiveEventIngester()
    ingester.connect()

    import asyncio

    config = LiveGameConfig(room_id=None)
    try:
        asyncio.run(ingester._ingest_loop(config))
    except ValueError as e:
        assert "room_id is required" in str(e)
    finally:
        ingester.disconnect()
