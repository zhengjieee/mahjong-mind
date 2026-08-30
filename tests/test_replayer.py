"""Tests for historical game log replayer."""

from pathlib import Path
from unittest.mock import MagicMock

from mahjong_mind.kafka_events.replayer import HistoricalReplayer, ReplayConfig


def test_replayer_publishes_events_with_game_id_key(tmp_path: Path) -> None:
    """Test that replayer publishes events to Kafka with game_id as key."""
    # Create a minimal MJAI file with one hand
    game_file = tmp_path / "test_game.mjson"
    events = [
        '{"type":"start_game","names":["A","B","C","D"],"kyoku_first":0,"aka_flag":false}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"1m","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"]]}',
        '{"type":"tsumo","actor":0,"pai":"1m"}',
        '{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}',
        '{"type":"end_kyoku"}',
        '{"type":"end_game"}',
    ]

    import gzip

    with gzip.open(game_file, "wt") as f:
        for event in events:
            f.write(event + "\n")

    # Mock Kafka producer
    mock_producer = MagicMock()
    game_id = "test-game-001"

    replayer = HistoricalReplayer()
    replayer.producer = mock_producer

    config = ReplayConfig(game_id=game_id, replay_speed=0.0)  # No delay for testing
    replayer.replay_game(game_file, config)

    # Verify all events were published
    assert mock_producer.send.call_count == len(events)
    mock_producer.flush.assert_called_once()

    # Verify key is game_id (encoded)
    calls = mock_producer.send.call_args_list
    for call in calls:
        assert call[1]["key"] == game_id.encode("utf-8")
        assert call[0][0] == "riichi.game-events"


def test_replayer_respects_start_event_index(tmp_path: Path) -> None:
    """Test that replayer can skip to a specific event."""
    game_file = tmp_path / "test_game.mjson"
    events = [
        '{"type":"start_game","names":["A","B","C","D"],"kyoku_first":0,"aka_flag":false}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"1m","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"]]}',
        '{"type":"tsumo","actor":0,"pai":"1m"}',
        '{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}',
        '{"type":"end_kyoku"}',
        '{"type":"end_game"}',
    ]

    import gzip

    with gzip.open(game_file, "wt") as f:
        for event in events:
            f.write(event + "\n")

    mock_producer = MagicMock()
    replayer = HistoricalReplayer()
    replayer.producer = mock_producer

    # Start from event index 2, skipping first two events
    config = ReplayConfig(game_id="test", replay_speed=0.0, start_event_index=2)
    replayer.replay_game(game_file, config)

    # Should publish events 2-5 (4 events total)
    assert mock_producer.send.call_count == 4
    calls = mock_producer.send.call_args_list
    assert calls[0][1]["value"]["event_index"] == 2


def test_replayer_list_games(tmp_path: Path) -> None:
    """Test listing available games."""
    # Create test files
    year_dir = tmp_path / "2018"
    year_dir.mkdir()
    (year_dir / "game1.mjson").touch()
    (year_dir / "game2.mjson").touch()
    (year_dir / "game3.mjson").touch()

    replayer = HistoricalReplayer()
    games = list(replayer.list_games(tmp_path, year=2018))

    assert len(games) == 3
    game_ids = [g[0] for g in games]
    assert sorted(game_ids) == ["game1", "game2", "game3"]
