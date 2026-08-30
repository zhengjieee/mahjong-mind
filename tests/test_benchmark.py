"""Tests for consumer benchmarking."""

from pathlib import Path

from mahjong_mind.kafka_events.benchmark import ConsumerBenchmark


def test_benchmark_event_parsing(tmp_path: Path) -> None:
    """Test event parsing benchmark."""
    import gzip

    game_file = tmp_path / "test_game.mjson"
    events = [
        '{"type":"start_game","names":["A","B","C","D"],"kyoku_first":0,"aka_flag":false}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"1m","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"]]}',
        '{"type":"tsumo","actor":0,"pai":"1m"}',
        '{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}',
    ]

    with gzip.open(game_file, "wt") as f:
        for event in events:
            f.write(event + "\n")

    benchmark = ConsumerBenchmark(game_file)
    results = benchmark.benchmark_event_parsing()

    assert results["total_events"] == len(events)
    assert results["total_time_ms"] > 0
    assert results["avg_time_per_event_us"] > 0


def test_benchmark_state_reconstruction(tmp_path: Path) -> None:
    """Test state reconstruction benchmark."""
    import gzip

    game_file = tmp_path / "test_game.mjson"
    events = [
        '{"type":"start_game","names":["A","B","C","D"],"kyoku_first":0,"aka_flag":false}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"1m","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"]]}',
        '{"type":"tsumo","actor":0,"pai":"1m"}',
        '{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}',
    ]

    with gzip.open(game_file, "wt") as f:
        for event in events:
            f.write(event + "\n")

    benchmark = ConsumerBenchmark(game_file)
    results = benchmark.benchmark_state_reconstruction()

    assert results["total_events"] == len(events)
    assert results["total_time_ms"] > 0
    assert results["avg_time_per_event_us"] > 0


def test_benchmark_full_pipeline(tmp_path: Path) -> None:
    """Test full pipeline benchmark."""
    import gzip

    game_file = tmp_path / "test_game.mjson"
    events = [
        '{"type":"start_game","names":["A","B","C","D"],"kyoku_first":0,"aka_flag":false}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"1m","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","1s","E","S"]]}',
        '{"type":"tsumo","actor":0,"pai":"1m"}',
        '{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}',
    ]

    with gzip.open(game_file, "wt") as f:
        for event in events:
            f.write(event + "\n")

    benchmark = ConsumerBenchmark(game_file)
    results = benchmark.benchmark_full_pipeline()

    assert results["total_events"] == len(events)
    assert results["total_time_ms"] > 0
    assert results["throughput_events_per_sec"] > 0
