"""Tests for the agreement metrics consumer."""

from mahjong_mind.kafka_events.metrics_consumer import MetricsTracker


def _record(rank: int | None, *, draws: int = 60, riichi: str = "none") -> dict:
    """One predictions-topic record with a resolved outcome."""
    return {
        "game_id": "g",
        "event_index": 3,
        "event": {"type": "dahai"},
        "observation": {
            "draws_remaining": draws,
            "dealer": 0,
            "players": [
                {"melds": [], "riichi": "none"},
                {"melds": [], "riichi": riichi},
                {"melds": [], "riichi": "none"},
                {"melds": [], "riichi": "none"},
            ],
        },
        "outcome": {
            "actor": 0,
            "actual_tile": "1m",
            "rank": rank,
            "top_1": rank == 1,
            "top_3": rank is not None,
        },
    }


def test_ignores_records_without_an_outcome() -> None:
    """Only resolved decisions are scored; a tsumo carries no outcome yet."""
    tracker = MetricsTracker()

    assert tracker.record({"game_id": "g", "event": {"type": "tsumo"}}) is False
    assert tracker.overall.decisions == 0


def test_tracks_top_1_and_top_3_agreement() -> None:
    """Agreement rates count first-choice and within-top-3 matches separately."""
    tracker = MetricsTracker()
    for rank in (1, 2, None, 1):
        tracker.record(_record(rank))

    assert tracker.overall.decisions == 4
    assert tracker.overall.top_1_rate == 0.5
    assert tracker.overall.top_3_rate == 0.75


def test_segments_split_by_riichi_pressure() -> None:
    """Decisions are bucketed so weak segments can be compared."""
    tracker = MetricsTracker()
    tracker.record(_record(1, riichi="none"))
    tracker.record(_record(None, riichi="accepted"))

    assert tracker.per_segment["riichi:none"].top_1_rate == 1.0
    assert tracker.per_segment["riichi:opponent"].top_1_rate == 0.0
    assert tracker.per_segment["seat:dealer"].decisions == 2
