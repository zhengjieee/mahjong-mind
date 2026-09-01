"""Agreement metrics consumer for the predictions topic.

Reads the same records that drive the live viewer, but with its own consumer
group and offset, and scores how often the model's ranking matched the human's
actual discard. Nothing in the inference or viewer path knows this exists.
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kafka import KafkaConsumer  # type: ignore[import-untyped]
from kafka.errors import KafkaError  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

TOPIC_PREDICTIONS = "riichi.predictions"


@dataclass
class AgreementTally:
    """Running top-1 and top-3 agreement over resolved decisions."""

    decisions: int = 0
    top_1: int = 0
    top_3: int = 0

    def add(self, outcome: dict[str, Any]) -> None:
        """Record one resolved decision."""
        self.decisions += 1
        if outcome.get("top_1"):
            self.top_1 += 1
        if outcome.get("top_3"):
            self.top_3 += 1

    @property
    def top_1_rate(self) -> float:
        """Share of decisions where the model's first choice matched."""
        return self.top_1 / self.decisions if self.decisions else 0.0

    @property
    def top_3_rate(self) -> float:
        """Share of decisions where the human's tile was in the top 3."""
        return self.top_3 / self.decisions if self.decisions else 0.0

    def summary(self) -> str:
        """One-line readout."""
        return (
            f"{self.decisions} decisions | "
            f"top-1 {self.top_1_rate:.1%} | top-3 {self.top_3_rate:.1%}"
        )


@dataclass
class MetricsTracker:
    """Overall and per-segment agreement."""

    overall: AgreementTally = field(default_factory=AgreementTally)
    per_game: dict[str, AgreementTally] = field(
        default_factory=lambda: defaultdict(AgreementTally)
    )
    per_segment: dict[str, AgreementTally] = field(
        default_factory=lambda: defaultdict(AgreementTally)
    )

    def record(self, message: dict[str, Any]) -> bool:
        """Score one record; returns True if it carried a resolved decision."""
        outcome = message.get("outcome")
        if not outcome:
            return False

        self.overall.add(outcome)
        self.per_game[message.get("game_id", "unknown")].add(outcome)
        for segment in self._segments(message.get("observation"), outcome):
            self.per_segment[segment].add(outcome)
        return True

    @staticmethod
    def _segments(observation: dict[str, Any] | None, outcome: dict[str, Any]) -> list[str]:
        """Segment labels for one decision, derived from observable state only.

        Mirrors the dimensions in the Week 7 segmented error analysis, where
        opponent riichi pressure was the model's clearest weak point.
        """
        if not observation:
            return []

        segments = []

        draws = observation.get("draws_remaining", 0)
        segments.append(
            "phase:early" if draws > 46 else "phase:mid" if draws > 23 else "phase:late"
        )

        actor = outcome.get("actor")
        players = observation.get("players") or []
        opponents_riichi = any(
            player.get("riichi") == "accepted"
            for index, player in enumerate(players)
            if index != actor
        )
        segments.append(
            "riichi:opponent" if opponents_riichi else "riichi:none"
        )

        if actor is not None and actor < len(players):
            open_hand = bool(players[actor].get("melds"))
            segments.append("hand:open" if open_hand else "hand:closed")

        if actor is not None:
            segments.append(
                "seat:dealer" if actor == observation.get("dealer") else "seat:non-dealer"
            )

        return segments

    def report(self) -> str:
        """Multi-line summary of overall and per-segment agreement."""
        lines = [f"OVERALL  {self.overall.summary()}"]
        for segment in sorted(self.per_segment):
            lines.append(f"  {segment:<20} {self.per_segment[segment].summary()}")
        return "\n".join(lines)


def main() -> int:
    """Consume the predictions topic and report agreement as it goes."""
    parser = argparse.ArgumentParser(
        description="Score model agreement against human discards",
    )
    parser.add_argument(
        "--kafka-brokers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--group-id",
        default="mahjong-mind-metrics",
        help="Kafka consumer group id (default: mahjong-mind-metrics)",
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Score the whole topic instead of only new records",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print the full segment report every N decisions (default: 25)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        consumer = KafkaConsumer(
            TOPIC_PREDICTIONS,
            bootstrap_servers=args.kafka_brokers.split(","),
            group_id=args.group_id,
            auto_offset_reset="earliest" if args.from_beginning else "latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
    except (KafkaError, OSError) as e:
        print(f"Error: could not connect to Kafka at {args.kafka_brokers}: {e}")
        print("Is Kafka running? Start it with: docker-compose up -d")
        return 1

    tracker = MetricsTracker()
    print(f"Scoring {TOPIC_PREDICTIONS} from {args.kafka_brokers}")
    print("Waiting for decisions (Ctrl+C to stop)", flush=True)

    try:
        for message in consumer:
            if not tracker.record(message.value):
                continue
            print(f"\r{tracker.overall.summary()}", end="", flush=True)
            if tracker.overall.decisions % args.report_every == 0:
                print(f"\n{tracker.report()}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        if tracker.overall.decisions:
            print(f"\n\nFinal report\n{tracker.report()}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
