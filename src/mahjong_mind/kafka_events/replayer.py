"""Historical game log replayer for Kafka event streaming."""

import json
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from kafka import KafkaProducer  # type: ignore[import-untyped]

from mahjong_mind.mjai.parser import iter_mjai_events


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for game replay."""

    game_id: str
    replay_speed: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed
    start_event_index: int = 0


class HistoricalReplayer:
    """Replays MJAI game logs to Kafka."""

    TOPIC_GAME_EVENTS = "riichi.game-events"

    def __init__(self, kafka_brokers: list[str] | str = "localhost:9092"):
        """Initialize replayer with Kafka connection."""
        if isinstance(kafka_brokers, str):
            kafka_brokers = [kafka_brokers]
        self.kafka_brokers = kafka_brokers
        self.producer: KafkaProducer | None = None
        self._paused = False

    def connect(self) -> None:
        """Connect to Kafka."""
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None

    def replay_game(
        self,
        game_path: Path,
        config: ReplayConfig,
    ) -> None:
        """Replay a single game to Kafka."""
        if not self.producer:
            raise RuntimeError("Not connected to Kafka; call connect() first")

        event_count = 0
        for parsed in iter_mjai_events(game_path):
            if parsed.event_index < config.start_event_index:
                continue

            # Pause support: simple blocking loop
            while self._paused:
                time.sleep(0.1)

            # Publish event to Kafka with game_id as key
            self.producer.send(
                self.TOPIC_GAME_EVENTS,
                key=config.game_id.encode("utf-8"),
                value={
                    "game_id": config.game_id,
                    "event_index": parsed.event_index,
                    "event": parsed.event.model_dump(),
                },
            )
            event_count += 1

            # Apply replay speed delay
            if config.replay_speed > 0:
                # Assume ~0.5s per event at 1x speed (typical decision cadence)
                delay = 0.5 / config.replay_speed
                time.sleep(delay)

        self.producer.flush()

    def pause(self) -> None:
        """Pause replay."""
        self._paused = True

    def resume(self) -> None:
        """Resume replay."""
        self._paused = False

    def list_games(self, data_directory: Path, year: int | None = None) -> Generator[tuple[str, Path], None, None]:
        """Iterate over available game files, optionally filtered by year."""
        search_dir = data_directory
        if year:
            search_dir = data_directory / str(year)
        if not search_dir.exists():
            return

        for game_file in sorted(search_dir.glob("*.mjson")):
            # Extract game_id from filename (stem, without .mjson extension)
            game_id = game_file.stem
            yield game_id, game_file
