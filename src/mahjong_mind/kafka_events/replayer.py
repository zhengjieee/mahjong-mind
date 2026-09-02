"""Historical game log replayer for Kafka event streaming."""

import json
import threading
import time
from collections.abc import Callable, Generator
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
    # In step mode the replay publishes one event per advance() call instead
    # of running on a timer.
    step_mode: bool = False


class HistoricalReplayer:
    """Replays MJAI game logs to Kafka."""

    TOPIC_GAME_EVENTS = "riichi.game-events"

    def __init__(
        self,
        kafka_brokers: list[str] | str = "localhost:9092",
        sink: Callable[[dict], None] | None = None,
    ):
        """Initialize replayer, publishing to Kafka or to a given sink.

        A sink replaces the broker entirely, which is how the deployed service
        replays games with no Kafka available.
        """
        if isinstance(kafka_brokers, str):
            kafka_brokers = [kafka_brokers]
        self.kafka_brokers = kafka_brokers
        self.sink = sink
        self.producer: KafkaProducer | None = None
        self._paused = False
        self._stopped = False
        # Set by advance() to release one event in step mode.
        self._step = threading.Event()
        self.events_sent = 0
        # Whether this replay advances on demand, so a UI can show the right
        # controls for a game it did not start itself.
        self.step_mode = False

    def connect(self) -> None:
        """Connect to Kafka, unless a sink is already taking the events."""
        if self.sink is not None:
            return
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
        if not self.producer and self.sink is None:
            raise RuntimeError("Not connected to Kafka; call connect() first")

        self.step_mode = config.step_mode
        event_count = 0
        for parsed in iter_mjai_events(game_path):
            if self._stopped:
                break

            if parsed.event_index < config.start_event_index:
                continue

            if config.step_mode:
                # Block until advance() releases the next event.
                self._step.wait()
                self._step.clear()
                if self._stopped:
                    break
            else:
                # Pause support: simple blocking loop
                while self._paused and not self._stopped:
                    time.sleep(0.1)
                if self._stopped:
                    break

            self._emit(
                config.game_id,
                {
                    "game_id": config.game_id,
                    "event_index": parsed.event_index,
                    "event": parsed.event.model_dump(),
                },
            )
            event_count += 1
            self.events_sent = event_count

            # Apply replay speed delay (step mode waits on advance() instead)
            if not config.step_mode and config.replay_speed > 0:
                # Assume ~0.5s per event at 1x speed (typical decision cadence)
                delay = 0.5 / config.replay_speed
                time.sleep(delay)

        if self.producer:
            self.producer.flush()

    def _emit(self, game_id: str, payload: dict) -> None:
        """Hand one event to whichever transport this replay is using."""
        if self.sink is not None:
            self.sink(payload)
        elif self.producer:
            self.producer.send(
                self.TOPIC_GAME_EVENTS,
                key=game_id.encode("utf-8"),
                value=payload,
            )

    def pause(self) -> None:
        """Pause replay."""
        self._paused = True

    def resume(self) -> None:
        """Resume replay."""
        self._paused = False

    def advance(self) -> None:
        """Release one event in step mode."""
        self._step.set()

    def stop(self) -> None:
        """Stop the replay; a step-mode wait is released so it can exit."""
        self._stopped = True
        self._paused = False
        self._step.set()

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
