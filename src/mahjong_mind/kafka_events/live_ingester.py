"""Live game event ingester from Majsoul servers."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from kafka import KafkaProducer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveGameConfig:
    """Configuration for live game ingestion."""

    room_id: int | None = None
    player_id: int | None = None
    spectator_mode: bool = True


class LiveEventIngester:
    """Capture live game events from Majsoul and publish to Kafka."""

    TOPIC_GAME_EVENTS = "riichi.game-events"

    def __init__(
        self,
        kafka_brokers: list[str] | str = "localhost:9092",
    ):
        """Initialize ingester with Kafka connection."""
        if isinstance(kafka_brokers, str):
            kafka_brokers = [kafka_brokers]
        self.kafka_brokers = kafka_brokers

        self.producer: KafkaProducer | None = None
        self.ws_connection: Any = None
        self.running = False

        self.current_game_id: str | None = None
        self.event_index = 0

    def connect(self) -> None:
        """Initialize Kafka producer."""
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def disconnect(self) -> None:
        """Close Kafka producer and WebSocket."""
        self.running = False
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
        if self.ws_connection:
            try:
                asyncio.run(self._close_websocket())
            except (RuntimeError, OSError) as e:
                logger.error(f"Error closing WebSocket: {e}")
            self.ws_connection = None

    async def _connect_majsoul(self, room_id: int) -> None:
        """Connect to Majsoul WebSocket (requires reverse-engineered protocol)."""
        # This is a placeholder implementation using the reverse-engineered Majsoul API.
        # The actual implementation would use websockets library to connect to:
        # wss://gateway.majsoul.com/gateway
        # and handle Majsoul's protobuf-based message format.
        logger.info(f"Connecting to Majsoul room {room_id} (placeholder)")

    async def _close_websocket(self) -> None:
        """Close WebSocket connection gracefully."""
        if self.ws_connection:
            await self.ws_connection.close()

    def ingest_game(self, config: LiveGameConfig) -> None:
        """Start ingesting a live game (blocking call)."""
        if not self.producer:
            raise RuntimeError("Not connected; call connect() first")

        self.running = True
        try:
            asyncio.run(self._ingest_loop(config))
        except KeyboardInterrupt:
            logger.info("Live ingester interrupted")
        finally:
            self.running = False

    async def _ingest_loop(self, config: LiveGameConfig) -> None:
        """Main loop: connect to Majsoul and ingest live events."""
        if config.room_id is None:
            raise ValueError("room_id is required for live ingestion")

        # Connect to Majsoul
        try:
            await self._connect_majsoul(config.room_id)
        except Exception as e:
            logger.error(f"Failed to connect to Majsoul: {e}")
            raise

        # Listen for game events and translate to MJAI format
        try:
            while self.running:
                # This is where we would receive events from Majsoul WebSocket
                # For now, this is a placeholder
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error during live ingestion: {e}")
            raise

    def _translate_majsoul_event(self, majsoul_event: dict[str, Any]) -> dict[str, Any]:
        """Translate Majsoul event format to MJAI format.

        Majsoul uses a different event structure; this translates to MJAI
        for compatibility with the rest of the pipeline.
        """
        # This would map Majsoul events to MJAI equivalents
        # For example: majsoul "dahai" -> MJAI "dahai"
        # with field conversions as needed
        raise NotImplementedError("Event translation requires Majsoul protocol reverse-engineering")

    def _publish_event(
        self,
        game_id: str,
        event_index: int,
        event_dict: dict[str, Any],
    ) -> None:
        """Publish an event to Kafka."""
        if not self.producer:
            return

        message = {
            "game_id": game_id,
            "event_index": event_index,
            "event": event_dict,
        }

        try:
            self.producer.send(
                self.TOPIC_GAME_EVENTS,
                key=game_id.encode("utf-8"),
                value=message,
            )
            logger.debug(f"Published {game_id} event {event_index}")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to publish event for {game_id}: {e}")


def run_live_ingester(room_id: int) -> None:
    """Run a live game ingester for a specific room."""
    ingester = LiveEventIngester()
    try:
        ingester.connect()
        config = LiveGameConfig(room_id=room_id)
        ingester.ingest_game(config)
    except KeyboardInterrupt:
        logger.info("Live ingester stopped by user")
    finally:
        ingester.disconnect()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m mahjong_mind.kafka_events.live_ingester <room_id>")
        sys.exit(1)

    try:
        room_id = int(sys.argv[1])
        run_live_ingester(room_id)
    except ValueError:
        print("Invalid room_id; must be an integer")
        sys.exit(1)
