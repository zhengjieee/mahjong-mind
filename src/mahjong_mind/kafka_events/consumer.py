"""Kafka transport around the shared game-event processor.

This module owns everything Kafka: reading the raw event topic, publishing
enriched records, and routing failures to the dead-letter queue. The work of
rebuilding state and producing predictions lives in
`mahjong_mind.event_processing`, which the deployed API service drives directly
without a broker.
"""

import json
import logging
from typing import Any

import httpx
from kafka import KafkaConsumer, KafkaProducer  # type: ignore[import-untyped]
from kafka.errors import KafkaError  # type: ignore[import-untyped]

from mahjong_mind.event_processing import (
    GameEventProcessor,
    PredictionResult,
    parse_event,
)
from mahjong_mind.game_state.reconstructor import StateReconstructor
from mahjong_mind.mjai.parser import ParsedEvent

__all__ = ["GameStateConsumer", "PredictionResult", "main"]

logger = logging.getLogger(__name__)


class GameStateConsumer:
    """Consumes game events, runs the processor, and publishes the results."""

    TOPIC_GAME_EVENTS = "riichi.game-events"
    TOPIC_PREDICTIONS = "riichi.predictions"
    TOPIC_DLQ = "riichi.game-events-dlq"

    def __init__(
        self,
        api_base_url: str,
        kafka_brokers: list[str] | str = "localhost:9092",
        group_id: str = "mahjong-mind-consumer",
        from_beginning: bool = False,
    ):
        """Initialize consumer with API endpoint and Kafka connection."""
        if isinstance(kafka_brokers, str):
            kafka_brokers = [kafka_brokers]
        self.kafka_brokers = kafka_brokers
        self.api_base_url = api_base_url
        self.group_id = group_id
        self.from_beginning = from_beginning

        self.consumer: KafkaConsumer | None = None
        self.producer: KafkaProducer | None = None
        self.http_client: httpx.Client | None = None

        # This stage reaches the model over HTTP; the API service runs the same
        # processor with an in-process recommender instead.
        self.processor = GameEventProcessor(self._request_recommendation)

    @property
    def reconstructors(self) -> dict[str, StateReconstructor]:
        """Per-game reconstructed state held by the processor."""
        return self.processor.reconstructors

    @property
    def pending_predictions(self) -> dict[str, PredictionResult]:
        """Recommendations awaiting the discard that resolves them."""
        return self.processor.pending_predictions

    def connect(self) -> None:
        """Connect to Kafka and initialize HTTP client."""
        self.consumer = KafkaConsumer(
            self.TOPIC_GAME_EVENTS,
            bootstrap_servers=self.kafka_brokers,
            group_id=self.group_id,
            auto_offset_reset="earliest" if self.from_beginning else "latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.http_client = httpx.Client(base_url=self.api_base_url)

    def disconnect(self) -> None:
        """Disconnect from Kafka and close HTTP client."""
        if self.consumer:
            self.consumer.close()
            self.consumer = None
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
        if self.http_client:
            self.http_client.close()
            self.http_client = None

    def run(self) -> None:
        """Main loop: consume and process events."""
        if not self.consumer or not self.http_client:
            raise RuntimeError("Not connected; call connect() first")

        try:
            for message in self.consumer:
                self.process_event(message.value)
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        except KafkaError as e:
            # A broker restart, a group rebalance, or a failed offset commit
            # all surface here rather than inside process_event. Say so
            # loudly: a silent exit looks like the consumer stopped for no
            # reason.
            logger.error(f"Kafka connection lost, consumer stopping: {e}")
            raise

    def process_event(self, event_data: dict[str, Any]) -> None:
        """Process a single Kafka event and publish the enriched record."""
        game_id = event_data["game_id"]
        event_index = event_data["event_index"]
        event_dict = event_data["event"]

        try:
            # Kafka's own consumer-group offsets handle resume, so there is
            # nothing to skip beyond that.
            record = self.processor.process(game_id, event_index, event_dict)
            self._publish_enriched(game_id, event_index, record)
        except Exception as e:  # noqa: BLE001
            # Deliberately broad. The dead-letter queue exists so that one
            # unprocessable event cannot end the stream, and narrowing this
            # once let a TypeError kill the whole consumer.
            logger.error(f"Error processing {game_id} event {event_index}: {e}")
            self._send_to_dlq(game_id, event_index, event_dict, str(e))

    def _reconstruct_event(
        self,
        game_id: str,
        event_index: int,
        event_dict: dict[str, Any],
    ) -> ParsedEvent:
        """Convert a Kafka event dict back to ParsedEvent."""
        return parse_event(game_id, event_index, event_dict)

    def _request_recommendation(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Fulfil one recommendation request over HTTP."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized; call connect() first")

        try:
            response = self.http_client.post("/recommend", json=request)
            response.raise_for_status()
            return dict(response.json())
        except (httpx.HTTPError, ValueError) as e:
            logger.error(f"Failed to get a recommendation: {e}")
            # Don't send to DLQ for API errors; they may be transient.
            return None

    def _publish_enriched(
        self,
        game_id: str,
        event_index: int,
        record: dict[str, Any],
    ) -> None:
        """Publish the event enriched with state and prediction.

        This is the output of the inference stage. The API service reads it to
        drive the live viewer, and the metrics consumer reads the same records
        independently to score the model, each at its own offset.
        """
        if not self.producer:
            return

        try:
            self.producer.send(
                self.TOPIC_PREDICTIONS, key=game_id.encode("utf-8"), value=record
            )
        except (KafkaError, OSError, ValueError) as e:
            # Downstream readers are optional; never break event processing.
            logger.warning(f"Failed to publish {game_id} event {event_index}: {e}")

    def _send_to_dlq(
        self,
        game_id: str,
        event_index: int,
        event_data: dict[str, Any],
        error: str,
    ) -> None:
        """Send failed event to dead-letter queue."""
        if not self.producer:
            return

        dlq_message = {
            "game_id": game_id,
            "event_index": event_index,
            "event": event_data,
            "error": error,
            "timestamp": __import__("time").time(),
        }

        try:
            self.producer.send(
                self.TOPIC_DLQ,
                key=game_id.encode("utf-8"),
                value=dlq_message,
            )
            logger.warning(
                f"Sent event to DLQ: game {game_id} event {event_index}: {error}"
            )
        except (httpx.HTTPError, OSError) as e:
            logger.error(f"Failed to send to DLQ for {game_id} event {event_index}: {e}")


def main() -> int:
    """Run the consumer until interrupted."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Consume game events, run inference, and feed the live viewer",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Inference API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--kafka-brokers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--group-id",
        default="mahjong-mind-consumer",
        help="Kafka consumer group id (default: mahjong-mind-consumer)",
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Replay the whole topic instead of only new events",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    consumer = GameStateConsumer(
        api_base_url=args.api_url,
        kafka_brokers=args.kafka_brokers,
        group_id=args.group_id,
        from_beginning=args.from_beginning,
    )

    try:
        consumer.connect()
    except (KafkaError, RuntimeError, ValueError, OSError) as e:
        print(f"Error: could not connect to Kafka at {args.kafka_brokers}: {e}")
        print("Is Kafka running? Start it with: docker-compose up -d")
        return 1

    print(f"Consuming {GameStateConsumer.TOPIC_GAME_EVENTS} from {args.kafka_brokers}")
    print(f"Inference API: {args.api_url}")
    print("Waiting for events (Ctrl+C to stop)", flush=True)

    try:
        consumer.run()
        return 0
    finally:
        consumer.disconnect()


if __name__ == "__main__":
    import sys

    sys.exit(main())
