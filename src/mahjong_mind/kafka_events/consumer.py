"""Kafka event consumer for game state reconstruction and inference."""

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from kafka import KafkaConsumer  # type: ignore[import-untyped]

from mahjong_mind.game_state.legal_actions import legal_discard_mask
from mahjong_mind.game_state.player_observation import observation_for_player
from mahjong_mind.game_state.reconstructor import StateReconstructor
from mahjong_mind.mjai.events import DahaiEvent, TsumoEvent
from mahjong_mind.mjai.parser import ParsedEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Cached prediction result from /recommend endpoint."""

    game_id: str
    event_index: int
    actor: int
    predicted_tiles: list[str]
    predicted_probabilities: list[float]


class GameStateConsumer:
    """Consumes game events, reconstructs state, and calls inference API."""

    TOPIC_GAME_EVENTS = "riichi.game-events"

    def __init__(
        self,
        api_base_url: str,
        kafka_brokers: list[str] | str = "localhost:9092",
        group_id: str = "mahjong-mind-consumer",
    ):
        """Initialize consumer with API endpoint and Kafka connection."""
        if isinstance(kafka_brokers, str):
            kafka_brokers = [kafka_brokers]
        self.kafka_brokers = kafka_brokers
        self.api_base_url = api_base_url
        self.group_id = group_id

        self.consumer: KafkaConsumer | None = None
        self.http_client: httpx.Client | None = None

        # Per-game-id reconstructors to maintain state
        self.reconstructors: dict[str, StateReconstructor] = {}

        # Cache predictions from TsumoEvent to DahaiEvent
        self.pending_predictions: dict[str, PredictionResult] = {}

    def connect(self) -> None:
        """Connect to Kafka and initialize HTTP client."""
        self.consumer = KafkaConsumer(
            self.TOPIC_GAME_EVENTS,
            bootstrap_servers=self.kafka_brokers,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.http_client = httpx.Client(base_url=self.api_base_url)

    def disconnect(self) -> None:
        """Disconnect from Kafka and close HTTP client."""
        if self.consumer:
            self.consumer.close()
            self.consumer = None
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

    def process_event(self, event_data: dict[str, Any]) -> None:
        """Process a single Kafka event."""
        game_id = event_data["game_id"]
        event_index = event_data["event_index"]
        event_dict = event_data["event"]

        # Get or create state reconstructor for this game
        if game_id not in self.reconstructors:
            self.reconstructors[game_id] = StateReconstructor()

        reconstructor = self.reconstructors[game_id]

        # Reconstruct the event from dict
        parsed_event = self._reconstruct_event(game_id, event_index, event_dict)

        # Apply to state
        state = reconstructor.apply(parsed_event)

        # Detect decision points and call inference API
        if isinstance(parsed_event.event, TsumoEvent):
            self._handle_tsumo(game_id, event_index, parsed_event.event, state)
        elif isinstance(parsed_event.event, DahaiEvent):
            self._handle_dahai(game_id, event_index, parsed_event.event)

    def _reconstruct_event(
        self,
        game_id: str,
        event_index: int,
        event_dict: dict[str, Any],
    ) -> ParsedEvent:
        """Convert a Kafka event dict back to ParsedEvent."""
        event_type = event_dict.get("type")

        # Map event type to class
        event_classes = {
            "start_game": "StartGameEvent",
            "start_kyoku": "StartKyokuEvent",
            "tsumo": "TsumoEvent",
            "dahai": "DahaiEvent",
            "chi": "ChiEvent",
            "pon": "PonEvent",
            "daiminkan": "DaiminkanEvent",
            "ankan": "AnkanEvent",
            "kakan": "KakanEvent",
            "dora": "DoraEvent",
            "reach": "ReachEvent",
            "reach_accepted": "ReachAcceptedEvent",
            "hora": "HoraEvent",
            "ryukyoku": "RyukyokuEvent",
            "end_kyoku": "EndKyokuEvent",
            "end_game": "EndGameEvent",
        }

        if event_type not in event_classes:
            raise ValueError(f"Unknown event type: {event_type}")

        # Dynamically import the event class
        from mahjong_mind.mjai import events as events_module

        event_class = getattr(events_module, event_classes[event_type])
        event_obj = event_class(**event_dict)

        return ParsedEvent(
            match_id=game_id,
            event_index=event_index,
            event=event_obj,
        )

    def _handle_tsumo(
        self,
        game_id: str,
        event_index: int,
        event: TsumoEvent,
        state: Any,
    ) -> None:
        """Handle a draw event: call /recommend API."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized; call connect() first")

        actor = event.actor
        observation = observation_for_player(state, actor)

        # Compute turn index: how many discards has this player made?
        actor_turn_index = len(observation.players[actor].discards)

        # Compute seat wind
        seat_wind = self._seat_wind(actor, observation.dealer)

        # Get legal discard mask and identify which tile indices are legal
        mask = legal_discard_mask(observation)

        try:
            # Call /recommend API
            request_data = {
                "match_id": observation.match_id,
                "observer": actor,
                "names": list(observation.names),
                "aka_flag": observation.aka_flag,
                "hand_index": observation.hand_index,
                "bakaze": observation.bakaze,
                "kyoku": observation.kyoku,
                "honba": observation.honba,
                "kyotaku": observation.kyotaku,
                "dealer": observation.dealer,
                "scores": list(observation.scores),
                "dora_markers": list(observation.dora_markers),
                "draws_remaining": observation.draws_remaining,
                "hand_ended": observation.hand_ended,
                "own_hand": list(observation.own_hand),
                "own_last_draw": observation.own_last_draw,
                "players": [
                    {
                        "concealed_tile_count": p.concealed_tile_count,
                        "discards": [
                            {
                                "tile": d.tile,
                                "tsumogiri": d.tsumogiri,
                                "riichi": d.riichi,
                                "called": d.called,
                            }
                            for d in p.discards
                        ],
                        "melds": [
                            {
                                "type": m.type,
                                "tiles": list(m.tiles),
                                "called_tile": m.called_tile,
                                "source": m.target,
                            }
                            for m in p.melds
                        ],
                        "riichi": p.riichi,
                    }
                    for p in observation.players
                ],
                "actor_turn_index": actor_turn_index,
                "seat_wind": seat_wind,
                "legal_discard_mask": list(mask),
                "label_index": None,
            }

            response = self.http_client.post("/recommend", json=request_data)
            response.raise_for_status()
            result = response.json()

            # Extract top predictions
            top_3 = result.get("top_3_actions", [])
            predicted_tiles = [a["tile"] for a in top_3]
            predicted_probs = [a["probability"] for a in top_3]

            # Cache prediction
            prediction = PredictionResult(
                game_id=game_id,
                event_index=event_index,
                actor=actor,
                predicted_tiles=predicted_tiles,
                predicted_probabilities=predicted_probs,
            )
            self.pending_predictions[game_id] = prediction

            logger.debug(
                f"Game {game_id} event {event_index}: predicted {predicted_tiles} "
                f"for player {actor}"
            )

        except (httpx.HTTPError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to get prediction for {game_id} event {event_index}: {e}")

    def _handle_dahai(
        self,
        game_id: str,
        event_index: int,
        event: DahaiEvent,
    ) -> None:
        """Handle a discard event: log prediction vs actual outcome."""
        # Lookup the cached prediction from the previous TsumoEvent
        if game_id not in self.pending_predictions:
            logger.debug(f"No pending prediction for {game_id}")
            return

        prediction = self.pending_predictions[game_id]

        # The actual discard
        actual_tile = event.pai

        # Did the model predict correctly?
        correct = actual_tile in prediction.predicted_tiles
        rank = prediction.predicted_tiles.index(actual_tile) + 1 if correct else None

        logger.info(
            f"Game {game_id} event {event_index}: player {event.actor} discarded {actual_tile} "
            f"(predicted: {prediction.predicted_tiles}, rank: {rank})"
        )

        # Clean up
        del self.pending_predictions[game_id]

    @staticmethod
    def _seat_wind(player_id: int, dealer: int) -> str:
        """Compute seat wind for a player given the dealer."""
        winds = ("E", "S", "W", "N")
        offset = (player_id - dealer) % 4
        return winds[offset]
