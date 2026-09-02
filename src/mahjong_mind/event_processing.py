"""Turn a stream of MJAI events into enriched, prediction-carrying records.

This is the inference stage's logic, with no transport attached. It takes one
event at a time, rebuilds the game state, asks for a recommendation at each
draw, and resolves that recommendation once the discard arrives.

How events reach it and where the enriched records go is the caller's business:
the Kafka consumer reads a topic and publishes to another, while the deployed
API service feeds it from a replay thread and pushes results straight to its
WebSocket clients. Both share this file so neither reimplements the logic.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mahjong_mind.game_state.legal_actions import legal_discard_mask
from mahjong_mind.game_state.player_observation import (
    ObservationError,
    PlayerObservation,
    observation_for_player,
)
from mahjong_mind.game_state.reconstructor import StateReconstructor
from mahjong_mind.mjai.events import DahaiEvent, TsumoEvent
from mahjong_mind.mjai.parser import ParsedEvent

logger = logging.getLogger(__name__)

# Delivers one /recommend request and returns its response, or None if the
# recommendation could not be obtained. Implemented over HTTP by the Kafka
# consumer and in-process by the API service.
Recommender = Callable[[dict[str, Any]], dict[str, Any] | None]

_EVENT_CLASS_NAMES = {
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


@dataclass(frozen=True)
class PredictionResult:
    """A recommendation held from a draw until the matching discard arrives."""

    game_id: str
    event_index: int
    actor: int
    predicted_tiles: list[str]
    predicted_probabilities: list[float]


def seat_wind(player_id: int, dealer: int) -> str:
    """Compute seat wind for a player given the dealer."""
    winds = ("E", "S", "W", "N")
    return winds[(player_id - dealer) % 4]


def observation_payload(observation: PlayerObservation) -> dict[str, Any]:
    """Serialise an observation into the shape the viewer renders."""
    return {
        "match_id": observation.match_id,
        "observer": observation.observer,
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
    }


def recommendation_request(observation: PlayerObservation) -> dict[str, Any]:
    """Build the /recommend request body for one player's observation.

    The viewer payload already carries every observable field the model reads,
    so this extends it rather than restating it; the four extra fields are the
    ones /recommend needs and the viewer does not.
    """
    actor = observation.observer
    return {
        **observation_payload(observation),
        "actor_turn_index": len(observation.players[actor].discards),
        "seat_wind": seat_wind(actor, observation.dealer),
        "legal_discard_mask": list(legal_discard_mask(observation)),
        "label_index": None,
    }


def parse_event(game_id: str, event_index: int, event_dict: dict[str, Any]) -> ParsedEvent:
    """Rebuild a ParsedEvent from a serialised event dict."""
    event_type = event_dict.get("type")
    if event_type not in _EVENT_CLASS_NAMES:
        raise ValueError(f"Unknown event type: {event_type}")

    from mahjong_mind.mjai import events as events_module

    event_class = getattr(events_module, _EVENT_CLASS_NAMES[event_type])
    return ParsedEvent(
        match_id=game_id,
        event_index=event_index,
        event=event_class(**event_dict),
    )


class GameEventProcessor:
    """Rebuilds game state per game and enriches each event with predictions."""

    def __init__(self, recommender: Recommender):
        """Initialise with the callable that fulfils recommendation requests."""
        self.recommender = recommender

        # Per-game-id reconstructors to maintain state.
        self.reconstructors: dict[str, StateReconstructor] = {}

        # A recommendation is cached from its draw until the discard resolves it.
        self.pending_predictions: dict[str, PredictionResult] = {}

        # Last acting player per game, for events that carry no actor.
        self.last_actor: dict[str, int] = {}

    def process(
        self,
        game_id: str,
        event_index: int,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply one event and return the enriched record for it.

        Raises if the event cannot be applied; the caller decides whether that
        means a dead-letter queue, a log line, or something else.
        """
        # A start_game means this game is beginning again, so any state from an
        # earlier replay of the same id must be discarded. Without this,
        # replaying a game twice in one session fails on every event because the
        # reconstructor is still part-way through the first run.
        if event_dict.get("type") == "start_game":
            self.reset_game(game_id)

        if game_id not in self.reconstructors:
            self.reconstructors[game_id] = StateReconstructor()

        parsed_event = parse_event(game_id, event_index, event_dict)
        state = self.reconstructors[game_id].apply(parsed_event)

        # Whose perspective this event is seen from. Events like dora or
        # end_kyoku carry no actor, so fall back to the last one seen.
        actor = getattr(parsed_event.event, "actor", None)
        if actor is None:
            actor = self.last_actor.get(game_id, 0)
        else:
            self.last_actor[game_id] = actor

        observation = self._observation_or_none(state, actor)

        recommendation: dict[str, Any] | None = None
        outcome: dict[str, Any] | None = None
        if isinstance(parsed_event.event, TsumoEvent) and observation is not None:
            recommendation = self._handle_tsumo(game_id, event_index, observation)
        elif isinstance(parsed_event.event, DahaiEvent):
            outcome = self._handle_dahai(game_id, event_index, parsed_event.event)

        return {
            "game_id": game_id,
            "event_index": event_index,
            "event": event_dict,
            "observation": observation_payload(observation) if observation else None,
            "recommendations": recommendation,
            "outcome": outcome,
        }

    def reset_game(self, game_id: str) -> None:
        """Forget everything held for one game, so it can be replayed again."""
        self.reconstructors.pop(game_id, None)
        self.pending_predictions.pop(game_id, None)
        self.last_actor.pop(game_id, None)

    def _handle_tsumo(
        self,
        game_id: str,
        event_index: int,
        observation: PlayerObservation,
    ) -> dict[str, Any] | None:
        """Ask for a recommendation at a draw and hold it for the discard."""
        result = self.recommender(recommendation_request(observation))
        if result is None:
            return None

        top_3 = result.get("top_3_actions", [])
        predicted_tiles = [a["tile"] for a in top_3]

        self.pending_predictions[game_id] = PredictionResult(
            game_id=game_id,
            event_index=event_index,
            actor=observation.observer,
            predicted_tiles=predicted_tiles,
            predicted_probabilities=[a["probability"] for a in top_3],
        )

        logger.debug(
            f"Game {game_id} event {event_index}: predicted {predicted_tiles} "
            f"for player {observation.observer}"
        )
        return dict(result)

    def _handle_dahai(
        self,
        game_id: str,
        event_index: int,
        event: DahaiEvent,
    ) -> dict[str, Any] | None:
        """Resolve the held recommendation against the discard that happened."""
        prediction = self.pending_predictions.pop(game_id, None)
        if prediction is None:
            logger.debug(f"No pending prediction for {game_id}")
            return None

        actual_tile = event.pai
        correct = actual_tile in prediction.predicted_tiles
        rank = prediction.predicted_tiles.index(actual_tile) + 1 if correct else None

        logger.info(
            f"Game {game_id} event {event_index}: player {event.actor} discarded "
            f"{actual_tile} (predicted: {prediction.predicted_tiles}, rank: {rank})"
        )

        return {
            "actor": event.actor,
            "actual_tile": actual_tile,
            "predicted_tiles": prediction.predicted_tiles,
            "rank": rank,
            "top_1": rank == 1,
            "top_3": correct,
        }

    @staticmethod
    def _observation_or_none(state: Any, actor: int) -> PlayerObservation | None:
        """Build an observation, or None before the first hand has started."""
        if state.current_hand is None:
            return None
        try:
            return observation_for_player(state, actor)
        except ObservationError:
            return None
