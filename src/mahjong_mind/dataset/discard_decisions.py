import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from mahjong_mind.game_state.legal_actions import (
    DISCARD_TILE_TYPES,
    legal_discard_mask,
)
from mahjong_mind.game_state.player_observation import (
    PlayerObservation,
    observation_for_player,
)
from mahjong_mind.game_state.reconstructor import StateReconstructor
from mahjong_mind.mjai.events import DahaiEvent, PlayerId, StartGameEvent, Tile
from mahjong_mind.mjai.parser import ParsedEvent, iter_mjai_events

_SOURCE_YEAR_PATTERN = re.compile(r"^(\d{4})")
_SEAT_WINDS = ("E", "S", "W", "N")


class DecisionDatasetError(ValueError):
    """Raised when a discard decision cannot be extracted safely."""


@dataclass(frozen=True, slots=True)
class DiscardDecision:
    """The observable state and historical label at one discard decision."""

    decision_id: str
    match_id: str
    hand_id: str
    hand_index: int
    event_index: int
    actor: PlayerId
    source_year: int
    seat_wind: str
    actor_turn_index: int
    observation: PlayerObservation
    legal_discard_mask: tuple[bool, ...]
    actual_discard: Tile
    actual_tsumogiri: bool
    label_index: int


def source_year_from_match_id(match_id: str) -> int:
    """Read the four-digit source year at the start of a match ID."""
    match = _SOURCE_YEAR_PATTERN.match(match_id)
    if match is None:
        raise DecisionDatasetError(
            f"Match ID must begin with a four-digit source year: {match_id}"
        )
    return int(match.group(1))


def is_example_match(path: Path) -> bool:
    """Return whether a match is an artificial EXAMPLE fixture."""
    first = next(iter_mjai_events(path), None)
    if first is None or not isinstance(first.event, StartGameEvent):
        raise DecisionDatasetError(f"Match does not begin with start_game: {path}")
    return any(name.startswith("EXAMPLE") for name in first.event.names)


def iter_discard_decisions(
    events: Iterable[ParsedEvent],
) -> Iterator[DiscardDecision]:
    """Yield an observable record immediately before each discard is applied."""
    reconstructor = StateReconstructor(validate_after_event=True)

    for parsed in events:
        event = parsed.event
        if isinstance(event, DahaiEvent):
            state = reconstructor.state
            if state is None or state.current_hand is None:
                raise DecisionDatasetError(
                    f"Missing hand state in {parsed.match_id} "
                    f"at event {parsed.event_index}"
                )

            observation = observation_for_player(state, event.actor)
            legal_mask = legal_discard_mask(observation)
            try:
                label_index = DISCARD_TILE_TYPES.index(event.pai)
            except ValueError as exc:
                raise DecisionDatasetError(
                    f"Unknown discard label {event.pai} in {parsed.match_id} "
                    f"at event {parsed.event_index}"
                ) from exc
            if not legal_mask[label_index]:
                raise DecisionDatasetError(
                    f"Illegal discard label {event.pai} in {parsed.match_id} "
                    f"at event {parsed.event_index}"
                )

            hand_index = state.current_hand.hand_index
            hand_id = f"{parsed.match_id}:{hand_index}"
            yield DiscardDecision(
                decision_id=f"{parsed.match_id}:{parsed.event_index}",
                match_id=parsed.match_id,
                hand_id=hand_id,
                hand_index=hand_index,
                event_index=parsed.event_index,
                actor=event.actor,
                source_year=source_year_from_match_id(parsed.match_id),
                seat_wind=_SEAT_WINDS[(event.actor - state.current_hand.dealer) % 4],
                actor_turn_index=len(observation.players[event.actor].discards),
                observation=observation,
                legal_discard_mask=legal_mask,
                actual_discard=event.pai,
                actual_tsumogiri=event.tsumogiri,
                label_index=label_index,
            )

        reconstructor.apply(parsed)


def iter_match_discard_decisions(path: Path) -> Iterator[DiscardDecision]:
    """Stream discard decisions from one non-example MJAI match."""
    events = iter_mjai_events(path)
    first = next(events, None)
    if first is None or not isinstance(first.event, StartGameEvent):
        raise DecisionDatasetError(f"Match does not begin with start_game: {path}")
    if any(name.startswith("EXAMPLE") for name in first.event.names):
        raise DecisionDatasetError(f"Artificial EXAMPLE match is not modelling data: {path}")
    yield from iter_discard_decisions(chain((first,), events))
