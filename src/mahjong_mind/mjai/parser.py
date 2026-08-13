import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from mahjong_mind.mjai.events import MjaiEvent

_EVENT_ADAPTER: TypeAdapter[MjaiEvent] = TypeAdapter(MjaiEvent)


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    match_id: str
    event_index: int
    event: MjaiEvent


@dataclass(frozen=True, slots=True)
class MatchSummary:
    match_id: str
    event_count: int
    hand_count: int


class MjaiParseError(ValueError):
    """Raised when an MJAI event cannot be read or validated."""


def iter_mjai_events(path: Path) -> Iterator[ParsedEvent]:
    """Yield validated events from one gzip-compressed MJAI match."""
    match_id = path.stem
    event_index = 0

    try:
        with gzip.open(path, mode="rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue

                try:
                    event = _EVENT_ADAPTER.validate_json(line)
                except ValidationError as exc:
                    message = (
                        f"Failed to parse {match_id} at event {event_index} "
                        f"(line {line_number})"
                    )
                    raise MjaiParseError(message) from exc

                yield ParsedEvent(
                    match_id=match_id,
                    event_index=event_index,
                    event=event,
                )
                event_index += 1
    except OSError as exc:
        raise MjaiParseError(f"Failed to read gzip file: {path}") from exc


def validate_mjai_match(path: Path) -> MatchSummary:
    """Validate match and hand boundaries without retaining parsed events."""
    match_id = path.stem
    event_count = 0
    hand_count = 0
    hand_open = False
    game_started = False
    game_ended = False

    for parsed in iter_mjai_events(path):
        event_type = parsed.event.type

        if game_ended:
            raise MjaiParseError(
                f"Invalid structure in {match_id} at event {parsed.event_index}: "
                "event appears after end_game"
            )

        if event_type == "start_game":
            if game_started or parsed.event_index != 0:
                raise MjaiParseError(
                    f"Invalid structure in {match_id} at event {parsed.event_index}: "
                    "misplaced start_game"
                )
            game_started = True
        elif not game_started:
            raise MjaiParseError(
                f"Invalid structure in {match_id} at event {parsed.event_index}: "
                "event appears before start_game"
            )
        elif event_type == "start_kyoku":
            if hand_open:
                raise MjaiParseError(
                    f"Invalid structure in {match_id} at event {parsed.event_index}: "
                    "start_kyoku appears before the previous hand ended"
                )
            hand_open = True
            hand_count += 1
        elif event_type == "end_kyoku":
            if not hand_open:
                raise MjaiParseError(
                    f"Invalid structure in {match_id} at event {parsed.event_index}: "
                    "end_kyoku appears without an open hand"
                )
            hand_open = False
        elif event_type == "end_game":
            if hand_open:
                raise MjaiParseError(
                    f"Invalid structure in {match_id} at event {parsed.event_index}: "
                    "end_game appears before end_kyoku"
                )
            game_ended = True
        elif not hand_open:
            raise MjaiParseError(
                f"Invalid structure in {match_id} at event {parsed.event_index}: "
                f"{event_type} appears outside a hand"
            )

        event_count += 1

    if not game_started:
        raise MjaiParseError(f"Invalid structure in {match_id}: match is empty")
    if hand_open:
        raise MjaiParseError(f"Invalid structure in {match_id}: hand is not closed")
    if not game_ended:
        raise MjaiParseError(f"Invalid structure in {match_id}: missing end_game")

    return MatchSummary(
        match_id=match_id,
        event_count=event_count,
        hand_count=hand_count,
    )
