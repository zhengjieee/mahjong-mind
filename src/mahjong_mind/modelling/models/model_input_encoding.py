from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES

MODEL_INPUT_VERSION = "dense-observation-v1"

_TILE_TO_INDEX = {tile: index for index, tile in enumerate(DISCARD_TILE_TYPES)}
_WINDS = ("E", "S", "W", "N")
_RIICHI_STATES = ("none", "pending", "accepted")
_MELD_TYPES = ("chi", "pon", "daiminkan", "ankan", "kakan")


class ModelInputEncodingError(ValueError):
    """Raised when a decision row cannot be encoded safely."""


@dataclass(frozen=True, slots=True)
class EncodedModelInput:
    """Numeric observation features and the separate legal-action constraint."""

    features: tuple[float, ...]
    legal_discard_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class EncodedDiscardExample:
    """One model input paired with its historical discard target."""

    model_input: EncodedModelInput
    label_index: int


def _feature_names() -> tuple[str, ...]:
    names = ["aka_flag", "honba", "kyotaku", "draws_remaining", "actor_turn_index"]
    names.extend(f"bakaze.{wind}" for wind in _WINDS)
    names.extend(f"seat_wind.{wind}" for wind in _WINDS)
    names.extend(f"kyoku.{number}" for number in range(1, 5))
    names.extend(f"dealer_relative.{seat}" for seat in range(4))
    names.extend(f"score_relative.{seat}" for seat in range(4))
    names.extend(f"own_hand_count.{tile}" for tile in DISCARD_TILE_TYPES)
    names.extend(f"own_last_draw.{tile}" for tile in DISCARD_TILE_TYPES)
    names.extend(f"dora_marker_count.{tile}" for tile in DISCARD_TILE_TYPES)

    for seat in range(4):
        prefix = f"player_relative.{seat}"
        names.append(f"{prefix}.concealed_tile_count")
        names.extend(f"{prefix}.riichi.{state}" for state in _RIICHI_STATES)
        for field in (
            "discard_count",
            "tsumogiri_discard_count",
            "riichi_discard_count",
            "called_discard_count",
            "meld_tile_count",
        ):
            names.extend(f"{prefix}.{field}.{tile}" for tile in DISCARD_TILE_TYPES)
        names.extend(f"{prefix}.meld_type_count.{kind}" for kind in _MELD_TYPES)
        names.extend(f"{prefix}.meld_target_relative.{target}" for target in range(4))
    return tuple(names)


FEATURE_NAMES = _feature_names()


def encode_parquet_row(row: Mapping[str, Any]) -> EncodedDiscardExample:
    """Convert one structured decision row into the shared numeric encoding."""
    actor = _integer(row, "actor")
    dealer = _integer(row, "dealer")
    if actor not in range(4) or dealer not in range(4):
        raise ModelInputEncodingError("actor and dealer must be between 0 and 3")

    players = _mappings(row, "players", expected_length=4)
    scores = _integers(row, "scores", expected_length=4)
    legal_mask = _legal_mask(row)
    label_index = _integer(row, "label_index")
    if label_index not in range(len(DISCARD_TILE_TYPES)) or not legal_mask[label_index]:
        raise ModelInputEncodingError(f"label_index {label_index} is not legal")

    values: dict[str, float] = {
        "aka_flag": float(_boolean(row, "aka_flag")),
        "honba": float(_integer(row, "honba")),
        "kyotaku": float(_integer(row, "kyotaku")),
        "draws_remaining": float(_integer(row, "draws_remaining")),
        "actor_turn_index": float(_integer(row, "actor_turn_index")),
    }
    _add_one_hot(values, "bakaze", _WINDS, _string(row, "bakaze"))
    _add_one_hot(values, "seat_wind", _WINDS, _string(row, "seat_wind"))
    _add_one_hot(values, "kyoku", range(1, 5), _integer(row, "kyoku"))
    _add_one_hot(values, "dealer_relative", range(4), (dealer - actor) % 4)

    for relative_seat in range(4):
        absolute_seat = (actor + relative_seat) % 4
        values[f"score_relative.{relative_seat}"] = float(scores[absolute_seat])

    _add_tile_counts(values, "own_hand_count", _sequence(row, "own_hand"))
    _add_tile_one_hot(values, "own_last_draw", row.get("own_last_draw"))
    _add_tile_counts(values, "dora_marker_count", _sequence(row, "dora_markers"))

    for relative_seat in range(4):
        absolute_seat = (actor + relative_seat) % 4
        _add_player_features(
            values,
            players[absolute_seat],
            relative_seat=relative_seat,
            observer=actor,
        )

    if set(values) != set(FEATURE_NAMES):
        raise ModelInputEncodingError("Internal feature definition is inconsistent")

    return EncodedDiscardExample(
        model_input=EncodedModelInput(
            features=tuple(values[name] for name in FEATURE_NAMES),
            legal_discard_mask=legal_mask,
        ),
        label_index=label_index,
    )


def _add_player_features(
    values: dict[str, float],
    player: Mapping[str, Any],
    *,
    relative_seat: int,
    observer: int,
) -> None:
    prefix = f"player_relative.{relative_seat}"
    values[f"{prefix}.concealed_tile_count"] = float(
        _integer(player, "concealed_tile_count")
    )
    _add_one_hot(
        values,
        f"{prefix}.riichi",
        _RIICHI_STATES,
        _string(player, "riichi"),
    )

    discards = _mappings(player, "discards")
    _add_tile_counts(
        values,
        f"{prefix}.discard_count",
        [discard.get("tile") for discard in discards],
    )
    for flag, field in (
        ("tsumogiri", "tsumogiri_discard_count"),
        ("riichi", "riichi_discard_count"),
        ("called", "called_discard_count"),
    ):
        _add_tile_counts(
            values,
            f"{prefix}.{field}",
            [discard.get("tile") for discard in discards if _boolean(discard, flag)],
        )

    melds = _mappings(player, "melds")
    meld_tiles: list[Any] = []
    meld_type_counts = {kind: 0 for kind in _MELD_TYPES}
    meld_target_counts = [0, 0, 0, 0]
    for meld in melds:
        meld_tiles.extend(_sequence(meld, "tiles"))
        meld_type = _string(meld, "type")
        if meld_type not in meld_type_counts:
            raise ModelInputEncodingError(f"Unknown meld type: {meld_type}")
        meld_type_counts[meld_type] += 1
        target = meld.get("target")
        if target is not None:
            if type(target) is not int or target not in range(4):
                raise ModelInputEncodingError(f"Invalid meld target: {target}")
            meld_target_counts[(target - observer) % 4] += 1

    _add_tile_counts(values, f"{prefix}.meld_tile_count", meld_tiles)
    for kind in _MELD_TYPES:
        values[f"{prefix}.meld_type_count.{kind}"] = float(meld_type_counts[kind])
    for target, count in enumerate(meld_target_counts):
        values[f"{prefix}.meld_target_relative.{target}"] = float(count)


def _add_tile_counts(
    values: dict[str, float],
    prefix: str,
    tiles: Sequence[Any],
) -> None:
    counts = [0] * len(DISCARD_TILE_TYPES)
    for tile in tiles:
        if not isinstance(tile, str) or tile not in _TILE_TO_INDEX:
            raise ModelInputEncodingError(f"Unknown tile: {tile}")
        counts[_TILE_TO_INDEX[tile]] += 1
    for tile, count in zip(DISCARD_TILE_TYPES, counts, strict=True):
        values[f"{prefix}.{tile}"] = float(count)


def _add_tile_one_hot(
    values: dict[str, float],
    prefix: str,
    tile: Any,
) -> None:
    if tile is not None and (not isinstance(tile, str) or tile not in _TILE_TO_INDEX):
        raise ModelInputEncodingError(f"Unknown tile: {tile}")
    for candidate in DISCARD_TILE_TYPES:
        values[f"{prefix}.{candidate}"] = float(candidate == tile)


def _add_one_hot(
    values: dict[str, float],
    prefix: str,
    choices: Sequence[Any] | range,
    selected: Any,
) -> None:
    if selected not in choices:
        raise ModelInputEncodingError(f"Invalid {prefix}: {selected}")
    for choice in choices:
        values[f"{prefix}.{choice}"] = float(choice == selected)


def _sequence(row: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = row.get(key)
    if not isinstance(value, (list, tuple)):
        raise ModelInputEncodingError(f"{key} must be a sequence")
    return value


def _mappings(
    row: Mapping[str, Any],
    key: str,
    *,
    expected_length: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    value = _sequence(row, key)
    if expected_length is not None and len(value) != expected_length:
        raise ModelInputEncodingError(
            f"{key} has length {len(value)}, expected {expected_length}"
        )
    if not all(isinstance(item, Mapping) for item in value):
        raise ModelInputEncodingError(f"{key} must contain mappings")
    return tuple(value)


def _integers(
    row: Mapping[str, Any],
    key: str,
    *,
    expected_length: int,
) -> tuple[int, ...]:
    value = _sequence(row, key)
    if len(value) != expected_length or not all(type(item) is int for item in value):
        raise ModelInputEncodingError(f"{key} must contain {expected_length} integers")
    return tuple(value)


def _legal_mask(row: Mapping[str, Any]) -> tuple[bool, ...]:
    value = _sequence(row, "legal_discard_mask")
    if len(value) != len(DISCARD_TILE_TYPES) or not all(
        type(item) is bool for item in value
    ):
        raise ModelInputEncodingError(
            f"legal_discard_mask must contain {len(DISCARD_TILE_TYPES)} booleans"
        )
    if not any(value):
        raise ModelInputEncodingError("legal_discard_mask has no legal actions")
    return tuple(value)


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise ModelInputEncodingError(f"{key} must be an integer")
    return value


def _boolean(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ModelInputEncodingError(f"{key} must be a boolean")
    return value


def _string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ModelInputEncodingError(f"{key} must be a string")
    return value
