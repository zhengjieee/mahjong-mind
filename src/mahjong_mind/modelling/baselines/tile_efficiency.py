from collections.abc import Iterable

from mahjong.shanten import Shanten

from mahjong_mind.game_state.legal_actions import (
    DISCARD_TILE_TYPES,
    legal_discard_mask,
)
from mahjong_mind.game_state.player_observation import PlayerObservation
from mahjong_mind.mjai.events import Tile
from mahjong_mind.modelling.shared.metrics_evaluation import PolicyPrediction

BASE_TILE_TYPES = DISCARD_TILE_TYPES[:34]
_BASE_TILE_INDEX: dict[str, int] = {
    tile: index for index, tile in enumerate(BASE_TILE_TYPES)
}


class TileEfficiencyError(ValueError):
    """Raised when tile-efficiency inputs are inconsistent."""


class TileEfficiencyBaseline:
    """Rank legal discards by shanten and observable ukeire."""

    def predict(self, observation: PlayerObservation) -> PolicyPrediction:
        return self.predict_tiles(
            observation.own_hand,
            legal_discard_mask(observation),
            known_tiles_from_observation(observation),
        )

    def predict_tiles(
        self,
        hand: tuple[Tile, ...],
        legal_mask: tuple[bool, ...],
        known_tiles: Iterable[Tile],
    ) -> PolicyPrediction:
        """Predict from the tile fields stored in the Parquet dataset."""
        ranked_actions = rank_discards_by_tile_efficiency(
            hand,
            legal_mask,
            known_tiles,
        )
        rank_weights = {
            action: 1.0 / rank
            for rank, action in enumerate(ranked_actions, start=1)
        }
        total_weight = sum(rank_weights.values())
        probabilities = tuple(
            rank_weights.get(action, 0.0) / total_weight
            for action in range(len(DISCARD_TILE_TYPES))
        )
        return PolicyPrediction(ranked_actions, probabilities)


def tiles_to_34_counts(tiles: Iterable[Tile]) -> tuple[int, ...]:
    """Convert MahjongMind tiles to the shanten library's 34-count format."""
    counts = [0] * len(BASE_TILE_TYPES)
    for tile in tiles:
        counts[_BASE_TILE_INDEX[tile.removesuffix("r")]] += 1
    return tuple(counts)


def calculate_shanten(tiles: Iterable[Tile]) -> int:
    """Return shanten for a concealed MahjongMind hand, including red fives."""
    return Shanten.calculate_shanten(tiles_to_34_counts(tiles))


def known_tiles_from_observation(observation: PlayerObservation) -> tuple[Tile, ...]:
    """Collect physical tiles visible to the observing player."""
    known_tiles = [*observation.own_hand, *observation.dora_markers]
    for player in observation.players:
        known_tiles.extend(
            discard.tile for discard in player.discards if not discard.called
        )
        for meld in player.melds:
            known_tiles.extend(meld.tiles)
    return tuple(known_tiles)


def shanten_after_legal_discards(
    hand: tuple[Tile, ...],
    legal_mask: tuple[bool, ...],
) -> tuple[int | None, ...]:
    """Calculate resulting shanten for each legal discard action."""
    if len(legal_mask) != len(DISCARD_TILE_TYPES):
        raise TileEfficiencyError(
            f"Legal mask has {len(legal_mask)} actions, "
            f"expected {len(DISCARD_TILE_TYPES)}"
        )

    results: list[int | None] = []
    for tile, is_legal in zip(DISCARD_TILE_TYPES, legal_mask):
        if not is_legal:
            results.append(None)
            continue

        remaining_hand = _hand_after_discard(hand, tile)
        results.append(calculate_shanten(remaining_hand))

    return tuple(results)


def ukeire_after_legal_discards(
    hand: tuple[Tile, ...],
    legal_mask: tuple[bool, ...],
    known_tiles: Iterable[Tile],
) -> tuple[int | None, ...]:
    """Count unseen tiles that reduce shanten after each legal discard."""
    known_counts = tiles_to_34_counts(known_tiles)
    if any(count > 4 for count in known_counts):
        raise TileEfficiencyError("Known tiles contain more than four of one tile")
    remaining_counts = tuple(4 - count for count in known_counts)
    discard_shanten = shanten_after_legal_discards(hand, legal_mask)

    results: list[int | None] = []
    for action, shanten in enumerate(discard_shanten):
        if shanten is None:
            results.append(None)
            continue

        remaining_hand = _hand_after_discard(hand, DISCARD_TILE_TYPES[action])
        ukeire = sum(
            copies
            for tile, copies in zip(BASE_TILE_TYPES, remaining_counts)
            if copies > 0
            and calculate_shanten((*remaining_hand, tile)) < shanten
        )
        results.append(ukeire)

    return tuple(results)


def rank_discards_by_tile_efficiency(
    hand: tuple[Tile, ...],
    legal_mask: tuple[bool, ...],
    known_tiles: Iterable[Tile],
) -> tuple[int, ...]:
    """Rank legal actions by lower shanten, then higher ukeire."""
    discard_shanten = shanten_after_legal_discards(hand, legal_mask)
    discard_ukeire = ukeire_after_legal_discards(hand, legal_mask, known_tiles)
    scores: list[tuple[int, int, int]] = []
    for action, (shanten, ukeire) in enumerate(
        zip(discard_shanten, discard_ukeire)
    ):
        if shanten is None or ukeire is None:
            continue
        scores.append((action, shanten, ukeire))

    if not scores:
        raise TileEfficiencyError("Decision has no legal discard actions")

    scores.sort(key=lambda score: (score[1], -score[2], score[0]))
    return tuple(action for action, _, _ in scores)


def _hand_after_discard(hand: tuple[Tile, ...], tile: Tile) -> tuple[Tile, ...]:
    remaining_hand = list(hand)
    try:
        remaining_hand.remove(tile)
    except ValueError as error:
        raise TileEfficiencyError(
            f"Legal discard {tile} is not present in the hand"
        ) from error
    return tuple(remaining_hand)
