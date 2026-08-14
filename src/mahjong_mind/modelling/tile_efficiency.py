from collections.abc import Iterable

from mahjong.shanten import Shanten

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.mjai.events import Tile

BASE_TILE_TYPES = DISCARD_TILE_TYPES[:34]
_BASE_TILE_INDEX: dict[str, int] = {
    tile: index for index, tile in enumerate(BASE_TILE_TYPES)
}


def tiles_to_34_counts(tiles: Iterable[Tile]) -> tuple[int, ...]:
    """Convert MahjongMind tiles to the shanten library's 34-count format."""
    counts = [0] * len(BASE_TILE_TYPES)
    for tile in tiles:
        counts[_BASE_TILE_INDEX[tile.removesuffix("r")]] += 1
    return tuple(counts)


def calculate_shanten(tiles: Iterable[Tile]) -> int:
    """Return shanten for a concealed MahjongMind hand, including red fives."""
    return Shanten.calculate_shanten(tiles_to_34_counts(tiles))
