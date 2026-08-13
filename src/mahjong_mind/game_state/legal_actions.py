from mahjong_mind.game_state.player_observation import PlayerObservation
from mahjong_mind.mjai.events import Tile

DISCARD_TILE_TYPES: tuple[Tile, ...] = (
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
)


def legal_discard_tiles(observation: PlayerObservation) -> tuple[Tile, ...]:
    """Return the distinct tiles the observing player may discard now."""
    if observation.hand_ended:
        return ()

    public_player = observation.players[observation.observer]
    discard_ready_size = 14 - 3 * len(public_player.melds)
    if len(observation.own_hand) != discard_ready_size:
        return ()

    if public_player.riichi == "accepted":
        if observation.own_last_draw is None:
            return ()
        return (observation.own_last_draw,)

    held_tiles = set(observation.own_hand)
    return tuple(tile for tile in DISCARD_TILE_TYPES if tile in held_tiles)


def legal_discard_mask(observation: PlayerObservation) -> tuple[bool, ...]:
    """Return a fixed 37-position mask in ``DISCARD_TILE_TYPES`` order."""
    legal_tiles = set(legal_discard_tiles(observation))
    return tuple(tile in legal_tiles for tile in DISCARD_TILE_TYPES)
