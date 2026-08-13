from collections import Counter

from mahjong_mind.game_state.state_types import HandState, Meld
from mahjong_mind.mjai.events import Tile


class StateInvariantError(ValueError):
    """Raised when a reconstructed hand violates a Mahjong invariant."""


def base_tile(tile: Tile) -> str:
    return tile.removesuffix("r")


def validate_hand_state(hand: HandState) -> None:
    """Validate physical tiles, hand sizes, melds, and state references."""
    if not 0 <= hand.draws_remaining <= 70:
        raise StateInvariantError(
            f"draws_remaining must be between 0 and 70, got {hand.draws_remaining}"
        )
    if not 1 <= len(hand.dora_markers) <= 5:
        raise StateInvariantError(
            f"a hand must have 1 to 5 visible dora markers, got {len(hand.dora_markers)}"
        )

    physical_tiles: list[Tile] = list(hand.dora_markers)

    for player_id, player in enumerate(hand.players):
        meld_count = len(player.melds)
        expected_after_discard = 13 - 3 * meld_count
        expected_sizes = {expected_after_discard, expected_after_discard + 1}
        concealed_count = len(player.concealed_tiles)

        if expected_after_discard < 1 or concealed_count not in expected_sizes:
            raise StateInvariantError(
                f"player {player_id} has {concealed_count} concealed tiles with "
                f"{meld_count} melds; expected {sorted(expected_sizes)}"
            )
        if player.last_draw is not None:
            if player.last_draw not in player.concealed_tiles:
                raise StateInvariantError(
                    f"player {player_id}'s last draw is missing from their hand"
                )
            if concealed_count != expected_after_discard + 1:
                raise StateInvariantError(
                    f"player {player_id} has a pending draw but the wrong hand size"
                )

        physical_tiles.extend(player.concealed_tiles)
        physical_tiles.extend(
            discard.tile for discard in player.discards if not discard.called
        )
        for meld in player.melds:
            _validate_meld(meld, player_id)
            physical_tiles.extend(meld.tiles)

    base_counts = Counter(base_tile(tile) for tile in physical_tiles)
    for tile, count in base_counts.items():
        if count > 4:
            raise StateInvariantError(
                f"physical tile count for {tile} is {count}, maximum 4"
            )

    exact_counts = Counter(physical_tiles)
    for red_five in ("5mr", "5pr", "5sr"):
        if exact_counts[red_five] > 1:
            raise StateInvariantError(
                f"physical tile count for {red_five} is {exact_counts[red_five]}, maximum 1"
            )

    _validate_last_discard_reference(hand)


def _validate_meld(meld: Meld, actor: int) -> None:
    expected_length = 3 if meld.type in {"chi", "pon"} else 4
    if len(meld.tiles) != expected_length:
        raise StateInvariantError(
            f"{meld.type} meld has {len(meld.tiles)} tiles, expected {expected_length}"
        )

    if meld.type == "ankan":
        if meld.target is not None or meld.called_tile is not None:
            raise StateInvariantError("ankan cannot have a target or called tile")
    else:
        if meld.target is None or meld.called_tile is None:
            raise StateInvariantError(
                f"{meld.type} must retain its target and called tile"
            )
        if meld.target == actor:
            raise StateInvariantError(f"player {actor} cannot call their own tile")

    normalized = [base_tile(tile) for tile in meld.tiles]
    if meld.type == "chi":
        if meld.target != (actor - 1) % 4:
            raise StateInvariantError(
                f"player {actor} cannot chi a discard from player {meld.target}"
            )
        suits = {tile[-1] for tile in normalized}
        if suits not in ({"m"}, {"p"}, {"s"}):
            raise StateInvariantError("chi tiles must belong to one numbered suit")
        ranks = sorted(int(tile[0]) for tile in normalized)
        if ranks[1] != ranks[0] + 1 or ranks[2] != ranks[1] + 1:
            raise StateInvariantError("chi tiles must form a consecutive sequence")
    elif len(set(normalized)) != 1:
        raise StateInvariantError(f"{meld.type} tiles must have one base tile type")


def _validate_last_discard_reference(hand: HandState) -> None:
    actor = hand.last_discard_actor
    index = hand.last_discard_index
    if actor is None and index is None:
        return
    if actor is None or index is None:
        raise StateInvariantError("last discard actor and index must be set together")

    discards = hand.player(actor).discards
    if not 0 <= index < len(discards):
        raise StateInvariantError(
            "last discard index does not exist in the actor's river"
        )
