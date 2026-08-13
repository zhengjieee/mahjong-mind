from dataclasses import dataclass

from mahjong_mind.game_state.state_types import (
    Discard,
    MatchState,
    Meld,
    PlayerState,
    RiichiState,
)
from mahjong_mind.mjai.events import PlayerId, Tile, Wind


class ObservationError(ValueError):
    """Raised when an observation cannot be created from the current state."""


@dataclass(frozen=True, slots=True)
class PublicPlayerObservation:
    """Information about one player that is visible to everyone."""

    concealed_tile_count: int
    discards: tuple[Discard, ...]
    melds: tuple[Meld, ...]
    riichi: RiichiState


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """A read-only game-state snapshot from one player's perspective."""

    match_id: str
    observer: PlayerId
    names: tuple[str, str, str, str]
    hand_index: int
    bakaze: Wind
    kyoku: int
    honba: int
    kyotaku: int
    dealer: PlayerId
    scores: tuple[int, int, int, int]
    dora_markers: tuple[Tile, ...]
    draws_remaining: int
    hand_ended: bool
    own_hand: tuple[Tile, ...]
    own_last_draw: Tile | None
    players: tuple[
        PublicPlayerObservation,
        PublicPlayerObservation,
        PublicPlayerObservation,
        PublicPlayerObservation,
    ]


def _public_view(player: PlayerState) -> PublicPlayerObservation:
    return PublicPlayerObservation(
        concealed_tile_count=len(player.concealed_tiles),
        discards=tuple(player.discards),
        melds=tuple(player.melds),
        riichi=player.riichi,
    )


def observation_for_player(
    state: MatchState,
    observer: PlayerId,
) -> PlayerObservation:
    """Create a snapshot containing only information available to ``observer``."""
    if observer not in (0, 1, 2, 3):
        raise ObservationError(f"observer must be between 0 and 3, got {observer}")
    if state.current_hand is None:
        raise ObservationError("cannot create an observation before start_kyoku")

    hand = state.current_hand
    own_state = hand.player(observer)
    public_players = (
        _public_view(hand.players[0]),
        _public_view(hand.players[1]),
        _public_view(hand.players[2]),
        _public_view(hand.players[3]),
    )

    return PlayerObservation(
        match_id=state.match_id,
        observer=observer,
        names=state.names,
        hand_index=hand.hand_index,
        bakaze=hand.bakaze,
        kyoku=hand.kyoku,
        honba=hand.honba,
        kyotaku=hand.kyotaku,
        dealer=hand.dealer,
        scores=(hand.scores[0], hand.scores[1], hand.scores[2], hand.scores[3]),
        dora_markers=tuple(hand.dora_markers),
        draws_remaining=hand.draws_remaining,
        hand_ended=hand.ended,
        own_hand=tuple(own_state.concealed_tiles),
        own_last_draw=own_state.last_draw,
        players=public_players,
    )
