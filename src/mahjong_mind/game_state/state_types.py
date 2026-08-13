from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from mahjong_mind.mjai.events import PlayerId, Tile, Wind

MeldType: TypeAlias = Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
RiichiState: TypeAlias = Literal["none", "pending", "accepted"]


@dataclass(frozen=True, slots=True)
class Discard:
    tile: Tile
    tsumogiri: bool
    riichi: bool = False
    called: bool = False


@dataclass(frozen=True, slots=True)
class Meld:
    type: MeldType
    tiles: tuple[Tile, ...]
    called_tile: Tile | None
    target: PlayerId | None


@dataclass(slots=True)
class PlayerState:
    concealed_tiles: list[Tile] = field(default_factory=list)
    last_draw: Tile | None = None
    discards: list[Discard] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    riichi: RiichiState = "none"


@dataclass(slots=True)
class HandState:
    hand_index: int
    bakaze: Wind
    kyoku: int
    honba: int
    kyotaku: int
    dealer: PlayerId
    scores: list[int]
    dora_markers: list[Tile]
    players: tuple[PlayerState, PlayerState, PlayerState, PlayerState]
    draws_remaining: int = 70
    last_discard_actor: PlayerId | None = None
    last_discard_index: int | None = None
    ended: bool = False

    def player(self, player_id: PlayerId) -> PlayerState:
        return self.players[player_id]

    @property
    def last_discard(self) -> Discard | None:
        if self.last_discard_actor is None or self.last_discard_index is None:
            return None
        return self.player(self.last_discard_actor).discards[self.last_discard_index]


@dataclass(slots=True)
class MatchState:
    match_id: str
    names: tuple[str, str, str, str]
    kyoku_first: int
    aka_flag: bool
    current_hand: HandState | None = None
    hand_index: int = -1
    event_index: int = -1
    ended: bool = False
