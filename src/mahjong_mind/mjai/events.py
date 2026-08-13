from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

PlayerId: TypeAlias = Annotated[int, Field(ge=0, le=3)]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
Wind: TypeAlias = Literal["E", "S", "W", "N"]

Tile: TypeAlias = Literal[
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
]

InitialHand: TypeAlias = Annotated[
    tuple[Tile, ...],
    Field(min_length=13, max_length=13),
]


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StartGameEvent(EventBase):
    type: Literal["start_game"]
    names: tuple[str, str, str, str]
    kyoku_first: NonNegativeInt
    aka_flag: bool


class StartKyokuEvent(EventBase):
    type: Literal["start_kyoku"]
    bakaze: Wind
    dora_marker: Tile
    kyoku: Annotated[int, Field(ge=1, le=4)]
    honba: NonNegativeInt
    kyotaku: NonNegativeInt
    oya: PlayerId
    scores: tuple[int, int, int, int]
    tehais: tuple[InitialHand, InitialHand, InitialHand, InitialHand]


class TsumoEvent(EventBase):
    type: Literal["tsumo"]
    actor: PlayerId
    pai: Tile


class DahaiEvent(EventBase):
    type: Literal["dahai"]
    actor: PlayerId
    pai: Tile
    tsumogiri: bool


class ChiEvent(EventBase):
    type: Literal["chi"]
    actor: PlayerId
    target: PlayerId
    pai: Tile
    consumed: tuple[Tile, Tile]


class PonEvent(EventBase):
    type: Literal["pon"]
    actor: PlayerId
    target: PlayerId
    pai: Tile
    consumed: tuple[Tile, Tile]


class DaiminkanEvent(EventBase):
    type: Literal["daiminkan"]
    actor: PlayerId
    target: PlayerId
    pai: Tile
    consumed: tuple[Tile, Tile, Tile]


class AnkanEvent(EventBase):
    type: Literal["ankan"]
    actor: PlayerId
    consumed: tuple[Tile, Tile, Tile, Tile]


class KakanEvent(EventBase):
    type: Literal["kakan"]
    actor: PlayerId
    pai: Tile
    consumed: tuple[Tile, Tile, Tile]


class DoraEvent(EventBase):
    type: Literal["dora"]
    dora_marker: Tile


class ReachEvent(EventBase):
    type: Literal["reach"]
    actor: PlayerId


class ReachAcceptedEvent(EventBase):
    type: Literal["reach_accepted"]
    actor: PlayerId


class HoraEvent(EventBase):
    type: Literal["hora"]
    actor: PlayerId
    target: PlayerId
    deltas: tuple[int, int, int, int]
    ura_markers: tuple[Tile, ...]


class RyukyokuEvent(EventBase):
    type: Literal["ryukyoku"]
    deltas: tuple[int, int, int, int]


class EndKyokuEvent(EventBase):
    type: Literal["end_kyoku"]


class EndGameEvent(EventBase):
    type: Literal["end_game"]


MjaiEvent: TypeAlias = Annotated[
    StartGameEvent
    | StartKyokuEvent
    | TsumoEvent
    | DahaiEvent
    | ChiEvent
    | PonEvent
    | DaiminkanEvent
    | AnkanEvent
    | KakanEvent
    | DoraEvent
    | ReachEvent
    | ReachAcceptedEvent
    | HoraEvent
    | RyukyokuEvent
    | EndKyokuEvent
    | EndGameEvent,
    Field(discriminator="type"),
]
