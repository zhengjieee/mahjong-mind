from collections import Counter
from dataclasses import replace
from typing import NoReturn

from mahjong_mind.game_state.state_types import (
    Discard,
    HandState,
    MatchState,
    Meld,
    MeldType,
    PlayerState,
)
from mahjong_mind.game_state.state_validation import (
    StateInvariantError,
    validate_hand_state,
)
from mahjong_mind.mjai.events import (
    AnkanEvent,
    ChiEvent,
    DahaiEvent,
    DaiminkanEvent,
    DoraEvent,
    EndGameEvent,
    EndKyokuEvent,
    HoraEvent,
    KakanEvent,
    PonEvent,
    ReachAcceptedEvent,
    ReachEvent,
    RyukyokuEvent,
    StartGameEvent,
    StartKyokuEvent,
    Tile,
    TsumoEvent,
)
from mahjong_mind.mjai.parser import ParsedEvent


class StateReconstructionError(ValueError):
    """Raised when a valid MJAI event is impossible in the current state."""


class StateReconstructor:
    def __init__(self, *, validate_after_event: bool = False) -> None:
        self.state: MatchState | None = None
        self.validate_after_event = validate_after_event

    def apply(self, parsed: ParsedEvent) -> MatchState:
        if isinstance(parsed.event, StartGameEvent):
            self._start_game(parsed)
            return self._require_match(parsed)

        match = self._require_match(parsed)
        self._check_event_identity(parsed, match)
        event = parsed.event

        if isinstance(event, StartKyokuEvent):
            self._start_hand(match, event)
        elif isinstance(event, TsumoEvent):
            self._apply_draw(self._require_hand(parsed), event)
        elif isinstance(event, DahaiEvent):
            self._apply_discard(self._require_hand(parsed), event)
        elif isinstance(event, ChiEvent):
            self._apply_open_meld(self._require_hand(parsed), event, "chi")
        elif isinstance(event, PonEvent):
            self._apply_open_meld(self._require_hand(parsed), event, "pon")
        elif isinstance(event, DaiminkanEvent):
            self._apply_open_meld(self._require_hand(parsed), event, "daiminkan")
        elif isinstance(event, AnkanEvent):
            self._apply_ankan(self._require_hand(parsed), event)
        elif isinstance(event, KakanEvent):
            self._apply_kakan(self._require_hand(parsed), event)
        elif isinstance(event, DoraEvent):
            self._require_hand(parsed).dora_markers.append(event.dora_marker)
        elif isinstance(event, ReachEvent):
            self._apply_reach(self._require_hand(parsed), event)
        elif isinstance(event, ReachAcceptedEvent):
            self._accept_reach(self._require_hand(parsed), event)
        elif isinstance(event, HoraEvent):
            self._apply_hora(self._require_hand(parsed), event)
        elif isinstance(event, RyukyokuEvent):
            self._apply_ryukyoku(self._require_hand(parsed), event)
        elif isinstance(event, EndKyokuEvent):
            self._require_hand(parsed).ended = True
        elif isinstance(event, EndGameEvent):
            if match.current_hand is not None and not match.current_hand.ended:
                self._fail(parsed, "end_game appears before the current hand ended")
            match.ended = True

        if self.validate_after_event and match.current_hand is not None:
            try:
                validate_hand_state(match.current_hand)
            except StateInvariantError as exc:
                self._fail(parsed, str(exc))

        match.event_index = parsed.event_index
        return match

    def _start_game(self, parsed: ParsedEvent) -> None:
        if self.state is not None:
            self._fail(parsed, "start_game appears more than once")
        if parsed.event_index != 0:
            self._fail(parsed, "start_game is not event 0")

        event = parsed.event
        assert isinstance(event, StartGameEvent)
        self.state = MatchState(
            match_id=parsed.match_id,
            names=event.names,
            kyoku_first=event.kyoku_first,
            aka_flag=event.aka_flag,
            event_index=parsed.event_index,
        )

    def _start_hand(self, match: MatchState, event: StartKyokuEvent) -> None:
        if match.current_hand is not None and not match.current_hand.ended:
            raise StateReconstructionError(
                f"Invalid state in {match.match_id}: start_kyoku appears before end_kyoku"
            )

        match.hand_index += 1
        match.current_hand = HandState(
            hand_index=match.hand_index,
            bakaze=event.bakaze,
            kyoku=event.kyoku,
            honba=event.honba,
            kyotaku=event.kyotaku,
            dealer=event.oya,
            scores=list(event.scores),
            dora_markers=[event.dora_marker],
            players=(
                PlayerState(concealed_tiles=list(event.tehais[0])),
                PlayerState(concealed_tiles=list(event.tehais[1])),
                PlayerState(concealed_tiles=list(event.tehais[2])),
                PlayerState(concealed_tiles=list(event.tehais[3])),
            ),
        )

    def _apply_draw(self, hand: HandState, event: TsumoEvent) -> None:
        player = hand.player(event.actor)
        if player.last_draw is not None:
            raise StateReconstructionError(
                f"Player {event.actor} draws before resolving the previous draw"
            )
        if hand.draws_remaining <= 0:
            raise StateReconstructionError(
                "Draw occurs after the live wall is exhausted"
            )

        player.concealed_tiles.append(event.pai)
        player.last_draw = event.pai
        hand.draws_remaining -= 1

    def _apply_discard(self, hand: HandState, event: DahaiEvent) -> None:
        player = hand.player(event.actor)
        if event.tsumogiri and player.last_draw != event.pai:
            raise StateReconstructionError(
                f"Player {event.actor} marks {event.pai} as tsumogiri after drawing "
                f"{player.last_draw}"
            )

        self._remove_tiles(player, (event.pai,), event.actor)
        player.discards.append(
            Discard(
                tile=event.pai,
                tsumogiri=event.tsumogiri,
                riichi=player.riichi == "pending",
            )
        )
        player.last_draw = None
        hand.last_discard_actor = event.actor
        hand.last_discard_index = len(player.discards) - 1

    def _apply_open_meld(
        self,
        hand: HandState,
        event: ChiEvent | PonEvent | DaiminkanEvent,
        meld_type: MeldType,
    ) -> None:
        if event.actor == event.target:
            raise StateReconstructionError(
                f"Player {event.actor} calls their own discard"
            )

        self._mark_last_discard_called(hand, event.target, event.pai)
        player = hand.player(event.actor)
        self._remove_tiles(player, event.consumed, event.actor)
        player.last_draw = None
        player.melds.append(
            Meld(
                type=meld_type,
                tiles=(*event.consumed, event.pai),
                called_tile=event.pai,
                target=event.target,
            )
        )

    def _apply_ankan(self, hand: HandState, event: AnkanEvent) -> None:
        player = hand.player(event.actor)
        self._remove_tiles(player, event.consumed, event.actor)
        player.last_draw = None
        player.melds.append(
            Meld(
                type="ankan",
                tiles=event.consumed,
                called_tile=None,
                target=None,
            )
        )

    def _apply_kakan(self, hand: HandState, event: KakanEvent) -> None:
        player = hand.player(event.actor)
        self._remove_tiles(player, (event.pai,), event.actor)
        player.last_draw = None

        expected_tiles = Counter(event.consumed)
        for index, meld in enumerate(player.melds):
            if meld.type == "pon" and Counter(meld.tiles) == expected_tiles:
                player.melds[index] = Meld(
                    type="kakan",
                    tiles=(*meld.tiles, event.pai),
                    called_tile=meld.called_tile,
                    target=meld.target,
                )
                return

        raise StateReconstructionError(
            f"Player {event.actor} uses kakan without a matching pon"
        )

    def _apply_reach(self, hand: HandState, event: ReachEvent) -> None:
        player = hand.player(event.actor)
        if player.riichi != "none":
            raise StateReconstructionError(
                f"Player {event.actor} declares Riichi more than once"
            )
        player.riichi = "pending"

    def _accept_reach(self, hand: HandState, event: ReachAcceptedEvent) -> None:
        player = hand.player(event.actor)
        if player.riichi != "pending":
            raise StateReconstructionError(
                f"Player {event.actor} has reach_accepted without pending Riichi"
            )
        player.riichi = "accepted"
        hand.scores[event.actor] -= 1000
        hand.kyotaku += 1

    def _apply_hora(self, hand: HandState, event: HoraEvent) -> None:
        for player in hand.players:
            if player.riichi == "pending":
                player.riichi = "none"
        self._apply_score_deltas(hand, event.deltas)

    def _apply_ryukyoku(self, hand: HandState, event: RyukyokuEvent) -> None:
        for player_id, player in enumerate(hand.players):
            if player.riichi == "pending":
                player.riichi = "accepted"
                hand.scores[player_id] -= 1000
                hand.kyotaku += 1
        self._apply_score_deltas(hand, event.deltas)

    @staticmethod
    def _apply_score_deltas(hand: HandState, deltas: tuple[int, int, int, int]) -> None:
        for player_id, delta in enumerate(deltas):
            hand.scores[player_id] += delta

    @staticmethod
    def _remove_tiles(
        player: PlayerState,
        tiles: tuple[Tile, ...],
        player_id: int,
    ) -> None:
        for tile in tiles:
            try:
                player.concealed_tiles.remove(tile)
            except ValueError as exc:
                raise StateReconstructionError(
                    f"Player {player_id} does not hold required tile {tile}"
                ) from exc

    @staticmethod
    def _mark_last_discard_called(
        hand: HandState,
        target: int,
        tile: Tile,
    ) -> None:
        if hand.last_discard_actor != target or hand.last_discard_index is None:
            raise StateReconstructionError(
                f"Call targets player {target}, who did not make the latest discard"
            )

        discards = hand.player(target).discards
        discard = discards[hand.last_discard_index]
        if discard.tile != tile or discard.called:
            raise StateReconstructionError(
                f"Call claims {tile}, but the latest available discard is {discard.tile}"
            )
        discards[hand.last_discard_index] = replace(discard, called=True)

    def _require_match(self, parsed: ParsedEvent) -> MatchState:
        if self.state is None:
            self._fail(parsed, "event appears before start_game")
        return self.state

    def _require_hand(self, parsed: ParsedEvent) -> HandState:
        match = self._require_match(parsed)
        if match.current_hand is None:
            self._fail(parsed, "event appears before start_kyoku")
        if match.current_hand.ended:
            self._fail(parsed, "event appears after end_kyoku")
        return match.current_hand

    def _check_event_identity(self, parsed: ParsedEvent, match: MatchState) -> None:
        if parsed.match_id != match.match_id:
            self._fail(parsed, f"expected match_id {match.match_id}")
        expected_index = match.event_index + 1
        if parsed.event_index != expected_index:
            self._fail(parsed, f"expected event index {expected_index}")

    @staticmethod
    def _fail(parsed: ParsedEvent, reason: str) -> NoReturn:
        raise StateReconstructionError(
            f"Invalid state in {parsed.match_id} at event {parsed.event_index}: {reason}"
        )
