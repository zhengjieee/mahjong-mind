import pytest

from mahjong_mind.game_state.legal_actions import (
    DISCARD_TILE_TYPES,
    legal_discard_mask,
    legal_discard_tiles,
)
from mahjong_mind.game_state.player_observation import observation_for_player
from mahjong_mind.game_state.reconstructor import (
    StateReconstructionError,
    StateReconstructor,
)
from mahjong_mind.mjai.events import (
    DahaiEvent,
    InitialHand,
    MjaiEvent,
    PonEvent,
    StartGameEvent,
    StartKyokuEvent,
    TsumoEvent,
)
from mahjong_mind.mjai.parser import ParsedEvent
from mahjong_mind.modelling.tile_efficiency import (
    TileEfficiencyBaseline,
    known_tiles_from_observation,
    tiles_to_34_counts,
)


def start_game() -> StartGameEvent:
    return StartGameEvent(
        type="start_game",
        names=("A", "B", "C", "D"),
        kyoku_first=0,
        aka_flag=True,
    )


def real_start_kyoku() -> StartKyokuEvent:
    return StartKyokuEvent(
        type="start_kyoku",
        bakaze="E",
        dora_marker="9s",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        tehais=(
            ("4m", "7m", "9m", "1s", "3s", "3s", "3s", "7s", "8s", "W", "N", "P", "F"),
            (
                "1m",
                "3m",
                "1p",
                "3p",
                "4p",
                "7p",
                "8p",
                "9p",
                "6s",
                "6s",
                "8s",
                "P",
                "P",
            ),
            ("1m", "2m", "2m", "2m", "6m", "6m", "2p", "4p", "5p", "8s", "E", "P", "C"),
            (
                "3m",
                "3m",
                "4m",
                "4m",
                "7m",
                "8m",
                "8m",
                "9m",
                "6p",
                "9p",
                "1s",
                "5s",
                "S",
            ),
        ),
    )


def test_reconstructs_draw_discard_and_pon() -> None:
    reconstructor = StateReconstructor(validate_after_event=True)
    events: list[MjaiEvent] = [
        start_game(),
        real_start_kyoku(),
        TsumoEvent(type="tsumo", actor=0, pai="N"),
        DahaiEvent(type="dahai", actor=0, pai="P", tsumogiri=False),
        PonEvent(type="pon", actor=1, target=0, pai="P", consumed=("P", "P")),
        DahaiEvent(type="dahai", actor=1, pai="1p", tsumogiri=False),
    ]

    for index, event in enumerate(events):
        state = reconstructor.apply(
            ParsedEvent(match_id="sample-match", event_index=index, event=event)
        )

    assert state.current_hand is not None
    hand = state.current_hand
    assert hand.player(0).discards[0].called is True
    assert hand.player(1).melds[0].tiles == ("P", "P", "P")
    assert len(hand.player(1).concealed_tiles) == 10
    assert hand.last_discard_actor == 1
    assert hand.last_discard is not None
    assert hand.last_discard.tile == "1p"

    observation = observation_for_player(state, 1)
    known_tiles = known_tiles_from_observation(observation)
    known_counts = tiles_to_34_counts(known_tiles)

    assert len(known_tiles) == 15
    assert known_counts[9] == 1
    assert known_counts[31] == 3


def test_rejects_a_fifth_copy_of_a_tile() -> None:
    reconstructor = StateReconstructor(validate_after_event=True)
    reconstructor.apply(ParsedEvent("sample-match", 0, start_game()))
    repeated_hand: InitialHand = (
        "1m",
        "2m",
        "3m",
        "4m",
        "5m",
        "6m",
        "7m",
        "8m",
        "9m",
        "E",
        "S",
        "P",
        "C",
    )
    invalid_start = StartKyokuEvent(
        type="start_kyoku",
        bakaze="E",
        dora_marker="1m",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        tehais=(repeated_hand, repeated_hand, repeated_hand, repeated_hand),
    )

    with pytest.raises(
        StateReconstructionError, match="physical tile count for 1m is 5"
    ):
        reconstructor.apply(ParsedEvent("sample-match", 1, invalid_start))


def test_observation_hides_opponents_hands_and_is_a_snapshot() -> None:
    reconstructor = StateReconstructor(validate_after_event=True)
    events: list[MjaiEvent] = [
        start_game(),
        real_start_kyoku(),
        TsumoEvent(type="tsumo", actor=0, pai="N"),
    ]
    for index, event in enumerate(events):
        state = reconstructor.apply(
            ParsedEvent(match_id="sample-match", event_index=index, event=event)
        )

    observation = observation_for_player(state, 0)
    legal_mask = legal_discard_mask(observation)
    tile_efficiency_prediction = TileEfficiencyBaseline().predict(observation)

    assert state.current_hand is not None
    assert observation.own_hand == tuple(
        state.current_hand.player(0).concealed_tiles
    )
    assert observation.own_last_draw == "N"
    assert not hasattr(observation.players[1], "concealed_tiles")
    assert observation.players[1].concealed_tile_count == 13
    assert set(tile_efficiency_prediction.ranked_actions) == {
        action for action, is_legal in enumerate(legal_mask) if is_legal
    }
    assert sum(tile_efficiency_prediction.probabilities) == pytest.approx(1.0)
    assert all(
        probability > 0.0 if is_legal else probability == 0.0
        for probability, is_legal in zip(
            tile_efficiency_prediction.probabilities,
            legal_mask,
        )
    )

    reconstructor.apply(
        ParsedEvent(
            match_id="sample-match",
            event_index=3,
            event=DahaiEvent(type="dahai", actor=0, pai="N", tsumogiri=True),
        )
    )

    assert len(observation.own_hand) == 14
    assert observation.players[0].discards == ()


def test_legal_discard_mask_respects_hand_and_riichi() -> None:
    reconstructor = StateReconstructor(validate_after_event=True)
    events: list[MjaiEvent] = [
        start_game(),
        real_start_kyoku(),
        TsumoEvent(type="tsumo", actor=0, pai="2s"),
    ]
    for index, event in enumerate(events):
        state = reconstructor.apply(
            ParsedEvent(match_id="sample-match", event_index=index, event=event)
        )

    observation = observation_for_player(state, 0)
    legal_tiles = legal_discard_tiles(observation)
    mask = legal_discard_mask(observation)

    assert len(mask) == 37
    assert {tile for tile, allowed in zip(DISCARD_TILE_TYPES, mask) if allowed} == set(
        legal_tiles
    )
    assert set(legal_tiles) == set(observation.own_hand)

    assert state.current_hand is not None
    state.current_hand.player(0).riichi = "accepted"
    riichi_observation = observation_for_player(state, 0)

    assert legal_discard_tiles(riichi_observation) == ("2s",)
