import json
import math
from pathlib import Path

import pytest

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.mjai.events import Tile
from mahjong_mind.modelling.baseline_predictions import (
    MostCommonLegalBaseline,
    RandomLegalBaseline,
    fit_most_common_baseline,
)
from mahjong_mind.modelling.metrics_evaluation import (
    ACTION_COUNT,
    PolicyPrediction,
    RankingMetricsAccumulator,
)
from mahjong_mind.modelling.model_input_encoding import (
    FEATURE_NAMES,
    MODEL_INPUT_VERSION,
    encode_parquet_row,
)
from mahjong_mind.modelling.tile_efficiency import (
    calculate_shanten,
    rank_discards_by_tile_efficiency,
    shanten_after_legal_discards,
    tiles_to_34_counts,
    ukeire_after_legal_discards,
)


def legal_mask(*actions: int) -> tuple[bool, ...]:
    return tuple(action in actions for action in range(ACTION_COUNT))


def test_model_input_encoding_separates_features_mask_and_label() -> None:
    players = [
        {
            "concealed_tile_count": 13,
            "discards": [],
            "melds": [],
            "riichi": "none",
        }
        for _ in range(4)
    ]
    players[2] = {
        "concealed_tile_count": 14,
        "discards": [
            {
                "tile": "5mr",
                "tsumogiri": True,
                "riichi": False,
                "called": False,
            }
        ],
        "melds": [],
        "riichi": "pending",
    }
    row = {
        "actor": 2,
        "aka_flag": True,
        "bakaze": "E",
        "seat_wind": "S",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 1,
        "dealer": 1,
        "scores": [10_000, 20_000, 30_000, 40_000],
        "dora_markers": ["C"],
        "draws_remaining": 50,
        "actor_turn_index": 1,
        "own_hand": ["1m", "5mr"],
        "own_last_draw": "5mr",
        "players": players,
        "legal_discard_mask": legal_mask(0, 34),
        "label_index": 34,
        "actual_discard": "5mr",
    }

    encoded = encode_parquet_row(row)
    other_label = encode_parquet_row({**row, "label_index": 0, "actual_discard": "1m"})

    assert MODEL_INPUT_VERSION == "dense-observation-v1"
    assert len(encoded.model_input.features) == len(FEATURE_NAMES)
    assert encoded.model_input.features == other_label.model_input.features
    assert encoded.model_input.legal_discard_mask == legal_mask(0, 34)
    assert encoded.label_index == 34
    assert other_label.label_index == 0
    assert encoded.model_input.features[FEATURE_NAMES.index("own_hand_count.5mr")] == 1
    assert (
        encoded.model_input.features[FEATURE_NAMES.index("score_relative.0")] == 30_000
    )
    assert (
        encoded.model_input.features[
            FEATURE_NAMES.index("player_relative.0.riichi.pending")
        ]
        == 1
    )


def test_ranking_metrics_calculate_expected_values() -> None:
    mask = legal_mask(0, 1, 2)
    accumulator = RankingMetricsAccumulator()
    accumulator.update(
        PolicyPrediction(
            ranked_actions=(1, 2, 0),
            probabilities=(0.25, 0.5, 0.25, *([0.0] * (ACTION_COUNT - 3))),
        ),
        label_index=1,
        legal_mask=mask,
    )
    accumulator.update(
        PolicyPrediction(
            ranked_actions=(1, 2, 0),
            probabilities=(0.25, 0.5, 0.25, *([0.0] * (ACTION_COUNT - 3))),
        ),
        label_index=2,
        legal_mask=mask,
    )

    metrics = accumulator.compute()

    assert metrics.decisions == 2
    assert metrics.top_1_accuracy == 0.5
    assert metrics.top_3_accuracy == 1.0
    assert metrics.mean_reciprocal_rank == 0.75
    assert metrics.cross_entropy == pytest.approx(
        (-math.log(0.5) - math.log(0.25)) / 2
    )


def test_baselines_only_rank_legal_actions_with_valid_probabilities() -> None:
    mask = legal_mask(0, 2, 34)
    first_random = RandomLegalBaseline(seed=7).predict(mask)
    second_random = RandomLegalBaseline(seed=7).predict(mask)

    assert first_random == second_random
    assert set(first_random.ranked_actions) == {0, 2, 34}
    assert sum(first_random.probabilities) == pytest.approx(1.0)
    assert all(
        first_random.probabilities[action] == pytest.approx(1 / 3)
        for action in (0, 2, 34)
    )

    frequency = MostCommonLegalBaseline.fit([0, 0, 1, 1, 1, 2])
    prediction = frequency.predict(mask)

    assert prediction.ranked_actions == (0, 2, 34)
    assert prediction.probabilities[1] == 0.0
    assert prediction.probabilities[0] > prediction.probabilities[2]
    assert prediction.probabilities[34] > 0.0
    assert sum(prediction.probabilities) == pytest.approx(1.0)


def test_frequency_baseline_uses_manifest_counts(tmp_path: Path) -> None:
    action_counts = {tile: 0 for tile in DISCARD_TILE_TYPES}
    action_counts["1m"] = 3
    action_counts["2m"] = 1
    (tmp_path / "manifest.json").write_text(
        json.dumps({"records": {"action_counts": action_counts}}),
        encoding="utf-8",
    )

    baseline = fit_most_common_baseline(tmp_path)

    assert baseline.action_counts[0] == 3
    assert baseline.action_counts[1] == 1


def test_shanten_adapter_normalizes_red_fives() -> None:
    complete_hand: tuple[Tile, ...] = (
        "1m",
        "2m",
        "3m",
        "1p",
        "2p",
        "3p",
        "1s",
        "2s",
        "3s",
        "E",
        "E",
        "E",
        "5pr",
        "5p",
    )

    counts = tiles_to_34_counts(complete_hand)

    assert len(counts) == 34
    assert counts[13] == 2
    assert calculate_shanten(complete_hand) == -1

    discard_shanten = shanten_after_legal_discards(
        complete_hand,
        legal_mask(13, 35),
    )

    assert len(discard_shanten) == ACTION_COUNT
    assert discard_shanten[13] == 0
    assert discard_shanten[35] == 0
    assert discard_shanten[0] is None

    discard_ukeire = ukeire_after_legal_discards(
        complete_hand,
        legal_mask(13, 35),
        known_tiles=complete_hand,
    )

    assert discard_ukeire[13] == 2
    assert discard_ukeire[35] == 2
    assert discard_ukeire[0] is None

    ranked_actions = rank_discards_by_tile_efficiency(
        complete_hand,
        legal_mask(0, 13, 35),
        known_tiles=complete_hand,
    )

    assert ranked_actions == (0, 13, 35)
