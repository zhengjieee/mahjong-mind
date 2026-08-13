import math

import pytest

from mahjong_mind.modeling.baseline_predictions import (
    MostCommonLegalBaseline,
    RandomLegalBaseline,
)
from mahjong_mind.modeling.metrics_evaluation import (
    ACTION_COUNT,
    PolicyPrediction,
    RankingMetricsAccumulator,
)


def legal_mask(*actions: int) -> tuple[bool, ...]:
    return tuple(action in actions for action in range(ACTION_COUNT))


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
