import math
from dataclasses import dataclass

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES

ACTION_COUNT = len(DISCARD_TILE_TYPES)


class RankingEvaluationError(ValueError):
    """Raised when a policy prediction cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class PolicyPrediction:
    """A legal action ranking and probability distribution for one decision."""

    ranked_actions: tuple[int, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Aggregate behavioral-cloning ranking metrics."""

    decisions: int
    top_1_accuracy: float
    top_3_accuracy: float
    mean_reciprocal_rank: float
    cross_entropy: float


class RankingMetricsAccumulator:
    """Accumulate ranking metrics without retaining individual predictions."""

    def __init__(self) -> None:
        self._decisions = 0
        self._top_1_hits = 0
        self._top_3_hits = 0
        self._reciprocal_rank_sum = 0.0
        self._cross_entropy_sum = 0.0

    def update(
        self,
        prediction: PolicyPrediction,
        label_index: int,
        legal_mask: tuple[bool, ...],
    ) -> None:
        """Validate and score one prediction against its historical label."""
        self._validate(prediction, label_index, legal_mask)
        rank = prediction.ranked_actions.index(label_index) + 1
        label_probability = prediction.probabilities[label_index]

        self._decisions += 1
        self._top_1_hits += rank == 1
        self._top_3_hits += rank <= 3
        self._reciprocal_rank_sum += 1.0 / rank
        self._cross_entropy_sum += -math.log(label_probability)

    def compute(self) -> RankingMetrics:
        """Return the mean metrics accumulated so far."""
        if self._decisions == 0:
            raise RankingEvaluationError("Cannot compute metrics without decisions")
        return RankingMetrics(
            decisions=self._decisions,
            top_1_accuracy=self._top_1_hits / self._decisions,
            top_3_accuracy=self._top_3_hits / self._decisions,
            mean_reciprocal_rank=self._reciprocal_rank_sum / self._decisions,
            cross_entropy=self._cross_entropy_sum / self._decisions,
        )

    @staticmethod
    def _validate(
        prediction: PolicyPrediction,
        label_index: int,
        legal_mask: tuple[bool, ...],
    ) -> None:
        if len(legal_mask) != ACTION_COUNT:
            raise RankingEvaluationError(
                f"Legal mask has {len(legal_mask)} actions, expected {ACTION_COUNT}"
            )
        legal_actions = tuple(
            action for action, is_legal in enumerate(legal_mask) if is_legal
        )
        if not legal_actions:
            raise RankingEvaluationError("Decision has no legal actions")
        if not 0 <= label_index < ACTION_COUNT or not legal_mask[label_index]:
            raise RankingEvaluationError(f"Label {label_index} is not legal")
        if len(prediction.probabilities) != ACTION_COUNT:
            raise RankingEvaluationError(
                f"Prediction has {len(prediction.probabilities)} probabilities, "
                f"expected {ACTION_COUNT}"
            )
        if set(prediction.ranked_actions) != set(legal_actions) or len(
            prediction.ranked_actions
        ) != len(legal_actions):
            raise RankingEvaluationError(
                "Ranked actions must contain every legal action exactly once"
            )

        probability_sum = 0.0
        for action, probability in enumerate(prediction.probabilities):
            if not math.isfinite(probability) or probability < 0.0:
                raise RankingEvaluationError(
                    f"Action {action} has invalid probability {probability}"
                )
            if not legal_mask[action] and probability != 0.0:
                raise RankingEvaluationError(
                    f"Illegal action {action} has non-zero probability"
                )
            probability_sum += probability

        if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise RankingEvaluationError(
                f"Prediction probabilities sum to {probability_sum}, expected 1"
            )
        if prediction.probabilities[label_index] <= 0.0:
            raise RankingEvaluationError("Historical label has zero probability")
