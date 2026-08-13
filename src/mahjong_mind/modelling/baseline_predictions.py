import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.modelling.metrics_evaluation import (
    ACTION_COUNT,
    PolicyPrediction,
    RankingMetrics,
    RankingMetricsAccumulator,
)


class BaselineError(ValueError):
    """Raised when a discard baseline cannot be fitted or evaluated."""


class DiscardBaseline(Protocol):
    def predict(self, legal_mask: tuple[bool, ...]) -> PolicyPrediction:
        """Return a ranked probability distribution over legal discards."""


class RandomLegalBaseline:
    """Rank legal actions randomly and assign each equal probability."""

    def __init__(self, *, seed: int = 0) -> None:
        self._random = random.Random(seed)

    def predict(self, legal_mask: tuple[bool, ...]) -> PolicyPrediction:
        legal_actions = _legal_actions(legal_mask)
        ranked_actions = list(legal_actions)
        self._random.shuffle(ranked_actions)
        legal_probability = 1.0 / len(legal_actions)
        probabilities = tuple(
            legal_probability if is_legal else 0.0 for is_legal in legal_mask
        )
        return PolicyPrediction(tuple(ranked_actions), probabilities)


class MostCommonLegalBaseline:
    """Rank currently legal actions by their frequency in fitting data."""

    def __init__(self, action_counts: tuple[int, ...], *, smoothing: float = 1.0):
        if len(action_counts) != ACTION_COUNT:
            raise BaselineError(
                f"Received {len(action_counts)} action counts, expected {ACTION_COUNT}"
            )
        if any(count < 0 for count in action_counts):
            raise BaselineError("Action counts cannot be negative")
        if sum(action_counts) == 0:
            raise BaselineError("Cannot fit a frequency baseline without labels")
        if not 0.0 < smoothing:
            raise BaselineError("smoothing must be greater than zero")
        self.action_counts = action_counts
        self.smoothing = smoothing

    @classmethod
    def fit(cls, labels: Iterable[int], *, smoothing: float = 1.0) -> "MostCommonLegalBaseline":
        counts: Counter[int] = Counter()
        for label in labels:
            if not 0 <= label < ACTION_COUNT:
                raise BaselineError(f"Label {label} is outside the action space")
            counts[label] += 1
        return cls(
            tuple(counts[action] for action in range(ACTION_COUNT)),
            smoothing=smoothing,
        )

    def predict(self, legal_mask: tuple[bool, ...]) -> PolicyPrediction:
        legal_actions = _legal_actions(legal_mask)
        ranked_actions = tuple(
            sorted(legal_actions, key=lambda action: (-self.action_counts[action], action))
        )
        legal_weights = {
            action: self.action_counts[action] + self.smoothing
            for action in legal_actions
        }
        total_weight = sum(legal_weights.values())
        probabilities = tuple(
            legal_weights[action] / total_weight if action in legal_weights else 0.0
            for action in range(ACTION_COUNT)
        )
        return PolicyPrediction(ranked_actions, probabilities)


def _legal_actions(legal_mask: tuple[bool, ...]) -> tuple[int, ...]:
    if len(legal_mask) != ACTION_COUNT:
        raise BaselineError(
            f"Legal mask has {len(legal_mask)} actions, expected {ACTION_COUNT}"
        )
    actions = tuple(action for action, is_legal in enumerate(legal_mask) if is_legal)
    if not actions:
        raise BaselineError("Decision has no legal discard actions")
    return actions


def iter_parquet_examples(
    dataset_directory: Path,
    *,
    max_decisions: int | None = None,
) -> Iterator[tuple[tuple[bool, ...], int]]:
    """Stream legal masks and labels from sorted Parquet shards."""
    if max_decisions is not None and max_decisions < 1:
        raise BaselineError("max_decisions must be at least 1")
    shard_paths = sorted(dataset_directory.glob("source_year=*/part-*.parquet"))
    if not shard_paths:
        raise BaselineError(f"No Parquet shards found in {dataset_directory}")

    yielded = 0
    for path in shard_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["legal_discard_mask", "label_index"],
            batch_size=65_536,
        ):
            masks = batch.column(0).to_pylist()
            labels = batch.column(1).to_pylist()
            for mask, label in zip(masks, labels, strict=True):
                if max_decisions is not None and yielded >= max_decisions:
                    return
                yield tuple(mask), label
                yielded += 1


def fit_most_common_baseline(
    dataset_directory: Path,
) -> MostCommonLegalBaseline:
    """Fit global discard frequencies from a dataset's label column."""
    return MostCommonLegalBaseline.fit(
        label for _, label in iter_parquet_examples(dataset_directory)
    )


def evaluate_baselines(
    baselines: Mapping[str, DiscardBaseline],
    examples: Iterable[tuple[tuple[bool, ...], int]],
) -> dict[str, RankingMetrics]:
    """Evaluate several baselines in one streaming pass."""
    if not baselines:
        raise BaselineError("At least one baseline is required")
    accumulators = {name: RankingMetricsAccumulator() for name in baselines}
    for legal_mask, label_index in examples:
        for name, baseline in baselines.items():
            prediction = baseline.predict(legal_mask)
            accumulators[name].update(prediction, label_index, legal_mask)
    return {name: accumulator.compute() for name, accumulator in accumulators.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate simple legal-discard baselines on a Parquet dataset."
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument(
        "--frequency-fit-dataset",
        type=Path,
        help="Dataset whose labels fit the most-common baseline (defaults to input).",
    )
    parser.add_argument("--limit", type=int, dest="max_decisions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fit_directory = args.frequency_fit_dataset or args.dataset_directory
    frequency_baseline = fit_most_common_baseline(fit_directory)
    metrics = evaluate_baselines(
        {
            "random_legal": RandomLegalBaseline(seed=args.seed),
            "most_common_legal": frequency_baseline,
        },
        iter_parquet_examples(
            args.dataset_directory,
            max_decisions=args.max_decisions,
        ),
    )
    same_dataset = fit_directory.resolve() == args.dataset_directory.resolve()
    report = {
        "evaluation_dataset": str(args.dataset_directory),
        "frequency_fit_dataset": str(fit_directory),
        "decision_limit": args.max_decisions,
        "development_only": same_dataset,
        "note": (
            "The frequency baseline was fitted and evaluated on the same development "
            "dataset; these are pipeline sanity metrics, not validation results."
            if same_dataset
            else None
        ),
        "action_order": list(DISCARD_TILE_TYPES),
        "metrics": {name: asdict(result) for name, result in metrics.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
