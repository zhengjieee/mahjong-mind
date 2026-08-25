import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.mjai.events import Tile
from mahjong_mind.modelling.baselines.tile_efficiency import TileEfficiencyBaseline
from mahjong_mind.modelling.shared.metrics_evaluation import (
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


def sample_shard_paths(
    dataset_directory: Path,
    *,
    shard_count: int,
    seed: int = 0,
    exclude: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """Randomly select a fixed number of whole shard files.

    Sampling at shard granularity (rather than truncating to the first N rows)
    avoids biasing an evaluation sample toward whichever matches happened to be
    written into the earliest shards. Spreading the sample across many shards,
    each contributing only a capped number of decisions (see
    max_decisions_per_shard on the iteration functions below), keeps any single
    contiguous block of matches from dominating the sample. Pass exclude (e.g.
    a set already used as a final comparison set) to guarantee this sample
    never overlaps with it.
    """
    if shard_count < 1:
        raise BaselineError("shard_count must be at least 1")
    excluded = {path.resolve() for path in exclude}
    shard_paths = [
        path
        for path in sorted(dataset_directory.glob("source_year=*/part-*.parquet"))
        if path.resolve() not in excluded
    ]
    if not shard_paths:
        raise BaselineError(f"No Parquet shards found in {dataset_directory}")
    if shard_count > len(shard_paths):
        raise BaselineError(
            f"Requested {shard_count} shards but only {len(shard_paths)} remain "
            "after exclusions"
        )

    shuffled = list(shard_paths)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled[:shard_count])


def iter_parquet_examples(
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
) -> Iterator[tuple[tuple[bool, ...], int]]:
    """Stream legal masks and labels from Parquet shards.

    Reads every shard under dataset_directory in sorted order unless an
    explicit shard_paths subset (e.g. from sample_shard_paths) is given.
    max_decisions caps the overall stream; max_decisions_per_shard caps how
    many rows are read from each individual shard file, so a random sample of
    shards can be spread thin instead of read in full.
    """
    if max_decisions is not None and max_decisions < 1:
        raise BaselineError("max_decisions must be at least 1")
    if max_decisions_per_shard is not None and max_decisions_per_shard < 1:
        raise BaselineError("max_decisions_per_shard must be at least 1")
    resolved_shard_paths = (
        tuple(shard_paths)
        if shard_paths is not None
        else tuple(sorted(dataset_directory.glob("source_year=*/part-*.parquet")))
    )
    if not resolved_shard_paths:
        raise BaselineError(f"No Parquet shards found in {dataset_directory}")

    yielded = 0
    for path in resolved_shard_paths:
        yielded_in_shard = 0
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
                if (
                    max_decisions_per_shard is not None
                    and yielded_in_shard >= max_decisions_per_shard
                ):
                    break
                yield tuple(mask), label
                yielded += 1
                yielded_in_shard += 1
            if (
                max_decisions_per_shard is not None
                and yielded_in_shard >= max_decisions_per_shard
            ):
                break


def fit_most_common_baseline(
    dataset_directory: Path,
) -> MostCommonLegalBaseline:
    """Fit global discard frequencies from a dataset's label column."""
    manifest_path = dataset_directory / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            action_counts = manifest["records"]["action_counts"]
            return MostCommonLegalBaseline(
                tuple(action_counts[tile] for tile in DISCARD_TILE_TYPES)
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise BaselineError(
                f"Invalid action counts in {manifest_path}"
            ) from error

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


def evaluate_tile_efficiency_baseline(
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
) -> RankingMetrics:
    """Evaluate tile efficiency from observable Parquet fields."""
    baseline = TileEfficiencyBaseline()
    accumulator = RankingMetricsAccumulator()
    for hand, legal_mask, known_tiles, label_index in (
        _iter_tile_efficiency_examples(
            dataset_directory,
            shard_paths=shard_paths,
            max_decisions=max_decisions,
            max_decisions_per_shard=max_decisions_per_shard,
        )
    ):
        prediction = baseline.predict_tiles(hand, legal_mask, known_tiles)
        accumulator.update(prediction, label_index, legal_mask)
    return accumulator.compute()


def _iter_tile_efficiency_examples(
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
) -> Iterator[tuple[tuple[Tile, ...], tuple[bool, ...], tuple[Tile, ...], int]]:
    if max_decisions is not None and max_decisions < 1:
        raise BaselineError("max_decisions must be at least 1")
    if max_decisions_per_shard is not None and max_decisions_per_shard < 1:
        raise BaselineError("max_decisions_per_shard must be at least 1")
    resolved_shard_paths = (
        tuple(shard_paths)
        if shard_paths is not None
        else tuple(sorted(dataset_directory.glob("source_year=*/part-*.parquet")))
    )
    if not resolved_shard_paths:
        raise BaselineError(f"No Parquet shards found in {dataset_directory}")

    yielded = 0
    columns = [
        "own_hand",
        "dora_markers",
        "players",
        "legal_discard_mask",
        "label_index",
    ]
    for path in resolved_shard_paths:
        yielded_in_shard = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=4_096):
            for row in batch.to_pylist():
                if max_decisions is not None and yielded >= max_decisions:
                    return
                if (
                    max_decisions_per_shard is not None
                    and yielded_in_shard >= max_decisions_per_shard
                ):
                    break
                hand = tuple(row["own_hand"])
                known_tiles = _known_tiles_from_parquet_row(row)
                yield hand, tuple(row["legal_discard_mask"]), known_tiles, row[
                    "label_index"
                ]
                yielded += 1
                yielded_in_shard += 1
            if (
                max_decisions_per_shard is not None
                and yielded_in_shard >= max_decisions_per_shard
            ):
                break


def _known_tiles_from_parquet_row(row: dict[str, Any]) -> tuple[Tile, ...]:
    known_tiles = [*row["own_hand"], *row["dora_markers"]]
    for player in row["players"]:
        known_tiles.extend(
            discard["tile"]
            for discard in player["discards"]
            if not discard["called"]
        )
        for meld in player["melds"]:
            known_tiles.extend(meld["tiles"])
    return tuple(known_tiles)


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
    parser.add_argument(
        "--sample-shard-count",
        type=int,
        help=(
            "Randomly select this many whole shard files, instead of reading "
            "every shard in the dataset."
        ),
    )
    parser.add_argument(
        "--max-decisions-per-shard",
        type=int,
        help=(
            "Cap how many decisions are read from each sampled shard, so the "
            "sample is spread across many shards instead of a few full ones."
        ),
    )
    parser.add_argument(
        "--tile-efficiency",
        action="store_true",
        help="Also evaluate the shanten-and-ukeire baseline.",
    )
    args = parser.parse_args()

    shard_paths = (
        sample_shard_paths(
            args.dataset_directory,
            shard_count=args.sample_shard_count,
            seed=args.seed,
        )
        if args.sample_shard_count is not None
        else None
    )

    fit_directory = args.frequency_fit_dataset or args.dataset_directory
    frequency_baseline = fit_most_common_baseline(fit_directory)
    metrics = evaluate_baselines(
        {
            "random_legal": RandomLegalBaseline(seed=args.seed),
            "most_common_legal": frequency_baseline,
        },
        iter_parquet_examples(
            args.dataset_directory,
            shard_paths=shard_paths,
            max_decisions=args.max_decisions,
            max_decisions_per_shard=args.max_decisions_per_shard,
        ),
    )
    if args.tile_efficiency:
        metrics["tile_efficiency"] = evaluate_tile_efficiency_baseline(
            args.dataset_directory,
            shard_paths=shard_paths,
            max_decisions=args.max_decisions,
            max_decisions_per_shard=args.max_decisions_per_shard,
        )
    same_dataset = fit_directory.resolve() == args.dataset_directory.resolve()
    report = {
        "evaluation_dataset": str(args.dataset_directory),
        "frequency_fit_dataset": str(fit_directory),
        "sample_shard_count": args.sample_shard_count,
        "max_decisions_per_shard": args.max_decisions_per_shard,
        "sampled_shard_count": len(shard_paths) if shard_paths is not None else None,
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
