import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from mahjong_mind.modelling.baseline_predictions import sample_shard_paths
from mahjong_mind.modelling.metrics_evaluation import (
    ACTION_COUNT,
    PolicyPrediction,
    RankingMetrics,
    RankingMetricsAccumulator,
)
from mahjong_mind.modelling.model_input_encoding import (
    FEATURE_NAMES,
    MODEL_INPUT_VERSION,
    encode_parquet_row,
)

INPUT_DIM = len(FEATURE_NAMES)

_PARQUET_COLUMNS = [
    "actor",
    "dealer",
    "players",
    "scores",
    "aka_flag",
    "honba",
    "kyotaku",
    "draws_remaining",
    "actor_turn_index",
    "bakaze",
    "seat_wind",
    "kyoku",
    "own_hand",
    "own_last_draw",
    "dora_markers",
    "legal_discard_mask",
    "label_index",
]


class MlpBaselineError(ValueError):
    """Raised when the MLP baseline cannot be trained safely."""


class DiscardMLP(nn.Module):
    """Feed-forward discard policy over the shared dense observation encoding."""

    def __init__(self, *, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ACTION_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def mask_illegal_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Replace illegal-action logits with a large negative value before softmax."""
    return logits.masked_fill(~legal_mask, -1e9)


_NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    """Per-feature mean and standard deviation, used to standardize inputs.

    Raw features mix wildly different scales (0/1 one-hots next to scores in
    the tens of thousands), which destabilizes early training. Standardizing
    to zero mean and unit variance keeps every feature on a comparable scale.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]


def compute_feature_statistics(
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
) -> FeatureStatistics:
    """Compute per-feature mean and standard deviation over a Parquet dataset."""
    dataset = ParquetDiscardDataset(
        dataset_directory,
        shard_paths=shard_paths,
        max_decisions=max_decisions,
        max_decisions_per_shard=max_decisions_per_shard,
    )
    total = torch.zeros(INPUT_DIM, dtype=torch.float64)
    total_squared = torch.zeros(INPUT_DIM, dtype=torch.float64)
    count = 0
    for features, _legal_mask, _label_index in dataset:
        values = features.to(torch.float64)
        total += values
        total_squared += values * values
        count += 1
    if count == 0:
        raise MlpBaselineError("No examples were found to compute feature statistics")

    mean = total / count
    variance = torch.clamp((total_squared / count) - mean * mean, min=0.0)
    std = torch.sqrt(variance)
    return FeatureStatistics(mean=tuple(mean.tolist()), std=tuple(std.tolist()))


def _normalization_tensors(
    statistics: FeatureStatistics,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(statistics.mean, dtype=torch.float32)
    std = torch.tensor(statistics.std, dtype=torch.float32) + _NORMALIZATION_EPSILON
    return mean, std


_PROBABILITY_FLOOR = 1e-12


def _mlp_prediction(
    logits: torch.Tensor, legal_mask: tuple[bool, ...]
) -> PolicyPrediction:
    """Convert one row of legal-masked logits into a PolicyPrediction.

    Softmax is computed in float64 so the resulting probabilities sum to 1
    within the tight tolerance RankingMetricsAccumulator requires; illegal
    actions carry logit -1e9 and underflow to exactly 0.0 probability. An
    undertrained model can still be confidently wrong enough that a *legal*
    action's probability underflows to exactly 0.0 too, which would make its
    cross-entropy undefined if that happens to be the historical label. Every
    legal action is floored to a tiny positive probability and renormalized,
    the same guarantee the smoothed non-learned baselines already provide.
    """
    probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
    legal_tensor = torch.tensor(legal_mask, dtype=torch.bool)
    floored = torch.where(
        legal_tensor,
        torch.clamp(probabilities, min=_PROBABILITY_FLOOR),
        probabilities,
    )
    normalized = (floored / floored.sum()).tolist()
    legal_actions = [action for action, is_legal in enumerate(legal_mask) if is_legal]
    ranked_actions = tuple(sorted(legal_actions, key=lambda action: -normalized[action]))
    return PolicyPrediction(
        ranked_actions=ranked_actions, probabilities=tuple(normalized)
    )


def _evaluate_model(
    model: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None,
    max_decisions: int | None,
    max_decisions_per_shard: int | None,
    batch_size: int,
) -> RankingMetrics:
    dataset = ParquetDiscardDataset(
        dataset_directory,
        shard_paths=shard_paths,
        max_decisions=max_decisions,
        max_decisions_per_shard=max_decisions_per_shard,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    accumulator = RankingMetricsAccumulator()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for features, legal_mask, labels in loader:
            normalized_features = (features - mean) / std
            logits = mask_illegal_logits(model(normalized_features), legal_mask)
            for row in range(logits.shape[0]):
                row_legal_mask = tuple(legal_mask[row].tolist())
                prediction = _mlp_prediction(logits[row], row_legal_mask)
                accumulator.update(prediction, int(labels[row]), row_legal_mask)
    if was_training:
        model.train()
    return accumulator.compute()


def load_checkpoint(checkpoint_path: Path) -> tuple[DiscardMLP, torch.Tensor, torch.Tensor]:
    """Load a saved checkpoint's model weights and normalization statistics."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = DiscardMLP(
        hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    statistics = FeatureStatistics(
        mean=tuple(checkpoint["feature_mean"]), std=tuple(checkpoint["feature_std"])
    )
    mean, std = _normalization_tensors(statistics)
    return model, mean, std


def evaluate_mlp_checkpoint(
    checkpoint_path: Path,
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
    batch_size: int = 256,
) -> RankingMetrics:
    """Evaluate one saved checkpoint's ranking metrics on a Parquet dataset."""
    model, mean, std = load_checkpoint(checkpoint_path)
    return _evaluate_model(
        model,
        mean,
        std,
        dataset_directory,
        shard_paths=shard_paths,
        max_decisions=max_decisions,
        max_decisions_per_shard=max_decisions_per_shard,
        batch_size=batch_size,
    )


class ParquetDiscardDataset(IterableDataset):
    """Streams encoded discard-decision examples from Parquet shards."""

    def __init__(
        self,
        dataset_directory: Path,
        *,
        shard_paths: Sequence[Path] | None = None,
        max_decisions: int | None = None,
        max_decisions_per_shard: int | None = None,
    ) -> None:
        self._dataset_directory = dataset_directory
        self._shard_paths = shard_paths
        self._max_decisions = max_decisions
        self._max_decisions_per_shard = max_decisions_per_shard

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        resolved_shard_paths = (
            tuple(self._shard_paths)
            if self._shard_paths is not None
            else tuple(
                sorted(self._dataset_directory.glob("source_year=*/part-*.parquet"))
            )
        )
        if not resolved_shard_paths:
            raise MlpBaselineError(f"No Parquet shards found in {self._dataset_directory}")

        yielded = 0
        for path in resolved_shard_paths:
            yielded_in_shard = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=_PARQUET_COLUMNS, batch_size=4_096):
                for row in batch.to_pylist():
                    if self._max_decisions is not None and yielded >= self._max_decisions:
                        return
                    if (
                        self._max_decisions_per_shard is not None
                        and yielded_in_shard >= self._max_decisions_per_shard
                    ):
                        break
                    example = encode_parquet_row(row)
                    features = torch.tensor(
                        example.model_input.features, dtype=torch.float32
                    )
                    legal_mask = torch.tensor(
                        example.model_input.legal_discard_mask, dtype=torch.bool
                    )
                    yield features, legal_mask, example.label_index
                    yielded += 1
                    yielded_in_shard += 1
                if (
                    self._max_decisions_per_shard is not None
                    and yielded_in_shard >= self._max_decisions_per_shard
                ):
                    break


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary of one training run.

    validation_metrics/best_epoch/best_checkpoint_path are populated only when
    a validation set was given; otherwise no checkpoint has been chosen over
    any other, since nothing evaluated the model against held-out data.
    """

    epochs: int
    epoch_losses: tuple[float, ...]
    final_loss: float
    examples_seen: int
    checkpoint_paths: tuple[Path, ...]
    validation_metrics: tuple[RankingMetrics, ...] | None
    best_epoch: int | None
    best_checkpoint_path: Path | None


def train_mlp(
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
    validation_dataset_directory: Path | None = None,
    validation_shard_paths: Sequence[Path] | None = None,
    validation_max_decisions: int | None = None,
    validation_max_decisions_per_shard: int | None = None,
) -> TrainingResult:
    """Fit the MLP discard policy on a Parquet decision dataset.

    Without validation_dataset_directory, this only fits the model: it saves
    one checkpoint per epoch but never picks a "best" one, since nothing
    evaluated the model against held-out data. With it, the model is scored
    against that separate dataset after every epoch (early stopping's
    selection signal), and the epoch with the highest validation Top-1
    accuracy is reported as best. checkpoint_dir is required whenever
    validation_dataset_directory is given, since selecting a best epoch is
    only meaningful if each epoch's weights were actually saved.
    """
    if epochs < 1:
        raise MlpBaselineError("epochs must be at least 1")
    if validation_dataset_directory is not None and checkpoint_dir is None:
        raise MlpBaselineError(
            "checkpoint_dir is required when validation_dataset_directory is given"
        )

    statistics = compute_feature_statistics(
        dataset_directory,
        shard_paths=shard_paths,
        max_decisions=max_decisions,
        max_decisions_per_shard=max_decisions_per_shard,
    )
    mean, std = _normalization_tensors(statistics)

    torch.manual_seed(seed)
    model = DiscardMLP(hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    checkpoint_paths: list[Path] = []
    validation_metrics: list[RankingMetrics] = []
    examples_seen = 0
    for epoch in range(1, epochs + 1):
        dataset = ParquetDiscardDataset(
            dataset_directory,
            shard_paths=shard_paths,
            max_decisions=max_decisions,
            max_decisions_per_shard=max_decisions_per_shard,
        )
        loader = DataLoader(dataset, batch_size=batch_size)
        total_loss = 0.0
        batch_count = 0
        model.train()
        for features, legal_mask, labels in loader:
            optimizer.zero_grad()
            normalized_features = (features - mean) / std
            logits = mask_illegal_logits(model(normalized_features), legal_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            examples_seen += features.shape[0]
        if batch_count == 0:
            raise MlpBaselineError("No training examples were found")
        epoch_losses.append(total_loss / batch_count)

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_checkpoint_path = checkpoint_dir / f"epoch-{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "feature_names": FEATURE_NAMES,
                    "model_input_version": MODEL_INPUT_VERSION,
                    "feature_mean": statistics.mean,
                    "feature_std": statistics.std,
                },
                epoch_checkpoint_path,
            )
            checkpoint_paths.append(epoch_checkpoint_path)

        if validation_dataset_directory is not None:
            validation_metrics.append(
                _evaluate_model(
                    model,
                    mean,
                    std,
                    validation_dataset_directory,
                    shard_paths=validation_shard_paths,
                    max_decisions=validation_max_decisions,
                    max_decisions_per_shard=validation_max_decisions_per_shard,
                    batch_size=batch_size,
                )
            )

    best_epoch: int | None = None
    best_checkpoint_path: Path | None = None
    if validation_metrics:
        best_epoch = 1 + max(
            range(len(validation_metrics)),
            key=lambda index: validation_metrics[index].top_1_accuracy,
        )
        best_checkpoint_path = checkpoint_paths[best_epoch - 1]

    return TrainingResult(
        epochs=epochs,
        epoch_losses=tuple(epoch_losses),
        final_loss=epoch_losses[-1],
        examples_seen=examples_seen,
        checkpoint_paths=tuple(checkpoint_paths),
        validation_metrics=tuple(validation_metrics) if validation_metrics else None,
        best_epoch=best_epoch,
        best_checkpoint_path=best_checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the dense MLP discard baseline on a Parquet dataset, with "
            "optional early stopping against a separate validation dataset."
        )
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int)
    parser.add_argument("--max-decisions-per-shard", type=int)
    parser.add_argument("--limit", type=int, dest="max_decisions")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--validation-dataset-directory",
        type=Path,
        help="Enables early stopping: score every epoch's checkpoint against this dataset.",
    )
    parser.add_argument("--validation-sample-shard-count", type=int)
    parser.add_argument("--validation-max-decisions-per-shard", type=int)
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=1,
        help="Seed for sampling validation shards; kept separate from --seed.",
    )
    parser.add_argument(
        "--exclude-sample-shard-count",
        type=int,
        default=60,
        help=(
            "Shard count used to regenerate the final comparison sample (set A) "
            "so validation sampling never overlaps it."
        ),
    )
    parser.add_argument(
        "--exclude-sample-seed",
        type=int,
        default=0,
        help="Seed used to regenerate the final comparison sample (set A).",
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

    validation_shard_paths = None
    if (
        args.validation_dataset_directory is not None
        and args.validation_sample_shard_count is not None
    ):
        excluded = sample_shard_paths(
            args.validation_dataset_directory,
            shard_count=args.exclude_sample_shard_count,
            seed=args.exclude_sample_seed,
        )
        validation_shard_paths = sample_shard_paths(
            args.validation_dataset_directory,
            shard_count=args.validation_sample_shard_count,
            seed=args.validation_seed,
            exclude=excluded,
        )

    result = train_mlp(
        args.dataset_directory,
        shard_paths=shard_paths,
        max_decisions=args.max_decisions,
        max_decisions_per_shard=args.max_decisions_per_shard,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        validation_dataset_directory=args.validation_dataset_directory,
        validation_shard_paths=validation_shard_paths,
        validation_max_decisions_per_shard=args.validation_max_decisions_per_shard,
    )
    print(
        json.dumps(
            {
                "epochs": result.epochs,
                "epoch_losses": list(result.epoch_losses),
                "final_loss": result.final_loss,
                "examples_seen": result.examples_seen,
                "checkpoint_paths": [str(path) for path in result.checkpoint_paths],
                "validation_metrics": (
                    [asdict(metrics) for metrics in result.validation_metrics]
                    if result.validation_metrics is not None
                    else None
                ),
                "best_epoch": result.best_epoch,
                "best_checkpoint_path": (
                    str(result.best_checkpoint_path)
                    if result.best_checkpoint_path
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
