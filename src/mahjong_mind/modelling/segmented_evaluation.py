from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch import nn

from mahjong_mind.modelling.models.transformer_model import (
    _PARQUET_COLUMNS,
    collate_transformer_batch,
    encode_transformer_row,
    load_checkpoint,
)
from mahjong_mind.modelling.shared.logits_decoding import (
    logits_to_policy_prediction,
    mask_illegal_logits,
)
from mahjong_mind.modelling.shared.metrics_evaluation import (
    RankingMetrics,
    RankingMetricsAccumulator,
)

SEGMENT_DIMENSIONS: tuple[str, ...] = (
    "game_phase",
    "hand_openness",
    "dealer_status",
    "opponent_riichi",
    "candidate_set_size",
)


class SegmentedEvaluationError(ValueError):
    """Raised when a segmented evaluation cannot be run safely."""


def compute_segment_labels(row: dict[str, Any]) -> dict[str, str]:
    """Derive this decision's bucket label along each analysis dimension.

    Every field used here already exists in the Parquet decision schema; no
    new data collection is needed, only grouping what's already recorded.
    """
    actor = int(row["actor"])
    dealer = int(row["dealer"])
    draws_remaining = int(row["draws_remaining"])
    players = row["players"]

    if draws_remaining >= 47:
        game_phase = "early"
    elif draws_remaining >= 24:
        game_phase = "mid"
    else:
        game_phase = "late"

    hand_openness = "open" if players[actor]["melds"] else "closed"
    dealer_status = "dealer" if actor == dealer else "non_dealer"

    opponent_riichi = (
        "riichi_present"
        if any(
            players[(actor + relative_seat) % 4]["riichi"] == "accepted"
            for relative_seat in (1, 2, 3)
        )
        else "no_riichi"
    )

    legal_count = sum(row["legal_discard_mask"])
    if legal_count <= 3:
        candidate_set_size = "1-3"
    elif legal_count <= 6:
        candidate_set_size = "4-6"
    elif legal_count <= 9:
        candidate_set_size = "7-9"
    else:
        candidate_set_size = "10+"

    return {
        "game_phase": game_phase,
        "hand_openness": hand_openness,
        "dealer_status": dealer_status,
        "opponent_riichi": opponent_riichi,
        "candidate_set_size": candidate_set_size,
    }


@dataclass(frozen=True, slots=True)
class SegmentedEvaluationResult:
    overall: RankingMetrics
    by_dimension: dict[str, dict[str, RankingMetrics]]


def _score_rows(
    model: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    rows: list[dict[str, Any]],
    overall: RankingMetricsAccumulator,
    by_dimension: dict[str, dict[str, RankingMetricsAccumulator]],
) -> None:
    """Tokenize, score, and accumulate metrics for one buffered batch of rows."""
    if not rows:
        return
    examples = [encode_transformer_row(row) for row in rows]
    segment_labels = [compute_segment_labels(row) for row in rows]
    batch_items = [
        (
            torch.tensor(example.tile_tokens, dtype=torch.long),
            torch.tensor(example.segment_ids, dtype=torch.long),
            torch.tensor(example.flags, dtype=torch.float32),
            torch.tensor(example.context_features, dtype=torch.float32),
            torch.tensor(example.legal_discard_mask, dtype=torch.bool),
            example.label_index,
        )
        for example in examples
    ]
    tokens, segment_ids, flags, context, padding_mask, legal_mask, labels = (
        collate_transformer_batch(batch_items)
    )
    with torch.no_grad():
        normalized_context = (context - mean) / std
        logits = mask_illegal_logits(
            model(tokens, segment_ids, flags, normalized_context, padding_mask), legal_mask
        )

    for row_index in range(logits.shape[0]):
        row_legal_mask = tuple(legal_mask[row_index].tolist())
        prediction = logits_to_policy_prediction(logits[row_index], row_legal_mask)
        label = int(labels[row_index])
        overall.update(prediction, label, row_legal_mask)
        for dimension, bucket in segment_labels[row_index].items():
            by_dimension[dimension][bucket].update(prediction, label, row_legal_mask)


def evaluate_transformer_checkpoint_segmented(
    checkpoint_path: Path,
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
    batch_size: int = 256,
) -> SegmentedEvaluationResult:
    """Evaluate a checkpoint overall and broken down by SEGMENT_DIMENSIONS.

    Reuses the same tokenizer, collate function, masking, and prediction
    logic as evaluate_transformer_checkpoint; the only new work is deriving
    each decision's segment labels from fields already in the Parquet row
    and updating one extra accumulator per (dimension, bucket) alongside the
    overall one. Row-by-row limit tracking mirrors TransformerDiscardDataset,
    buffering rows into batch_size chunks for model inference.
    """
    model, mean, std = load_checkpoint(checkpoint_path)
    model.eval()

    resolved_shard_paths = (
        tuple(shard_paths)
        if shard_paths is not None
        else tuple(sorted(dataset_directory.glob("source_year=*/part-*.parquet")))
    )
    if not resolved_shard_paths:
        raise SegmentedEvaluationError(f"No Parquet shards found in {dataset_directory}")

    overall = RankingMetricsAccumulator()
    by_dimension: dict[str, dict[str, RankingMetricsAccumulator]] = {
        dimension: defaultdict(RankingMetricsAccumulator) for dimension in SEGMENT_DIMENSIONS
    }

    yielded = 0
    for path in resolved_shard_paths:
        yielded_in_shard = 0
        parquet = pq.ParquetFile(path)
        buffer: list[dict[str, Any]] = []
        for batch in parquet.iter_batches(columns=_PARQUET_COLUMNS, batch_size=batch_size):
            for row in batch.to_pylist():
                if max_decisions is not None and yielded >= max_decisions:
                    _score_rows(model, mean, std, buffer, overall, by_dimension)
                    return _finalize(overall, by_dimension)
                if (
                    max_decisions_per_shard is not None
                    and yielded_in_shard >= max_decisions_per_shard
                ):
                    break
                buffer.append(row)
                yielded += 1
                yielded_in_shard += 1
                if len(buffer) >= batch_size:
                    _score_rows(model, mean, std, buffer, overall, by_dimension)
                    buffer = []
            if (
                max_decisions_per_shard is not None
                and yielded_in_shard >= max_decisions_per_shard
            ):
                break
        _score_rows(model, mean, std, buffer, overall, by_dimension)

    return _finalize(overall, by_dimension)


def _finalize(
    overall: RankingMetricsAccumulator,
    by_dimension: dict[str, dict[str, RankingMetricsAccumulator]],
) -> SegmentedEvaluationResult:
    return SegmentedEvaluationResult(
        overall=overall.compute(),
        by_dimension={
            dimension: {bucket: accumulator.compute() for bucket, accumulator in buckets.items()}
            for dimension, buckets in by_dimension.items()
        },
    )
