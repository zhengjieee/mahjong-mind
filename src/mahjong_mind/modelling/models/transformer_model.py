import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.modelling.baselines.baseline_predictions import sample_shard_paths
from mahjong_mind.modelling.shared.feature_normalisation import (
    FeatureStatistics,
    compute_vector_statistics,
    normalisation_tensors,
)
from mahjong_mind.modelling.shared.logits_decoding import (
    logits_to_policy_prediction,
    mask_illegal_logits,
)
from mahjong_mind.modelling.shared.metrics_evaluation import (
    ACTION_COUNT,
    RankingMetrics,
    RankingMetricsAccumulator,
)

MODEL_INPUT_VERSION = "tokenized-sequence-v1"

_TILE_TO_INDEX = {tile: index for index, tile in enumerate(DISCARD_TILE_TYPES)}
_WINDS = ("E", "S", "W", "N")
_RIICHI_STATES = ("none", "pending", "accepted")

# Token vocabulary: 0 = padding, 1 = an opponent's unidentified concealed
# tile, 2.. = the 37 discard tile types.
PAD_TILE_TOKEN = 0
UNKNOWN_TILE_TOKEN = 1
_TILE_TOKEN_OFFSET = 2
TILE_VOCAB_SIZE = _TILE_TOKEN_OFFSET + len(DISCARD_TILE_TYPES)

# Segment ids: what role a token plays. 0 is reserved for padding.
SEGMENT_PAD = 0
SEGMENT_CONTEXT = 1
SEGMENT_OWN_HAND = 2
SEGMENT_HAND_SEAT1 = 3
SEGMENT_HAND_SEAT2 = 4
SEGMENT_HAND_SEAT3 = 5
SEGMENT_DORA = 6
SEGMENT_DISCARD_SEAT0 = 7
SEGMENT_DISCARD_SEAT1 = 8
SEGMENT_DISCARD_SEAT2 = 9
SEGMENT_DISCARD_SEAT3 = 10
SEGMENT_MELD_SEAT0 = 11
SEGMENT_MELD_SEAT1 = 12
SEGMENT_MELD_SEAT2 = 13
SEGMENT_MELD_SEAT3 = 14
NUM_SEGMENTS = 15

_OPPONENT_HAND_SEGMENTS = (SEGMENT_HAND_SEAT1, SEGMENT_HAND_SEAT2, SEGMENT_HAND_SEAT3)
_DISCARD_SEGMENTS = (
    SEGMENT_DISCARD_SEAT0,
    SEGMENT_DISCARD_SEAT1,
    SEGMENT_DISCARD_SEAT2,
    SEGMENT_DISCARD_SEAT3,
)
_MELD_SEGMENTS = (
    SEGMENT_MELD_SEAT0,
    SEGMENT_MELD_SEAT1,
    SEGMENT_MELD_SEAT2,
    SEGMENT_MELD_SEAT3,
)

# Flags carried per content token: (is_last_draw, is_tsumogiri, is_riichi_discard, is_called).
FLAG_DIM = 4

# Generous bound on real hands (own hand + dora + opponents' unknown tiles +
# every discard + every meld tile); real games stay well under this.
MAX_SEQUENCE_LENGTH = 256

CONTEXT_DIM = 37

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


class TransformerModelError(ValueError):
    """Raised when a decision row cannot be tokenized or trained safely."""


@dataclass(frozen=True, slots=True)
class EncodedTransformerExample:
    """One decision as a token sequence plus a separate context vector."""

    tile_tokens: tuple[int, ...]
    segment_ids: tuple[int, ...]
    flags: tuple[tuple[float, float, float, float], ...]
    context_features: tuple[float, ...]
    legal_discard_mask: tuple[bool, ...]
    label_index: int


def _tile_token_id(tile: str) -> int:
    if tile not in _TILE_TO_INDEX:
        raise TransformerModelError(f"Unknown tile: {tile}")
    return _TILE_TO_INDEX[tile] + _TILE_TOKEN_OFFSET


def _one_hot(choices: Sequence[Any], selected: Any) -> list[float]:
    if selected not in choices:
        raise TransformerModelError(f"Invalid value {selected}, expected one of {choices}")
    return [float(choice == selected) for choice in choices]


def _context_features(row: dict[str, Any], *, actor: int, dealer: int) -> tuple[float, ...]:
    """Scalar/categorical context that isn't naturally a tile token."""
    values: list[float] = [
        float(bool(row["aka_flag"])),
        float(row["honba"]),
        float(row["kyotaku"]),
        float(row["draws_remaining"]),
        float(row["actor_turn_index"]),
    ]
    values.extend(_one_hot(_WINDS, row["bakaze"]))
    values.extend(_one_hot(_WINDS, row["seat_wind"]))
    values.extend(_one_hot(range(1, 5), row["kyoku"]))
    values.extend(_one_hot(range(4), (dealer - actor) % 4))

    scores = row["scores"]
    for relative_seat in range(4):
        absolute_seat = (actor + relative_seat) % 4
        values.append(float(scores[absolute_seat]))

    players = row["players"]
    for relative_seat in range(4):
        absolute_seat = (actor + relative_seat) % 4
        values.extend(_one_hot(_RIICHI_STATES, players[absolute_seat]["riichi"]))

    return tuple(values)


def encode_transformer_row(row: dict[str, Any]) -> EncodedTransformerExample:
    """Convert one structured decision row into a token sequence + context vector."""
    actor = int(row["actor"])
    dealer = int(row["dealer"])
    if actor not in range(4) or dealer not in range(4):
        raise TransformerModelError("actor and dealer must be between 0 and 3")

    tokens: list[int] = []
    segments: list[int] = []
    flags: list[tuple[float, float, float, float]] = []

    own_last_draw = row.get("own_last_draw")
    marked_last_draw = False
    for tile in row["own_hand"]:
        is_last_draw = not marked_last_draw and tile == own_last_draw
        marked_last_draw = marked_last_draw or is_last_draw
        tokens.append(_tile_token_id(tile))
        segments.append(SEGMENT_OWN_HAND)
        flags.append((float(is_last_draw), 0.0, 0.0, 0.0))

    for marker in row["dora_markers"]:
        tokens.append(_tile_token_id(marker))
        segments.append(SEGMENT_DORA)
        flags.append((0.0, 0.0, 0.0, 0.0))

    players = row["players"]
    for relative_seat in (1, 2, 3):
        absolute_seat = (actor + relative_seat) % 4
        concealed_count = int(players[absolute_seat]["concealed_tile_count"])
        for _ in range(concealed_count):
            tokens.append(UNKNOWN_TILE_TOKEN)
            segments.append(_OPPONENT_HAND_SEGMENTS[relative_seat - 1])
            flags.append((0.0, 0.0, 0.0, 0.0))

    for relative_seat in range(4):
        absolute_seat = (actor + relative_seat) % 4
        for discard in players[absolute_seat]["discards"]:
            tokens.append(_tile_token_id(discard["tile"]))
            segments.append(_DISCARD_SEGMENTS[relative_seat])
            flags.append(
                (
                    0.0,
                    float(discard["tsumogiri"]),
                    float(discard["riichi"]),
                    float(discard["called"]),
                )
            )

    for relative_seat in range(4):
        absolute_seat = (actor + relative_seat) % 4
        for meld in players[absolute_seat]["melds"]:
            for tile in meld["tiles"]:
                tokens.append(_tile_token_id(tile))
                segments.append(_MELD_SEGMENTS[relative_seat])
                flags.append((0.0, 0.0, 0.0, 0.0))

    if len(tokens) > MAX_SEQUENCE_LENGTH:
        raise TransformerModelError(
            f"Decision sequence length {len(tokens)} exceeds MAX_SEQUENCE_LENGTH "
            f"{MAX_SEQUENCE_LENGTH}"
        )

    legal_mask = tuple(row["legal_discard_mask"])
    label_index = int(row["label_index"])
    if label_index not in range(len(DISCARD_TILE_TYPES)) or not legal_mask[label_index]:
        raise TransformerModelError(f"label_index {label_index} is not legal")

    return EncodedTransformerExample(
        tile_tokens=tuple(tokens),
        segment_ids=tuple(segments),
        flags=tuple(flags),
        context_features=_context_features(row, actor=actor, dealer=dealer),
        legal_discard_mask=legal_mask,
        label_index=label_index,
    )


class DiscardTransformer(nn.Module):
    """Transformer discard policy over a tokenized observation sequence.

    Every content token (hand tile, dora marker, opponent's unknown tile,
    discard, meld tile) gets a tile embedding + segment embedding + position
    embedding + a projection of its flags (is_last_draw/tsumogiri/riichi/
    called). A separate context vector (scores, winds, riichi state, etc. —
    values that aren't naturally a tile) is linearly projected and prepended
    as one extra token, and its encoded output is used as the pooled
    representation for the final head, the same role a BERT-style [CLS]
    token plays.
    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.tile_embedding = nn.Embedding(
            TILE_VOCAB_SIZE, d_model, padding_idx=PAD_TILE_TOKEN
        )
        self.segment_embedding = nn.Embedding(
            NUM_SEGMENTS, d_model, padding_idx=SEGMENT_PAD
        )
        self.flag_projection = nn.Linear(FLAG_DIM, d_model, bias=False)
        self.position_embedding = nn.Embedding(MAX_SEQUENCE_LENGTH + 1, d_model)
        self.context_projection = nn.Linear(CONTEXT_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, ACTION_COUNT)

    def forward(
        self,
        tile_tokens: torch.Tensor,
        segment_ids: torch.Tensor,
        flags: torch.Tensor,
        context_features: torch.Tensor,
        content_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = tile_tokens.shape
        device = tile_tokens.device

        content_positions = (
            torch.arange(1, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        content_embed = (
            self.tile_embedding(tile_tokens)
            + self.segment_embedding(segment_ids)
            + self.flag_projection(flags)
            + self.position_embedding(content_positions)
        )

        context_segment = torch.full(
            (batch_size,), SEGMENT_CONTEXT, dtype=torch.long, device=device
        )
        context_position = torch.zeros((batch_size,), dtype=torch.long, device=device)
        context_token = (
            self.context_projection(context_features)
            + self.segment_embedding(context_segment)
            + self.position_embedding(context_position)
        ).unsqueeze(1)

        full_embed = torch.cat([context_token, content_embed], dim=1)
        context_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        full_padding_mask = torch.cat([context_padding, content_padding_mask], dim=1)

        encoded = self.encoder(full_embed, src_key_padding_mask=full_padding_mask)
        pooled = encoded[:, 0, :]
        return self.output_head(pooled)


class TransformerDiscardDataset(IterableDataset):
    """Streams tokenized discard-decision examples from Parquet shards."""

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

    def __iter__(
        self,
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ]:
        resolved_shard_paths = (
            tuple(self._shard_paths)
            if self._shard_paths is not None
            else tuple(
                sorted(self._dataset_directory.glob("source_year=*/part-*.parquet"))
            )
        )
        if not resolved_shard_paths:
            raise TransformerModelError(
                f"No Parquet shards found in {self._dataset_directory}"
            )

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
                    example = encode_transformer_row(row)
                    yield (
                        torch.tensor(example.tile_tokens, dtype=torch.long),
                        torch.tensor(example.segment_ids, dtype=torch.long),
                        torch.tensor(example.flags, dtype=torch.float32),
                        torch.tensor(example.context_features, dtype=torch.float32),
                        torch.tensor(example.legal_discard_mask, dtype=torch.bool),
                        example.label_index,
                    )
                    yielded += 1
                    yielded_in_shard += 1
                if (
                    self._max_decisions_per_shard is not None
                    and yielded_in_shard >= self._max_decisions_per_shard
                ):
                    break


def collate_transformer_batch(
    batch: Sequence[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length token sequences to the batch's max length."""
    tokens_list, segments_list, flags_list, context_list, legal_mask_list, label_list = zip(
        *batch, strict=True
    )
    batch_size = len(batch)
    max_length = max(tokens.shape[0] for tokens in tokens_list)

    padded_tokens = torch.full((batch_size, max_length), PAD_TILE_TOKEN, dtype=torch.long)
    padded_segments = torch.full((batch_size, max_length), SEGMENT_PAD, dtype=torch.long)
    padded_flags = torch.zeros((batch_size, max_length, FLAG_DIM), dtype=torch.float32)
    content_padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)

    for index, (tokens, segments, flags) in enumerate(
        zip(tokens_list, segments_list, flags_list, strict=True)
    ):
        length = tokens.shape[0]
        padded_tokens[index, :length] = tokens
        padded_segments[index, :length] = segments
        padded_flags[index, :length] = flags
        content_padding_mask[index, :length] = False

    return (
        padded_tokens,
        padded_segments,
        padded_flags,
        torch.stack(context_list),
        content_padding_mask,
        torch.stack(legal_mask_list),
        torch.tensor(label_list, dtype=torch.long),
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
    dataset = TransformerDiscardDataset(
        dataset_directory,
        shard_paths=shard_paths,
        max_decisions=max_decisions,
        max_decisions_per_shard=max_decisions_per_shard,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_transformer_batch)
    accumulator = RankingMetricsAccumulator()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for tokens, segments, flags, context, padding_mask, legal_mask, labels in loader:
            normalized_context = (context - mean) / std
            logits = mask_illegal_logits(
                model(tokens, segments, flags, normalized_context, padding_mask), legal_mask
            )
            for row in range(logits.shape[0]):
                row_legal_mask = tuple(legal_mask[row].tolist())
                prediction = logits_to_policy_prediction(logits[row], row_legal_mask)
                accumulator.update(prediction, int(labels[row]), row_legal_mask)
    if was_training:
        model.train()
    return accumulator.compute()


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[DiscardTransformer, torch.Tensor, torch.Tensor]:
    """Load a saved checkpoint's model weights and context normalization stats."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = DiscardTransformer(
        d_model=checkpoint["d_model"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        dim_feedforward=checkpoint["dim_feedforward"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    statistics = FeatureStatistics(
        mean=tuple(checkpoint["context_mean"]), std=tuple(checkpoint["context_std"])
    )
    mean, std = normalisation_tensors(statistics)
    return model, mean, std


def evaluate_transformer_checkpoint(
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


def train_transformer(
    dataset_directory: Path,
    *,
    shard_paths: Sequence[Path] | None = None,
    max_decisions: int | None = None,
    max_decisions_per_shard: int | None = None,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    d_model: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
    validation_dataset_directory: Path | None = None,
    validation_shard_paths: Sequence[Path] | None = None,
    validation_max_decisions: int | None = None,
    validation_max_decisions_per_shard: int | None = None,
    initial_checkpoint: Path | None = None,
    starting_epoch: int = 1,
) -> TrainingResult:
    """Fit the Transformer discard policy on a Parquet decision dataset.

    Mirrors mlp_baseline.train_mlp's structure and the same early-stopping
    contract: without validation_dataset_directory, this only fits the
    model. With it, every epoch's checkpoint is scored against that separate
    dataset, and the epoch with the highest validation Top-1 accuracy is
    reported as best. checkpoint_dir is required whenever validation is
    requested, since selecting a best epoch is only meaningful if each
    epoch's weights were actually saved.

    initial_checkpoint continues training from a previously saved checkpoint
    instead of starting fresh, reusing its weights and normalization
    statistics (skipping the statistics pass) rather than redoing already-
    completed epochs. The architecture is read from that checkpoint, so
    d_model/num_layers/num_heads/dim_feedforward/dropout are ignored in that
    case. The optimizer itself always starts fresh (its momentum/variance
    state isn't saved), a deliberate simplification. starting_epoch controls
    the epoch numbers used for this call's new checkpoint filenames and the
    returned best_epoch, so they continue rather than overwrite the earlier
    run's files; best_epoch/best_checkpoint_path only reflect this call's
    epochs, not any prior run's.
    """
    if epochs < 1:
        raise TransformerModelError("epochs must be at least 1")
    if validation_dataset_directory is not None and checkpoint_dir is None:
        raise TransformerModelError(
            "checkpoint_dir is required when validation_dataset_directory is given"
        )

    if initial_checkpoint is not None:
        previous = torch.load(initial_checkpoint, weights_only=False)
        d_model = previous["d_model"]
        num_layers = previous["num_layers"]
        num_heads = previous["num_heads"]
        dim_feedforward = previous["dim_feedforward"]
        dropout = previous["dropout"]
        statistics = FeatureStatistics(
            mean=tuple(previous["context_mean"]), std=tuple(previous["context_std"])
        )
        mean, std = normalisation_tensors(statistics)
        torch.manual_seed(seed)
        model = DiscardTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        model.load_state_dict(previous["model_state_dict"])
    else:
        stats_dataset = TransformerDiscardDataset(
            dataset_directory,
            shard_paths=shard_paths,
            max_decisions=max_decisions,
            max_decisions_per_shard=max_decisions_per_shard,
        )
        statistics = compute_vector_statistics(
            (context for _t, _s, _f, context, _l, _lbl in stats_dataset), CONTEXT_DIM
        )
        mean, std = normalisation_tensors(statistics)

        torch.manual_seed(seed)
        model = DiscardTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    checkpoint_paths: list[Path] = []
    validation_metrics: list[RankingMetrics] = []
    examples_seen = 0
    for epoch in range(starting_epoch, starting_epoch + epochs):
        dataset = TransformerDiscardDataset(
            dataset_directory,
            shard_paths=shard_paths,
            max_decisions=max_decisions,
            max_decisions_per_shard=max_decisions_per_shard,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_transformer_batch
        )
        total_loss = 0.0
        batch_count = 0
        model.train()
        for tokens, segments, flags, context, padding_mask, legal_mask, labels in loader:
            optimizer.zero_grad()
            normalized_context = (context - mean) / std
            logits = mask_illegal_logits(
                model(tokens, segments, flags, normalized_context, padding_mask), legal_mask
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            examples_seen += tokens.shape[0]
        if batch_count == 0:
            raise TransformerModelError("No training examples were found")
        epoch_losses.append(total_loss / batch_count)

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_checkpoint_path = checkpoint_dir / f"epoch-{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "d_model": d_model,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                    "model_input_version": MODEL_INPUT_VERSION,
                    "context_mean": statistics.mean,
                    "context_std": statistics.std,
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
        best_index = max(
            range(len(validation_metrics)),
            key=lambda index: validation_metrics[index].top_1_accuracy,
        )
        best_epoch = starting_epoch + best_index
        best_checkpoint_path = checkpoint_paths[best_index]

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
            "Fit the Transformer discard policy on a Parquet dataset, with "
            "optional early stopping against a separate validation dataset."
        )
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
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
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Continue training from this checkpoint instead of starting fresh.",
    )
    parser.add_argument(
        "--starting-epoch",
        type=int,
        default=1,
        help="Epoch number new checkpoints/best_epoch start counting from.",
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

    result = train_transformer(
        args.dataset_directory,
        shard_paths=shard_paths,
        max_decisions=args.max_decisions,
        max_decisions_per_shard=args.max_decisions_per_shard,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        validation_dataset_directory=args.validation_dataset_directory,
        validation_shard_paths=validation_shard_paths,
        validation_max_decisions_per_shard=args.validation_max_decisions_per_shard,
        initial_checkpoint=args.initial_checkpoint,
        starting_epoch=args.starting_epoch,
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
