import math
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from mahjong_mind.modelling.transformer_model import (
    CONTEXT_DIM,
    SEGMENT_DISCARD_SEAT0,
    SEGMENT_DISCARD_SEAT2,
    SEGMENT_DORA,
    SEGMENT_HAND_SEAT1,
    SEGMENT_HAND_SEAT2,
    SEGMENT_MELD_SEAT1,
    SEGMENT_OWN_HAND,
    UNKNOWN_TILE_TOKEN,
    TransformerModelError,
    collate_transformer_batch,
    encode_transformer_row,
    train_transformer,
)


def _row() -> dict:
    return {
        "actor": 0,
        "dealer": 0,
        "aka_flag": False,
        "bakaze": "E",
        "seat_wind": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "scores": [25_000, 26_000, 24_000, 25_000],
        "dora_markers": ["1p"],
        "draws_remaining": 50,
        "actor_turn_index": 0,
        "own_hand": ["5m", "5m", "6m"],
        "own_last_draw": "5m",
        "players": [
            {
                "concealed_tile_count": 3,
                "discards": [
                    {"tile": "9s", "tsumogiri": True, "riichi": False, "called": False}
                ],
                "melds": [],
                "riichi": "none",
            },
            {
                "concealed_tile_count": 2,
                "discards": [],
                "melds": [
                    {
                        "type": "pon",
                        "tiles": ["7p", "7p", "7p"],
                        "called_tile": "7p",
                        "target": 0,
                    }
                ],
                "riichi": "none",
            },
            {
                "concealed_tile_count": 1,
                "discards": [
                    {"tile": "E", "tsumogiri": False, "riichi": True, "called": False}
                ],
                "melds": [],
                "riichi": "accepted",
            },
            {
                "concealed_tile_count": 0,
                "discards": [],
                "melds": [],
                "riichi": "none",
            },
        ],
        "legal_discard_mask": [tile in (4, 5) for tile in range(37)],
        "label_index": 4,
    }


def _write_synthetic_shard(path: Path, *, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row() for _ in range(row_count)]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_encode_transformer_row_builds_expected_sequence() -> None:
    example = encode_transformer_row(_row())

    # own_hand (3) + dora (1) + opponent unknown tiles (2 + 1 + 0) +
    # discards (1 + 0 + 1 + 0) + meld tiles (0 + 3 + 0 + 0) = 12
    assert len(example.tile_tokens) == 12
    assert len(example.segment_ids) == 12
    assert len(example.flags) == 12
    assert len(example.context_features) == CONTEXT_DIM
    assert example.label_index == 4

    # Only the first matching hand tile is marked as the last draw, even
    # though the hand holds two copies of "5m".
    assert example.segment_ids[0:3] == (
        SEGMENT_OWN_HAND,
        SEGMENT_OWN_HAND,
        SEGMENT_OWN_HAND,
    )
    assert example.flags[0][0] == 1.0
    assert example.flags[1][0] == 0.0
    assert example.flags[2][0] == 0.0

    assert example.segment_ids[3] == SEGMENT_DORA

    # Opponent relative seat 1 contributes 2 unknown tiles, seat 2 contributes 1.
    assert example.segment_ids[4:6] == (SEGMENT_HAND_SEAT1, SEGMENT_HAND_SEAT1)
    assert example.tile_tokens[4] == UNKNOWN_TILE_TOKEN
    assert example.segment_ids[6] == SEGMENT_HAND_SEAT2

    # Seat 0's tsumogiri discard and seat 2's riichi discard.
    discard_segments = [
        segment
        for segment in example.segment_ids
        if segment in (SEGMENT_DISCARD_SEAT0, SEGMENT_DISCARD_SEAT2)
    ]
    assert discard_segments == [SEGMENT_DISCARD_SEAT0, SEGMENT_DISCARD_SEAT2]
    discard_flags = [
        flag
        for segment, flag in zip(example.segment_ids, example.flags, strict=True)
        if segment in (SEGMENT_DISCARD_SEAT0, SEGMENT_DISCARD_SEAT2)
    ]
    assert discard_flags[0] == (0.0, 1.0, 0.0, 0.0)  # tsumogiri
    assert discard_flags[1] == (0.0, 0.0, 1.0, 0.0)  # riichi discard

    # Seat 1's pon contributes 3 meld tiles.
    meld_segments = [
        segment for segment in example.segment_ids if segment == SEGMENT_MELD_SEAT1
    ]
    assert len(meld_segments) == 3


def test_encode_transformer_row_rejects_illegal_label() -> None:
    row = _row()
    row["label_index"] = 10  # not in the legal mask

    with pytest.raises(TransformerModelError, match="not legal"):
        encode_transformer_row(row)


def test_collate_transformer_batch_pads_variable_length_sequences() -> None:
    short_example = encode_transformer_row(_row())
    long_row = _row()
    long_row["own_hand"] = ["5m", "5m", "6m", "7m"]
    long_row["legal_discard_mask"] = [tile in (4, 5, 6) for tile in range(37)]
    long_row["label_index"] = 4
    long_example = encode_transformer_row(long_row)
    assert len(long_example.tile_tokens) > len(short_example.tile_tokens)

    def _as_batch_item(example):
        return (
            torch.tensor(example.tile_tokens, dtype=torch.long),
            torch.tensor(example.segment_ids, dtype=torch.long),
            torch.tensor(example.flags, dtype=torch.float32),
            torch.tensor(example.context_features, dtype=torch.float32),
            torch.tensor(example.legal_discard_mask, dtype=torch.bool),
            example.label_index,
        )

    tokens, segments, flags, context, padding_mask, _legal_mask, labels = (
        collate_transformer_batch([_as_batch_item(short_example), _as_batch_item(long_example)])
    )

    max_length = len(long_example.tile_tokens)
    assert tokens.shape == (2, max_length)
    assert segments.shape == (2, max_length)
    assert flags.shape == (2, max_length, 4)
    assert context.shape == (2, CONTEXT_DIM)
    assert padding_mask.shape == (2, max_length)
    assert labels.tolist() == [short_example.label_index, long_example.label_index]

    padding_length = max_length - len(short_example.tile_tokens)
    assert padding_mask[0, -padding_length:].all()
    assert not padding_mask[0, : len(short_example.tile_tokens)].any()
    assert not padding_mask[1].any()


def test_train_transformer_runs_end_to_end_on_synthetic_dataset(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "2017"
    _write_synthetic_shard(
        dataset_directory / "source_year=2017" / "part-00000.parquet",
        row_count=20,
    )
    checkpoint_dir = tmp_path / "checkpoints"

    result = train_transformer(
        dataset_directory,
        epochs=2,
        batch_size=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    assert result.epochs == 2
    assert len(result.epoch_losses) == 2
    assert all(not math.isnan(loss) for loss in result.epoch_losses)
    assert result.examples_seen == 40
    assert len(result.checkpoint_paths) == 2
    assert result.validation_metrics is None

    checkpoint = torch.load(result.checkpoint_paths[-1], weights_only=False)
    assert checkpoint["d_model"] == 8
    assert "model_state_dict" in checkpoint
    assert len(checkpoint["context_mean"]) == CONTEXT_DIM
    assert len(checkpoint["context_std"]) == CONTEXT_DIM


def test_train_transformer_with_validation_picks_a_best_epoch(tmp_path: Path) -> None:
    train_directory = tmp_path / "2017"
    _write_synthetic_shard(
        train_directory / "source_year=2017" / "part-00000.parquet",
        row_count=20,
    )
    validation_directory = tmp_path / "2018"
    _write_synthetic_shard(
        validation_directory / "source_year=2018" / "part-00000.parquet",
        row_count=10,
    )
    checkpoint_dir = tmp_path / "checkpoints"

    result = train_transformer(
        train_directory,
        epochs=3,
        batch_size=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        seed=0,
        checkpoint_dir=checkpoint_dir,
        validation_dataset_directory=validation_directory,
    )

    assert len(result.checkpoint_paths) == 3
    assert result.validation_metrics is not None
    assert len(result.validation_metrics) == 3
    assert result.best_epoch in (1, 2, 3)
    assert result.best_checkpoint_path == result.checkpoint_paths[result.best_epoch - 1]


def test_train_transformer_requires_checkpoint_dir_for_validation(tmp_path: Path) -> None:
    train_directory = tmp_path / "2017"
    _write_synthetic_shard(
        train_directory / "source_year=2017" / "part-00000.parquet",
        row_count=10,
    )
    validation_directory = tmp_path / "2018"
    _write_synthetic_shard(
        validation_directory / "source_year=2018" / "part-00000.parquet",
        row_count=10,
    )

    with pytest.raises(TransformerModelError, match="checkpoint_dir is required"):
        train_transformer(
            train_directory,
            batch_size=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            dim_feedforward=16,
            validation_dataset_directory=validation_directory,
        )
