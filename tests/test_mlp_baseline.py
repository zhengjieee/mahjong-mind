import math
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from mahjong_mind.modelling.models.mlp_baseline import (
    INPUT_DIM,
    MlpBaselineError,
    compute_feature_statistics,
    train_mlp,
)
from mahjong_mind.modelling.models.model_input_encoding import FEATURE_NAMES


def _players() -> list[dict]:
    return [
        {
            "concealed_tile_count": 13,
            "discards": [],
            "melds": [],
            "riichi": "none",
        }
        for _ in range(4)
    ]


def _row(*, label_index: int) -> dict:
    return {
        "actor": 2,
        "dealer": 1,
        "aka_flag": True,
        "bakaze": "E",
        "seat_wind": "S",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "scores": [25_000, 25_000, 25_000, 25_000],
        "dora_markers": ["C"],
        "draws_remaining": 50,
        "actor_turn_index": 1,
        "own_hand": ["1m", "2m"],
        "own_last_draw": "2m",
        "players": _players(),
        "legal_discard_mask": [
            tile in (0, 1) for tile in range(37)
        ],
        "label_index": label_index,
    }


def _write_synthetic_shard(path: Path, *, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row(label_index=index % 2) for index in range(row_count)]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_compute_feature_statistics_matches_manual_calculation(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "2017"
    _write_synthetic_shard(
        dataset_directory / "source_year=2017" / "part-00000.parquet",
        row_count=10,
    )

    statistics = compute_feature_statistics(dataset_directory)

    assert len(statistics.mean) == INPUT_DIM
    assert len(statistics.std) == INPUT_DIM
    # own_hand is identical ("1m", "2m") in every synthetic row, so this
    # feature column has mean 1 and zero variance.
    index = FEATURE_NAMES.index("own_hand_count.1m")
    assert statistics.mean[index] == pytest.approx(1.0)
    assert statistics.std[index] == pytest.approx(0.0)


def test_train_mlp_runs_end_to_end_on_synthetic_dataset(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "2017"
    _write_synthetic_shard(
        dataset_directory / "source_year=2017" / "part-00000.parquet",
        row_count=20,
    )
    checkpoint_dir = tmp_path / "checkpoints"

    result = train_mlp(
        dataset_directory,
        epochs=2,
        batch_size=4,
        hidden_dim=8,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    assert result.epochs == 2
    assert len(result.epoch_losses) == 2
    assert all(not math.isnan(loss) for loss in result.epoch_losses)
    assert result.examples_seen == 40
    assert len(result.checkpoint_paths) == 2
    assert result.validation_metrics is None
    assert result.best_epoch is None
    assert result.best_checkpoint_path is None

    checkpoint = torch.load(result.checkpoint_paths[-1], weights_only=False)
    assert checkpoint["hidden_dim"] == 8
    assert "model_state_dict" in checkpoint
    assert len(checkpoint["feature_mean"]) == INPUT_DIM
    assert len(checkpoint["feature_std"]) == INPUT_DIM


def test_train_mlp_with_validation_picks_a_best_epoch(tmp_path: Path) -> None:
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

    result = train_mlp(
        train_directory,
        epochs=3,
        batch_size=4,
        hidden_dim=8,
        seed=0,
        checkpoint_dir=checkpoint_dir,
        validation_dataset_directory=validation_directory,
    )

    assert len(result.checkpoint_paths) == 3
    assert result.validation_metrics is not None
    assert len(result.validation_metrics) == 3
    assert result.best_epoch in (1, 2, 3)
    assert result.best_checkpoint_path == result.checkpoint_paths[result.best_epoch - 1]


def test_train_mlp_requires_checkpoint_dir_for_validation(tmp_path: Path) -> None:
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

    with pytest.raises(MlpBaselineError, match="checkpoint_dir is required"):
        train_mlp(
            train_directory,
            batch_size=4,
            hidden_dim=8,
            validation_dataset_directory=validation_directory,
        )
