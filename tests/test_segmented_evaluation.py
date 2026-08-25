from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from mahjong_mind.modelling.segmented_evaluation import (
    SEGMENT_DIMENSIONS,
    compute_segment_labels,
    evaluate_transformer_checkpoint_segmented,
)
from mahjong_mind.modelling.transformer_model import train_transformer


def _players(*, actor_melds: list | None = None, riichi_states: tuple[str, str, str, str] = ("none",) * 4) -> list[dict]:
    melds = actor_melds if actor_melds is not None else []
    return [
        {
            "concealed_tile_count": 2,
            "discards": [],
            "melds": melds if seat == 0 else [],
            "riichi": riichi_states[seat],
        }
        for seat in range(4)
    ]


def _row(
    *,
    draws_remaining: int = 50,
    actor: int = 0,
    dealer: int = 0,
    actor_melds: list | None = None,
    riichi_states: tuple[str, str, str, str] = ("none",) * 4,
    legal_indices: tuple[int, ...] = (4, 5),
    label_index: int = 4,
) -> dict:
    return {
        "actor": actor,
        "dealer": dealer,
        "aka_flag": False,
        "bakaze": "E",
        "seat_wind": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "scores": [25_000, 25_000, 25_000, 25_000],
        "dora_markers": ["1p"],
        "draws_remaining": draws_remaining,
        "actor_turn_index": 0,
        "own_hand": ["5m", "6m"],
        "own_last_draw": "5m",
        "players": _players(actor_melds=actor_melds, riichi_states=riichi_states),
        "legal_discard_mask": [tile in legal_indices for tile in range(37)],
        "label_index": label_index,
    }


def test_compute_segment_labels_buckets_correctly() -> None:
    early = compute_segment_labels(_row(draws_remaining=50))
    mid = compute_segment_labels(_row(draws_remaining=30))
    late = compute_segment_labels(_row(draws_remaining=10))
    assert early["game_phase"] == "early"
    assert mid["game_phase"] == "mid"
    assert late["game_phase"] == "late"

    closed = compute_segment_labels(_row(actor_melds=[]))
    open_hand = compute_segment_labels(
        _row(actor_melds=[{"type": "pon", "tiles": ["7p", "7p", "7p"], "called_tile": "7p", "target": 1}])
    )
    assert closed["hand_openness"] == "closed"
    assert open_hand["hand_openness"] == "open"

    dealer = compute_segment_labels(_row(actor=1, dealer=1))
    non_dealer = compute_segment_labels(_row(actor=1, dealer=2))
    assert dealer["dealer_status"] == "dealer"
    assert non_dealer["dealer_status"] == "non_dealer"

    no_riichi = compute_segment_labels(_row(riichi_states=("none", "none", "none", "none")))
    riichi_present = compute_segment_labels(_row(riichi_states=("none", "accepted", "none", "none")))
    assert no_riichi["opponent_riichi"] == "no_riichi"
    assert riichi_present["opponent_riichi"] == "riichi_present"

    small = compute_segment_labels(_row(legal_indices=(4, 5), label_index=4))
    large = compute_segment_labels(_row(legal_indices=tuple(range(12)), label_index=4))
    assert small["candidate_set_size"] == "1-3"
    assert large["candidate_set_size"] == "10+"


def _write_synthetic_shard(path: Path, *, row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row() for _ in range(row_count)]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_evaluate_transformer_checkpoint_segmented_covers_every_decision(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "2017"
    _write_synthetic_shard(
        dataset_directory / "source_year=2017" / "part-00000.parquet",
        row_count=20,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    training_result = train_transformer(
        dataset_directory,
        epochs=1,
        batch_size=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    result = evaluate_transformer_checkpoint_segmented(
        training_result.checkpoint_paths[-1],
        dataset_directory,
        batch_size=4,
    )

    assert result.overall.decisions == 20
    assert set(result.by_dimension) == set(SEGMENT_DIMENSIONS)
    for dimension in SEGMENT_DIMENSIONS:
        buckets = result.by_dimension[dimension]
        assert sum(metrics.decisions for metrics in buckets.values()) == 20
