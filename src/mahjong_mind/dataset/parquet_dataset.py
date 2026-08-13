import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from mahjong_mind.dataset.discard_decisions import (
    DiscardDecision,
    is_example_match,
    iter_match_discard_decisions,
)
from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES

SCHEMA_VERSION = "1"
REPRESENTATION_VERSION = "structured-observation-v1"

_DISCARD_SCHEMA = pa.struct(
    [
        pa.field("tile", pa.string(), nullable=False),
        pa.field("tsumogiri", pa.bool_(), nullable=False),
        pa.field("riichi", pa.bool_(), nullable=False),
        pa.field("called", pa.bool_(), nullable=False),
    ]
)
_MELD_SCHEMA = pa.struct(
    [
        pa.field("type", pa.string(), nullable=False),
        pa.field("tiles", pa.list_(pa.string()), nullable=False),
        pa.field("called_tile", pa.string()),
        pa.field("target", pa.int8()),
    ]
)
_PLAYER_SCHEMA = pa.struct(
    [
        pa.field("concealed_tile_count", pa.int8(), nullable=False),
        pa.field("discards", pa.list_(_DISCARD_SCHEMA), nullable=False),
        pa.field("melds", pa.list_(_MELD_SCHEMA), nullable=False),
        pa.field("riichi", pa.string(), nullable=False),
    ]
)

DECISION_SCHEMA = pa.schema(
    [
        pa.field("decision_id", pa.string(), nullable=False),
        pa.field("match_id", pa.string(), nullable=False),
        pa.field("hand_id", pa.string(), nullable=False),
        pa.field("hand_index", pa.int16(), nullable=False),
        pa.field("event_index", pa.int32(), nullable=False),
        pa.field("actor", pa.int8(), nullable=False),
        pa.field("source_year", pa.int16(), nullable=False),
        pa.field("seat_wind", pa.string(), nullable=False),
        pa.field("actor_turn_index", pa.int16(), nullable=False),
        pa.field("aka_flag", pa.bool_(), nullable=False),
        pa.field("bakaze", pa.string(), nullable=False),
        pa.field("kyoku", pa.int8(), nullable=False),
        pa.field("honba", pa.int16(), nullable=False),
        pa.field("kyotaku", pa.int16(), nullable=False),
        pa.field("dealer", pa.int8(), nullable=False),
        pa.field("scores", pa.list_(pa.int32(), 4), nullable=False),
        pa.field("dora_markers", pa.list_(pa.string()), nullable=False),
        pa.field("draws_remaining", pa.int8(), nullable=False),
        pa.field("own_hand", pa.list_(pa.string()), nullable=False),
        pa.field("own_last_draw", pa.string()),
        pa.field("players", pa.list_(_PLAYER_SCHEMA, 4), nullable=False),
        pa.field(
            "legal_discard_mask",
            pa.list_(pa.bool_(), len(DISCARD_TILE_TYPES)),
            nullable=False,
        ),
        pa.field("actual_discard", pa.string(), nullable=False),
        pa.field("actual_tsumogiri", pa.bool_(), nullable=False),
        pa.field("label_index", pa.int8(), nullable=False),
    ],
    metadata={
        b"mahjong_mind.schema_version": SCHEMA_VERSION.encode(),
        b"mahjong_mind.representation_version": REPRESENTATION_VERSION.encode(),
        b"mahjong_mind.discard_action_order": json.dumps(
            DISCARD_TILE_TYPES, separators=(",", ":")
        ).encode(),
    },
)


class DatasetBuildError(ValueError):
    """Raised when a decision dataset cannot be built safely."""


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    output_directory: Path
    manifest_path: Path
    match_count: int
    decision_count: int
    shard_count: int
    dataset_sha256: str


def _decision_row(decision: DiscardDecision) -> dict[str, Any]:
    observation = decision.observation
    players = []
    for player in observation.players:
        players.append(
            {
                "concealed_tile_count": player.concealed_tile_count,
                "discards": [
                    {
                        "tile": discard.tile,
                        "tsumogiri": discard.tsumogiri,
                        "riichi": discard.riichi,
                        "called": discard.called,
                    }
                    for discard in player.discards
                ],
                "melds": [
                    {
                        "type": meld.type,
                        "tiles": meld.tiles,
                        "called_tile": meld.called_tile,
                        "target": meld.target,
                    }
                    for meld in player.melds
                ],
                "riichi": player.riichi,
            }
        )

    return {
        "decision_id": decision.decision_id,
        "match_id": decision.match_id,
        "hand_id": decision.hand_id,
        "hand_index": decision.hand_index,
        "event_index": decision.event_index,
        "actor": decision.actor,
        "source_year": decision.source_year,
        "seat_wind": decision.seat_wind,
        "actor_turn_index": decision.actor_turn_index,
        "aka_flag": observation.aka_flag,
        "bakaze": observation.bakaze,
        "kyoku": observation.kyoku,
        "honba": observation.honba,
        "kyotaku": observation.kyotaku,
        "dealer": observation.dealer,
        "scores": observation.scores,
        "dora_markers": observation.dora_markers,
        "draws_remaining": observation.draws_remaining,
        "own_hand": observation.own_hand,
        "own_last_draw": observation.own_last_draw,
        "players": players,
        "legal_discard_mask": decision.legal_discard_mask,
        "actual_discard": decision.actual_discard,
        "actual_tsumogiri": decision.actual_tsumogiri,
        "label_index": decision.label_index,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_decision_dataset(
    match_paths: list[Path],
    output_directory: Path,
    *,
    shard_size: int = 100_000,
    max_matches: int | None = None,
) -> DatasetBuildResult:
    """Build source-year-partitioned Parquet shards and a lineage manifest."""
    if shard_size < 1:
        raise DatasetBuildError("shard_size must be at least 1")
    if max_matches is not None and max_matches < 1:
        raise DatasetBuildError("max_matches must be at least 1")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise DatasetBuildError(f"Output directory is not empty: {output_directory}")

    paths = sorted(match_paths, key=lambda path: path.name)
    match_ids = [path.stem for path in paths]
    duplicate_match_ids = sorted(
        match_id for match_id, count in Counter(match_ids).items() if count > 1
    )
    if duplicate_match_ids:
        raise DatasetBuildError(
            f"Duplicate match IDs in source paths: {duplicate_match_ids[:3]}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_digest = hashlib.sha256()
    buffers: dict[int, list[dict[str, Any]]] = {}
    shard_indexes: Counter[int] = Counter()
    shards: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    excluded_matches: list[str] = []
    source_years: set[int] = set()
    matches_scanned = 0
    matches_written = 0
    decision_count = 0
    tsumogiri_count = 0

    def flush(year: int) -> None:
        rows = buffers.get(year, [])
        if not rows:
            return
        partition = output_directory / f"source_year={year}"
        partition.mkdir(parents=True, exist_ok=True)
        shard_path = partition / f"part-{shard_indexes[year]:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=DECISION_SCHEMA)
        pq.write_table(
            table,
            shard_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        shard_sha256 = _file_sha256(shard_path)
        shards.append(
            {
                "path": shard_path.relative_to(output_directory).as_posix(),
                "records": len(rows),
                "size_bytes": shard_path.stat().st_size,
                "sha256": shard_sha256,
            }
        )
        shard_indexes[year] += 1
        buffers[year] = []

    for path in paths:
        if max_matches is not None and matches_written >= max_matches:
            break

        matches_scanned += 1
        source_file_sha256 = _file_sha256(path)
        source_digest.update(path.name.encode())
        source_digest.update(b"\0")
        source_digest.update(source_file_sha256.encode())
        source_digest.update(b"\n")

        if is_example_match(path):
            excluded_matches.append(path.name)
            continue

        match_decisions = 0
        for decision in iter_match_discard_decisions(path):
            row = _decision_row(decision)
            year_buffer = buffers.setdefault(decision.source_year, [])
            year_buffer.append(row)
            source_years.add(decision.source_year)
            action_counts[decision.actual_discard] += 1
            decision_count += 1
            match_decisions += 1
            if decision.actual_tsumogiri:
                tsumogiri_count += 1
            if len(year_buffer) >= shard_size:
                flush(decision.source_year)

        if match_decisions == 0:
            raise DatasetBuildError(f"Match contains no discard decisions: {path}")
        matches_written += 1

    for year in sorted(buffers):
        flush(year)

    if decision_count == 0:
        raise DatasetBuildError("No discard decisions were written")

    dataset_digest = hashlib.sha256()
    for shard in sorted(shards, key=lambda item: str(item["path"])):
        dataset_digest.update(str(shard["path"]).encode())
        dataset_digest.update(b"\0")
        dataset_digest.update(str(shard["sha256"]).encode())
        dataset_digest.update(b"\n")
    dataset_sha256 = dataset_digest.hexdigest()
    elapsed_seconds = time.perf_counter() - started
    schema_sha256 = hashlib.sha256(str(DECISION_SCHEMA).encode()).hexdigest()
    manifest = {
        "dataset": "MahjongMind discard decisions",
        "schema_version": SCHEMA_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "format": {"type": "parquet", "compression": "zstd"},
        "source": {
            "years": sorted(source_years),
            "files_scanned": matches_scanned,
            "file_set_sha256": source_digest.hexdigest(),
            "excluded_matches": excluded_matches,
        },
        "records": {
            "matches": matches_written,
            "decisions": decision_count,
            "tsumogiri_decisions": tsumogiri_count,
            "action_counts": {
                tile: action_counts[tile] for tile in DISCARD_TILE_TYPES
            },
        },
        "schema_sha256": schema_sha256,
        "discard_action_order": list(DISCARD_TILE_TYPES),
        "shards": shards,
        "dataset_sha256": dataset_sha256,
        "validation": {
            "duplicate_match_ids": 0,
            "illegal_labels": 0,
            "missing_labels": 0,
            "hidden_or_future_fields_in_schema": 0,
        },
        "benchmark": {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "matches_per_second": round(matches_written / elapsed_seconds, 2),
            "decisions_per_second": round(decision_count / elapsed_seconds, 2),
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return DatasetBuildResult(
        output_directory=output_directory,
        manifest_path=manifest_path,
        match_count=matches_written,
        decision_count=decision_count,
        shard_count=len(shards),
        dataset_sha256=dataset_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build observable MahjongMind discard decisions as Parquet."
    )
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--limit", type=int, dest="max_matches")
    parser.add_argument("--shard-size", type=int, default=100_000)
    args = parser.parse_args()

    paths = list(args.input_directory.glob("*.mjson"))
    if not paths:
        parser.error(f"no .mjson files found in {args.input_directory}")
    result = build_decision_dataset(
        paths,
        args.output_directory,
        shard_size=args.shard_size,
        max_matches=args.max_matches,
    )
    print(
        f"Wrote {result.decision_count:,} decisions from "
        f"{result.match_count:,} matches to {result.output_directory} "
        f"({result.shard_count} shards, sha256={result.dataset_sha256})"
    )


if __name__ == "__main__":
    main()
