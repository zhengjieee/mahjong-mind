import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from mahjong_mind.dataset.discard_decisions import iter_discard_decisions
from mahjong_mind.dataset.parquet_dataset import build_decision_dataset
from mahjong_mind.mjai.events import (
    DahaiEvent,
    EndGameEvent,
    EndKyokuEvent,
    MjaiEvent,
    StartGameEvent,
    StartKyokuEvent,
    TsumoEvent,
)
from mahjong_mind.mjai.parser import ParsedEvent


def match_events(*, example: bool = False) -> list[MjaiEvent]:
    return [
        StartGameEvent(
            type="start_game",
            names=("EXAMPLE A", "B", "C", "D") if example else ("A", "B", "C", "D"),
            kyoku_first=0,
            aka_flag=True,
        ),
        StartKyokuEvent(
            type="start_kyoku",
            bakaze="E",
            dora_marker="9s",
            kyoku=1,
            honba=0,
            kyotaku=0,
            oya=0,
            scores=(25000, 25000, 25000, 25000),
            tehais=(
                (
                    "4m",
                    "7m",
                    "9m",
                    "1s",
                    "3s",
                    "3s",
                    "3s",
                    "7s",
                    "8s",
                    "W",
                    "N",
                    "P",
                    "F",
                ),
                (
                    "1m",
                    "3m",
                    "1p",
                    "3p",
                    "4p",
                    "7p",
                    "8p",
                    "9p",
                    "6s",
                    "6s",
                    "8s",
                    "P",
                    "P",
                ),
                (
                    "1m",
                    "2m",
                    "2m",
                    "2m",
                    "6m",
                    "6m",
                    "2p",
                    "4p",
                    "5p",
                    "8s",
                    "E",
                    "P",
                    "C",
                ),
                (
                    "3m",
                    "3m",
                    "4m",
                    "4m",
                    "7m",
                    "8m",
                    "8m",
                    "9m",
                    "6p",
                    "9p",
                    "1s",
                    "5s",
                    "S",
                ),
            ),
        ),
        TsumoEvent(type="tsumo", actor=0, pai="2s"),
        DahaiEvent(type="dahai", actor=0, pai="2s", tsumogiri=True),
        EndKyokuEvent(type="end_kyoku"),
        EndGameEvent(type="end_game"),
    ]


def write_match(path: Path, events: list[MjaiEvent]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8") as stream:
        for event in events:
            stream.write(event.model_dump_json())
            stream.write("\n")


def test_captures_observable_state_before_the_historical_discard() -> None:
    match_id = "2009010100gm-test"
    parsed = [
        ParsedEvent(match_id=match_id, event_index=index, event=event)
        for index, event in enumerate(match_events())
    ]

    decisions = list(iter_discard_decisions(parsed))

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.decision_id == f"{match_id}:3"
    assert decision.hand_id == f"{match_id}:0"
    assert decision.actual_discard == "2s"
    assert decision.actual_tsumogiri is True
    assert decision.observation.own_last_draw == "2s"
    assert len(decision.observation.own_hand) == 14
    assert decision.legal_discard_mask[decision.label_index] is True
    assert not hasattr(decision.observation.players[1], "concealed_tiles")


def test_builds_reproducible_parquet_and_excludes_example_matches(
    tmp_path: Path,
) -> None:
    real_path = tmp_path / "2009010100gm-real.mjson"
    example_path = tmp_path / "2009010100gm-example.mjson"
    write_match(real_path, match_events())
    write_match(example_path, match_events(example=True))

    first = build_decision_dataset(
        [real_path, example_path], tmp_path / "first", shard_size=1
    )
    second = build_decision_dataset(
        [example_path, real_path], tmp_path / "second", shard_size=1
    )

    assert first.match_count == 1
    assert first.decision_count == 1
    assert first.dataset_sha256 == second.dataset_sha256

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["excluded_matches"] == [example_path.name]
    assert manifest["validation"]["illegal_labels"] == 0

    shard_path = first.output_directory / manifest["shards"][0]["path"]
    table = pq.ParquetFile(shard_path).read()
    assert table.num_rows == 1
    assert "names" not in table.schema.names
    assert "ura_markers" not in table.schema.names
    assert "actual_discard" in table.schema.names
    assert table.column("own_last_draw").to_pylist() == ["2s"]

    player_type = table.schema.field("players").type.value_type
    assert "concealed_tiles" not in [field.name for field in player_type]
    assert "last_draw" not in [field.name for field in player_type]
