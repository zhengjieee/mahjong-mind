import gzip
import json
from pathlib import Path

import pytest

from mahjong_mind.mjai.parser import (
    MjaiParseError,
    iter_mjai_events,
    validate_mjai_match,
)


def write_match(path: Path, events: list[dict[str, object]]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event))
            stream.write("\n")


def start_kyoku_event() -> dict[str, object]:
    initial_hand = [
        "1m",
        "2m",
        "3m",
        "4m",
        "5m",
        "6m",
        "7m",
        "8m",
        "9m",
        "E",
        "S",
        "P",
        "C",
    ]
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "7m",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [initial_hand] * 4,
    }


def test_valid_match_is_parsed_in_order(tmp_path: Path) -> None:
    path = tmp_path / "sample-match.mjson"
    write_match(
        path,
        [
            {
                "type": "start_game",
                "names": ["A", "B", "C", "D"],
                "kyoku_first": 0,
                "aka_flag": True,
            },
            start_kyoku_event(),
            {"type": "tsumo", "actor": 2, "pai": "5sr"},
            {"type": "dahai", "actor": 2, "pai": "8s", "tsumogiri": False},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ],
    )

    parsed = list(iter_mjai_events(path))

    assert [item.event.type for item in parsed] == [
        "start_game",
        "start_kyoku",
        "tsumo",
        "dahai",
        "end_kyoku",
        "end_game",
    ]
    assert [item.event_index for item in parsed] == [0, 1, 2, 3, 4, 5]
    assert {item.match_id for item in parsed} == {"sample-match"}
    assert validate_mjai_match(path).event_count == 6
    assert validate_mjai_match(path).hand_count == 1


@pytest.mark.parametrize(
    "invalid_event",
    [
        {"type": "tsumo", "actor": 4, "pai": "5s"},
        {"type": "unknown_event"},
    ],
    ids=["invalid-field", "unknown-type"],
)
def test_invalid_event_reports_location(
    tmp_path: Path,
    invalid_event: dict[str, object],
) -> None:
    path = tmp_path / "invalid-match.mjson"
    write_match(
        path,
        [
            {
                "type": "start_game",
                "names": ["A", "B", "C", "D"],
                "kyoku_first": 0,
                "aka_flag": True,
            },
            invalid_event,
        ],
    )

    with pytest.raises(
        MjaiParseError,
        match=r"invalid-match at event 1 \(line 2\)",
    ):
        list(iter_mjai_events(path))


def test_invalid_hand_boundary_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid-boundary.mjson"
    write_match(
        path,
        [
            {
                "type": "start_game",
                "names": ["A", "B", "C", "D"],
                "kyoku_first": 0,
                "aka_flag": True,
            },
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ],
    )

    with pytest.raises(
        MjaiParseError,
        match="end_kyoku appears without an open hand",
    ):
        validate_mjai_match(path)
