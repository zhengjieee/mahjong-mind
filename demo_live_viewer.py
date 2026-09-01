#!/usr/bin/env python
"""Demo script for testing the live game viewer with mock events."""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def send_mock_events(api_url: str = "http://localhost:8000") -> None:
    """Send mock game events to the API service."""
    async with httpx.AsyncClient(base_url=api_url) as client:
        game_id = f"demo-game-{int(time.time())}"

        # Event 1: Game start
        print("Sending: Game start event")
        await client.post(
            "/api/publish-event",
            json={
                "game_id": game_id,
                "event_index": 0,
                "event": {
                    "type": "start_game",
                    "names": ["Player A", "Player B", "Player C", "Player D"],
                    "kyoku_first": 0,
                    "aka_flag": False,
                },
                "observation": None,
                "recommendations": None,
            },
        )
        await asyncio.sleep(1)

        # Event 2: Round start
        print("Sending: Round start event")
        await client.post(
            "/api/publish-event",
            json={
                "game_id": game_id,
                "event_index": 1,
                "event": {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "dora_marker": "5m",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "tehais": [
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                    ],
                },
                "observation": {
                    "match_id": game_id,
                    "hand_index": 0,
                    "kyoku": 1,
                    "bakaze": "E",
                    "dealer": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "names": ["Player A", "Player B", "Player C", "Player D"],
                    "aka_flag": False,
                    "dora_markers": ["5m"],
                    "draws_remaining": 70,
                    "hand_ended": False,
                    "own_hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                    "own_last_draw": None,
                    "players": [
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                    ],
                },
                "recommendations": None,
            },
        )
        await asyncio.sleep(2)

        # Event 3: First draw
        print("Sending: Draw event")
        await client.post(
            "/api/publish-event",
            json={
                "game_id": game_id,
                "event_index": 2,
                "event": {"type": "tsumo", "actor": 0, "pai": "2p"},
                "observation": {
                    "match_id": game_id,
                    "hand_index": 0,
                    "kyoku": 1,
                    "bakaze": "E",
                    "dealer": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "names": ["Player A", "Player B", "Player C", "Player D"],
                    "aka_flag": False,
                    "dora_markers": ["5m"],
                    "draws_remaining": 69,
                    "hand_ended": False,
                    "own_hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                    "own_last_draw": "2p",
                    "players": [
                        {
                            "concealed_tile_count": 14,
                            "discards": [],
                            "melds": [],
                            "riichi": "none",
                        },
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                    ],
                },
                "recommendations": {
                    "model_version": "transformer-epoch-10",
                    "top_3_actions": [
                        {"tile": "1m", "probability": 0.45},
                        {"tile": "E", "probability": 0.30},
                        {"tile": "S", "probability": 0.15},
                    ],
                    "inference_ms": 12.5,
                },
            },
        )
        await asyncio.sleep(2)

        # Event 4: Discard
        print("Sending: Discard event")
        await client.post(
            "/api/publish-event",
            json={
                "game_id": game_id,
                "event_index": 3,
                "event": {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
                "observation": {
                    "match_id": game_id,
                    "hand_index": 0,
                    "kyoku": 1,
                    "bakaze": "E",
                    "dealer": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "names": ["Player A", "Player B", "Player C", "Player D"],
                    "aka_flag": False,
                    "dora_markers": ["5m"],
                    "draws_remaining": 69,
                    "hand_ended": False,
                    "own_hand": ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1s", "E", "S"],
                    "own_last_draw": None,
                    "players": [
                        {
                            "concealed_tile_count": 13,
                            "discards": [{"tile": "1m", "tsumogiri": False, "riichi": False, "called": False}],
                            "melds": [],
                            "riichi": "none",
                        },
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                        {"concealed_tile_count": 13, "discards": [], "melds": [], "riichi": "none"},
                    ],
                },
                "recommendations": None,
            },
        )
        await asyncio.sleep(1)

        print("\nDemo events sent! Open your browser to http://localhost:8000/static/index.html")
        print("If the page is already open, you should see the events appear in real-time.")


def main() -> None:
    """Run the demo."""
    print("MahjongMind Live Viewer Demo")
    print("=" * 50)
    print("\nThis script will:")
    print("1. Send mock game events to the API service")
    print("2. Display events on the live viewer website")
    print("\nMake sure the API service is running first:")
    print("  cd /Users/zhengjie/Documents/mahjong-mind")
    print("  python -m mahjong_mind.api.service")
    print("\nThen open your browser to:")
    print("  http://localhost:8000/static/index.html")
    print("\n" + "=" * 50)

    try:
        asyncio.run(send_mock_events())
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the API service is running on http://localhost:8000")
        sys.exit(1)


if __name__ == "__main__":
    main()
