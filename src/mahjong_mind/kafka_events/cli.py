"""CLI for running game replays."""

import argparse
import sys
from pathlib import Path

from mahjong_mind.kafka_events.replayer import HistoricalReplayer, ReplayConfig


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Replay MJAI game logs to Kafka event stream",
    )
    parser.add_argument(
        "game_id",
        help="Game ID to replay (e.g., 2018010100gm-00a9-0000-0d318262)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "data" / "raw",
        help="Path to raw data directory containing .mjson files",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Filter games by year (optional)",
    )
    parser.add_argument(
        "--kafka-brokers",
        default="localhost:9092",
        help="Kafka bootstrap servers (comma-separated)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (1.0 = real-time, 2.0 = 2x speed)",
    )
    parser.add_argument(
        "--start-event",
        type=int,
        default=0,
        help="Start from event index (for resuming)",
    )

    args = parser.parse_args()

    # Find the game file
    if args.year:
        game_dir = args.data_dir / str(args.year)
    else:
        # Try to extract year from game_id
        try:
            year = int(args.game_id[:4])
            game_dir = args.data_dir / str(year)
        except (ValueError, IndexError):
            game_dir = args.data_dir

    game_file = game_dir / f"{args.game_id}.mjson"
    if not game_file.exists():
        print(f"Error: Game file not found: {game_file}", file=sys.stderr)
        return 1

    # Connect and replay
    replayer = HistoricalReplayer(args.kafka_brokers)
    try:
        replayer.connect()
        config = ReplayConfig(
            game_id=args.game_id,
            replay_speed=args.speed,
            start_event_index=args.start_event,
        )
        replayer.replay_game(game_file, config)
        print(f"Successfully replayed {args.game_id}")
        return 0
    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        replayer.disconnect()


if __name__ == "__main__":
    sys.exit(main())
