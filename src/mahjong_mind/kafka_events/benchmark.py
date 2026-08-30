"""Benchmarks for consumer performance."""

import time
from pathlib import Path

from mahjong_mind.kafka_events.consumer import GameStateConsumer
from mahjong_mind.mjai.parser import iter_mjai_events


class ConsumerBenchmark:
    """Benchmark consumer performance on real game data."""

    def __init__(self, game_file: Path, api_base_url: str = "http://localhost:8000"):
        """Initialize benchmark with a game file."""
        self.game_file = game_file
        self.api_base_url = api_base_url
        self.consumer = GameStateConsumer(api_base_url=api_base_url)
        self.consumer.http_client = None  # Don't actually call API

    def benchmark_event_parsing(self) -> dict[str, float]:
        """Measure event reconstruction overhead."""
        results = {"total_events": 0, "total_time_ms": 0.0, "avg_time_per_event_us": 0.0}

        game_id = self.game_file.stem
        start = time.perf_counter()

        for parsed in iter_mjai_events(self.game_file):
            self.consumer._reconstruct_event(
                game_id, parsed.event_index, parsed.event.model_dump()
            )
            results["total_events"] += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        results["total_time_ms"] = elapsed_ms
        results["avg_time_per_event_us"] = (elapsed_ms * 1000) / results["total_events"]

        return results

    def benchmark_state_reconstruction(self) -> dict[str, float]:
        """Measure state reconstruction cost."""
        from mahjong_mind.game_state.reconstructor import StateReconstructor

        results = {
            "total_events": 0,
            "total_time_ms": 0.0,
            "avg_time_per_event_us": 0.0,
        }

        reconstructor = StateReconstructor()
        start = time.perf_counter()

        for parsed in iter_mjai_events(self.game_file):
            reconstructor.apply(parsed)
            results["total_events"] += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        results["total_time_ms"] = elapsed_ms
        results["avg_time_per_event_us"] = (elapsed_ms * 1000) / results["total_events"]

        return results

    def benchmark_full_pipeline(self) -> dict[str, float]:
        """Measure end-to-end consumer performance (without API calls)."""
        from unittest.mock import MagicMock

        results = {
            "total_events": 0,
            "total_time_ms": 0.0,
            "throughput_events_per_sec": 0.0,
        }

        game_id = self.game_file.stem
        events_data = []

        for parsed in iter_mjai_events(self.game_file):
            events_data.append(
                {
                    "game_id": game_id,
                    "event_index": parsed.event_index,
                    "event": parsed.event.model_dump(),
                }
            )

        # Mock HTTP client to avoid actual API calls
        self.consumer.http_client = MagicMock()

        start = time.perf_counter()

        for event_data in events_data:
            self.consumer.process_event(event_data)
            results["total_events"] += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        results["total_time_ms"] = elapsed_ms
        results["throughput_events_per_sec"] = (results["total_events"] / elapsed_ms) * 1000

        return results

    def print_results(self, name: str, results: dict[str, float]) -> None:
        """Print benchmark results in a readable format."""
        print(f"\n{name}")
        print("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                if "time" in key.lower() or "us" in key.lower():
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


def run_benchmarks(game_file: Path) -> None:
    """Run all benchmarks on a game file."""
    print(f"Running benchmarks on {game_file.name}")

    benchmark = ConsumerBenchmark(game_file)

    # Benchmark 1: Event parsing
    results = benchmark.benchmark_event_parsing()
    benchmark.print_results("Event Parsing", results)

    # Benchmark 2: State reconstruction
    results = benchmark.benchmark_state_reconstruction()
    benchmark.print_results("State Reconstruction", results)

    # Benchmark 3: Full pipeline
    results = benchmark.benchmark_full_pipeline()
    benchmark.print_results("Full Pipeline (no API)", results)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m mahjong_mind.kafka_events.benchmark <game_file>")
        sys.exit(1)

    game_file = Path(sys.argv[1])
    if not game_file.exists():
        print(f"Game file not found: {game_file}")
        sys.exit(1)

    run_benchmarks(game_file)
