"""FastAPI inference service for discard recommendations."""

import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from kafka import KafkaConsumer  # type: ignore[import-untyped]
from kafka.errors import KafkaError  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from mahjong_mind.event_processing import GameEventProcessor
from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.kafka_events.replayer import HistoricalReplayer, ReplayConfig
from mahjong_mind.modelling.models.transformer_model import (
    DiscardTransformer,
    encode_transformer_row,
)
from mahjong_mind.modelling.shared.feature_normalisation import (
    FeatureStatistics,
    normalisation_tensors,
)
from mahjong_mind.modelling.shared.logits_decoding import (
    logits_to_policy_prediction,
    mask_illegal_logits,
)

logger = logging.getLogger(__name__)

# Pydantic schemas for API requests/responses


class DiscardInfo(BaseModel):
    """One player's discard."""

    tile: str
    tsumogiri: bool = False
    riichi: bool = False
    called: bool = False


class MeldInfo(BaseModel):
    """One open meld."""

    type: str
    tiles: list[str]
    called_tile: str | None = None
    source: int | None = None


class PublicPlayerInfo(BaseModel):
    """Public information about one player from an observer's perspective."""

    concealed_tile_count: int
    discards: list[DiscardInfo]
    melds: list[MeldInfo]
    riichi: str  # "none", "pending", or "accepted"


class PlayerObservationRequest(BaseModel):
    """Request body for /recommend endpoint."""

    match_id: str
    observer: int = Field(..., ge=0, le=3)
    names: list[str] = Field(..., min_length=4, max_length=4)
    aka_flag: bool
    hand_index: int
    bakaze: str
    kyoku: int
    honba: int
    kyotaku: int
    dealer: int = Field(..., ge=0, le=3)
    scores: list[int] = Field(..., min_length=4, max_length=4)
    dora_markers: list[str]
    draws_remaining: int
    hand_ended: bool
    own_hand: list[str]
    own_last_draw: str | None = None
    players: list[PublicPlayerInfo] = Field(..., min_length=4, max_length=4)
    actor_turn_index: int
    seat_wind: str
    legal_discard_mask: list[bool]
    label_index: int | None = None


class ActionProbability(BaseModel):
    """One discard action with its probability."""

    tile: str
    probability: float


class GameEventPublish(BaseModel):
    """Request to publish a game event to WebSocket."""

    game_id: str
    event_index: int
    event: dict[str, Any]
    observation: dict[str, Any] | None = None
    recommendations: dict[str, Any] | None = None


class WatchRequest(BaseModel):
    """Request to start replaying a recorded game."""

    game_id: str
    speed: float = Field(default=1.0, gt=0, le=20)
    step_mode: bool = False


class GameControl(BaseModel):
    """Request naming the running replay to control."""

    game_id: str


class DiscarRecommendation(BaseModel):
    """Response from /recommend endpoint."""

    model_version: str
    top_3_actions: list[ActionProbability]
    inference_ms: float


# Service state


@dataclass(frozen=True)
class InferenceService:
    """Holds model and normalisation stats."""

    model: DiscardTransformer
    context_mean: torch.Tensor
    context_std: torch.Tensor
    model_version: str


_service: InferenceService | None = None


class GameEventBroadcaster:
    """Routes game events to the WebSocket clients watching each game."""

    # A game with no events for this long is no longer considered live.
    LIVE_GAME_TIMEOUT_SECONDS = 120.0

    def __init__(self) -> None:
        """Initialize broadcaster."""
        # Each connection maps to the game_id it watches, or None for all games.
        self.connections: dict[WebSocket, str | None] = {}
        # game_id -> {"event_count": int, "last_event": float, "last_type": str}
        self.live_games: dict[str, dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection, initially watching all games."""
        await websocket.accept()
        self.connections[websocket] = None
        logger.info(f"Client connected. Total connections: {len(self.connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        self.connections.pop(websocket, None)
        logger.info(f"Client disconnected. Total connections: {len(self.connections)}")

    def subscribe(self, websocket: WebSocket, game_id: str | None) -> None:
        """Point one connection at a single game, or at all games if None."""
        if websocket in self.connections:
            self.connections[websocket] = game_id
            logger.info(f"Client now watching: {game_id or 'all games'}")

    def record_activity(self, game_id: str, event_type: str) -> None:
        """Note that a game produced an event, so it counts as live."""
        entry = self.live_games.setdefault(game_id, {"event_count": 0})
        entry["event_count"] += 1
        entry["last_event"] = time.time()
        entry["last_type"] = event_type

    def forget_game(self, game_id: str) -> None:
        """Drop a game from the playing list the moment it stops.

        The timeout below is only a fallback for a producer that dies without
        saying so; a game we stopped ourselves should disappear at once.
        """
        self.live_games.pop(game_id, None)

    def active_games(self, only: set[str] | None = None) -> list[dict[str, Any]]:
        """List games that produced an event recently, newest activity first.

        `only` restricts the result to games still known to be running. Events
        stay in flight briefly after a replay stops, so without this a stopped
        game reappears the moment one of them lands.
        """
        now = time.time()
        for game_id, entry in list(self.live_games.items()):
            if now - entry["last_event"] > self.LIVE_GAME_TIMEOUT_SECONDS:
                del self.live_games[game_id]

        games = [
            {
                "game_id": game_id,
                "event_count": entry["event_count"],
                "last_type": entry["last_type"],
                "seconds_ago": round(now - entry["last_event"], 1),
            }
            for game_id, entry in self.live_games.items()
            if only is None or game_id in only
        ]
        return sorted(games, key=lambda g: g["seconds_ago"])

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send an event to the clients watching that game."""
        game_id = message.get("game_id")

        disconnected = []
        for connection, watching in list(self.connections.items()):
            if watching is not None and watching != game_id:
                continue
            try:
                await connection.send_json(message)
            except (RuntimeError, OSError, ValueError) as e:
                logger.warning(f"Error sending message to client: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            await self.disconnect(connection)


_broadcaster = GameEventBroadcaster()


def load_service(checkpoint_path: Path, model_version: str) -> InferenceService:
    """Load model checkpoint and create inference service."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = DiscardTransformer(
        d_model=checkpoint["d_model"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        dim_feedforward=checkpoint["dim_feedforward"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    statistics = FeatureStatistics(
        mean=tuple(checkpoint["context_mean"]),
        std=tuple(checkpoint["context_std"]),
    )
    mean, std = normalisation_tensors(statistics)

    return InferenceService(
        model=model,
        context_mean=mean,
        context_std=std,
        model_version=model_version,
    )


# FastAPI app


app = FastAPI(title="MahjongMind Inference Service")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model and start consuming enriched events."""
    global _service
    checkpoint_path = CHECKPOINT_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    _service = load_service(checkpoint_path, model_version="transformer-epoch-10")
    logger.info("Model loaded successfully")

    global _event_loop
    _event_loop = asyncio.get_running_loop()
    if broker_reachable():
        _start_predictions_reader(_event_loop)
    else:
        logger.info("No Kafka broker; replays will run in-process")


TOPIC_PREDICTIONS = "riichi.predictions"

KAFKA_BOOTSTRAP = os.environ.get("MAHJONG_MIND_KAFKA", "localhost:9092")

# The served checkpoint. Repo-relative by default, which is where a local
# checkout keeps it; an installed copy has no repo around it, so the image
# points this at the weights it ships with.
CHECKPOINT_PATH = Path(
    os.environ.get("MAHJONG_MIND_CHECKPOINT")
    or Path(__file__).parent.parent.parent.parent
    / "data"
    / "checkpoints"
    / "transformer_model"
    / "epoch-10.pt"
)

# The loop worker threads hand their broadcasts back to.
_event_loop: asyncio.AbstractEventLoop | None = None


def broker_reachable(timeout: float = 0.3) -> bool:
    """Whether a Kafka broker is listening, decided by a plain TCP probe.

    kafka-python retries an unreachable bootstrap server indefinitely rather
    than failing, so asking it is not a usable signal: it hangs the caller. A
    deployed instance has no broker at all and takes the in-process path.
    """
    host, _, port = KAFKA_BOOTSTRAP.partition(":")
    try:
        with socket.create_connection((host, int(port or "9092")), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _start_predictions_reader(loop: asyncio.AbstractEventLoop) -> None:
    """Read the predictions topic on a worker thread and broadcast each record.

    The viewer is one consumer of that topic among others, so this holds its own
    offset and never talks to the inference consumer directly.
    """

    def reader() -> None:
        try:
            consumer = KafkaConsumer(
                TOPIC_PREDICTIONS,
                bootstrap_servers=["localhost:9092"],
                group_id=f"viewer-{uuid.uuid4().hex[:8]}",
                auto_offset_reset="latest",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
        except (KafkaError, OSError) as e:
            # The viewer still serves pages and /recommend without live updates.
            logger.warning(f"Live updates unavailable, could not reach Kafka: {e}")
            return

        logger.info(f"Reading {TOPIC_PREDICTIONS} for live updates")
        for message in consumer:
            record = message.value
            _broadcaster.record_activity(
                record.get("game_id", ""), str((record.get("event") or {}).get("type", ""))
            )
            asyncio.run_coroutine_threadsafe(_broadcaster.broadcast(record), loop)

    threading.Thread(target=reader, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the live viewer page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Viewer not available</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


def run_inference(request: PlayerObservationRequest) -> DiscarRecommendation:
    """Rank legal discard actions for the given game state.

    Shared by the HTTP endpoint and by the in-process pipeline the service
    runs when no Kafka broker is available.
    """
    if _service is None:
        raise RuntimeError("Service not initialised; startup failed")

    # Convert request to a dict format that the encoder expects.
    # The encoder expects fields matching what encode_transformer_row uses.
    row_dict: dict[str, Any] = {
        "actor": request.observer,
        "dealer": request.dealer,
        "aka_flag": request.aka_flag,
        "honba": request.honba,
        "kyotaku": request.kyotaku,
        "draws_remaining": request.draws_remaining,
        "actor_turn_index": request.actor_turn_index,
        "bakaze": request.bakaze,
        "seat_wind": request.seat_wind,
        "kyoku": request.kyoku,
        "scores": request.scores,
        "own_hand": request.own_hand,
        "own_last_draw": request.own_last_draw,
        "dora_markers": request.dora_markers,
        "legal_discard_mask": request.legal_discard_mask,
        # The encoder requires a legal label even though inference ignores it,
        # so use the first legal action rather than a fixed index.
        "label_index": next(
            (i for i, legal in enumerate(request.legal_discard_mask) if legal), 0
        ),
        "players": [
            {
                "concealed_tile_count": p.concealed_tile_count,
                "discards": [
                    {
                        "tile": d.tile,
                        "tsumogiri": d.tsumogiri,
                        "riichi": d.riichi,
                        "called": d.called,
                    }
                    for d in p.discards
                ],
                "melds": [
                    {"tiles": m.tiles, "type": m.type}
                    for m in p.melds
                ],
                "riichi": p.riichi,
            }
            for p in request.players
        ],
    }

    # Encode the observation into tokens and context.
    encoded = encode_transformer_row(row_dict)

    # Convert to tensors and run inference.
    tokens = torch.tensor([encoded.tile_tokens], dtype=torch.long)
    segments = torch.tensor([encoded.segment_ids], dtype=torch.long)
    flags = torch.tensor([encoded.flags], dtype=torch.float32)
    context = torch.tensor([encoded.context_features], dtype=torch.float32)
    padding_mask = torch.zeros(1, len(encoded.tile_tokens), dtype=torch.bool)

    legal_mask = tuple(encoded.legal_discard_mask)

    with torch.no_grad():
        normalised_context = (context - _service.context_mean) / _service.context_std
        logits = _service.model(tokens, segments, flags, normalised_context, padding_mask)
        # Illegal actions must be masked before softmax, or they can outrank
        # the tiles the player actually holds.
        masked = mask_illegal_logits(
            logits[0], torch.tensor(legal_mask, dtype=torch.bool)
        )

    # Decode logits into a ranked prediction. ranked_actions is already sorted
    # by descending probability and contains only legal actions.
    prediction = logits_to_policy_prediction(masked, legal_mask)

    top_3 = [
        ActionProbability(
            tile=DISCARD_TILE_TYPES[action],
            probability=float(prediction.probabilities[action]),
        )
        for action in prediction.ranked_actions[:3]
    ]

    return DiscarRecommendation(
        model_version=_service.model_version,
        top_3_actions=top_3,
        inference_ms=0.0,  # TODO: measure actual latency in Day 5-7
    )


@app.post("/recommend")
async def recommend(request: PlayerObservationRequest) -> DiscarRecommendation:
    """Rank legal discard actions for the given observable game state."""
    return run_inference(request)


# In-process pipeline, used when there is no broker to carry events.


_processor: GameEventProcessor | None = None


def _local_recommendation(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Fulfil a recommendation request in this process, without HTTP."""
    return run_inference(PlayerObservationRequest(**payload)).model_dump()


def _direct_sink(event_payload: dict[str, Any]) -> None:
    """Process one replayed event here and push the result to the viewer.

    This is the deployed path. It does the same work the Kafka consumer does,
    through the same processor, with the broker and the predictions topic
    replaced by a direct call and a WebSocket push.
    """
    global _processor
    if _processor is None:
        _processor = GameEventProcessor(_local_recommendation)

    game_id = event_payload["game_id"]
    try:
        record = _processor.process(
            game_id, event_payload["event_index"], event_payload["event"]
        )
    except Exception as e:  # noqa: BLE001
        # Matches the consumer's dead-letter behaviour: one unprocessable
        # event must not end the replay. There is no DLQ here, so it is logged.
        logger.error(f"Error processing {game_id} event {event_payload['event_index']}: {e}")
        return

    _broadcaster.record_activity(game_id, str(record["event"].get("type", "")))
    if _event_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcaster.broadcast(record), _event_loop)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for live game updates.

    Clients choose what to watch by sending {"action": "subscribe",
    "game_id": "..."}; a null game_id means every game.
    """
    await _broadcaster.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                command = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(command, dict) and command.get("action") == "subscribe":
                _broadcaster.subscribe(websocket, command.get("game_id"))
    except WebSocketDisconnect:
        await _broadcaster.disconnect(websocket)


@app.post("/api/publish-event")
async def publish_event(request: GameEventPublish) -> dict[str, str]:
    """Publish a game event to WebSocket subscribers."""
    message = {
        "game_id": request.game_id,
        "event_index": request.event_index,
        "event": request.event,
        "observation": request.observation,
        "recommendations": request.recommendations,
    }
    _broadcaster.record_activity(request.game_id, str(request.event.get("type", "")))
    await _broadcaster.broadcast(message)
    return {"status": "published"}


# Games come from the full local corpus when it is present, and from the small
# bundled sample otherwise -- which is what a deployed instance serves, since
# the raw archive is far too large to ship and is not committed. Setting
# MAHJONG_MIND_GAMES_DIR forces one or the other, so the deployed behaviour can
# be exercised locally.
BUNDLED_GAMES_DIRECTORY = Path(__file__).parent / "sample_games"
_RAW_DIRECTORY = Path(__file__).parent.parent.parent.parent / "data" / "raw"
DATA_DIRECTORY = Path(
    os.environ.get("MAHJONG_MIND_GAMES_DIR")
    or (_RAW_DIRECTORY if _RAW_DIRECTORY.is_dir() else BUNDLED_GAMES_DIRECTORY)
)


# 2009 was the parser and pipeline development corpus, and no model was
# trained, validated, or tested on it. It is excluded from the viewer.
EXCLUDED_YEARS = {"2009"}


def _available_years() -> list[str]:
    """Year directories present under data/raw, excluding development corpora."""
    if not DATA_DIRECTORY.exists():
        return []
    return sorted(
        p.name
        for p in DATA_DIRECTORY.iterdir()
        if p.is_dir() and p.name.isdigit() and p.name not in EXCLUDED_YEARS
    )


def _recorded_games(year: str, limit: int) -> list[str]:
    """Up to `limit` recorded game ids for one year.

    Deliberately lazy: some year directories hold ~170,000 files, so this
    never materialises or sorts the full listing.
    """
    year_dir = DATA_DIRECTORY / year
    if not year_dir.is_dir():
        return []
    return sorted(path.stem for path in islice(year_dir.glob("*.mjson"), limit))


@app.get("/api/games")
async def list_games(year: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List games currently streaming, plus recorded games available to replay."""
    years = _available_years()
    selected = year if year in years else (years[0] if years else None)

    # Annotate each running game with how it advances, so the viewer can show
    # the right transport controls for a game it did not start itself.
    live = _broadcaster.active_games(only=set(_active_replays))
    for game in live:
        replayer = _active_replays.get(game["game_id"])
        game["step_mode"] = replayer.step_mode if replayer else False

    return {
        "live": live,
        "years": years,
        "selected_year": selected,
        "recorded": _recorded_games(selected, min(limit, 200)) if selected else [],
        "replaying": sorted(_active_replays),
    }


# Replays this service is currently running, so they can be controlled.
_active_replays: dict[str, HistoricalReplayer] = {}


def _replay_worker(
    game_id: str, game_path: Path, replayer: HistoricalReplayer, config: ReplayConfig
) -> None:
    """Replay one game into Kafka, or in-process. Runs on a worker thread."""
    try:
        replayer.connect()
        replayer.replay_game(game_path, config)
        logger.info(f"Replay of {game_id} ended after {replayer.events_sent} events")
    except (KafkaError, RuntimeError, ValueError, OSError) as e:
        logger.error(f"Replay of {game_id} failed: {e}")
    finally:
        replayer.disconnect()
        _active_replays.pop(game_id, None)
        # The game is over, so it should leave the playing list immediately
        # rather than lingering until the activity timeout expires.
        _broadcaster.forget_game(game_id)


def _running_replay(game_id: str) -> HistoricalReplayer:
    """Look up a running replay, or 404."""
    replayer = _active_replays.get(game_id)
    if replayer is None:
        raise HTTPException(status_code=404, detail=f"No replay running for {game_id}")
    return replayer


@app.post("/api/watch")
async def watch_game(request: WatchRequest) -> dict[str, Any]:
    """Start replaying a recorded game so subscribers can watch it."""
    game_id = request.game_id

    if game_id in _active_replays:
        return {"status": "already_replaying", "game_id": game_id}

    # Game files live under data/raw/<year>/, and the year prefixes the id.
    year = game_id[:4]
    game_path = DATA_DIRECTORY / year / f"{game_id}.mjson"
    if not game_path.exists():
        raise HTTPException(status_code=404, detail=f"Game not found: {game_id}")

    # Only one game plays at a time, so starting this one retires the rest.
    # Each is dropped from the registry immediately rather than waiting for
    # its worker to notice, so it leaves the playing list at once.
    for other_id in [gid for gid in _active_replays if gid != game_id]:
        _active_replays.pop(other_id).stop()
        _broadcaster.forget_game(other_id)
        logger.info(f"Stopped {other_id} to make way for {game_id}")

    # Probed per replay rather than cached, so stopping or starting the broker
    # takes effect without restarting the service.
    replayer = HistoricalReplayer(
        kafka_brokers=KAFKA_BOOTSTRAP,
        sink=None if broker_reachable() else _direct_sink,
    )
    config = ReplayConfig(
        game_id=game_id, replay_speed=request.speed, step_mode=request.step_mode
    )
    _active_replays[game_id] = replayer
    threading.Thread(
        target=_replay_worker,
        args=(game_id, game_path, replayer, config),
        daemon=True,
    ).start()

    mode = "step mode" if request.step_mode else f"{request.speed}x"
    logger.info(f"Started replay of {game_id} in {mode}")
    return {"status": "started", "game_id": game_id, "step_mode": request.step_mode}


@app.post("/api/step")
async def step_game(request: GameControl) -> dict[str, Any]:
    """Release the next event of a step-mode replay."""
    _running_replay(request.game_id).advance()
    return {"status": "stepped", "game_id": request.game_id}


@app.post("/api/pause")
async def pause_game(request: GameControl) -> dict[str, Any]:
    """Pause a timed replay."""
    _running_replay(request.game_id).pause()
    return {"status": "paused", "game_id": request.game_id}


@app.post("/api/resume")
async def resume_game(request: GameControl) -> dict[str, Any]:
    """Resume a paused replay."""
    _running_replay(request.game_id).resume()
    return {"status": "resumed", "game_id": request.game_id}


@app.post("/api/stop")
async def stop_game(request: GameControl) -> dict[str, Any]:
    """Stop a replay and free its worker thread."""
    _running_replay(request.game_id).stop()
    _broadcaster.forget_game(request.game_id)
    return {"status": "stopped", "game_id": request.game_id}


if __name__ == "__main__":
    import uvicorn

    # Without this the service's own log lines are dropped, which hides both
    # the model load and the choice between the Kafka and in-process paths.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Bind every interface on the port the platform assigns: Cloud Run and
    # most other hosts route to $PORT, and reach the container from outside.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
