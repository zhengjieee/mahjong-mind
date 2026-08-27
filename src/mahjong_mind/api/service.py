"""FastAPI inference service for discard recommendations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mahjong_mind.game_state.legal_actions import DISCARD_TILE_TYPES
from mahjong_mind.modelling.models.transformer_model import (
    DiscardTransformer,
    encode_transformer_row,
)
from mahjong_mind.modelling.shared.feature_normalisation import (
    FeatureStatistics,
    normalisation_tensors,
)
from mahjong_mind.modelling.shared.logits_decoding import logits_to_policy_prediction

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


@app.on_event("startup")
async def startup_event() -> None:
    """Load model at startup."""
    global _service
    checkpoint_path = Path(__file__).parent.parent.parent.parent / "data" / "checkpoints" / "transformer_model" / "epoch-10.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    _service = load_service(checkpoint_path, model_version="transformer-epoch-10")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/recommend")
async def recommend(request: PlayerObservationRequest) -> DiscarRecommendation:
    """
    Rank legal discard actions for the given game state.

    Input: PlayerObservationRequest (complete observable game state)
    Output: DiscarRecommendation (top-3 discard actions with probabilities)
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
        "label_index": 0,  # Placeholder; encoder requires it but won't use it for inference
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

    with torch.no_grad():
        normalised_context = (context - _service.context_mean) / _service.context_std
        logits = _service.model(tokens, segments, flags, normalised_context, padding_mask)

    # Decode logits into a ranked prediction.
    prediction = logits_to_policy_prediction(logits[0], tuple(encoded.legal_discard_mask))

    # Extract top-3 actions.
    top_3 = sorted(
        zip(DISCARD_TILE_TYPES, prediction.probabilities),
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    return DiscarRecommendation(
        model_version=_service.model_version,
        top_3_actions=[ActionProbability(tile=tile, probability=float(prob)) for tile, prob in top_3],
        inference_ms=0.0,  # TODO: measure actual latency in Day 5-7
    )
