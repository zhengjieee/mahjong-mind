from collections.abc import Iterable
from dataclasses import dataclass

import torch

_NORMALISATION_EPSILON = 1e-6


class FeatureNormalisationError(ValueError):
    """Raised when feature statistics cannot be computed safely."""


@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    """Per-dimension mean and standard deviation, used to standardize inputs.

    Raw features often mix wildly different scales (0/1 one-hots next to
    scores in the tens of thousands), which destabilizes early training.
    Standardizing to zero mean and unit variance keeps every dimension on a
    comparable scale. Shared by any model with a fixed-size float input
    vector (the MLP's dense encoding, the Transformer's context vector).
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]


def compute_vector_statistics(
    vectors: Iterable[torch.Tensor], dim: int
) -> FeatureStatistics:
    """Compute per-dimension mean and standard deviation over float vectors."""
    total = torch.zeros(dim, dtype=torch.float64)
    total_squared = torch.zeros(dim, dtype=torch.float64)
    count = 0
    for vector in vectors:
        values = vector.to(torch.float64)
        total += values
        total_squared += values * values
        count += 1
    if count == 0:
        raise FeatureNormalisationError("No vectors were found to compute statistics")

    mean = total / count
    variance = torch.clamp((total_squared / count) - mean * mean, min=0.0)
    std = torch.sqrt(variance)
    return FeatureStatistics(mean=tuple(mean.tolist()), std=tuple(std.tolist()))


def normalisation_tensors(
    statistics: FeatureStatistics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) tensors ready for `(x - mean) / std`, std floored."""
    mean = torch.tensor(statistics.mean, dtype=torch.float32)
    std = torch.tensor(statistics.std, dtype=torch.float32) + _NORMALISATION_EPSILON
    return mean, std
