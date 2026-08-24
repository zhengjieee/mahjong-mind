import torch

from mahjong_mind.modelling.metrics_evaluation import PolicyPrediction

_PROBABILITY_FLOOR = 1e-12


def mask_illegal_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Replace illegal-action logits with a large negative value before softmax."""
    return logits.masked_fill(~legal_mask, -1e9)


def logits_to_policy_prediction(
    logits: torch.Tensor, legal_mask: tuple[bool, ...]
) -> PolicyPrediction:
    """Convert one row of legal-masked logits into a PolicyPrediction.

    Architecture-agnostic: works the same for the MLP, the Transformer, or
    any future model, since it only depends on the output logits.

    Softmax is computed in float64 so the resulting probabilities sum to 1
    within the tight tolerance RankingMetricsAccumulator requires; illegal
    actions carry logit -1e9 and underflow to exactly 0.0 probability. An
    undertrained model can still be confidently wrong enough that a *legal*
    action's probability underflows to exactly 0.0 too, which would make its
    cross-entropy undefined if that happens to be the historical label. Every
    legal action is floored to a tiny positive probability and renormalised,
    the same guarantee the smoothed non-learned baselines already provide.
    """
    probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
    legal_tensor = torch.tensor(legal_mask, dtype=torch.bool)
    floored = torch.where(
        legal_tensor,
        torch.clamp(probabilities, min=_PROBABILITY_FLOOR),
        probabilities,
    )
    normalised = (floored / floored.sum()).tolist()
    legal_actions = [action for action, is_legal in enumerate(legal_mask) if is_legal]
    ranked_actions = tuple(sorted(legal_actions, key=lambda action: -normalised[action]))
    return PolicyPrediction(
        ranked_actions=ranked_actions, probabilities=tuple(normalised)
    )
