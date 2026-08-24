import pytest
import torch

from mahjong_mind.modelling.logits_decoding import (
    logits_to_policy_prediction,
    mask_illegal_logits,
)


def test_mask_illegal_logits_zeroes_out_illegal_probabilities() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    legal_mask = torch.tensor([[True, False, True]])

    masked = mask_illegal_logits(logits, legal_mask)
    probabilities = torch.softmax(masked, dim=-1)

    assert probabilities[0, 1].item() == 0.0
    assert probabilities[0, 0].item() > 0.0
    assert probabilities[0, 2].item() > 0.0


def test_logits_to_policy_prediction_floors_legal_actions_to_avoid_zero_probability() -> (
    None
):
    logits = torch.tensor([1000.0, -1000.0, -1e9])
    legal_mask = (True, True, False)

    prediction = logits_to_policy_prediction(logits, legal_mask)

    assert prediction.probabilities[1] > 0.0
    assert prediction.probabilities[2] == 0.0
    assert sum(prediction.probabilities) == pytest.approx(1.0)
