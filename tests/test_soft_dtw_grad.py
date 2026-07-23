from __future__ import annotations

import torch

from hdtw_rope.alignment.hard_dtw import hard_dtw
from hdtw_rope.alignment.soft_dtw import SoftDTWAligner, soft_dtw_value


def test_soft_dtw_approaches_hard_dtw() -> None:
    cost = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    hard = hard_dtw(cost.float(), mask, mask).alignment.value.double()
    soft = soft_dtw_value(cost, mask, mask, temperature=1e-3)
    torch.testing.assert_close(soft, hard, atol=1e-4, rtol=1e-4)


def test_soft_dtw_gradcheck() -> None:
    cost = torch.tensor(
        [[[0.1, 0.7, 1.2], [0.5, 0.2, 0.8], [1.0, 0.4, 0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.ones(1, 3, dtype=torch.bool)

    def function(value: torch.Tensor) -> torch.Tensor:
        return soft_dtw_value(value, mask, mask, temperature=0.2)

    assert torch.autograd.gradcheck(function, (cost,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_alignment_mass_is_nonnegative_and_normalizable() -> None:
    cost = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    output = SoftDTWAligner(temperature=0.1)(cost, mask, mask)
    assert torch.all(output.mass >= 0)
    normalized = output.mass / output.mass.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(normalized.sum(dim=-1), torch.ones(1, 2), atol=1e-6, rtol=1e-6)
