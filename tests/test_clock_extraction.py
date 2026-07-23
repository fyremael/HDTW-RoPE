from __future__ import annotations

import pytest
import torch

from hdtw_rope.clocks.extract import LatentClockExtractor
from hdtw_rope.clocks.monotone import isotonic_projection, monotonicity_violation_count
from hdtw_rope.errors import InsufficientAlignmentMass
from hdtw_rope.types import AlignmentOutput


def _alignment(mass: torch.Tensor) -> AlignmentOutput:
    return AlignmentOutput(cost=torch.zeros_like(mass), value=torch.zeros(mass.shape[0]), mass=mass, valid_rows=mass.sum(dim=-1) > 0, valid_cols=mass.sum(dim=-2) > 0)


def test_hard_path_clock_extraction_is_monotone() -> None:
    mass = torch.zeros(1, 4, 3)
    mass[0, 0, 0] = 1
    mass[0, 1, 1] = 1
    mass[0, 2, 1] = 1
    mass[0, 3, 2] = 1
    target_coordinate = torch.tensor([[0.0, 0.5, 1.0]])
    source_mask = torch.ones(1, 4, dtype=torch.bool)
    target_mask = torch.ones(1, 3, dtype=torch.bool)
    clocks = LatentClockExtractor(["aligned_word"])({"aligned_word": _alignment(mass)}, {}, {"aligned_word": target_coordinate}, {}, {"aligned_word": target_mask}, {})
    torch.testing.assert_close(clocks.source_clock[0, :, 0], torch.tensor([0.0, 0.5, 0.5, 1.0]))
    assert monotonicity_violation_count(clocks.source_clock[..., 0], source_mask).item() == 0
    assert clocks.target_valid[..., 0].all()


def test_invalid_mass_fails_closed() -> None:
    mass = torch.zeros(1, 2, 2)
    alignment = AlignmentOutput(cost=torch.zeros_like(mass), value=torch.zeros(1), mass=mass, valid_rows=torch.ones(1, 2, dtype=torch.bool), valid_cols=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(InsufficientAlignmentMass):
        LatentClockExtractor(["aligned_word"], strict=True)({"aligned_word": alignment}, {}, {"aligned_word": torch.tensor([[0.0, 1.0]])}, {}, {"aligned_word": torch.ones(1, 2, dtype=torch.bool)}, {})


def test_isotonic_projection_removes_violations() -> None:
    values = torch.tensor([[0.0, 0.7, 0.4, 1.0]])
    mask = torch.ones_like(values, dtype=torch.bool)
    projected = isotonic_projection(values, mask)
    assert monotonicity_violation_count(projected, mask).item() == 0
    assert projected[0, 1].item() == pytest.approx(0.55)
    assert projected[0, 2].item() == pytest.approx(0.55)
