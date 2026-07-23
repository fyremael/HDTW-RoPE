from __future__ import annotations

import torch

from hdtw_rope.clocks.hierarchy import containment_loss, containment_violation_rate
from hdtw_rope.data.hierarchy import parent_intervals


def test_hierarchy_containment_zero_and_positive() -> None:
    child = torch.tensor([[0.2, 0.7]])
    lower = torch.tensor([[0.0, 0.5]])
    upper = torch.tensor([[0.4, 0.9]])
    mask = torch.ones_like(child, dtype=torch.bool)
    assert containment_loss(child, lower, upper, mask).item() == 0.0
    invalid = child.clone()
    invalid[0, 1] = 1.1
    assert containment_loss(invalid, lower, upper, mask).item() > 0.0
    assert containment_violation_rate(invalid, lower, upper, mask).item() == 0.5


def test_parent_interval_gather() -> None:
    lower, upper, valid = parent_intervals(torch.tensor([0, 0, 1]), torch.tensor([0.0, 0.5]), torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(lower, torch.tensor([0.0, 0.0, 0.5]))
    torch.testing.assert_close(upper, torch.tensor([0.5, 0.5, 1.0]))
    assert valid.all()
