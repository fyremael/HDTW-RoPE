"""Hierarchical clock containment constraints."""

from __future__ import annotations

import torch
from torch import Tensor


def containment_loss(
    child_clock: Tensor,
    parent_lower: Tensor,
    parent_upper: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if not (
        child_clock.shape == parent_lower.shape == parent_upper.shape == valid_mask.shape
    ):
        raise ValueError("containment tensors must share a shape")
    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool")
    if torch.any((parent_upper < parent_lower) & valid_mask):
        raise ValueError("parent interval has negative width")
    violation = torch.relu(parent_lower - child_clock) + torch.relu(
        child_clock - parent_upper
    )
    return (violation * valid_mask).sum() / valid_mask.sum().clamp_min(1)


def containment_violation_rate(
    child_clock: Tensor,
    parent_lower: Tensor,
    parent_upper: Tensor,
    valid_mask: Tensor,
    *,
    tolerance: float = 1e-6,
) -> Tensor:
    violation = (
        (child_clock < parent_lower - tolerance)
        | (child_clock > parent_upper + tolerance)
    ) & valid_mask
    return violation.sum().to(torch.float32) / valid_mask.sum().clamp_min(1)
