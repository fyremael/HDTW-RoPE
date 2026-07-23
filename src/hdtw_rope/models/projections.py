"""Feature and clock projections with explicit shared-space contracts."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FeatureProjection(nn.Module):
    """Layer-normalized projection into a shared feature space."""

    def __init__(
        self, input_dim: int, output_dim: int, *, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear(self.norm(x)))


class ClockAdapter(nn.Module):
    """Optional modality adapter initialized to the identity map."""

    def __init__(self, clock_dim: int, *, learnable: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(clock_dim, clock_dim)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(clock_dim))
            self.linear.bias.zero_()
        for parameter in self.parameters():
            parameter.requires_grad = learnable

    def forward(self, clock: Tensor, mask: Tensor) -> Tensor:
        if clock.shape != mask.shape or mask.dtype is not torch.bool:
            raise ValueError("clock adapter expects matching clock and bool mask")
        adapted = self.linear(clock.float())
        return torch.where(mask, adapted, torch.zeros_like(adapted))
