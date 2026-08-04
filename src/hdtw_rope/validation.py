"""Shared validation helpers for public tensor interfaces."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from hdtw_rope.errors import InvalidShapeError


def require_rank(name: str, tensor: Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise InvalidShapeError(f"{name} must have rank {rank}; got shape {tuple(tensor.shape)}")


def require_bool_mask(name: str, mask: Tensor, shape: Iterable[int]) -> None:
    expected = tuple(shape)
    if mask.dtype is not torch.bool:
        raise InvalidShapeError(f"{name} must be bool; got {mask.dtype}")
    if tuple(mask.shape) != expected:
        raise InvalidShapeError(f"{name} must have shape {expected}; got {tuple(mask.shape)}")


def require_floating(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise InvalidShapeError(f"{name} must be floating point; got {tensor.dtype}")


def require_finite(name: str, tensor: Tensor, mask: Tensor | None = None) -> None:
    selected = tensor if mask is None else tensor.masked_select(mask)
    if selected.numel() and not torch.isfinite(selected).all():
        raise ValueError(f"{name} contains NaN or Inf")


def lengths_from_mask(mask: Tensor) -> Tensor:
    """Return valid prefix lengths and reject masks with holes."""

    require_rank("mask", mask, 2)
    if mask.dtype is not torch.bool:
        raise InvalidShapeError("mask must be bool")
    lengths = mask.sum(dim=-1)
    positions = torch.arange(mask.shape[1], device=mask.device)
    expected = positions.unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise InvalidShapeError("sequence masks must be contiguous valid prefixes")
    return lengths
