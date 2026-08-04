"""Clock normalization that retains validity separately from numeric values."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_unit_interval(values: Tensor, mask: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize each batch row to ``[0,1]`` over valid entries."""

    if values.shape != mask.shape:
        raise ValueError("values and mask shapes differ")
    if values.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("normalize_unit_interval expects [B,N] values and bool mask")
    safe_min = torch.where(mask, values, torch.full_like(values, torch.inf)).amin(
        dim=1, keepdim=True
    )
    safe_max = torch.where(mask, values, torch.full_like(values, -torch.inf)).amax(
        dim=1, keepdim=True
    )
    has_valid = mask.any(dim=1, keepdim=True)
    safe_min = torch.where(has_valid, safe_min, torch.zeros_like(safe_min))
    safe_max = torch.where(has_valid, safe_max, torch.ones_like(safe_max))
    scale = safe_max - safe_min
    normalized = torch.where(
        scale > eps,
        (values - safe_min) / scale.clamp_min(eps),
        torch.zeros_like(values),
    )
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def normalized_index(mask: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return per-sample prefix index coordinates in ``[0,1]``."""

    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("mask must be bool [B,N]")
    length = mask.sum(dim=-1).clamp_min(1)
    index = torch.arange(mask.shape[1], device=mask.device, dtype=dtype).unsqueeze(0)
    denominator = (length - 1).clamp_min(1).to(dtype).unsqueeze(1)
    values = index / denominator
    return torch.where(mask, values, torch.zeros_like(values))
