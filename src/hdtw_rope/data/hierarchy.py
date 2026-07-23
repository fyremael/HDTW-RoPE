"""Hierarchy utilities and repeated-section semantics."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from hdtw_rope.errors import InvalidHierarchyError


def parent_intervals(
    child_parent_id: Tensor,
    parent_start: Tensor,
    parent_end: Tensor,
    *,
    child_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather parent intervals for child units.

    Inputs may be ``[N]`` or batched ``[B,N]``. Root children (parent ``-1``)
    are invalid for containment and return zeros with a false mask.
    """

    squeeze = child_parent_id.ndim == 1
    if squeeze:
        child_parent_id = child_parent_id.unsqueeze(0)
        parent_start = parent_start.unsqueeze(0)
        parent_end = parent_end.unsqueeze(0)
        if child_valid is not None:
            child_valid = child_valid.unsqueeze(0)
    if child_parent_id.ndim != 2 or parent_start.ndim != 2 or parent_end.ndim != 2:
        raise InvalidHierarchyError("hierarchy tensors must be rank 1 or 2")
    if parent_start.shape != parent_end.shape:
        raise InvalidHierarchyError("parent interval shapes differ")
    parent_count = parent_start.shape[1]
    valid = (child_parent_id >= 0) & (child_parent_id < parent_count)
    if child_valid is not None:
        valid &= child_valid
    safe_ids = child_parent_id.clamp(min=0, max=max(parent_count - 1, 0))
    lower = parent_start.gather(1, safe_ids)
    upper = parent_end.gather(1, safe_ids)
    lower = torch.where(valid, lower, torch.zeros_like(lower))
    upper = torch.where(valid, upper, torch.zeros_like(upper))
    if torch.any((upper < lower) & valid):
        raise InvalidHierarchyError("invalid parent interval")
    if squeeze:
        return lower[0], upper[0], valid[0]
    return lower, upper, valid


def section_identity_coordinates(
    type_id: Tensor,
    occurrence_id: Tensor,
    valid_mask: Tensor,
) -> Mapping[str, Tensor]:
    """Return separate structural-type and absolute-occurrence coordinates."""

    if type_id.shape != occurrence_id.shape or type_id.shape != valid_mask.shape:
        raise InvalidHierarchyError("section identity shapes differ")
    return {
        "section_type": torch.where(
            valid_mask,
            type_id.to(torch.float32),
            torch.zeros_like(type_id, dtype=torch.float32),
        ),
        "section_occurrence": torch.where(
            valid_mask,
            occurrence_id.to(torch.float32),
            torch.zeros_like(occurrence_id, dtype=torch.float32),
        ),
    }
