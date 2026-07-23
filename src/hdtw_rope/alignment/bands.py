"""Fail-closed DTW band construction."""

from __future__ import annotations

import torch
from torch import Tensor

from hdtw_rope.validation import lengths_from_mask, require_bool_mask


def diagonal_band_mask(
    source_mask: Tensor,
    target_mask: Tensor,
    half_width: int,
) -> Tensor:
    """Construct a band around the length-normalized diagonal.

    ``half_width`` is measured in target-grid cells. Start and end cells are
    always included for nonempty samples, but no path is fabricated when the
    intervening band is disconnected.
    """

    if half_width < 0:
        raise ValueError("half_width must be nonnegative")
    source_lengths = lengths_from_mask(source_mask)
    target_lengths = lengths_from_mask(target_mask)
    batch, max_source = source_mask.shape
    max_target = target_mask.shape[1]
    source_index = torch.arange(max_source, device=source_mask.device, dtype=torch.float32)
    target_index = torch.arange(max_target, device=source_mask.device, dtype=torch.float32)
    masks: list[Tensor] = []
    for b in range(batch):
        t_len = int(source_lengths[b].item())
        s_len = int(target_lengths[b].item())
        if t_len == 0 or s_len == 0:
            masks.append(
                torch.zeros((max_source, max_target), dtype=torch.bool, device=source_mask.device)
            )
            continue
        denominator = max(t_len - 1, 1)
        center = source_index * float(max(s_len - 1, 0)) / float(denominator)
        row_band = (target_index[None, :] - center[:, None]).abs() <= float(half_width)
        row_band &= source_mask[b, :, None] & target_mask[b, None, :]
        row_band[0, 0] = True
        row_band[t_len - 1, s_len - 1] = True
        masks.append(row_band)
    return torch.stack(masks, dim=0)


def coordinate_band_mask(
    source_coordinate: Tensor,
    target_coordinate: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    half_width: float,
) -> Tensor:
    """Band by distance in a shared coarse coordinate system."""

    require_bool_mask("source_mask", source_mask, source_coordinate.shape)
    require_bool_mask("target_mask", target_mask, target_coordinate.shape)
    if half_width < 0:
        raise ValueError("half_width must be nonnegative")
    distance = (source_coordinate[:, :, None] - target_coordinate[:, None, :]).abs()
    return (distance <= half_width) & source_mask[:, :, None] & target_mask[:, None, :]


def parent_window_band_mask(
    source_parent_coordinate: Tensor,
    target_parent_coordinate: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    margin: float = 0.0,
) -> Tensor:
    """Coarse-to-fine band requiring parent-clock proximity."""

    return coordinate_band_mask(
        source_parent_coordinate,
        target_parent_coordinate,
        source_mask,
        target_mask,
        half_width=margin,
    )
