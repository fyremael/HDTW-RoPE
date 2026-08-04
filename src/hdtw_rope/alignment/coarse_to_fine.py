"""Hierarchical coarse-to-fine band orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from hdtw_rope.alignment.bands import coordinate_band_mask


@dataclass(frozen=True)
class LevelBand:
    level: str
    mask: Tensor


def build_coarse_to_fine_bands(
    source_parent_clocks: Mapping[str, Tensor],
    target_parent_clocks: Mapping[str, Tensor],
    source_masks: Mapping[str, Tensor],
    target_masks: Mapping[str, Tensor],
    margins: Mapping[str, float],
) -> dict[str, Tensor]:
    """Build child bands from parent-level shared coordinates."""

    keys = set(source_parent_clocks) & set(target_parent_clocks) & set(margins)
    return {
        level: coordinate_band_mask(
            source_parent_clocks[level],
            target_parent_clocks[level],
            source_masks[level],
            target_masks[level],
            margins[level],
        )
        for level in sorted(keys)
    }
