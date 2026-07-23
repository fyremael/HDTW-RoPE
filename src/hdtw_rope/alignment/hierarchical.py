"""Projection-aware single- and multi-level hierarchical aligners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from hdtw_rope.alignment.costs import PairwiseCost
from hdtw_rope.alignment.soft_dtw import SoftDTWAligner
from hdtw_rope.models.projections import FeatureProjection
from hdtw_rope.types import AlignmentOutput


class HierarchicalAligner(nn.Module):
    """One hierarchy-level aligner matching the normative semantic interface."""

    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        shared_dim: int,
        cost_mode: str = "cosine",
        temperature: float = 0.1,
        horizontal_penalty: float = 0.0,
        vertical_penalty: float = 0.0,
        diagonal_penalty: float = 0.0,
        min_alignment_mass: float = 1e-6,
        differentiable_mass: bool = False,
    ) -> None:
        super().__init__()
        self.source_projection = FeatureProjection(source_dim, shared_dim)
        self.target_projection = FeatureProjection(target_dim, shared_dim)
        self.cost = PairwiseCost(cost_mode)  # type: ignore[arg-type]
        self.aligner = SoftDTWAligner(
            temperature=temperature,
            horizontal_penalty=horizontal_penalty,
            vertical_penalty=vertical_penalty,
            diagonal_penalty=diagonal_penalty,
            min_alignment_mass=min_alignment_mass,
            differentiable_mass=differentiable_mass,
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
        band: Tensor | None = None,
    ) -> AlignmentOutput:
        source_common = self.source_projection(source)
        target_common = self.target_projection(target)
        cost = self.cost(source_common, target_common, source_mask, target_mask)
        return self.aligner.forward(cost, source_mask, target_mask, band)


class MultiLevelHierarchicalAligner(nn.Module):
    """Explicit level registry; parent-derived bands are supplied by the caller."""

    def __init__(
        self,
        aligners: Mapping[str, HierarchicalAligner],
        level_order: Sequence[str],
    ) -> None:
        super().__init__()
        if set(aligners) != set(level_order):
            raise ValueError("aligner keys and level_order must match exactly")
        self.aligners = nn.ModuleDict(dict(aligners))
        self.level_order = tuple(level_order)

    def forward(
        self,
        source: Mapping[str, Tensor],
        target: Mapping[str, Tensor],
        source_masks: Mapping[str, Tensor],
        target_masks: Mapping[str, Tensor],
        bands: Mapping[str, Tensor | None] | None = None,
    ) -> dict[str, AlignmentOutput]:
        bands = bands or {}
        outputs: dict[str, AlignmentOutput] = {}
        for level in self.level_order:
            missing = [
                name
                for name, mapping in (
                    ("source", source),
                    ("target", target),
                    ("source_masks", source_masks),
                    ("target_masks", target_masks),
                )
                if level not in mapping
            ]
            if missing:
                raise KeyError(f"level {level!r} missing from {', '.join(missing)}")
            aligner = self.aligners[level]
            if not isinstance(aligner, HierarchicalAligner):
                raise TypeError(f"level {level!r} does not contain a HierarchicalAligner")
            outputs[level] = aligner.forward(
                source[level],
                target[level],
                source_masks[level],
                target_masks[level],
                bands.get(level),
            )
        return outputs
