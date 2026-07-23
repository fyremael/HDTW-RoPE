"""Required and recommended HDTW-RoPE losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hdtw_rope.clocks.hierarchy import containment_loss
from hdtw_rope.clocks.monotone import monotonicity_loss, slope_loss, smoothness_loss
from hdtw_rope.types import AlignmentOutput, ClockOutput


def symmetric_contrastive_loss(similarity: Tensor) -> Tensor:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("paired contrastive loss requires a square [B,B] similarity matrix")
    target = torch.arange(similarity.shape[0], device=similarity.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(similarity, target)
        + torch.nn.functional.cross_entropy(similarity.transpose(0, 1), target)
    )


def soft_dtw_loss(alignments: Mapping[str, AlignmentOutput]) -> Tensor:
    if not alignments:
        raise ValueError("at least one alignment is required")
    return torch.stack([alignment.value.float().mean() for alignment in alignments.values()]).mean()


def anchor_loss(clock: Tensor, target: Tensor, mask: Tensor, *, delta: float = 1.0) -> Tensor:
    if not (clock.shape == target.shape == mask.shape):
        raise ValueError("anchor tensors must share a shape")
    element = torch.nn.functional.huber_loss(clock, target, delta=delta, reduction="none")
    return (element * mask).sum() / mask.sum().clamp_min(1)


def alignment_entropy_loss(alignment: AlignmentOutput, *, eps: float = 1e-12) -> Tensor:
    row_mass = alignment.mass.sum(dim=-1, keepdim=True)
    probabilities = alignment.mass / row_mass.clamp_min(eps)
    entropy = -(probabilities.clamp_min(eps) * probabilities.clamp_min(eps).log()).sum(dim=-1)
    return (entropy * alignment.valid_rows).sum() / alignment.valid_rows.sum().clamp_min(1)


def _linear_sample(values: Tensor, coordinates: Tensor) -> Tensor:
    if values.ndim != 2 or coordinates.ndim != 2 or values.shape[0] != coordinates.shape[0]:
        raise ValueError("linear sampling expects [B,N] and [B,M]")
    length = values.shape[1]
    scaled = coordinates.clamp(0.0, 1.0) * max(length - 1, 1)
    left = scaled.floor().long().clamp(0, length - 1)
    right = (left + 1).clamp(0, length - 1)
    weight = scaled - left.to(scaled.dtype)
    return values.gather(1, left) * (1.0 - weight) + values.gather(1, right) * weight


def cycle_consistency_loss(
    source_to_target: Tensor, target_to_source: Tensor, source_mask: Tensor, target_mask: Tensor
) -> Tensor:
    reconstructed_source = _linear_sample(target_to_source, source_to_target)
    reconstructed_target = _linear_sample(source_to_target, target_to_source)
    source_index = torch.arange(
        source_to_target.shape[1], device=source_to_target.device
    ).float() / max(source_to_target.shape[1] - 1, 1)
    target_index = torch.arange(
        target_to_source.shape[1], device=target_to_source.device
    ).float() / max(target_to_source.shape[1] - 1, 1)
    source_error = (reconstructed_source - source_index[None, :]).abs()
    target_error = (reconstructed_target - target_index[None, :]).abs()
    return 0.5 * (
        (source_error * source_mask).sum() / source_mask.sum().clamp_min(1)
        + (target_error * target_mask).sum() / target_mask.sum().clamp_min(1)
    )


@dataclass(frozen=True)
class LossWeights:
    task: float = 1.0
    soft_dtw: float = 0.2
    anchor: float = 0.5
    monotonicity: float = 0.05
    smoothness: float = 0.01
    slope: float = 0.01
    hierarchy: float = 0.1
    cycle: float = 0.05
    entropy: float = 0.001


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    terms: Mapping[str, Tensor]


class HDTWRoPELoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        *,
        similarity: Tensor,
        alignments: Mapping[str, AlignmentOutput],
        clocks: ClockOutput,
        anchor_targets: Mapping[str, tuple[Tensor, Tensor, Tensor]] | None = None,
        hierarchy_terms: Mapping[str, tuple[Tensor, Tensor, Tensor, Tensor]] | None = None,
        source_index_coordinate: Tensor | None = None,
        target_index_coordinate: Tensor | None = None,
    ) -> LossBreakdown:
        terms: dict[str, Tensor] = {
            "task": symmetric_contrastive_loss(similarity),
            "soft_dtw": soft_dtw_loss(alignments),
        }
        source_mono = [
            monotonicity_loss(clocks.source_clock[..., c], clocks.source_valid[..., c])
            for c in range(clocks.source_clock.shape[-1])
        ]
        target_mono = [
            monotonicity_loss(clocks.target_clock[..., c], clocks.target_valid[..., c])
            for c in range(clocks.target_clock.shape[-1])
        ]
        source_smooth = [
            smoothness_loss(clocks.source_clock[..., c], clocks.source_valid[..., c])
            for c in range(clocks.source_clock.shape[-1])
        ]
        target_smooth = [
            smoothness_loss(clocks.target_clock[..., c], clocks.target_valid[..., c])
            for c in range(clocks.target_clock.shape[-1])
        ]
        terms["monotonicity"] = torch.stack(source_mono + target_mono).mean()
        terms["smoothness"] = torch.stack(source_smooth + target_smooth).mean()
        if source_index_coordinate is not None and target_index_coordinate is not None:
            source_slopes = [
                slope_loss(
                    clocks.source_clock[..., c],
                    source_index_coordinate,
                    clocks.source_valid[..., c],
                )
                for c in range(clocks.source_clock.shape[-1])
            ]
            target_slopes = [
                slope_loss(
                    clocks.target_clock[..., c],
                    target_index_coordinate,
                    clocks.target_valid[..., c],
                )
                for c in range(clocks.target_clock.shape[-1])
            ]
            terms["slope"] = torch.stack(source_slopes + target_slopes).mean()
        else:
            terms["slope"] = similarity.new_zeros(())
        terms["anchor"] = (
            torch.stack([anchor_loss(*term) for term in anchor_targets.values()]).mean()
            if anchor_targets
            else similarity.new_zeros(())
        )
        terms["hierarchy"] = (
            torch.stack([containment_loss(*term) for term in hierarchy_terms.values()]).mean()
            if hierarchy_terms
            else similarity.new_zeros(())
        )
        terms["cycle"] = cycle_consistency_loss(
            clocks.source_clock[..., 0],
            clocks.target_clock[..., 0],
            clocks.source_valid[..., 0],
            clocks.target_valid[..., 0],
        )
        terms["entropy"] = torch.stack(
            [alignment_entropy_loss(alignment) for alignment in alignments.values()]
        ).mean()
        total = sum(getattr(self.weights, name) * value for name, value in terms.items())
        if not torch.isfinite(total):
            raise ValueError("loss produced NaN or Inf")
        return LossBreakdown(total=total, terms=terms)
