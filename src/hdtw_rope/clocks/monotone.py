"""Monotonicity, smoothness, slope constraints, and isotonic projection."""

from __future__ import annotations

import torch
from torch import Tensor


def monotonicity_loss(values: Tensor, mask: Tensor, *, margin: float = 0.0) -> Tensor:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("values and mask must have shape [B,N]")
    pair_mask = mask[:, :-1] & mask[:, 1:]
    violations = torch.relu(values[:, :-1] - values[:, 1:] + margin)
    denominator = pair_mask.sum().clamp_min(1)
    return (violations * pair_mask).sum() / denominator


def smoothness_loss(values: Tensor, mask: Tensor, *, delta: float = 1.0) -> Tensor:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("values and mask must have shape [B,N]")
    if values.shape[1] < 3:
        return values.new_zeros(())
    triplet_mask = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    second_difference = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
    loss = torch.nn.functional.huber_loss(
        second_difference,
        torch.zeros_like(second_difference),
        delta=delta,
        reduction="none",
    )
    return (loss * triplet_mask).sum() / triplet_mask.sum().clamp_min(1)


def slope_loss(
    values: Tensor,
    source_coordinate: Tensor,
    mask: Tensor,
    *,
    minimum: float = 0.0,
    maximum: float = 8.0,
    exempt_pair_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    if values.shape != source_coordinate.shape or values.shape != mask.shape:
        raise ValueError("slope tensors must share shape [B,N]")
    pair_mask = mask[:, :-1] & mask[:, 1:]
    if exempt_pair_mask is not None:
        if exempt_pair_mask.shape != pair_mask.shape:
            raise ValueError("exempt_pair_mask has wrong shape")
        pair_mask &= ~exempt_pair_mask
    delta_source = (source_coordinate[:, 1:] - source_coordinate[:, :-1]).clamp_min(eps)
    delta_clock = values[:, 1:] - values[:, :-1]
    slope = delta_clock / delta_source
    violation = torch.relu(minimum - slope) + torch.relu(slope - maximum)
    return (violation * pair_mask).sum() / pair_mask.sum().clamp_min(1)


def monotonicity_violation_count(
    values: Tensor, mask: Tensor, *, tolerance: float = 0.0
) -> Tensor:
    pair_mask = mask[:, :-1] & mask[:, 1:]
    return (((values[:, 1:] - values[:, :-1]) < -tolerance) & pair_mask).sum(dim=-1)


def _pav_1d(values: Tensor, weights: Tensor) -> Tensor:
    """Exact pool-adjacent-violators projection on a detached 1-D tensor."""

    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    block_ends: list[int] = []
    for index, (value, weight) in enumerate(
        zip(values.tolist(), weights.tolist(), strict=True)
    ):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_ends.append(index + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / max(total_weight, 1e-12)
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [total_weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()
    projected = torch.empty_like(values)
    for value, start, end in zip(
        block_values, block_starts, block_ends, strict=True
    ):
        projected[start:end] = value
    return projected


def isotonic_projection(
    values: Tensor,
    mask: Tensor,
    *,
    weights: Tensor | None = None,
    straight_through: bool = False,
) -> Tensor:
    """Project valid prefixes onto the nondecreasing cone."""

    if values.shape != mask.shape or values.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("values and mask must have shape [B,N]")
    if weights is None:
        weights = torch.ones_like(values)
    if weights.shape != values.shape:
        raise ValueError("weights shape differs")
    projected = torch.zeros_like(values)
    for batch_index in range(values.shape[0]):
        valid_count = int(mask[batch_index].sum().item())
        if valid_count == 0:
            continue
        row = values[batch_index, :valid_count].detach().to("cpu", torch.float64)
        row_weights = weights[batch_index, :valid_count].detach().to(
            "cpu", torch.float64
        )
        row_projected = _pav_1d(row, row_weights).to(values.device, values.dtype)
        projected[batch_index, :valid_count] = row_projected
    if straight_through:
        return values + (projected - values).detach()
    return projected
