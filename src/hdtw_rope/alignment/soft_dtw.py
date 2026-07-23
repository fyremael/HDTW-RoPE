"""Numerically stable reference Soft-DTW with alignment-mass extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hdtw_rope.errors import NoValidAlignmentPath
from hdtw_rope.types import AlignmentOutput
from hdtw_rope.validation import lengths_from_mask, require_bool_mask, require_finite, require_rank


@dataclass(frozen=True)
class TransitionPenalties:
    horizontal: float = 0.0
    vertical: float = 0.0
    diagonal: float = 0.0


def _soft_min(values: list[Tensor], temperature: float) -> Tensor:
    stacked = torch.stack(values)
    return -temperature * torch.logsumexp(-stacked / temperature, dim=0)


def _single_soft_dtw(
    cost: Tensor,
    allowed: Tensor,
    temperature: float,
    penalties: TransitionPenalties,
) -> Tensor | None:
    rows, cols = cost.shape
    dp: list[list[Tensor | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if not bool(allowed[i, j]):
                continue
            if i == 0 and j == 0:
                dp[i][j] = cost[i, j]
                continue
            predecessors: list[Tensor] = []
            if i > 0 and dp[i - 1][j] is not None:
                predecessors.append(dp[i - 1][j] + penalties.vertical)
            if j > 0 and dp[i][j - 1] is not None:
                predecessors.append(dp[i][j - 1] + penalties.horizontal)
            if i > 0 and j > 0 and dp[i - 1][j - 1] is not None:
                predecessors.append(dp[i - 1][j - 1] + penalties.diagonal)
            if predecessors:
                dp[i][j] = cost[i, j] + _soft_min(predecessors, temperature)
    return dp[-1][-1]


def soft_dtw_value(
    cost: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    band_mask: Tensor | None = None,
    *,
    temperature: float = 0.1,
    penalties: TransitionPenalties | None = None,
) -> Tensor:
    """Return one differentiable Soft-DTW value per sample."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    require_rank("cost", cost, 3)
    batch, max_source, max_target = cost.shape
    require_bool_mask("source_mask", source_mask, (batch, max_source))
    require_bool_mask("target_mask", target_mask, (batch, max_target))
    require_finite("cost", cost, source_mask[:, :, None] & target_mask[:, None, :])
    if band_mask is not None:
        require_bool_mask("band_mask", band_mask, cost.shape)
    source_lengths = lengths_from_mask(source_mask)
    target_lengths = lengths_from_mask(target_mask)
    transition = penalties or TransitionPenalties()
    outputs: list[Tensor] = []
    for b in range(batch):
        t_len = int(source_lengths[b].item())
        s_len = int(target_lengths[b].item())
        if t_len == 0 or s_len == 0:
            raise NoValidAlignmentPath(f"sample {b} has an empty sequence")
        allowed = (
            torch.ones((t_len, s_len), dtype=torch.bool, device=cost.device)
            if band_mask is None
            else band_mask[b, :t_len, :s_len]
        )
        local_cost = cost[b, :t_len, :s_len]
        if local_cost.dtype in (torch.float16, torch.bfloat16):
            local_cost = local_cost.float()
        result = _single_soft_dtw(local_cost, allowed, temperature, transition)
        if result is None or not bool(torch.isfinite(result)):
            raise NoValidAlignmentPath(f"sample {b} has no path under the configured band")
        outputs.append(result)
    return torch.stack(outputs)


class SoftDTWAligner(nn.Module):
    """Soft-DTW plus nonnegative alignment mass from the cost gradient."""

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        horizontal_penalty: float = 0.0,
        vertical_penalty: float = 0.0,
        diagonal_penalty: float = 0.0,
        min_alignment_mass: float = 1e-6,
        differentiable_mass: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.penalties = TransitionPenalties(
            horizontal=horizontal_penalty,
            vertical=vertical_penalty,
            diagonal=diagonal_penalty,
        )
        self.min_alignment_mass = min_alignment_mass
        self.differentiable_mass = differentiable_mass

    def forward(
        self,
        cost: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
        band_mask: Tensor | None = None,
    ) -> AlignmentOutput:
        original_requires_grad = cost.requires_grad
        working_cost = cost if original_requires_grad else cost.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            values32 = soft_dtw_value(
                working_cost,
                source_mask,
                target_mask,
                band_mask,
                temperature=self.temperature,
                penalties=self.penalties,
            )
            mass = torch.autograd.grad(
                values32.sum(),
                working_cost,
                create_graph=self.differentiable_mass,
                retain_graph=True,
            )[0]
        mass = mass.clamp_min(0.0)
        valid_pairs = source_mask[:, :, None] & target_mask[:, None, :]
        if band_mask is not None:
            valid_pairs &= band_mask
        mass = torch.where(valid_pairs, mass, torch.zeros_like(mass))
        row_mass = mass.sum(dim=-1)
        col_mass = mass.sum(dim=-2)
        valid_rows = source_mask & (row_mass >= self.min_alignment_mass)
        valid_cols = target_mask & (col_mass >= self.min_alignment_mass)
        masked_cost = torch.where(valid_pairs, cost, torch.full_like(cost, torch.inf))
        return AlignmentOutput(
            cost=masked_cost,
            value=values32.to(cost.dtype),
            mass=mass,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
