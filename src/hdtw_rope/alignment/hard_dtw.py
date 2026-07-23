"""Deterministic reference hard DTW."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hdtw_rope.errors import NoValidAlignmentPath
from hdtw_rope.types import AlignmentOutput
from hdtw_rope.validation import lengths_from_mask, require_bool_mask, require_finite, require_rank


@dataclass(frozen=True)
class HardDTWPath:
    rows: Tensor
    cols: Tensor


@dataclass(frozen=True)
class HardDTWBatchOutput:
    alignment: AlignmentOutput
    paths: tuple[HardDTWPath, ...]


def hard_dtw(
    cost: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    band_mask: Tensor | None = None,
    *,
    horizontal_penalty: float = 0.0,
    vertical_penalty: float = 0.0,
    diagonal_penalty: float = 0.0,
) -> HardDTWBatchOutput:
    """Compute exact DTW paths with diagonal-first deterministic tie-breaking."""

    require_rank("cost", cost, 3)
    batch, max_source, max_target = cost.shape
    require_bool_mask("source_mask", source_mask, (batch, max_source))
    require_bool_mask("target_mask", target_mask, (batch, max_target))
    require_finite("cost", cost, source_mask[:, :, None] & target_mask[:, None, :])
    if band_mask is not None:
        require_bool_mask("band_mask", band_mask, cost.shape)
    source_lengths = lengths_from_mask(source_mask)
    target_lengths = lengths_from_mask(target_mask)
    mass = torch.zeros_like(cost, dtype=torch.float32)
    values: list[Tensor] = []
    paths: list[HardDTWPath] = []

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
        dp = torch.full((t_len, s_len), torch.inf, dtype=torch.float32, device=cost.device)
        predecessor = torch.full((t_len, s_len), -1, dtype=torch.int8, device=cost.device)
        local_cost = cost[b, :t_len, :s_len].float()
        for i in range(t_len):
            for j in range(s_len):
                if not bool(allowed[i, j]):
                    continue
                if i == 0 and j == 0:
                    dp[i, j] = local_cost[i, j]
                    predecessor[i, j] = 3
                    continue
                candidates: list[tuple[Tensor, int]] = []
                if i > 0 and j > 0 and torch.isfinite(dp[i - 1, j - 1]):
                    candidates.append((dp[i - 1, j - 1] + diagonal_penalty, 2))
                if i > 0 and torch.isfinite(dp[i - 1, j]):
                    candidates.append((dp[i - 1, j] + vertical_penalty, 0))
                if j > 0 and torch.isfinite(dp[i, j - 1]):
                    candidates.append((dp[i, j - 1] + horizontal_penalty, 1))
                if not candidates:
                    continue
                candidate_values = torch.stack([candidate[0] for candidate in candidates])
                chosen = int(torch.argmin(candidate_values).item())
                dp[i, j] = local_cost[i, j] + candidate_values[chosen]
                predecessor[i, j] = candidates[chosen][1]
        if not torch.isfinite(dp[-1, -1]):
            raise NoValidAlignmentPath(f"sample {b} has no path under the configured band")

        rows: list[int] = []
        cols: list[int] = []
        i, j = t_len - 1, s_len - 1
        while True:
            rows.append(i)
            cols.append(j)
            mass[b, i, j] = 1.0
            if i == 0 and j == 0:
                break
            step = int(predecessor[i, j].item())
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            elif step == 2:
                i -= 1
                j -= 1
            else:
                raise NoValidAlignmentPath(f"sample {b} path traceback failed at {(i, j)}")
        rows.reverse()
        cols.reverse()
        paths.append(
            HardDTWPath(
                rows=torch.tensor(rows, dtype=torch.long, device=cost.device),
                cols=torch.tensor(cols, dtype=torch.long, device=cost.device),
            )
        )
        values.append(dp[-1, -1].to(cost.dtype))

    masked_cost = torch.where(
        source_mask[:, :, None]
        & target_mask[:, None, :]
        & (band_mask if band_mask is not None else True),
        cost,
        torch.full_like(cost, torch.inf),
    )
    alignment = AlignmentOutput(
        cost=masked_cost,
        value=torch.stack(values),
        mass=mass,
        valid_rows=source_mask & (mass.sum(dim=-1) > 0),
        valid_cols=target_mask & (mass.sum(dim=-2) > 0),
    )
    return HardDTWBatchOutput(alignment=alignment, paths=tuple(paths))
