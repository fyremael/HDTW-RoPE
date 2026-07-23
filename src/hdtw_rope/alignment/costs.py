"""Pairwise alignment costs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from hdtw_rope.validation import require_bool_mask, require_floating, require_rank

CostMode = Literal["cosine", "squared_euclidean", "bilinear"]


class PairwiseCost(nn.Module):
    """Compute a masked pairwise cost tensor without inferring lengths from data."""

    def __init__(
        self,
        mode: CostMode = "cosine",
        *,
        dim: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.eps = eps
        if mode == "bilinear":
            if dim is None:
                raise ValueError("dim is required for bilinear cost")
            self.weight = nn.Parameter(torch.eye(dim))
        else:
            self.register_parameter("weight", None)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        require_rank("source", source, 3)
        require_rank("target", target, 3)
        require_floating("source", source)
        require_floating("target", target)
        if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
            raise ValueError("source and target batch/feature dimensions must match")
        require_bool_mask("source_mask", source_mask, source.shape[:2])
        require_bool_mask("target_mask", target_mask, target.shape[:2])

        source32 = source.float()
        target32 = target.float()
        if self.mode == "cosine":
            source32 = torch.nn.functional.normalize(source32, dim=-1, eps=self.eps)
            target32 = torch.nn.functional.normalize(target32, dim=-1, eps=self.eps)
            cost = 1.0 - torch.einsum("btd,bsd->bts", source32, target32)
        elif self.mode == "squared_euclidean":
            source_norm = (source32 * source32).sum(dim=-1, keepdim=True)
            target_norm = (target32 * target32).sum(dim=-1).unsqueeze(1)
            cost = (
                source_norm
                + target_norm
                - 2.0 * torch.einsum("btd,bsd->bts", source32, target32)
            ).clamp_min(0.0)
        elif self.mode == "bilinear":
            assert self.weight is not None
            similarity = torch.einsum("btd,de,bse->bts", source32, self.weight.float(), target32)
            cost = -similarity
        else:  # pragma: no cover
            raise ValueError(f"unknown cost mode: {self.mode}")

        valid_pairs = source_mask[:, :, None] & target_mask[:, None, :]
        return torch.where(valid_pairs, cost, torch.zeros_like(cost))
