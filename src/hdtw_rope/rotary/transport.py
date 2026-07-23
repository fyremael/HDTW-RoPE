"""Exact norm-preserving rotary transport over latent clocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hdtw_rope.rotary.frequencies import ClockFrequencyMap


def apply_rotary_pairs(x: Tensor, phases: Tensor, rotary_dim: int) -> Tensor:
    """Apply rotations to the first ``rotary_dim`` channels of ``x``."""

    if x.ndim != 4:
        raise ValueError("x must have shape [B,H,N,Dh]")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("invalid rotary_dim")
    if phases.shape != (*x.shape[:3], rotary_dim // 2):
        raise ValueError(
            f"phase shape {tuple(phases.shape)} incompatible with x "
            f"{tuple(x.shape)} and rotary_dim {rotary_dim}"
        )
    original_dtype = x.dtype
    x_rot = x[..., :rotary_dim].float().reshape(*x.shape[:3], rotary_dim // 2, 2)
    x_pass = x[..., rotary_dim:]
    cos = torch.cos(phases.float())
    sin = torch.sin(phases.float())
    even = x_rot[..., 0]
    odd = x_rot[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.reshape(*x.shape[:3], rotary_dim).to(original_dtype)
    result = torch.cat((rotated, x_pass), dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("rotary transport produced NaN or Inf")
    return result


class ClockRotaryEmbedding(nn.Module):
    """Shared unitary transport for query and key streams."""

    def __init__(
        self,
        frequency_map: ClockFrequencyMap,
        *,
        phase_modulo: bool = True,
    ) -> None:
        super().__init__()
        self.frequency_map = frequency_map
        self.phase_modulo = phase_modulo

    @property
    def rotary_dim(self) -> int:
        return self.frequency_map.rotary_dim

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_clock: Tensor,
        k_clock: Tensor,
        q_clock_mask: Tensor,
        k_clock_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k batch/head/head-dimension shapes must match")
        if q.shape[1] != self.frequency_map.num_heads:
            raise ValueError("attention head count differs from frequency map")
        q_phase = self.frequency_map(q_clock, q_clock_mask, modulo=self.phase_modulo)
        k_phase = self.frequency_map(k_clock, k_clock_mask, modulo=self.phase_modulo)
        return (
            apply_rotary_pairs(q, q_phase, self.rotary_dim),
            apply_rotary_pairs(k, k_phase, self.rotary_dim),
        )
