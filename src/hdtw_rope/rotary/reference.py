"""Reference standard RoPE used by equivalence tests and baselines."""

from __future__ import annotations

from torch import Tensor

from hdtw_rope.rotary.frequencies import standard_inverse_frequencies
from hdtw_rope.rotary.transport import apply_rotary_pairs


def reference_rope(
    x: Tensor,
    positions: Tensor,
    *,
    rotary_dim: int,
    base: float = 10000.0,
) -> Tensor:
    """Apply canonical log-frequency RoPE to ``x [B,H,N,Dh]``."""

    if positions.shape != (x.shape[0], x.shape[2]):
        raise ValueError("positions must have shape [B,N]")
    inv_freq = standard_inverse_frequencies(rotary_dim, base=base).to(x.device)
    phases = positions.float()[:, None, :, None] * inv_freq[None, None, None, :]
    phases = phases.expand(x.shape[0], x.shape[1], x.shape[2], rotary_dim // 2)
    return apply_rotary_pairs(x, phases, rotary_dim)
