from __future__ import annotations

import torch

from hdtw_rope.rotary.frequencies import ClockFrequencyMap
from hdtw_rope.rotary.reference import reference_rope
from hdtw_rope.rotary.transport import ClockRotaryEmbedding, apply_rotary_pairs


def _fixed_rotary(num_heads: int = 2, rotary_dim: int = 8) -> ClockRotaryEmbedding:
    return ClockRotaryEmbedding(
        ClockFrequencyMap(
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            component_names=["position"],
            mode="fixed_log",
        ),
        phase_modulo=False,
    )


def test_rotary_transport_preserves_norms() -> None:
    x = torch.randn(2, 2, 5, 12)
    phase = torch.randn(2, 2, 5, 4)
    rotated = apply_rotary_pairs(x, phase, rotary_dim=8)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_standard_rope_equivalence() -> None:
    q = torch.randn(2, 2, 5, 12)
    k = torch.randn(2, 2, 5, 12)
    positions = torch.arange(5).float().expand(2, -1)
    clock = positions.unsqueeze(-1)
    mask = torch.ones_like(clock, dtype=torch.bool)
    q_rot, k_rot = _fixed_rotary()(q, k, clock, clock, mask, mask)
    torch.testing.assert_close(
        q_rot, reference_rope(q, positions, rotary_dim=8), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        k_rot, reference_rope(k, positions, rotary_dim=8), atol=1e-5, rtol=1e-5
    )


def test_equal_clocks_preserve_query_key_inner_product() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    clock = torch.rand(1, 4, 1)
    mask = torch.ones_like(clock, dtype=torch.bool)
    q_rot, k_rot = _fixed_rotary()(q, k, clock, clock, mask, mask)
    torch.testing.assert_close(
        (q_rot * k_rot).sum(dim=-1), (q * k).sum(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_relative_phase_identity() -> None:
    x = torch.randn(1, 2, 3, 8)
    q_clock = torch.rand(1, 3, 1)
    k_clock = torch.rand(1, 3, 1)
    mask = torch.ones_like(q_clock, dtype=torch.bool)
    rotary = _fixed_rotary()
    q_rot, k_rot = rotary(x, x, q_clock, k_clock, mask, mask)
    direct_inner = torch.einsum("bhnd,bhnd->bhn", q_rot, k_rot)
    relative_phase = rotary.frequency_map(k_clock - q_clock, mask, modulo=False)
    relative_x = apply_rotary_pairs(x, relative_phase, rotary_dim=8)
    relative_inner = torch.einsum("bhnd,bhnd->bhn", x, relative_x)
    torch.testing.assert_close(direct_inner, relative_inner, atol=1e-5, rtol=1e-5)


def test_float16_forward_is_finite() -> None:
    q = torch.randn(1, 2, 4, 8, dtype=torch.float16)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float16)
    clock = torch.rand(1, 4, 1)
    mask = torch.ones_like(clock, dtype=torch.bool)
    q_rot, k_rot = _fixed_rotary()(q, k, clock, clock, mask, mask)
    assert torch.isfinite(q_rot).all() and torch.isfinite(k_rot).all()
