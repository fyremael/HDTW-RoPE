"""Shared clock-to-phase frequency maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

FrequencyMode = Literal["fixed_log", "learned_positive", "clock_partitioned"]


def standard_inverse_frequencies(rotary_dim: int, *, base: float = 10000.0) -> Tensor:
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be a positive even integer")
    pair_index = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    return base ** (-pair_index / rotary_dim)


def _allocate_pairs(total_pairs: int, allocation: Mapping[str, float]) -> dict[str, int]:
    if any(value < 0 for value in allocation.values()):
        raise ValueError("partition allocations must be nonnegative")
    total_fraction = sum(allocation.values())
    if total_fraction > 1.0 + 1e-6:
        raise ValueError("partition allocations must sum to at most 1")
    target = total_pairs * total_fraction
    floor_counts = {name: int(total_pairs * value) for name, value in allocation.items()}
    remaining = min(
        total_pairs - sum(floor_counts.values()),
        round(target) - sum(floor_counts.values()),
    )
    remainders = sorted(
        allocation,
        key=lambda name: (
            total_pairs * allocation[name] - floor_counts[name],
            name,
        ),
        reverse=True,
    )
    for name in remainders[: max(remaining, 0)]:
        floor_counts[name] += 1
    return floor_counts


class ClockFrequencyMap(nn.Module):
    """Map versioned clock vectors to rotary phases.

    Frequencies are shared between modalities by construction. Per-modality
    clock adapters belong before this module.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        rotary_dim: int,
        component_names: Sequence[str],
        mode: FrequencyMode = "clock_partitioned",
        base: float = 10000.0,
        partition: Mapping[str, float] | None = None,
        family_components: Mapping[str, Sequence[str]] | None = None,
        learn_phase_offset: bool = False,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive and even")
        if len(set(component_names)) != len(component_names):
            raise ValueError("component names must be unique")
        self.num_heads = num_heads
        self.rotary_dim = rotary_dim
        self.num_pairs = rotary_dim // 2
        self.component_names = tuple(component_names)
        self.mode = mode
        self.base = base
        component_count = len(self.component_names)
        if component_count == 0:
            raise ValueError("at least one clock component is required")

        if mode == "fixed_log":
            omega = torch.zeros(num_heads, self.num_pairs, component_count)
            omega[:, :, 0] = standard_inverse_frequencies(rotary_dim, base=base)[None, :] / (
                2.0 * torch.pi
            )
            self.register_buffer("fixed_omega", omega)
            self.register_parameter("raw_omega", None)
        elif mode == "learned_positive":
            initial = torch.full((num_heads, self.num_pairs, component_count), -6.0)
            initial[:, :, 0] = torch.log(
                torch.expm1(
                    standard_inverse_frequencies(rotary_dim, base=base) / (2.0 * torch.pi)
                ).clamp_min(1e-8)
            )[None, :]
            self.raw_omega = nn.Parameter(initial)
            self.register_buffer("fixed_omega", torch.empty(0))
        elif mode == "clock_partitioned":
            if partition is None or family_components is None:
                raise ValueError("clock_partitioned mode requires partition and family_components")
            component_index = {name: index for index, name in enumerate(self.component_names)}
            unknown = {
                component
                for family in family_components.values()
                for component in family
                if component not in component_index
            }
            if unknown:
                raise ValueError(f"partition references unknown components: {sorted(unknown)}")
            counts = _allocate_pairs(self.num_pairs, partition)
            omega = torch.zeros(num_heads, self.num_pairs, component_count)
            cursor = 0
            for family, count in counts.items():
                if count == 0:
                    continue
                components = tuple(family_components.get(family, ()))
                if not components:
                    raise ValueError(f"family {family!r} has allocated pairs but no components")
                local_frequencies = standard_inverse_frequencies(2 * count, base=base) / (
                    2.0 * torch.pi
                )
                for local_index in range(count):
                    component = components[local_index % len(components)]
                    omega[:, cursor + local_index, component_index[component]] = local_frequencies[
                        local_index
                    ]
                cursor += count
            self.register_buffer("fixed_omega", omega)
            self.register_parameter("raw_omega", None)
        else:  # pragma: no cover
            raise ValueError(f"unknown frequency mode: {mode}")

        if learn_phase_offset:
            self.phase_offset = nn.Parameter(torch.zeros(num_heads, self.num_pairs))
        else:
            self.register_buffer("phase_offset", torch.zeros(num_heads, self.num_pairs))

    @property
    def omega(self) -> Tensor:
        if self.mode == "learned_positive":
            assert self.raw_omega is not None
            return torch.nn.functional.softplus(self.raw_omega)
        return self.fixed_omega

    def forward(self, clock: Tensor, clock_mask: Tensor, *, modulo: bool = True) -> Tensor:
        """Return phases ``[B,H,N,P]`` in float32."""

        if clock.ndim != 3 or clock_mask.shape != clock.shape or clock_mask.dtype is not torch.bool:
            raise ValueError("clock and clock_mask must have shapes [B,N,R] with bool mask")
        if clock.shape[-1] != len(self.component_names):
            raise ValueError(
                f"clock schema has {clock.shape[-1]} components; "
                f"expected {len(self.component_names)}"
            )
        clock32 = clock.float()
        if not torch.isfinite(clock32[clock_mask]).all():
            raise ValueError("clock contains nonfinite valid values")
        weights = self.omega.float()
        mask = clock_mask[:, None, :, None, :]
        masked_weights = weights[None, :, None, :, :] * mask
        full_norm = weights.abs().sum(dim=-1)[None, :, None, :, None]
        valid_norm = masked_weights.abs().sum(dim=-1, keepdim=True)
        scale = torch.where(
            valid_norm > 0,
            full_norm / valid_norm.clamp_min(1e-12),
            torch.zeros_like(valid_norm),
        )
        effective_weights = masked_weights * scale
        phases = 2.0 * torch.pi * torch.einsum("bhnpr,bnr->bhnp", effective_weights, clock32)
        phases = phases + self.phase_offset.float()[None, :, None, :]
        if modulo:
            phases = torch.remainder(phases + torch.pi, 2.0 * torch.pi) - torch.pi
        if not torch.isfinite(phases).all():
            raise ValueError("phase computation produced NaN or Inf")
        return phases
