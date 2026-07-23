"""Extract vector-valued latent clocks from alignment mass."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from hdtw_rope.clocks.monotone import isotonic_projection, monotonicity_violation_count
from hdtw_rope.clocks.normalize import normalize_unit_interval, normalized_index
from hdtw_rope.errors import InsufficientAlignmentMass, InvalidClockError
from hdtw_rope.types import AlignmentOutput, ClockOutput


def _infer_shape(
    alignments: Mapping[str, AlignmentOutput],
    source_coordinates: Mapping[str, Tensor],
    target_coordinates: Mapping[str, Tensor],
) -> tuple[int, int, int, torch.device]:
    if alignments:
        output = next(iter(alignments.values()))
        batch, source_length, target_length = output.mass.shape
        return batch, source_length, target_length, output.mass.device
    if not source_coordinates or not target_coordinates:
        raise InvalidClockError(
            "cannot infer clock lengths without alignments and both coordinate maps"
        )
    source = next(iter(source_coordinates.values()))
    target = next(iter(target_coordinates.values()))
    return source.shape[0], source.shape[1], target.shape[1], source.device


def _normalize_coordinate(values: Tensor, mask: Tensor, mode: str) -> Tensor:
    if mode == "unit_interval":
        return normalize_unit_interval(values.float(), mask)
    if mode == "none":
        return torch.where(mask, values.float(), torch.zeros_like(values, dtype=torch.float32))
    raise ValueError(f"unsupported normalization mode: {mode}")


def _expectation(
    mass: Tensor,
    coordinate: Tensor,
    coordinate_mask: Tensor,
    row_valid: Tensor,
    *,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    weighted_mass = mass.float() * coordinate_mask[:, None, :]
    row_mass = weighted_mass.sum(dim=-1)
    normalized_mass = weighted_mass / row_mass.unsqueeze(-1).clamp_min(eps)
    clock = torch.einsum("bts,bs->bt", normalized_mass, coordinate.float())
    valid = row_valid & (row_mass >= eps)
    clock = torch.where(valid, clock, torch.zeros_like(clock))
    return clock, valid, row_mass


class LatentClockExtractor(nn.Module):
    """Construct a versioned clock vector with explicit component masks."""

    def __init__(
        self,
        components: Sequence[str],
        *,
        normalization: str = "unit_interval",
        min_alignment_mass: float = 1e-6,
        monotonicity_mode: str = "penalty",
        strict: bool = True,
    ) -> None:
        super().__init__()
        if len(set(components)) != len(components):
            raise ValueError("clock component names must be unique")
        if monotonicity_mode not in {"penalty", "project", "none"}:
            raise ValueError("monotonicity_mode must be penalty, project, or none")
        self.components = tuple(components)
        self.normalization = normalization
        self.min_alignment_mass = min_alignment_mass
        self.monotonicity_mode = monotonicity_mode
        self.strict = strict

    def forward(
        self,
        alignments: Mapping[str, AlignmentOutput],
        source_coordinates: Mapping[str, Tensor],
        target_coordinates: Mapping[str, Tensor],
        source_coordinate_masks: Mapping[str, Tensor],
        target_coordinate_masks: Mapping[str, Tensor],
        hierarchy: Mapping[str, Tensor] | None = None,
    ) -> ClockOutput:
        del hierarchy
        batch, source_length, target_length, device = _infer_shape(
            alignments, source_coordinates, target_coordinates
        )
        source_parts: list[Tensor] = []
        target_parts: list[Tensor] = []
        source_valid_parts: list[Tensor] = []
        target_valid_parts: list[Tensor] = []
        diagnostics: dict[str, Tensor] = {}

        for component in self.components:
            source_coordinate = source_coordinates.get(component)
            target_coordinate = target_coordinates.get(component)
            source_mask = source_coordinate_masks.get(component)
            target_mask = target_coordinate_masks.get(component)
            alignment = alignments.get(component)

            if source_coordinate is not None:
                if source_mask is None:
                    raise InvalidClockError(
                        f"source coordinate {component!r} lacks a validity mask"
                    )
                source_direct = _normalize_coordinate(
                    source_coordinate, source_mask, self.normalization
                )
            else:
                source_direct = None
            if target_coordinate is not None:
                if target_mask is None:
                    raise InvalidClockError(
                        f"target coordinate {component!r} lacks a validity mask"
                    )
                target_direct = _normalize_coordinate(
                    target_coordinate, target_mask, self.normalization
                )
            else:
                target_direct = None

            source_clock = torch.zeros(
                (batch, source_length), device=device, dtype=torch.float32
            )
            target_clock = torch.zeros(
                (batch, target_length), device=device, dtype=torch.float32
            )
            source_valid = torch.zeros(
                (batch, source_length), device=device, dtype=torch.bool
            )
            target_valid = torch.zeros(
                (batch, target_length), device=device, dtype=torch.bool
            )
            source_mass = torch.zeros_like(source_clock)
            target_mass = torch.zeros_like(target_clock)

            if source_direct is not None:
                if source_direct.shape != source_clock.shape:
                    raise InvalidClockError(
                        f"source coordinate {component!r} has incompatible shape"
                    )
                source_clock = source_direct
                source_valid = source_mask
            elif alignment is not None:
                if target_direct is None or target_mask is None:
                    target_direct = normalized_index(alignment.valid_cols)
                    target_mask = alignment.valid_cols
                source_clock, source_valid, source_mass = _expectation(
                    alignment.mass,
                    target_direct,
                    target_mask,
                    alignment.valid_rows,
                    eps=self.min_alignment_mass,
                )

            if target_direct is not None:
                if target_direct.shape != target_clock.shape:
                    raise InvalidClockError(
                        f"target coordinate {component!r} has incompatible shape"
                    )
                target_clock = target_direct
                target_valid = target_mask
            elif alignment is not None:
                if source_direct is None or source_mask is None:
                    source_direct = normalized_index(alignment.valid_rows)
                    source_mask = alignment.valid_rows
                target_clock, target_valid, target_mass = _expectation(
                    alignment.mass.transpose(1, 2),
                    source_direct,
                    source_mask,
                    alignment.valid_cols,
                    eps=self.min_alignment_mass,
                )

            if alignment is not None and self.strict:
                if source_coordinate is None and torch.any(
                    alignment.valid_rows & ~source_valid
                ):
                    raise InsufficientAlignmentMass(
                        f"component {component!r} has invalid source rows"
                    )
                if target_coordinate is None and torch.any(
                    alignment.valid_cols & ~target_valid
                ):
                    raise InsufficientAlignmentMass(
                        f"component {component!r} has invalid target columns"
                    )

            if self.monotonicity_mode == "project":
                source_clock = isotonic_projection(
                    source_clock, source_valid, straight_through=self.training
                )
                target_clock = isotonic_projection(
                    target_clock, target_valid, straight_through=self.training
                )

            if not torch.isfinite(source_clock[source_valid]).all() or not torch.isfinite(
                target_clock[target_valid]
            ).all():
                raise InvalidClockError(f"component {component!r} produced nonfinite values")

            diagnostics[f"{component}/source_min"] = torch.where(
                source_valid, source_clock, torch.full_like(source_clock, torch.inf)
            ).amin(dim=1)
            diagnostics[f"{component}/source_max"] = torch.where(
                source_valid, source_clock, torch.full_like(source_clock, -torch.inf)
            ).amax(dim=1)
            diagnostics[f"{component}/target_min"] = torch.where(
                target_valid, target_clock, torch.full_like(target_clock, torch.inf)
            ).amin(dim=1)
            diagnostics[f"{component}/target_max"] = torch.where(
                target_valid, target_clock, torch.full_like(target_clock, -torch.inf)
            ).amax(dim=1)
            diagnostics[f"{component}/source_monotonicity_violations"] = (
                monotonicity_violation_count(source_clock, source_valid)
            )
            diagnostics[f"{component}/target_monotonicity_violations"] = (
                monotonicity_violation_count(target_clock, target_valid)
            )
            diagnostics[f"{component}/source_alignment_mass_min"] = source_mass.masked_fill(
                ~source_valid, torch.inf
            ).amin(dim=1)
            diagnostics[f"{component}/target_alignment_mass_min"] = target_mass.masked_fill(
                ~target_valid, torch.inf
            ).amin(dim=1)

            source_parts.append(source_clock)
            target_parts.append(target_clock)
            source_valid_parts.append(source_valid)
            target_valid_parts.append(target_valid)

        return ClockOutput(
            source_clock=torch.stack(source_parts, dim=-1),
            target_clock=torch.stack(target_parts, dim=-1),
            source_valid=torch.stack(source_valid_parts, dim=-1),
            target_valid=torch.stack(target_valid_parts, dim=-1),
            diagnostics=diagnostics,
        )
