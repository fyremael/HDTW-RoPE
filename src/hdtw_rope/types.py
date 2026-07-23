"""Public immutable result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class AlignmentOutput:
    """Alignment state for one hierarchy level.

    Attributes follow the normative implementation specification. ``mass`` is
    nonnegative but not assumed to be row-stochastic.
    """

    cost: Tensor
    value: Tensor
    mass: Tensor
    valid_rows: Tensor
    valid_cols: Tensor


@dataclass(frozen=True)
class ClockOutput:
    """Vector-valued clocks and their component validity masks."""

    source_clock: Tensor
    target_clock: Tensor
    source_valid: Tensor
    target_valid: Tensor
    diagnostics: Mapping[str, Tensor]
