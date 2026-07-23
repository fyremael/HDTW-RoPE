"""Required scalar diagnostics and serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from hdtw_rope.types import AlignmentOutput, ClockOutput


def alignment_diagnostics(
    alignment: AlignmentOutput, *, prefix: str = "alignment"
) -> dict[str, Tensor]:
    row_mass = alignment.mass.sum(dim=-1)
    probabilities = alignment.mass / row_mass.unsqueeze(-1).clamp_min(1e-12)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        f"{prefix}/soft_dtw": alignment.value.detach().mean(),
        f"{prefix}/entropy": (entropy * alignment.valid_rows).sum()
        / alignment.valid_rows.sum().clamp_min(1),
        f"{prefix}/valid_row_rate": alignment.valid_rows.float().mean(),
        f"{prefix}/valid_col_rate": alignment.valid_cols.float().mean(),
        f"{prefix}/mass_sum": alignment.mass.detach().sum(),
    }


def clock_diagnostics(clocks: ClockOutput, *, prefix: str = "clock") -> dict[str, Tensor]:
    source_delta = clocks.source_clock[:, 1:] - clocks.source_clock[:, :-1]
    target_delta = clocks.target_clock[:, 1:] - clocks.target_clock[:, :-1]
    source_pair_mask = clocks.source_valid[:, 1:] & clocks.source_valid[:, :-1]
    target_pair_mask = clocks.target_valid[:, 1:] & clocks.target_valid[:, :-1]
    return {
        f"{prefix}/source_valid_rate": clocks.source_valid.float().mean(),
        f"{prefix}/target_valid_rate": clocks.target_valid.float().mean(),
        f"{prefix}/source_mean_slope": source_delta.masked_select(source_pair_mask).mean()
        if source_pair_mask.any()
        else source_delta.new_zeros(()),
        f"{prefix}/target_mean_slope": target_delta.masked_select(target_pair_mask).mean()
        if target_pair_mask.any()
        else target_delta.new_zeros(()),
        f"{prefix}/source_max_slope": source_delta.masked_select(source_pair_mask).max()
        if source_pair_mask.any()
        else source_delta.new_zeros(()),
        f"{prefix}/target_max_slope": target_delta.masked_select(target_pair_mask).max()
        if target_pair_mask.any()
        else target_delta.new_zeros(()),
    }


def tensor_scalars(values: Mapping[str, Tensor | float | int]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                continue
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = value
    return result


def write_metrics(path: str | Path, values: Mapping[str, Tensor | float | int]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(tensor_scalars(values), indent=2, sort_keys=True) + "\n")
