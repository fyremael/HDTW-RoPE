"""Latent clock extraction and constraints."""

from hdtw_rope.clocks.extract import LatentClockExtractor
from hdtw_rope.clocks.hierarchy import containment_loss, containment_violation_rate
from hdtw_rope.clocks.monotone import (
    isotonic_projection,
    monotonicity_loss,
    monotonicity_violation_count,
    slope_loss,
    smoothness_loss,
)
from hdtw_rope.clocks.normalize import normalize_unit_interval, normalized_index

__all__ = [
    "LatentClockExtractor",
    "containment_loss",
    "containment_violation_rate",
    "isotonic_projection",
    "monotonicity_loss",
    "monotonicity_violation_count",
    "normalize_unit_interval",
    "normalized_index",
    "slope_loss",
    "smoothness_loss",
]
