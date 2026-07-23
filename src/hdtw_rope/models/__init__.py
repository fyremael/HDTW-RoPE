"""Cross-modal models."""

from hdtw_rope.models.cross_attention import (
    BidirectionalCrossAttentionLayer,
    HDTWRoPECrossAttention,
)
from hdtw_rope.models.hdtw_rope_model import (
    DifferentiableClockBuilder,
    HDTWRoPEModel,
    ModelOutput,
)
from hdtw_rope.models.projections import ClockAdapter, FeatureProjection

__all__ = [
    "BidirectionalCrossAttentionLayer",
    "ClockAdapter",
    "DifferentiableClockBuilder",
    "FeatureProjection",
    "HDTWRoPECrossAttention",
    "HDTWRoPEModel",
    "ModelOutput",
]
