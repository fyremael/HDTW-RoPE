"""Hard and differentiable alignment primitives."""

from hdtw_rope.alignment.bands import coordinate_band_mask, diagonal_band_mask
from hdtw_rope.alignment.costs import PairwiseCost
from hdtw_rope.alignment.hard_dtw import HardDTWBatchOutput, HardDTWPath, hard_dtw
from hdtw_rope.alignment.hierarchical import HierarchicalAligner, MultiLevelHierarchicalAligner
from hdtw_rope.alignment.soft_dtw import SoftDTWAligner, TransitionPenalties, soft_dtw_value

__all__ = [
    "HardDTWBatchOutput",
    "HardDTWPath",
    "HierarchicalAligner",
    "MultiLevelHierarchicalAligner",
    "PairwiseCost",
    "SoftDTWAligner",
    "TransitionPenalties",
    "coordinate_band_mask",
    "diagonal_band_mask",
    "hard_dtw",
    "soft_dtw_value",
]
