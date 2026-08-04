# Reference Python Interfaces

```python
from dataclasses import dataclass
from typing import Mapping
from torch import Tensor, nn

@dataclass(frozen=True)
class AlignmentOutput:
    cost: Tensor
    value: Tensor
    mass: Tensor
    valid_rows: Tensor
    valid_cols: Tensor

@dataclass(frozen=True)
class ClockOutput:
    source_clock: Tensor
    target_clock: Tensor
    source_valid: Tensor
    target_valid: Tensor
    diagnostics: Mapping[str, Tensor]

class PairwiseCost(nn.Module):
    def forward(self, source: Tensor, target: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor: ...

class SoftDTWAligner(nn.Module):
    def forward(self, cost: Tensor, source_mask: Tensor, target_mask: Tensor, band_mask: Tensor | None = None) -> AlignmentOutput: ...

class LatentClockExtractor(nn.Module):
    def forward(self, alignments: Mapping[str, AlignmentOutput], source_coordinates: Mapping[str, Tensor], target_coordinates: Mapping[str, Tensor], source_coordinate_masks: Mapping[str, Tensor], target_coordinate_masks: Mapping[str, Tensor], hierarchy: Mapping[str, Tensor]) -> ClockOutput: ...

class ClockRotaryEmbedding(nn.Module):
    def forward(self, q: Tensor, k: Tensor, q_clock: Tensor, k_clock: Tensor, q_clock_mask: Tensor, k_clock_mask: Tensor) -> tuple[Tensor, Tensor]: ...
```

## Shape rules

- Source and target features are batch-major `[B,N,D]`.
- Attention tensors are `[B,H,N,Dh]` internally.
- Pairwise costs and masses are `[B,T,S]`.
- Clocks and component masks are `[B,N,R]`.
- `True` means valid for every mask.
- Clock component order is determined by a versioned schema, never dictionary iteration order.
- Public functions validate rank, dtype, finite values, masks, and compatible lengths.
- Masked or invalid rows never contribute to reductions.
