from __future__ import annotations

import pytest
import torch

from hdtw_rope.alignment.bands import diagonal_band_mask
from hdtw_rope.alignment.hard_dtw import hard_dtw
from hdtw_rope.errors import NoValidAlignmentPath


def test_hard_dtw_known_path() -> None:
    cost = torch.tensor([[[0.0, 3.0, 4.0], [2.0, 0.0, 3.0], [4.0, 2.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = hard_dtw(cost, mask, mask)
    path = list(zip(result.paths[0].rows.tolist(), result.paths[0].cols.tolist(), strict=True))
    assert path == [(0, 0), (1, 1), (2, 2)]
    assert result.alignment.value.item() == pytest.approx(0.0)
    assert result.alignment.mass.sum().item() == 3.0


def test_banded_dtw_rejects_disconnected_path() -> None:
    cost = torch.zeros(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    band = torch.zeros_like(cost, dtype=torch.bool)
    band[0, 0, 0] = True
    band[0, 2, 2] = True
    with pytest.raises(NoValidAlignmentPath):
        hard_dtw(cost, mask, mask, band)


def test_diagonal_band_respects_padding() -> None:
    source_mask = torch.tensor([[True, True, True, False]])
    target_mask = torch.tensor([[True, True, False]])
    band = diagonal_band_mask(source_mask, target_mask, half_width=1)
    assert band.shape == (1, 4, 3)
    assert not band[0, 3].any()
    assert not band[0, :, 2].any()
    assert band[0, 0, 0]
    assert band[0, 2, 1]
