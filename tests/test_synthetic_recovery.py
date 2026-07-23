from __future__ import annotations

import pytest
import torch

from hdtw_rope.alignment.costs import PairwiseCost
from hdtw_rope.alignment.hard_dtw import hard_dtw
from hdtw_rope.clocks.extract import LatentClockExtractor
from hdtw_rope.data.synthetic import generate_synthetic_pair


@pytest.mark.parametrize("family", ["affine", "piecewise", "pause", "smooth", "melisma"])
def test_noiseless_synthetic_clock_recovery(family: str) -> None:
    pair = generate_synthetic_pair(audio_length=80, lyric_length=60, feature_dim=48, warp_family=family, noise_std=0.0, seed=7)  # type: ignore[arg-type]
    source = pair.sample.audio_features.unsqueeze(0)
    target = pair.sample.lyric_features.unsqueeze(0)
    source_mask = pair.sample.audio_mask.unsqueeze(0)
    target_mask = pair.sample.lyric_mask.unsqueeze(0)
    cost = PairwiseCost("cosine")(source, target, source_mask, target_mask)
    alignment = hard_dtw(cost, source_mask, target_mask).alignment
    clock = LatentClockExtractor(["aligned_word"])({"aligned_word": alignment}, {}, {"aligned_word": torch.linspace(0.0, 1.0, target.shape[1]).unsqueeze(0)}, {}, {"aligned_word": target_mask}, {}).source_clock[0, :, 0]
    assert (clock - pair.audio_to_lyric_clock).abs().mean().item() < 0.03


def test_moderate_noise_synthetic_clock_recovery() -> None:
    pair = generate_synthetic_pair(audio_length=80, lyric_length=60, feature_dim=48, warp_family="piecewise", noise_std=0.05, seed=19)
    source = pair.sample.audio_features.unsqueeze(0)
    target = pair.sample.lyric_features.unsqueeze(0)
    source_mask = pair.sample.audio_mask.unsqueeze(0)
    target_mask = pair.sample.lyric_mask.unsqueeze(0)
    alignment = hard_dtw(PairwiseCost("cosine")(source, target, source_mask, target_mask), source_mask, target_mask).alignment
    clock = LatentClockExtractor(["aligned_word"])({"aligned_word": alignment}, {}, {"aligned_word": torch.linspace(0.0, 1.0, target.shape[1]).unsqueeze(0)}, {}, {"aligned_word": target_mask}, {}).source_clock[0, :, 0]
    assert (clock - pair.audio_to_lyric_clock).abs().mean().item() < 0.08
