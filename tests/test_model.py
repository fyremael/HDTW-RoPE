from __future__ import annotations

import torch

from hdtw_rope.models.hdtw_rope_model import HDTWRoPEModel
from hdtw_rope.types import ClockOutput


def test_bidirectional_model_forward_is_finite() -> None:
    components = ["absolute_seconds", "aligned_word"]
    model = HDTWRoPEModel(
        audio_dim=12,
        lyric_dim=12,
        model_dim=16,
        num_heads=2,
        rotary_dim=8,
        clock_components=components,
        cross_attention_layers=1,
        dropout=0.0,
        frequency_mode="clock_partitioned",
        partition={"absolute_seconds": 0.5, "lyric_local": 0.5},
        family_components={
            "absolute_seconds": ("absolute_seconds",),
            "lyric_local": ("aligned_word",),
        },
    )
    audio = torch.randn(3, 5, 12)
    lyrics = torch.randn(3, 4, 12)
    audio_mask = torch.ones(3, 5, dtype=torch.bool)
    lyric_mask = torch.ones(3, 4, dtype=torch.bool)
    clocks = ClockOutput(
        source_clock=torch.rand(3, 5, 2),
        target_clock=torch.rand(3, 4, 2),
        source_valid=torch.ones(3, 5, 2, dtype=torch.bool),
        target_valid=torch.ones(3, 4, 2, dtype=torch.bool),
        diagnostics={},
    )
    output = model(audio, lyrics, audio_mask, lyric_mask, clocks)
    assert output.similarity.shape == (3, 3)
    assert torch.isfinite(output.similarity).all()
