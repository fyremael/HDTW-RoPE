"""Required positional baselines expressed through explicit clock contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

import torch
from torch import Tensor, nn

from hdtw_rope.clocks.normalize import normalize_unit_interval, normalized_index
from hdtw_rope.types import ClockOutput


class BaselineMode(StrEnum):
    NONE = "none"
    ADDITIVE = "additive"
    RAW_INDEX_ROPE = "raw_index_rope"
    TIMESTAMP_ROPE = "timestamp_rope"
    MULTI_CLOCK_ROPE = "multi_clock_rope"
    HDTW_SINGLE_CLOCK = "hdtw_single_clock"
    HDTW_HIERARCHICAL = "hdtw_hierarchical"


class AdditivePositionEmbedding(nn.Module):
    def __init__(self, model_dim: int, max_audio_tokens: int, max_lyric_tokens: int) -> None:
        super().__init__()
        self.audio = nn.Embedding(max_audio_tokens, model_dim)
        self.lyrics = nn.Embedding(max_lyric_tokens, model_dim)

    def forward(
        self, audio: Tensor, lyrics: Tensor, audio_mask: Tensor, lyric_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        audio_index = torch.arange(audio.shape[1], device=audio.device)
        lyric_index = torch.arange(lyrics.shape[1], device=lyrics.device)
        audio_out = audio + self.audio(audio_index)[None, :, :]
        lyric_out = lyrics + self.lyrics(lyric_index)[None, :, :]
        return (
            torch.where(audio_mask[:, :, None], audio_out, torch.zeros_like(audio_out)),
            torch.where(lyric_mask[:, :, None], lyric_out, torch.zeros_like(lyric_out)),
        )


def baseline_clock_output(
    *,
    mode: BaselineMode,
    component_names: Sequence[str],
    audio_mask: Tensor,
    lyric_mask: Tensor,
    audio_coordinates: Mapping[str, Tensor] | None = None,
    lyric_coordinates: Mapping[str, Tensor] | None = None,
    learned_clocks: ClockOutput | None = None,
) -> ClockOutput:
    batch, audio_length = audio_mask.shape
    lyric_length = lyric_mask.shape[1]
    component_count = len(component_names)
    audio_clock = torch.zeros((batch, audio_length, component_count), device=audio_mask.device)
    lyric_clock = torch.zeros((batch, lyric_length, component_count), device=lyric_mask.device)
    audio_valid = torch.zeros_like(audio_clock, dtype=torch.bool)
    lyric_valid = torch.zeros_like(lyric_clock, dtype=torch.bool)
    audio_coordinates = audio_coordinates or {}
    lyric_coordinates = lyric_coordinates or {}
    if mode == BaselineMode.NONE:
        pass
    elif mode == BaselineMode.RAW_INDEX_ROPE:
        audio_clock[..., 0] = normalized_index(audio_mask)
        lyric_clock[..., 0] = normalized_index(lyric_mask)
        audio_valid[..., 0] = audio_mask
        lyric_valid[..., 0] = lyric_mask
    elif mode == BaselineMode.TIMESTAMP_ROPE:
        if (
            "absolute_seconds" not in audio_coordinates
            or "absolute_seconds" not in lyric_coordinates
        ):
            raise ValueError("timestamp RoPE requires seconds for both modalities")
        audio_clock[..., 0] = normalize_unit_interval(
            audio_coordinates["absolute_seconds"], audio_mask
        )
        lyric_clock[..., 0] = normalize_unit_interval(
            lyric_coordinates["absolute_seconds"], lyric_mask
        )
        audio_valid[..., 0] = audio_mask
        lyric_valid[..., 0] = lyric_mask
    elif mode == BaselineMode.MULTI_CLOCK_ROPE:
        for index, name in enumerate(component_names):
            if name in audio_coordinates:
                audio_clock[..., index] = normalize_unit_interval(
                    audio_coordinates[name], audio_mask
                )
                audio_valid[..., index] = audio_mask
            if name in lyric_coordinates:
                lyric_clock[..., index] = normalize_unit_interval(
                    lyric_coordinates[name], lyric_mask
                )
                lyric_valid[..., index] = lyric_mask
    elif mode in {BaselineMode.HDTW_SINGLE_CLOCK, BaselineMode.HDTW_HIERARCHICAL}:
        if learned_clocks is None:
            raise ValueError("HDTW baselines require learned_clocks")
        if mode == BaselineMode.HDTW_HIERARCHICAL:
            return learned_clocks
        audio_clock[..., 0] = learned_clocks.source_clock[..., 0]
        lyric_clock[..., 0] = learned_clocks.target_clock[..., 0]
        audio_valid[..., 0] = learned_clocks.source_valid[..., 0]
        lyric_valid[..., 0] = learned_clocks.target_valid[..., 0]
    elif mode == BaselineMode.ADDITIVE:
        raise ValueError(
            "additive positions are applied to features, not represented as rotary clocks"
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported baseline mode: {mode}")
    return ClockOutput(
        source_clock=audio_clock,
        target_clock=lyric_clock,
        source_valid=audio_valid,
        target_valid=lyric_valid,
        diagnostics={
            "baseline_mode": torch.tensor(list(BaselineMode).index(mode), device=audio_mask.device)
        },
    )
