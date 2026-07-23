"""Synthetic lyric-music pairs with known monotone warps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from hdtw_rope.data.schema import LyricMusicSample, LyricUnitType

WarpFamily = Literal["affine", "piecewise", "pause", "smooth", "melisma", "missing"]


@dataclass(frozen=True)
class SyntheticPair:
    sample: LyricMusicSample
    audio_to_lyric_clock: Tensor
    lyric_to_audio_clock: Tensor
    latent_audio_coordinate: Tensor
    latent_lyric_coordinate: Tensor


def _monotone_warp(u: Tensor, family: WarpFamily) -> Tensor:
    if family == "affine":
        warped = 0.05 + 0.90 * u
    elif family == "piecewise":
        warped = torch.where(u < 0.4, 0.55 * u, 0.22 + 1.30 * (u - 0.4))
    elif family == "pause":
        warped = torch.where(
            u < 0.35,
            u,
            torch.where(u < 0.55, torch.full_like(u, 0.35), 0.35 + 1.45 * (u - 0.55)),
        )
    elif family == "smooth":
        warped = u + 0.08 * torch.sin(2.0 * torch.pi * u)
    elif family == "melisma":
        warped = torch.where(
            u < 0.45,
            u,
            torch.where(u < 0.72, 0.45 + 0.25 * (u - 0.45), 0.5175 + 1.72 * (u - 0.72)),
        )
    elif family == "missing":
        warped = torch.where(u < 0.45, u, 0.12 + 0.88 * u)
    else:  # pragma: no cover
        raise ValueError(f"unknown warp family: {family}")
    warped = warped - warped[0]
    return warped / warped[-1].clamp_min(torch.finfo(warped.dtype).eps)


def _latent_content(u: Tensor, dim: int) -> Tensor:
    frequencies = torch.arange(1, dim // 4 + 1, dtype=u.dtype, device=u.device)
    phase = 2.0 * torch.pi * u[:, None] * frequencies[None, :]
    features = torch.cat(
        (torch.sin(phase), torch.cos(phase), u[:, None], u[:, None] ** 2), dim=-1
    )
    if features.shape[1] < dim:
        features = torch.nn.functional.pad(features, (0, dim - features.shape[1]))
    return features[:, :dim]


def _interpolate_rows(grid: Tensor, values: Tensor, query: Tensor) -> Tensor:
    query = query.clamp(grid[0], grid[-1])
    right = torch.searchsorted(grid, query, right=False).clamp(1, grid.numel() - 1)
    left = right - 1
    denominator = (grid[right] - grid[left]).clamp_min(torch.finfo(grid.dtype).eps)
    weight = ((query - grid[left]) / denominator)[:, None]
    return values[left] * (1.0 - weight) + values[right] * weight


def generate_synthetic_pair(
    *,
    audio_length: int = 96,
    lyric_length: int = 48,
    feature_dim: int = 32,
    warp_family: WarpFamily = "piecewise",
    noise_std: float = 0.01,
    seed: int = 0,
) -> SyntheticPair:
    """Generate two modalities sampled under different monotone clocks."""

    if audio_length < 2 or lyric_length < 2:
        raise ValueError("synthetic lengths must be at least 2")
    generator = torch.Generator().manual_seed(seed)
    latent_grid = torch.linspace(0.0, 1.0, 2048)
    latent = _latent_content(latent_grid, feature_dim)
    audio_index = torch.linspace(0.0, 1.0, audio_length)
    lyric_index = torch.linspace(0.0, 1.0, lyric_length)
    latent_audio = _monotone_warp(audio_index, warp_family)
    latent_lyric = lyric_index
    audio_features = _interpolate_rows(latent_grid, latent, latent_audio)
    lyric_features = _interpolate_rows(latent_grid, latent, latent_lyric)
    audio_features = audio_features + noise_std * torch.randn(
        audio_features.shape, generator=generator
    )
    lyric_features = lyric_features + noise_std * torch.randn(
        lyric_features.shape, generator=generator
    )
    audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
    lyric_features = torch.nn.functional.normalize(lyric_features, dim=-1)

    audio_to_lyric = latent_audio
    lyric_to_audio = (
        torch.interp(lyric_index, latent_audio, audio_index)
        if hasattr(torch, "interp")
        else _inverse_monotone(latent_audio, audio_index, lyric_index)
    )
    audio_time = 30.0 * audio_index
    sample = LyricMusicSample(
        sample_id=f"synthetic-{warp_family}-{seed}",
        audio_features=audio_features,
        audio_mask=torch.ones(audio_length, dtype=torch.bool),
        audio_time_seconds=audio_time,
        lyric_features=lyric_features,
        lyric_mask=torch.ones(lyric_length, dtype=torch.bool),
        lyric_text=" ".join(f"word-{index}" for index in range(lyric_length)),
        lyric_unit_type=LyricUnitType.WORD,
        audio_coordinates={
            "absolute_seconds": audio_time,
            "beat": 16.0 * audio_index,
            "bar": 4.0 * audio_index,
        },
        lyric_coordinates={
            "aligned_word": lyric_index,
            "aligned_line": torch.floor(8.0 * lyric_index),
            "aligned_section": torch.floor(2.0 * lyric_index),
        },
        metadata={"warp_family": warp_family, "noise_std": noise_std},
    )
    sample.validate()
    return SyntheticPair(
        sample=sample,
        audio_to_lyric_clock=audio_to_lyric,
        lyric_to_audio_clock=lyric_to_audio,
        latent_audio_coordinate=latent_audio,
        latent_lyric_coordinate=latent_lyric,
    )


def _inverse_monotone(x: Tensor, y: Tensor, query: Tensor) -> Tensor:
    right = torch.searchsorted(x.contiguous(), query.contiguous(), right=False).clamp(
        1, x.numel() - 1
    )
    left = right - 1
    denominator = (x[right] - x[left]).clamp_min(torch.finfo(x.dtype).eps)
    weight = (query - x[left]) / denominator
    return y[left] * (1.0 - weight) + y[right] * weight
