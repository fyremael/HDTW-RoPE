"""Reference offline-clock HDTW-RoPE model and differentiable clock builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hdtw_rope.alignment.costs import PairwiseCost
from hdtw_rope.alignment.soft_dtw import SoftDTWAligner
from hdtw_rope.clocks.extract import LatentClockExtractor
from hdtw_rope.models.cross_attention import BidirectionalCrossAttentionLayer
from hdtw_rope.models.projections import ClockAdapter, FeatureProjection
from hdtw_rope.rotary.frequencies import ClockFrequencyMap, FrequencyMode
from hdtw_rope.rotary.transport import ClockRotaryEmbedding
from hdtw_rope.types import AlignmentOutput, ClockOutput


@dataclass(frozen=True)
class ModelOutput:
    audio_states: Tensor
    lyric_states: Tensor
    audio_embedding: Tensor
    lyric_embedding: Tensor
    similarity: Tensor
    diagnostics: Mapping[str, Tensor]


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class DifferentiableClockBuilder(nn.Module):
    """Single-resolution differentiable path used before hierarchy expansion."""

    def __init__(
        self,
        *,
        audio_dim: int,
        lyric_dim: int,
        shared_dim: int,
        components: Sequence[str],
        aligned_components: Sequence[str],
        temperature: float = 0.1,
        min_alignment_mass: float = 1e-6,
        differentiable_mass: bool = False,
    ) -> None:
        super().__init__()
        self.audio_projection = FeatureProjection(audio_dim, shared_dim)
        self.lyric_projection = FeatureProjection(lyric_dim, shared_dim)
        self.cost = PairwiseCost("cosine")
        self.aligner = SoftDTWAligner(
            temperature=temperature,
            min_alignment_mass=min_alignment_mass,
            differentiable_mass=differentiable_mass,
        )
        self.extractor = LatentClockExtractor(
            components, min_alignment_mass=min_alignment_mass, monotonicity_mode="penalty"
        )
        self.aligned_components = tuple(aligned_components)

    def forward(
        self,
        audio: Tensor,
        lyrics: Tensor,
        audio_mask: Tensor,
        lyric_mask: Tensor,
        source_coordinates: Mapping[str, Tensor],
        target_coordinates: Mapping[str, Tensor],
        source_coordinate_masks: Mapping[str, Tensor],
        target_coordinate_masks: Mapping[str, Tensor],
        band_mask: Tensor | None = None,
    ) -> tuple[ClockOutput, AlignmentOutput]:
        audio_common = self.audio_projection(audio)
        lyric_common = self.lyric_projection(lyrics)
        cost = self.cost(audio_common, lyric_common, audio_mask, lyric_mask)
        alignment = self.aligner(cost, audio_mask, lyric_mask, band_mask)
        alignments = {component: alignment for component in self.aligned_components}
        clocks = self.extractor(
            alignments,
            source_coordinates,
            target_coordinates,
            source_coordinate_masks,
            target_coordinate_masks,
            hierarchy={},
        )
        return clocks, alignment


class HDTWRoPEModel(nn.Module):
    """Frozen-feature cross-modal model driven by external or learned clocks."""

    def __init__(
        self,
        *,
        audio_dim: int,
        lyric_dim: int,
        model_dim: int,
        num_heads: int,
        rotary_dim: int,
        clock_components: Sequence[str],
        cross_attention_layers: int = 2,
        dropout: float = 0.1,
        frequency_mode: FrequencyMode = "clock_partitioned",
        partition: Mapping[str, float] | None = None,
        family_components: Mapping[str, Sequence[str]] | None = None,
        learn_clock_adapters: bool = False,
    ) -> None:
        super().__init__()
        if cross_attention_layers <= 0:
            raise ValueError("cross_attention_layers must be positive")
        self.clock_components = tuple(clock_components)
        self.audio_input = FeatureProjection(audio_dim, model_dim, dropout=dropout)
        self.lyric_input = FeatureProjection(lyric_dim, model_dim, dropout=dropout)
        self.audio_clock_adapter = ClockAdapter(
            len(clock_components), learnable=learn_clock_adapters
        )
        self.lyric_clock_adapter = ClockAdapter(
            len(clock_components), learnable=learn_clock_adapters
        )
        frequency_map = ClockFrequencyMap(
            num_heads=num_heads,
            rotary_dim=rotary_dim,
            component_names=clock_components,
            mode=frequency_mode,
            partition=partition,
            family_components=family_components,
        )
        rotary = ClockRotaryEmbedding(frequency_map)
        self.layers = nn.ModuleList(
            BidirectionalCrossAttentionLayer(
                model_dim=model_dim, num_heads=num_heads, rotary=rotary, dropout=dropout
            )
            for _ in range(cross_attention_layers)
        )
        self.audio_output_norm = nn.LayerNorm(model_dim)
        self.lyric_output_norm = nn.LayerNorm(model_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())

    def forward(
        self,
        audio: Tensor,
        lyrics: Tensor,
        audio_mask: Tensor,
        lyric_mask: Tensor,
        clocks: ClockOutput,
    ) -> ModelOutput:
        audio_states = self.audio_input(audio)
        lyric_states = self.lyric_input(lyrics)
        audio_clock = self.audio_clock_adapter(clocks.source_clock, clocks.source_valid)
        lyric_clock = self.lyric_clock_adapter(clocks.target_clock, clocks.target_valid)
        diagnostics: dict[str, Tensor] = dict(clocks.diagnostics)
        for index, layer in enumerate(self.layers):
            audio_states, lyric_states, layer_diagnostics = layer(
                audio_states,
                lyric_states,
                audio_mask,
                lyric_mask,
                audio_clock,
                lyric_clock,
                clocks.source_valid,
                clocks.target_valid,
            )
            diagnostics.update(
                {f"layer_{index}/{key}": value for key, value in layer_diagnostics.items()}
            )
        audio_states = self.audio_output_norm(audio_states)
        lyric_states = self.lyric_output_norm(lyric_states)
        audio_embedding = torch.nn.functional.normalize(
            masked_mean(audio_states, audio_mask), dim=-1
        )
        lyric_embedding = torch.nn.functional.normalize(
            masked_mean(lyric_states, lyric_mask), dim=-1
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        similarity = scale * audio_embedding @ lyric_embedding.transpose(0, 1)
        diagnostics["logit_scale"] = scale.detach()
        return ModelOutput(
            audio_states=audio_states,
            lyric_states=lyric_states,
            audio_embedding=audio_embedding,
            lyric_embedding=lyric_embedding,
            similarity=similarity,
            diagnostics=diagnostics,
        )
