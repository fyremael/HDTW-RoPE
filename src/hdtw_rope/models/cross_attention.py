"""Ordinary attention after DTW-clock unitary transport."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from hdtw_rope.rotary.transport import ClockRotaryEmbedding


class HDTWRoPECrossAttention(nn.Module):
    """Cross-attention preserving RoPE factorization after clock construction."""

    def __init__(self, *, model_dim: int, num_heads: int, rotary: ClockRotaryEmbedding, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        head_dim = model_dim // num_heads
        if rotary.rotary_dim > head_dim:
            raise ValueError("rotary_dim cannot exceed attention head dimension")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary = rotary
        self.q_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)

    def _heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, key_value: Tensor, query_mask: Tensor, key_value_mask: Tensor, query_clock: Tensor, key_clock: Tensor, query_clock_mask: Tensor, key_clock_mask: Tensor) -> tuple[Tensor, Mapping[str, Tensor]]:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("query and key_value must have shape [B,N,D]")
        if query.shape[0] != key_value.shape[0] or query.shape[-1] != self.model_dim or key_value.shape[-1] != self.model_dim:
            raise ValueError("cross-attention feature shapes are incompatible")
        if query_mask.shape != query.shape[:2] or key_value_mask.shape != key_value.shape[:2]:
            raise ValueError("sequence masks have incompatible shapes")
        if query_mask.dtype is not torch.bool or key_value_mask.dtype is not torch.bool:
            raise ValueError("sequence masks must be bool")
        if torch.any(query_mask.any(dim=-1) & ~key_value_mask.any(dim=-1)):
            raise ValueError("a valid query sequence has no valid key/value tokens")
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(key_value))
        v = self._heads(self.v_proj(key_value))
        q, k = self.rotary(q, k, query_clock, key_clock, query_clock_mask, key_clock_mask)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.head_dim ** -0.5)
        key_mask = key_value_mask[:, None, None, :]
        logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = torch.where(key_mask, probabilities, torch.zeros_like(probabilities))
        probabilities = torch.where(query_mask[:, None, :, None], probabilities, torch.zeros_like(probabilities))
        probabilities = self.attention_dropout(probabilities)
        attended = torch.matmul(probabilities.to(v.dtype), v)
        attended = attended.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.model_dim)
        output = self.out_proj(attended)
        output = torch.where(query_mask[:, :, None], output, torch.zeros_like(output))
        if not torch.isfinite(output).all() or not torch.isfinite(probabilities).all():
            raise ValueError("attention produced NaN or Inf")
        entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        return output, {
            "attention_probabilities": probabilities,
            "attention_entropy": entropy,
            "attention_logits_max": logits.masked_fill(~key_mask, -torch.inf).amax(dim=-1),
        }


class BidirectionalCrossAttentionLayer(nn.Module):
    """Symmetric audio↔lyrics transport and attention layer."""

    def __init__(self, *, model_dim: int, num_heads: int, rotary: ClockRotaryEmbedding, dropout: float = 0.0, feedforward_multiplier: int = 4) -> None:
        super().__init__()
        self.audio_norm = nn.LayerNorm(model_dim)
        self.lyric_norm = nn.LayerNorm(model_dim)
        self.audio_to_lyrics = HDTWRoPECrossAttention(model_dim=model_dim, num_heads=num_heads, rotary=rotary, dropout=dropout)
        self.lyrics_to_audio = HDTWRoPECrossAttention(model_dim=model_dim, num_heads=num_heads, rotary=rotary, dropout=dropout)
        hidden_dim = model_dim * feedforward_multiplier
        self.audio_ffn_norm = nn.LayerNorm(model_dim)
        self.lyric_ffn_norm = nn.LayerNorm(model_dim)
        self.audio_ffn = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, model_dim), nn.Dropout(dropout))
        self.lyric_ffn = nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, model_dim), nn.Dropout(dropout))

    def forward(self, audio: Tensor, lyrics: Tensor, audio_mask: Tensor, lyric_mask: Tensor, audio_clock: Tensor, lyric_clock: Tensor, audio_clock_mask: Tensor, lyric_clock_mask: Tensor) -> tuple[Tensor, Tensor, Mapping[str, Tensor]]:
        audio_delta, audio_diag = self.audio_to_lyrics(self.audio_norm(audio), self.lyric_norm(lyrics), audio_mask, lyric_mask, audio_clock, lyric_clock, audio_clock_mask, lyric_clock_mask)
        lyric_delta, lyric_diag = self.lyrics_to_audio(self.lyric_norm(lyrics), self.audio_norm(audio), lyric_mask, audio_mask, lyric_clock, audio_clock, lyric_clock_mask, audio_clock_mask)
        audio = audio + audio_delta
        lyrics = lyrics + lyric_delta
        audio = audio + self.audio_ffn(self.audio_ffn_norm(audio))
        lyrics = lyrics + self.lyric_ffn(self.lyric_ffn_norm(lyrics))
        audio = torch.where(audio_mask[:, :, None], audio, torch.zeros_like(audio))
        lyrics = torch.where(lyric_mask[:, :, None], lyrics, torch.zeros_like(lyrics))
        diagnostics = {**{f"audio_to_lyrics/{key}": value for key, value in audio_diag.items()}, **{f"lyrics_to_audio/{key}": value for key, value in lyric_diag.items()}}
        return audio, lyrics, diagnostics
