from __future__ import annotations

import torch

from hdtw_rope.models.cross_attention import HDTWRoPECrossAttention
from hdtw_rope.rotary.frequencies import ClockFrequencyMap
from hdtw_rope.rotary.transport import ClockRotaryEmbedding


def test_padding_receives_no_attention_probability() -> None:
    frequency = ClockFrequencyMap(num_heads=2, rotary_dim=4, component_names=["position"], mode="fixed_log")
    module = HDTWRoPECrossAttention(model_dim=16, num_heads=2, rotary=ClockRotaryEmbedding(frequency))
    query = torch.randn(1, 3, 16)
    key_value = torch.randn(1, 4, 16)
    query_mask = torch.tensor([[True, True, False]])
    key_mask = torch.tensor([[True, True, False, False]])
    q_clock = torch.arange(3).float().view(1, 3, 1)
    k_clock = torch.arange(4).float().view(1, 4, 1)
    output, diagnostics = module(query, key_value, query_mask, key_mask, q_clock, k_clock, query_mask.unsqueeze(-1), key_mask.unsqueeze(-1))
    probability = diagnostics["attention_probabilities"]
    assert not probability[..., 2:].any()
    assert not probability[:, :, 2].any()
    assert not output[:, 2].any()
