from __future__ import annotations

import torch

from hdtw_rope.alignment.hierarchical import HierarchicalAligner, MultiLevelHierarchicalAligner


def test_multilevel_hierarchical_aligner_shapes() -> None:
    module = MultiLevelHierarchicalAligner({"section": HierarchicalAligner(source_dim=8, target_dim=8, shared_dim=6, temperature=0.2), "word": HierarchicalAligner(source_dim=8, target_dim=8, shared_dim=6, temperature=0.2)}, level_order=["section", "word"])
    source = {"section": torch.randn(1, 3, 8), "word": torch.randn(1, 5, 8)}
    target = {"section": torch.randn(1, 2, 8), "word": torch.randn(1, 4, 8)}
    source_masks = {"section": torch.ones(1, 3, dtype=torch.bool), "word": torch.ones(1, 5, dtype=torch.bool)}
    target_masks = {"section": torch.ones(1, 2, dtype=torch.bool), "word": torch.ones(1, 4, dtype=torch.bool)}
    output = module(source, target, source_masks, target_masks)
    assert output["section"].mass.shape == (1, 3, 2)
    assert output["word"].mass.shape == (1, 5, 4)
