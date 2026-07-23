"""Alignment, retrieval, and stress-test metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlignmentMetrics:
    mean_absolute_error_seconds: float
    median_absolute_error_seconds: float
    within_tolerance: Mapping[float, float]


def word_onset_metrics(predicted_seconds: Tensor, target_seconds: Tensor, mask: Tensor, *, tolerances: Sequence[float] = (0.1, 0.3, 0.5, 1.0)) -> AlignmentMetrics:
    if not (predicted_seconds.shape == target_seconds.shape == mask.shape):
        raise ValueError("onset metric tensors must share a shape")
    error = (predicted_seconds - target_seconds).abs()[mask]
    if error.numel() == 0:
        raise ValueError("onset metrics require at least one valid label")
    return AlignmentMetrics(mean_absolute_error_seconds=float(error.mean().item()), median_absolute_error_seconds=float(error.median().item()), within_tolerance={float(tol): float((error <= tol).float().mean().item()) for tol in tolerances})


def boundary_f1(predicted: Tensor, target: Tensor, mask: Tensor, *, tolerance: float) -> float:
    predicted_values = predicted[mask].detach().cpu().sort().values.tolist()
    target_values = target[mask].detach().cpu().sort().values.tolist()
    matched_target: set[int] = set()
    true_positive = 0
    for value in predicted_values:
        candidates = [(abs(value - target_value), index) for index, target_value in enumerate(target_values) if index not in matched_target and abs(value - target_value) <= tolerance]
        if candidates:
            _, index = min(candidates)
            matched_target.add(index)
            true_positive += 1
    precision = true_positive / max(len(predicted_values), 1)
    recall = true_positive / max(len(target_values), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def retrieval_metrics(similarity: Tensor, *, k_values: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("retrieval metrics require a square paired similarity matrix")
    count = similarity.shape[0]
    target = torch.arange(count, device=similarity.device)
    def direction(matrix: Tensor, prefix: str) -> dict[str, float]:
        order = matrix.argsort(dim=-1, descending=True)
        ranks = (order == target[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        result = {f"{prefix}/median_rank": float(ranks.float().median().item()), f"{prefix}/mrr": float((1.0 / ranks.float()).mean().item())}
        for k in k_values:
            result[f"{prefix}/recall_at_{k}"] = float((ranks <= min(k, count)).float().mean().item())
        return result
    return {**direction(similarity, "audio_to_lyric"), **direction(similarity.transpose(0, 1), "lyric_to_audio")}


def normalized_path_deviation(predicted_mass: Tensor, target_mass: Tensor, valid_pairs: Tensor) -> Tensor:
    if not (predicted_mass.shape == target_mass.shape == valid_pairs.shape):
        raise ValueError("path-deviation tensors must share a shape")
    return ((predicted_mass - target_mass).abs() * valid_pairs).sum() / valid_pairs.sum().clamp_min(1)


def paired_bootstrap_difference(candidate: Tensor, baseline: Tensor, *, samples: int = 10000, seed: int = 1337, confidence: float = 0.95) -> tuple[float, float, float]:
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("bootstrap inputs must be paired 1-D tensors")
    generator = torch.Generator(device=candidate.device).manual_seed(seed)
    indices = torch.randint(0, candidate.numel(), (samples, candidate.numel()), generator=generator, device=candidate.device)
    difference = candidate - baseline
    distribution = difference[indices].mean(dim=-1)
    alpha = (1.0 - confidence) / 2.0
    lower = torch.quantile(distribution, alpha)
    upper = torch.quantile(distribution, 1.0 - alpha)
    return float(difference.mean().item()), float(lower.item()), float(upper.item())
