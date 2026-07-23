"""Configuration loading and fail-closed validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    config = deepcopy(data)
    validate_config(config)
    return config


def _require(mapping: Mapping[str, Any], key: str, expected_type: type, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing {context}.{key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise ConfigError(f"{context}.{key} must be {expected_type.__name__}")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    project = _require(config, "project", dict, "config")
    runtime = _require(config, "runtime", dict, "config")
    alignment = _require(config, "alignment", dict, "config")
    clock = _require(config, "clock", dict, "config")
    rotary = _require(config, "rotary", dict, "config")
    model = _require(config, "model", dict, "config")
    _require(config, "loss", dict, "config")
    training = _require(config, "training", dict, "config")
    _require(config, "evaluation", dict, "config")

    _require(project, "seed", int, "project")
    if (
        runtime.get("alignment_precision") != "float32"
        or runtime.get("phase_precision") != "float32"
    ):
        raise ConfigError("alignment and phase precision must be float32 in version 0.1")
    if alignment.get("method") not in {"hard_dtw", "soft_dtw"}:
        raise ConfigError("alignment.method must be hard_dtw or soft_dtw")
    temperature = _require(alignment, "temperature", dict, "alignment")
    if float(temperature.get("initial", 0.0)) <= 0 or float(temperature.get("final", 0.0)) <= 0:
        raise ConfigError("Soft-DTW temperatures must be positive")
    components = _require(clock, "components", list, "clock")
    if not components or len(components) != len(set(components)):
        raise ConfigError("clock.components must be a nonempty unique list")
    model_dim = int(rotary.get("model_dim", 0))
    heads = int(rotary.get("num_heads", 0))
    rotary_dim = int(rotary.get("rotary_dim", 0))
    if model_dim <= 0 or heads <= 0 or model_dim % heads:
        raise ConfigError("rotary.model_dim must be positive and divisible by num_heads")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > model_dim // heads:
        raise ConfigError("rotary_dim must be positive, even, and no larger than head dimension")
    if not bool(rotary.get("shared_between_modalities", False)):
        raise ConfigError("version 0.1 requires a shared modality frequency map")
    partition = rotary.get("partition", {})
    if rotary.get("frequency_mode") == "clock_partitioned":
        if (
            not isinstance(partition, dict)
            or sum(float(value) for value in partition.values()) > 1.0 + 1e-6
        ):
            raise ConfigError("rotary.partition must be a mapping summing to at most one")
    if int(model.get("cross_attention_layers", 0)) <= 0:
        raise ConfigError("model.cross_attention_layers must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ConfigError("training.batch_size must be positive")
