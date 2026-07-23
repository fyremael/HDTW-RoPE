"""Run manifests and deterministic execution controls."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _git_state(repository: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repository,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def package_versions(
    names: tuple[str, ...] = ("torch", "PyYAML", "hdtw-rope"),
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    repository: str | Path = ".",
    dataset_split_manifest: str | None = None,
    encoder_checkpoints: Mapping[str, str] | None = None,
    clock_schema_version: str = "hdtw-clock-v1",
) -> dict[str, Any]:
    commit, dirty = _git_state(Path(repository))
    return {
        "config": config,
        "git": {"commit": commit, "dirty": dirty},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
        "dataset_split_manifest": dataset_split_manifest,
        "encoder_checkpoints": dict(encoder_checkpoints or {}),
        "clock_schema_version": clock_schema_version,
        "process": {"pid": os.getpid()},
    }


def save_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix in {".yaml", ".yml"}:
        destination.write_text(yaml.safe_dump(dict(manifest), sort_keys=False))
    else:
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
