from __future__ import annotations

from pathlib import Path

from hdtw_rope.config import load_config


def test_reference_config_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "prototype.yaml")
    assert config["rotary"]["shared_between_modalities"] is True
