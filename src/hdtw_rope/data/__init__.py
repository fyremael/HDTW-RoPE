"""Data contracts and synthetic fixtures."""

from hdtw_rope.data.collate import collate_lyric_music
from hdtw_rope.data.schema import (
    CLOCK_SCHEMA_VERSION,
    DEFAULT_CLOCK_COMPONENTS,
    HierarchyLevel,
    LyricMusicBatch,
    LyricMusicSample,
    LyricUnitType,
)
from hdtw_rope.data.synthetic import SyntheticPair, generate_synthetic_pair

__all__ = [
    "CLOCK_SCHEMA_VERSION",
    "DEFAULT_CLOCK_COMPONENTS",
    "HierarchyLevel",
    "LyricMusicBatch",
    "LyricMusicSample",
    "LyricUnitType",
    "SyntheticPair",
    "collate_lyric_music",
    "generate_synthetic_pair",
]
