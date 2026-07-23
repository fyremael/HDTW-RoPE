"""Validated sample and hierarchy schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

import torch
from torch import Tensor

from hdtw_rope.errors import InvalidHierarchyError, InvalidShapeError
from hdtw_rope.validation import require_finite, require_rank

CLOCK_SCHEMA_VERSION = "hdtw-clock-v1"
DEFAULT_CLOCK_COMPONENTS: tuple[str, ...] = (
    "absolute_seconds",
    "beat",
    "bar",
    "section_occurrence",
    "section_relative",
    "aligned_syllable",
    "aligned_word",
    "aligned_line",
    "aligned_section",
)


class LyricUnitType(StrEnum):
    PHONEME = "phoneme"
    SYLLABLE = "syllable"
    WORD = "word"
    MIXED = "mixed"


@dataclass(frozen=True)
class HierarchyLevel:
    """One supervised hierarchy level for a single sample."""

    unit_start: Tensor
    unit_end: Tensor
    parent_id: Tensor
    type_id: Tensor
    occurrence_id: Tensor
    valid_mask: Tensor

    def validate(self, *, name: str = "hierarchy") -> None:
        tensors = {
            "unit_start": self.unit_start,
            "unit_end": self.unit_end,
            "parent_id": self.parent_id,
            "type_id": self.type_id,
            "occurrence_id": self.occurrence_id,
            "valid_mask": self.valid_mask,
        }
        for key, tensor in tensors.items():
            require_rank(f"{name}.{key}", tensor, 1)
        sizes = {tensor.shape[0] for tensor in tensors.values()}
        if len(sizes) != 1:
            raise InvalidHierarchyError(f"{name} tensors must have the same length")
        if self.valid_mask.dtype is not torch.bool:
            raise InvalidHierarchyError(f"{name}.valid_mask must be bool")
        for key in ("parent_id", "type_id", "occurrence_id"):
            if tensors[key].dtype not in (torch.int32, torch.int64):
                raise InvalidHierarchyError(f"{name}.{key} must be integer")
        require_finite(f"{name}.unit_start", self.unit_start, self.valid_mask)
        require_finite(f"{name}.unit_end", self.unit_end, self.valid_mask)
        if torch.any(self.unit_end[self.valid_mask] < self.unit_start[self.valid_mask]):
            raise InvalidHierarchyError(f"{name} has an end before its start")
        n = self.unit_start.shape[0]
        valid_parent = (self.parent_id == -1) | ((self.parent_id >= 0) & (self.parent_id < n))
        if not torch.all(valid_parent | ~self.valid_mask):
            raise InvalidHierarchyError(f"{name} contains an out-of-range parent id")


@dataclass(frozen=True)
class LyricMusicSample:
    """Single paired feature sample with physical metadata retained."""

    sample_id: str
    audio_features: Tensor
    audio_mask: Tensor
    audio_time_seconds: Tensor
    lyric_features: Tensor
    lyric_mask: Tensor
    lyric_text: str
    lyric_unit_type: LyricUnitType
    hierarchy: Mapping[str, HierarchyLevel] = field(default_factory=dict)
    audio_coordinates: Mapping[str, Tensor] = field(default_factory=dict)
    lyric_coordinates: Mapping[str, Tensor] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        require_rank("audio_features", self.audio_features, 2)
        require_rank("lyric_features", self.lyric_features, 2)
        require_rank("audio_mask", self.audio_mask, 1)
        require_rank("lyric_mask", self.lyric_mask, 1)
        require_rank("audio_time_seconds", self.audio_time_seconds, 1)
        if self.audio_mask.dtype is not torch.bool or self.lyric_mask.dtype is not torch.bool:
            raise InvalidShapeError("audio_mask and lyric_mask must be bool")
        if self.audio_features.shape[0] != self.audio_mask.shape[0]:
            raise InvalidShapeError("audio feature length and mask length differ")
        if self.audio_time_seconds.shape[0] != self.audio_mask.shape[0]:
            raise InvalidShapeError("audio time length and mask length differ")
        if self.lyric_features.shape[0] != self.lyric_mask.shape[0]:
            raise InvalidShapeError("lyric feature length and mask length differ")
        require_finite(
            "audio_features",
            self.audio_features,
            self.audio_mask[:, None].expand_as(self.audio_features),
        )
        require_finite(
            "lyric_features",
            self.lyric_features,
            self.lyric_mask[:, None].expand_as(self.lyric_features),
        )
        require_finite("audio_time_seconds", self.audio_time_seconds, self.audio_mask)
        if self.audio_time_seconds[self.audio_mask].numel() > 1:
            delta = torch.diff(self.audio_time_seconds[self.audio_mask])
            if torch.any(delta < 0):
                raise InvalidShapeError("audio_time_seconds must be nondecreasing")
        for name, level in self.hierarchy.items():
            level.validate(name=f"hierarchy[{name!r}]")
        for modality, coordinates, mask in (
            ("audio", self.audio_coordinates, self.audio_mask),
            ("lyric", self.lyric_coordinates, self.lyric_mask),
        ):
            for name, coordinate in coordinates.items():
                require_rank(f"{modality}_coordinates[{name!r}]", coordinate, 1)
                if coordinate.shape[0] != mask.shape[0]:
                    raise InvalidShapeError(f"{modality} coordinate {name!r} has wrong length")
                require_finite(f"{modality}_coordinates[{name!r}]", coordinate, mask)


@dataclass(frozen=True)
class LyricMusicBatch:
    sample_ids: tuple[str, ...]
    audio_features: Tensor
    audio_mask: Tensor
    audio_time_seconds: Tensor
    lyric_features: Tensor
    lyric_mask: Tensor
    lyric_texts: tuple[str, ...]
    lyric_unit_types: tuple[LyricUnitType, ...]
    audio_coordinates: Mapping[str, Tensor]
    lyric_coordinates: Mapping[str, Tensor]
    audio_coordinate_masks: Mapping[str, Tensor]
    lyric_coordinate_masks: Mapping[str, Tensor]
    hierarchy: Mapping[str, Mapping[str, Tensor]]
    metadata: tuple[Mapping[str, object], ...]

    def to(self, device: torch.device | str) -> "LyricMusicBatch":
        def move_mapping(mapping: Mapping[str, Tensor]) -> dict[str, Tensor]:
            return {key: value.to(device) for key, value in mapping.items()}

        hierarchy = {
            level: {key: value.to(device) for key, value in fields.items()}
            for level, fields in self.hierarchy.items()
        }
        return LyricMusicBatch(
            sample_ids=self.sample_ids,
            audio_features=self.audio_features.to(device),
            audio_mask=self.audio_mask.to(device),
            audio_time_seconds=self.audio_time_seconds.to(device),
            lyric_features=self.lyric_features.to(device),
            lyric_mask=self.lyric_mask.to(device),
            lyric_texts=self.lyric_texts,
            lyric_unit_types=self.lyric_unit_types,
            audio_coordinates=move_mapping(self.audio_coordinates),
            lyric_coordinates=move_mapping(self.lyric_coordinates),
            audio_coordinate_masks=move_mapping(self.audio_coordinate_masks),
            lyric_coordinate_masks=move_mapping(self.lyric_coordinate_masks),
            hierarchy=hierarchy,
            metadata=self.metadata,
        )
