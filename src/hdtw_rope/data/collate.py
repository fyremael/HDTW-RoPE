"""Mask-preserving batch collation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from hdtw_rope.data.schema import LyricMusicBatch, LyricMusicSample


def _pad_sequence(values: Sequence[Tensor], *, padding_value: float | int | bool = 0) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        list(values), batch_first=True, padding_value=padding_value
    )


def collate_lyric_music(samples: Sequence[LyricMusicSample]) -> LyricMusicBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    for sample in samples:
        sample.validate()

    audio_coordinate_names = sorted(
        {name for sample in samples for name in sample.audio_coordinates}
    )
    lyric_coordinate_names = sorted(
        {name for sample in samples for name in sample.lyric_coordinates}
    )

    def collect_coordinates(
        modality: str,
        names: list[str],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        values: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for name in names:
            rows: list[Tensor] = []
            row_masks: list[Tensor] = []
            for sample in samples:
                base_mask = sample.audio_mask if modality == "audio" else sample.lyric_mask
                mapping = (
                    sample.audio_coordinates
                    if modality == "audio"
                    else sample.lyric_coordinates
                )
                if name in mapping:
                    rows.append(mapping[name])
                    row_masks.append(base_mask.clone())
                else:
                    rows.append(torch.zeros(base_mask.shape[0], dtype=torch.float32))
                    row_masks.append(torch.zeros_like(base_mask))
            values[name] = _pad_sequence(rows, padding_value=0.0)
            masks[name] = _pad_sequence(row_masks, padding_value=False)
        return values, masks

    audio_coordinates, audio_coordinate_masks = collect_coordinates(
        "audio", audio_coordinate_names
    )
    lyric_coordinates, lyric_coordinate_masks = collect_coordinates(
        "lyric", lyric_coordinate_names
    )

    hierarchy_names = sorted({name for sample in samples for name in sample.hierarchy})
    hierarchy: dict[str, dict[str, Tensor]] = {}
    for name in hierarchy_names:
        fields: dict[str, list[Tensor]] = {
            "unit_start": [],
            "unit_end": [],
            "parent_id": [],
            "type_id": [],
            "occurrence_id": [],
            "valid_mask": [],
        }
        for sample in samples:
            level = sample.hierarchy.get(name)
            if level is None:
                fields["unit_start"].append(torch.zeros(0))
                fields["unit_end"].append(torch.zeros(0))
                fields["parent_id"].append(torch.full((0,), -1, dtype=torch.long))
                fields["type_id"].append(torch.zeros(0, dtype=torch.long))
                fields["occurrence_id"].append(torch.zeros(0, dtype=torch.long))
                fields["valid_mask"].append(torch.zeros(0, dtype=torch.bool))
            else:
                for key in fields:
                    fields[key].append(getattr(level, key))
        hierarchy[name] = {
            "unit_start": _pad_sequence(fields["unit_start"], padding_value=0.0),
            "unit_end": _pad_sequence(fields["unit_end"], padding_value=0.0),
            "parent_id": _pad_sequence(fields["parent_id"], padding_value=-1),
            "type_id": _pad_sequence(fields["type_id"], padding_value=-1),
            "occurrence_id": _pad_sequence(fields["occurrence_id"], padding_value=-1),
            "valid_mask": _pad_sequence(fields["valid_mask"], padding_value=False),
        }

    return LyricMusicBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        audio_features=_pad_sequence([sample.audio_features for sample in samples]),
        audio_mask=_pad_sequence(
            [sample.audio_mask for sample in samples], padding_value=False
        ),
        audio_time_seconds=_pad_sequence(
            [sample.audio_time_seconds for sample in samples]
        ),
        lyric_features=_pad_sequence([sample.lyric_features for sample in samples]),
        lyric_mask=_pad_sequence(
            [sample.lyric_mask for sample in samples], padding_value=False
        ),
        lyric_texts=tuple(sample.lyric_text for sample in samples),
        lyric_unit_types=tuple(sample.lyric_unit_type for sample in samples),
        audio_coordinates=audio_coordinates,
        lyric_coordinates=lyric_coordinates,
        audio_coordinate_masks=audio_coordinate_masks,
        lyric_coordinate_masks=lyric_coordinate_masks,
        hierarchy=hierarchy,
        metadata=tuple(sample.metadata for sample in samples),
    )
