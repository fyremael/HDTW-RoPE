"""Reference command-line workflows over deterministic synthetic data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from hdtw_rope.alignment.bands import diagonal_band_mask
from hdtw_rope.config import load_config
from hdtw_rope.data.collate import collate_lyric_music
from hdtw_rope.data.schema import CLOCK_SCHEMA_VERSION
from hdtw_rope.data.synthetic import generate_synthetic_pair
from hdtw_rope.diagnostics import alignment_diagnostics, clock_diagnostics, write_metrics
from hdtw_rope.losses import HDTWRoPELoss
from hdtw_rope.metrics import retrieval_metrics
from hdtw_rope.models.hdtw_rope_model import DifferentiableClockBuilder, HDTWRoPEModel
from hdtw_rope.reproducibility import build_run_manifest, save_run_manifest, seed_everything

FAMILY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "absolute_seconds": ("absolute_seconds",),
    "beat": ("beat",),
    "bar": ("bar",),
    "lyric_local": ("aligned_syllable", "aligned_word"),
    "phrase_line": ("aligned_line",),
    "section_form": ("section_occurrence", "section_relative", "aligned_section"),
}


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["runtime"]["device"])
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _synthetic_batch(config: Mapping[str, Any], batch_size: int | None = None):
    count = batch_size or int(config["training"]["batch_size"])
    feature_dim = int(config["input"]["audio_feature_dim"])
    max_audio = min(int(config["input"]["max_audio_tokens"]), 96)
    max_lyrics = min(int(config["input"]["max_lyric_tokens"]), 48)
    families = ("affine", "piecewise", "pause", "smooth", "melisma", "missing")
    samples = [
        generate_synthetic_pair(
            audio_length=max_audio,
            lyric_length=max_lyrics,
            feature_dim=feature_dim,
            warp_family=families[index % len(families)],  # type: ignore[arg-type]
            seed=int(config["project"]["seed"]) + index,
        ).sample
        for index in range(count)
    ]
    return collate_lyric_music(samples)


def _clock_inputs(batch, components: list[str]):
    source_coordinates = dict(batch.audio_coordinates)
    target_coordinates = dict(batch.lyric_coordinates)
    source_masks = dict(batch.audio_coordinate_masks)
    target_masks = dict(batch.lyric_coordinate_masks)
    source_coordinates.setdefault("absolute_seconds", batch.audio_time_seconds)
    source_masks.setdefault("absolute_seconds", batch.audio_mask)
    aligned_components = [
        component
        for component in components
        if component in source_coordinates or component in target_coordinates
    ]
    return source_coordinates, target_coordinates, source_masks, target_masks, aligned_components


def _build_modules(config: Mapping[str, Any], device: torch.device):
    components = list(config["clock"]["components"])
    source_dim = int(config["input"]["audio_feature_dim"])
    target_dim = int(config["input"]["lyric_feature_dim"])
    builder = DifferentiableClockBuilder(
        audio_dim=source_dim,
        lyric_dim=target_dim,
        shared_dim=int(config["alignment"]["shared_dim"]),
        components=components,
        aligned_components=components,
        temperature=float(config["alignment"]["temperature"]["initial"]),
        min_alignment_mass=float(config["alignment"]["min_alignment_mass"]),
        differentiable_mass=False,
    ).to(device)
    model = HDTWRoPEModel(
        audio_dim=source_dim,
        lyric_dim=target_dim,
        model_dim=int(config["rotary"]["model_dim"]),
        num_heads=int(config["rotary"]["num_heads"]),
        rotary_dim=int(config["rotary"]["rotary_dim"]),
        clock_components=components,
        cross_attention_layers=int(config["model"]["cross_attention_layers"]),
        dropout=float(config["model"]["dropout"]),
        frequency_mode=str(config["rotary"]["frequency_mode"]),
        partition=config["rotary"].get("partition"),
        family_components=FAMILY_COMPONENTS,
        learn_clock_adapters=False,
    ).to(device)
    return builder, model


def _forward(config: Mapping[str, Any], batch, builder, model):
    source_coordinates, target_coordinates, source_masks, target_masks, aligned_components = (
        _clock_inputs(batch, list(config["clock"]["components"]))
    )
    builder.aligned_components = tuple(aligned_components)
    band = None
    if bool(config["alignment"]["band"]["enabled"]):
        band = diagonal_band_mask(
            batch.audio_mask,
            batch.lyric_mask,
            min(int(config["alignment"]["band"]["half_width"]), batch.lyric_mask.shape[1]),
        )
    clocks, alignment = builder(
        batch.audio_features,
        batch.lyric_features,
        batch.audio_mask,
        batch.lyric_mask,
        source_coordinates,
        target_coordinates,
        source_masks,
        target_masks,
        band,
    )
    output = model(
        batch.audio_features, batch.lyric_features, batch.audio_mask, batch.lyric_mask, clocks
    )
    alignments = {component: alignment for component in aligned_components}
    return clocks, alignment, alignments, output


def precompute_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute deterministic HDTW clocks")
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--output", default="artifacts/alignments/synthetic-clocks.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]), deterministic=True)
    device = _device(config)
    batch = _synthetic_batch(config, args.batch_size).to(device)
    builder, model = _build_modules(config, device)
    del model
    source_coordinates, target_coordinates, source_masks, target_masks, aligned_components = (
        _clock_inputs(batch, list(config["clock"]["components"]))
    )
    builder.aligned_components = tuple(aligned_components)
    band = diagonal_band_mask(
        batch.audio_mask, batch.lyric_mask, min(16, batch.lyric_mask.shape[1])
    )
    clocks, alignment = builder(
        batch.audio_features,
        batch.lyric_features,
        batch.audio_mask,
        batch.lyric_mask,
        source_coordinates,
        target_coordinates,
        source_masks,
        target_masks,
        band,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CLOCK_SCHEMA_VERSION,
            "sample_ids": batch.sample_ids,
            "source_clock": clocks.source_clock.detach().cpu(),
            "target_clock": clocks.target_clock.detach().cpu(),
            "source_valid": clocks.source_valid.cpu(),
            "target_valid": clocks.target_valid.cpu(),
            "alignment_mass": alignment.mass.detach().cpu(),
        },
        destination,
    )
    print(destination)


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the correctness-first synthetic training path"
    )
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/smoke-run")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed_everything(
        int(config["project"]["seed"]), deterministic=bool(config["runtime"]["deterministic"])
    )
    device = _device(config)
    batch = _synthetic_batch(config, args.batch_size).to(device)
    builder, model = _build_modules(config, device)
    criterion = HDTWRoPELoss()
    parameters = list(builder.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    final_metrics: dict[str, torch.Tensor | float] = {}
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        clocks, alignment, alignments, output = _forward(config, batch, builder, model)
        source_index = torch.linspace(0.0, 1.0, batch.audio_mask.shape[1], device=device).expand(
            batch.audio_mask.shape[0], -1
        )
        target_index = torch.linspace(0.0, 1.0, batch.lyric_mask.shape[1], device=device).expand(
            batch.lyric_mask.shape[0], -1
        )
        loss = criterion(
            similarity=output.similarity,
            alignments=alignments,
            clocks=clocks,
            source_index_coordinate=source_index,
            target_index_coordinate=target_index,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        final_metrics = {
            "step": step,
            "loss/total": loss.total.detach(),
            **{f"loss/{name}": value.detach() for name, value in loss.terms.items()},
            **alignment_diagnostics(alignment),
            **clock_diagnostics(clocks),
        }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"builder": builder.state_dict(), "model": model.state_dict(), "config": config},
        output_dir / "checkpoint.pt",
    )
    write_metrics(output_dir / "metrics.json", final_metrics)
    save_run_manifest(
        output_dir / "run_manifest.yaml",
        build_run_manifest(config=config, clock_schema_version=CLOCK_SCHEMA_VERSION),
    )
    print(json.dumps({"output_dir": str(output_dir), "metrics": str(output_dir / "metrics.json")}))


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the synthetic retrieval path")
    parser.add_argument("--config", default="configs/prototype.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="artifacts/metrics/synthetic-evaluation.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]), deterministic=True)
    device = _device(config)
    batch = _synthetic_batch(config, args.batch_size).to(device)
    builder, model = _build_modules(config, device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        builder.load_state_dict(checkpoint["builder"])
        model.load_state_dict(checkpoint["model"])
    builder.eval()
    model.eval()
    clocks, alignment, _, output = _forward(config, batch, builder, model)
    metrics: dict[str, torch.Tensor | float] = {
        **retrieval_metrics(output.similarity),
        **alignment_diagnostics(alignment),
        **clock_diagnostics(clocks),
    }
    write_metrics(args.output, metrics)
    print(args.output)


def render_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a serialized alignment artifact")
    parser.add_argument("artifact")
    parser.add_argument("--output", default="artifacts/figures/alignment.png")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise SystemExit("Install hdtw-rope[viz] to render alignments") from error
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    mass = artifact["alignment_mass"][args.sample]
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(mass.transpose(0, 1), origin="lower", aspect="auto")
    axis.set_xlabel("Audio token")
    axis.set_ylabel("Lyric token")
    axis.set_title(str(artifact["sample_ids"][args.sample]))
    figure.colorbar(image, ax=axis, label="Alignment mass")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    print(destination)
