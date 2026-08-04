# HDTW-RoPE

**DTW-derived hierarchical latent clocks for unitary rotary transport in lyric–music alignment.**

Lyrics and music should not be forced to share a raw index. HDTW-RoPE estimates monotone cross-modal coordinates from hard DTW or banded Soft-DTW, assembles those coordinates into a versioned hierarchical clock, and applies ordinary rotary transport before standard attention.

The reference identity is

```text
DTW alignment mass -> per-token latent clocks -> shared unitary rotation -> ordinary attention
```

The implementation is a correctness-first research prototype. It does not claim empirical superiority until the advancement gates in [`docs/SPEC.md`](docs/SPEC.md) have been met on held-out real data.

## Current status

Implemented:

- validated lyric–music sample, hierarchy, and batch schemas;
- cosine, squared-Euclidean, and bilinear alignment costs;
- deterministic hard DTW with transition penalties and fail-closed bands;
- differentiable Soft-DTW with gradient-derived alignment mass;
- normalized clock extraction in both alignment directions;
- monotonicity, smoothness, slope, containment, anchor, cycle, and entropy losses;
- exact float32 clock-to-phase accumulation and rotary transport;
- fixed-log, learned-positive, and clock-partitioned frequencies;
- bidirectional audio↔lyrics cross-attention;
- retrieval and alignment metrics, diagnostics, manifests, and synthetic stress fixtures;
- required positional baselines and synthetic command-line workflows;
- mathematical and integration tests.

Deferred until the correctness and empirical gates pass:

- fused CUDA/Triton Soft-DTW kernels;
- unconstrained long-sequence full-matrix alignment;
- causal online alignment;
- noncommuting path-ordered transport;
- end-to-end waveform generation;
- automatic song-form discovery as a prerequisite.

## Core invariants

1. **DTW constructs coordinates; it is not the primary pairwise attention bias.**
2. **Clock validity is represented by masks, never numeric sentinels.**
3. **Alignment bands are infeasibility constraints, not large finite penalties.**
4. **Audio and lyrics share the rotary frequency map.**
5. **Clock accumulation and trigonometry occur in float32.**
6. **Rotary blocks remain exact rotations; unrestricted learned 2×2 blocks are forbidden.**
7. **Every invalid path, insufficient-mass clock, nonfinite phase, and schema mismatch fails closed.**
8. **No fused optimization work precedes the reference mathematical gates.**

## Mathematical core

For an alignment mass `P` and target coordinate `u`, the audio-side clock is

```text
phi_A->Y(t) = sum_s normalize_rows(P)[t,s] * u_Y(s).
```

A clock vector `xi` is mapped to head- and pair-specific phases

```text
alpha[h,j](xi) = 2*pi*<omega[h,j], xi>.
```

The first rotary channels are transported by exact 2D rotations. With a shared frequency map and no phase offset,

```text
U(xi_A)^T U(xi_Y) = U(xi_Y - xi_A).
```

Thus absolute modality-native coordinates induce attention sensitive to relative displacement in the learned hierarchical clock.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The package requires Python 3.11 or newer and PyTorch 2.3 or newer.

## Validation

```bash
pytest -q
python -m compileall -q src tests scripts
```

The required mathematical tests include:

- canonical RoPE equivalence;
- norm preservation and relative-phase identity;
- hard-DTW path correctness and band rejection;
- double-precision Soft-DTW gradcheck;
- alignment-mass and mask safety;
- monotone clock extraction and isotonic projection;
- hierarchy containment;
- synthetic recovery under tempo scaling, local stretching, pauses, and melismas;
- finite float16/bfloat16 forward behavior with float32 phase arithmetic.

## Synthetic end-to-end path

The repository deliberately ships no copyrighted lyric or audio dataset. A deterministic synthetic path validates integration:

```bash
python scripts/precompute_clocks.py \
  --config configs/prototype.yaml \
  --output artifacts/alignments/synthetic-clocks.pt

python scripts/train.py \
  --config configs/prototype.yaml \
  --steps 1 \
  --output-dir artifacts/smoke-run

python scripts/evaluate.py \
  --config configs/prototype.yaml \
  --checkpoint artifacts/smoke-run/checkpoint.pt \
  --output artifacts/metrics/synthetic-evaluation.json

python scripts/render_alignment.py \
  artifacts/alignments/synthetic-clocks.pt \
  --output artifacts/figures/synthetic-alignment.png
```

When CUDA is requested but unavailable, the reference CLI uses CPU and records the actual environment in the run manifest.

## Real-data integration contract

Each sample provides:

- precomputed/frozen audio frame features;
- lyric features or encoded phoneme, syllable, or word units;
- explicit sequence masks;
- physical audio time in seconds;
- optional beat, bar, line, phrase, section, and form coordinates;
- hierarchy unit intervals, parent IDs, type IDs, and occurrence IDs;
- stable sample identifiers and dataset metadata.

Repeated sections must retain both structural identity and occurrence identity. “Chorus” and “second chorus” are different coordinates.

See [`src/hdtw_rope/data/schema.py`](src/hdtw_rope/data/schema.py) and [`docs/INTERFACES.md`](docs/INTERFACES.md).

## Training stages

The intended progression is fixed:

1. deterministic synthetic and hard-DTW validation;
2. frozen encoders with precomputed clocks;
3. banded differentiable Soft-DTW, initially detached from clock transport;
4. end-to-end differentiable alignment mass;
5. section/phrase-first hierarchical bands;
6. word/syllable/phoneme refinement;
7. optional top-layer encoder fine-tuning;
8. kernel optimization only after all correctness gates pass.

## Required comparisons

A real experiment must compare:

- no positional encoding;
- additive learned positions;
- raw-index RoPE;
- physical timestamp RoPE;
- non-warped multi-clock RoPE;
- single-clock HDTW-RoPE;
- full hierarchical HDTW-RoPE.

Recommended additional comparisons are specified in [`docs/SPEC.md`](docs/SPEC.md).

## Advancement gate

The mechanism is supported only after at least one primary and one secondary criterion hold on held-out real data.

Primary:

- at least 10% relative reduction in median word-onset error against timestamp RoPE; or
- at least 3 absolute Recall@1 points against non-warped multi-clock RoPE.

Secondary:

- significant improvement under tempo/local-warp stress;
- at least 20% relative reduction in repeated-section errors; or
- no more than 5% degradation on unwarped examples.

Every claim requires paired confidence intervals or bootstrap tests across songs.

## Repository map

```text
src/hdtw_rope/
  alignment/   hard DTW, Soft-DTW, bands, and hierarchy orchestration
  clocks/      extraction, normalization, projection, and constraints
  data/        sample contracts, collation, hierarchy, synthetic fixtures
  models/      projections, bidirectional attention, retrieval model
  rotary/      frequency maps, exact transport, canonical reference
  baselines.py required positional baselines
  losses.py    required and recommended objectives
  metrics.py   localization, retrieval, path, and bootstrap metrics
  diagnostics.py scalar logging and artifact serialization
  reproducibility.py deterministic controls and run manifests
```

Normative requirements live in [`docs/SPEC.md`](docs/SPEC.md). Development obligations live in [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
