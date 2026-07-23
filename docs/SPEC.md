# HDTW-RoPE Normative Implementation Specification

Status: **reference implementation, version 0.1**

Normative terms `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are interpreted as implementation obligations. The repository implements DTW-derived hierarchical latent clocks for unitary rotary transport in lyric–music alignment.

## 1. Scope and falsifiable claim

Lyrics and music MUST NOT be treated as though their raw indices were commensurate. The system constructs independent modality-native sequences, estimates monotone cross-modal correspondences, converts those correspondences into per-token scalar or vector clocks, and applies a shared unitary rotary action before ordinary attention.

The primary pipeline is:

```text
paired features -> alignment costs -> hard/Soft-DTW -> alignment mass
-> per-token latent clocks -> shared rotary transport -> standard attention
```

DTW MUST construct coordinates. The full DTW matrix MUST NOT be the primary pairwise attention bias in version 0.1.

No empirical superiority claim is authorized by synthetic tests alone.

## 2. Mathematical contract

For source features `a_t` and target features `y_s`, an alignment level defines

```text
C[t,s] = d(P_A a_t, P_Y y_s).
```

Hard DTW minimizes path cost under monotone transitions. Soft-DTW uses a positive temperature and differentiable soft minimum. Invalid cells are infeasible; they are not represented by merely large finite costs.

Given nonnegative alignment mass `P` and a target coordinate `u_Y`, the source-side clock is

```text
phi_A->Y(t) = sum_s row_normalize(P)[t,s] * u_Y(s).
```

The reverse clock is obtained from the transposed mass. Rows or columns with insufficient mass MUST be marked invalid or rejected in strict mode. A zero numeric coordinate MUST NOT encode invalidity.

A vector clock is

```text
xi = [absolute seconds, beat, bar, section occurrence,
      section-relative position, aligned syllable, aligned word,
      aligned line, aligned section].
```

The schema order is versioned as `hdtw-clock-v1` and MUST NOT depend on mapping iteration order.

For head `h` and rotary pair `j`, phase is

```text
alpha[h,j](xi) = 2*pi*<omega[h,j], xi>.
```

Queries and keys are transported by independent exact 2D rotations. The frequency map MUST be shared between modalities. With zero modality-specific phase offset,

```text
U(xi_A)^T U(xi_Y) = U(xi_Y - xi_A).
```

The first implementation MUST remain commuting and block-diagonal. Noncommuting path-ordered transport is deferred.

## 3. Data contract

Each sample MUST provide:

- a stable sample identifier;
- audio frame or token features `[T,D_A]`;
- lyric features `[S,D_Y]` at a declared unit type;
- explicit boolean masks where `True` means valid;
- physical audio timestamps in seconds;
- lyric text and unit metadata;
- optional beat, bar, line, phrase, section, and form coordinates;
- hierarchy intervals, parent identifiers, structural type identifiers, and occurrence identifiers when available;
- provenance metadata for dataset, preprocessing, and encoder checkpoints.

Sequence masks MUST be contiguous valid prefixes in the reference implementation. Lengths MUST NOT be inferred from zero-valued embeddings.

Repeated structures MUST retain both structural identity and absolute occurrence identity. `chorus` and `second chorus` are distinct coordinates.

The repository MUST NOT ship copyrighted lyric/audio datasets. Synthetic fixtures are permitted and required.

## 4. Alignment requirements

The reference implementation MUST include:

- cosine and squared-Euclidean costs;
- an optional learned bilinear cost;
- exact deterministic hard DTW;
- Soft-DTW with positive temperature;
- horizontal, vertical, and diagonal transition penalties;
- explicit source and target masks;
- optional diagonal or hierarchy-derived bands;
- fail-closed rejection when no valid path exists;
- nonnegative alignment-mass extraction;
- double-precision gradient checking.

Production recurrence arithmetic MUST be at least float32. Float16 and bfloat16 inputs MAY be accepted, but the recurrence MUST NOT execute at reduced precision. Float64 MUST remain available for numerical verification.

Tie-breaking in hard DTW MUST be deterministic. Banded alignment MUST NOT fabricate a diagonal fallback.

Hierarchical alignment SHOULD proceed coarse-to-fine: section, phrase/line, word, syllable/phoneme. Child alignments SHOULD be constrained by parent intervals or parent-clock bands.

## 5. Clock requirements

Clock extraction MUST:

- normalize alignment mass explicitly;
- carry component validity masks separately;
- support direct native coordinates and inferred cross-modal coordinates;
- support unit-interval normalization;
- check finite valid values;
- expose minimum mass and monotonicity diagnostics;
- fail closed in strict mode when expected rows or columns cannot define a coordinate.

The system MUST implement monotonicity, smoothness, slope, anchor, cycle, and hierarchy-containment objectives. Isotonic projection MAY be used at evaluation or through an explicitly declared straight-through estimator.

Missing components MUST use neutral numeric values and false component masks. The phase map MAY renormalize remaining active component weights, but the behavior MUST be deterministic and tested.

## 6. Rotary transport requirements

Rotary transport MUST:

- operate on `[B,H,N,Dh]` tensors;
- require positive even `rotary_dim <= Dh`;
- compute phases and trigonometric functions in float32;
- preserve non-rotary channels exactly;
- preserve vector norms within numerical tolerance;
- remain finite for float16 and bfloat16 activations;
- reject nonfinite valid clocks;
- use exact sine/cosine rotations, not unrestricted learned 2x2 matrices.

Supported frequency modes are:

1. `fixed_log`: canonical RoPE-style log frequencies on one clock component;
2. `learned_positive`: softplus-constrained frequencies;
3. `clock_partitioned`: declared rotary-pair allocations across physical, metrical, lyric-local, phrase/line, and section/form clock families.

Version 0.1 MUST use a shared frequency map for audio and lyrics.

## 7. Attention and model requirements

After transport, the model MUST use ordinary masked scaled dot-product attention. Padding keys MUST receive zero probability. Padding queries MUST produce zero output. A valid query sequence with no valid key/value token MUST be rejected.

The reference model MUST include:

- modality-specific feature projections;
- optional identity-initialized clock adapters;
- bidirectional audio-to-lyrics and lyrics-to-audio cross-attention;
- residual feed-forward blocks;
- masked pooled modality embeddings;
- a symmetric paired retrieval similarity matrix;
- layer-level attention diagnostics.

The primary prototype uses frozen/precomputed modality features. Encoder fine-tuning is deferred until the clock and transport mechanisms pass their correctness gates.

## 8. Loss requirements

The implementation MUST expose:

- symmetric contrastive retrieval loss;
- Soft-DTW loss;
- monotonicity loss;
- smoothness loss;
- slope-bound loss;
- hierarchy-containment loss;
- optional supervised anchor loss;
- cycle-consistency loss;
- alignment-entropy regularization.

All reductions MUST ignore invalid elements. A nonfinite aggregate loss MUST terminate the step.

## 9. Baselines and ablations

Every real-data experiment MUST include:

- no positional encoding;
- learned additive position embeddings;
- raw-index RoPE;
- physical timestamp RoPE;
- non-warped multi-clock RoPE;
- single-clock HDTW-RoPE;
- full hierarchical HDTW-RoPE.

Required ablations include hard versus Soft-DTW, banded versus unbanded alignment, detached versus differentiable mass, fixed versus learned frequencies, isotonic projection, and hierarchy removal.

Parameter counts, encoder checkpoints, optimizer budgets, splits, and evaluation code MUST be held constant where meaningful.

## 10. Verification requirements

The executable test suite MUST verify:

- known hard-DTW paths and deterministic values;
- disconnected-band rejection;
- Soft-DTW convergence toward hard DTW;
- float64 Soft-DTW gradcheck;
- nonnegative and normalizable mass;
- strict invalid-mass failure;
- monotone clock extraction;
- isotonic projection;
- hierarchy containment;
- canonical RoPE equivalence;
- norm preservation;
- equal-clock inner-product preservation;
- relative-phase identity;
- padding safety;
- finite mixed-precision rotary forward passes;
- complete bidirectional model forward behavior;
- synthetic recovery under affine, piecewise, pause, smooth, melisma, noise, and missing-segment conditions.

Local and CI gates are:

```bash
python -m compileall -q src tests scripts
pytest -q
ruff format --check .
ruff check .
mypy src/hdtw_rope
```

## 11. Reproducibility and artifacts

Every run MUST record:

- resolved configuration;
- random seed and deterministic settings;
- Git commit and dirty state when available;
- Python, PyTorch, CUDA, and package versions;
- data split manifest;
- encoder checkpoint identifiers;
- clock schema version;
- scalar metrics and diagnostic artifacts.

Generated checkpoints, alignment masses, metrics, and figures belong under `artifacts/` and are ignored by default. Empty artifact directories are retained for contract visibility.

## 12. Training progression

Development MUST proceed in this order:

1. exact synthetic and hard-DTW fixtures;
2. frozen encoders with offline/precomputed clocks;
3. banded Soft-DTW with detached mass;
4. differentiable alignment mass;
5. hierarchical coarse-to-fine bands;
6. word/syllable/phoneme refinement;
7. optional top-layer encoder fine-tuning;
8. fused kernels only after profiling identifies the reference recurrence as a bottleneck.

Fused CUDA/Triton kernels, causal online alignment, unconstrained long-sequence full matrices, noncommuting transport, and waveform generation are explicitly out of scope for version 0.1.

## 13. Empirical advancement gate

The mechanism is supported only after at least one primary and one secondary criterion hold on held-out real songs.

Primary criterion:

- at least 10% relative reduction in median word-onset error versus timestamp RoPE; or
- at least 3 absolute Recall@1 points versus non-warped multi-clock RoPE.

Secondary criterion:

- significant improvement under tempo or local-warp stress;
- at least 20% relative reduction in repeated-section errors; or
- no more than 5% degradation on unwarped examples.

Claims MUST report paired confidence intervals or bootstrap tests across songs. Qualitative attention maps are diagnostics, not advancement evidence.

## 14. Governance

Every substantial work package MUST update `agent_review.yaml`. Unresolved Adversary and Referee obligations remain open until real-data stress tests and empirical gates are completed. Optimization MAY NOT be used to conceal unresolved mathematical or evaluation failures.
