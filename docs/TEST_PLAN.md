# HDTW-RoPE Test Plan

## Unit tests

1. Hard DTW returns the known optimal path on hand-computed matrices.
2. Banded DTW rejects disconnected paths.
3. Soft-DTW approaches hard DTW as temperature decreases.
4. Soft-DTW passes double-precision gradient checking.
5. Alignment mass is nonnegative and normalizable.
6. Invalid rows are flagged rather than assigned fabricated coordinates.
7. Hard-path clock extraction is monotone.
8. Isotonic projection removes violations.
9. Hierarchy containment penalties are zero for valid nesting and positive for violations.
10. Rotary transport preserves vector norms.
11. Equal clocks preserve query-key inner products.
12. Shared frequency maps satisfy the relative-phase identity.
13. Raw-index single-clock mode matches reference RoPE.
14. Padding receives no alignment mass or attention probability.
15. Float16/bfloat16 forward passes remain finite with float32 recurrence and phase computation.

## Synthetic integration tests

Generate latent content `z(u)`, then sample modalities under known monotone warps. Required families are affine tempo scaling, piecewise stretching, pauses, smooth nonlinear warping, repeated motifs, missing segments, and melisma-like one-to-many regions. Report clock MAE, path deviation, monotonicity, and downstream correspondence accuracy.

## Regression fixtures

Every release retains a short exact DTW matrix, a Soft-DTW gradient fixture, a hierarchy containment example, a reference RoPE equivalence example, and one complete synthetic lyric–music sample. Numerical tolerances are versioned and justified when changed.
