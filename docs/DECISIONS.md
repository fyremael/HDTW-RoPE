# Decision Record

## DR-001 — Preserve attention factorization

DTW and Soft-DTW produce per-token coordinates. The primary model does not place the full DTW matrix into attention logits. This preserves rotary preprocessing, ordinary scaled dot-product attention, cacheability, and separation between alignment estimation and representation transport.

## DR-002 — Exact commuting transport first

Version 0.1 uses independent 2D rotations with a shared clock-frequency map. Unitarity is exact and the relative-phase identity remains testable. Noncommuting connection transport is deferred.

## DR-003 — Reference recurrence before kernels

The initial Soft-DTW recurrence is explicit PyTorch with float32 production arithmetic and float64 gradcheck support. Masks and failure semantics are fixed before CUDA or Triton optimization.

## DR-004 — Separate structural type from occurrence

Repeated sections retain both structural type and occurrence identity. Structural recurrence must not collapse distinct absolute occurrences.
