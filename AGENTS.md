# GCL Development Contract

This repository follows Grand Challenge Labs correctness-first development standards.

## Governing order

1. Preserve mathematical invariants.
2. Make failure states explicit and inspectable.
3. Establish deterministic reference behavior.
4. Measure before optimizing.
5. Optimize only proven bottlenecks.
6. Make empirical claims only after declared acceptance gates pass.

## Mandatory implementation obligations

- Public tensor interfaces validate rank, dtype, shape, masks, and finite values.
- Boolean masks use `True == valid` everywhere.
- Variable-length sequences use contiguous valid prefixes in the reference implementation.
- No function infers sequence length from zero-valued embeddings.
- Clock schemas are ordered and versioned; dictionary iteration never defines component order.
- Missing coordinates use a false mask and a neutral numeric value.
- Invalid alignments are never replaced by a diagonal path.
- Float16/bfloat16 activations do not change the float32 recurrence and phase contracts.
- Tests accompany every changed invariant or numerical tolerance.
- Random seeds, resolved configuration, package versions, data split, and commit state are recorded.

## Required local gates

```bash
python -m compileall -q src tests scripts
pytest -q
ruff format --check .
ruff check .
mypy src/hdtw_rope
```

When a tool is unavailable locally, record that fact in the pull request and rely on CI. A missing tool is not permission to waive its gate.

## Review roles

Every substantial work package records review status in `agent_review.yaml`:

- **Axiomatist:** mathematical assumptions and definitions.
- **Cartographer:** dependencies, module boundaries, and execution order.
- **Verifier:** tests, tolerances, and reproducibility.
- **Adversary:** counterexamples, failure modes, and stress tests.
- **Formalist:** interfaces, types, schemas, and invariants.
- **Amanuensis:** terminology, ledger, provenance, and document consistency.
- **Referee:** advancement-gate and claim acceptance.

Unresolved obligations remain explicit. Review records do not substitute for tests.
