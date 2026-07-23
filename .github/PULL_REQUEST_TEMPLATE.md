## Change

Describe the implemented behavior and the governing requirement.

## Invariants

- [ ] Masks remain explicit and fail closed.
- [ ] Clock schema order is unchanged or versioned.
- [ ] Float32 recurrence and phase contracts are preserved.
- [ ] Rotary transport remains exactly unitary.
- [ ] No empirical claim exceeds the available evidence.

## Validation

```text
python -m compileall -q src tests scripts
pytest -q
ruff format --check .
ruff check .
mypy src/hdtw_rope
```

## Review obligations

Summarize unresolved entries from `agent_review.yaml`.
