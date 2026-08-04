# Contributing

## Branch and commit discipline

Use focused branches named `agent/<scope>` or `feature/<scope>`. Keep commits reviewable and describe the complete behavioral change. Do not combine unrelated refactors with numerical changes.

## Change protocol

1. Identify the governing requirement in `docs/SPEC.md`.
2. State the invariant or falsifiable behavior being changed.
3. Add or update the smallest test that exposes the change.
4. Implement the reference behavior without premature fusion.
5. Run all local gates listed in `AGENTS.md`.
6. Update configuration, interface documentation, decision records, and artifact ledger when relevant.
7. Record remaining obligations in `agent_review.yaml` and the pull request.

## Numerical changes

Any tolerance, recurrence, mask, normalization, or dtype change requires:

- a frozen regression fixture or a justified new fixture;
- before/after error measurements;
- confirmation that masking and failure behavior remain closed;
- explicit documentation of precision and device behavior.

## Empirical claims

Qualitative attention maps are diagnostics, not evidence of advancement. Claims must satisfy the gates in `docs/SPEC.md`, include baseline parity, and report uncertainty across songs.
