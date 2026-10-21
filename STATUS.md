# Status

## Goal

SPEC.md §2: on an owned, known-answer testbed, find out whether a small local LLM (zero-shot, probed
or fine-tuned) detects abuse whose evidence is timeline-only better than the frozen detector and a
64-feature model, including on abuse families it never saw.

## State (2026-10-21)

- Handoff package created from the EntitlementShield postmortem (option A). Nothing is implemented.
- Vendored code at `fd5deca`: it imports from this location, and its feature-manifest check returns
  `fe21a7b052bea913fcbb880037a85c022d0e2be9d855a426bd449f60172a2021` (checked with the old repo's
  venv). The vendored test suite and an end-to-end smoke test have not been run.
- Roles and the three cross-vendor review points are set: AGENTS.md "Delegation and reviews",
  DECISIONS.md. Delegation starts at M1.

## Next

KICKOFF.md step 1 (the spec review), then the owner's decisions, then M0.

## Open owner decisions

SPEC.md §12, items 1–6.

- `graft/` (the code index's local cache) is untracked and not gitignored yet: add `/graft/` to
  `.gitignore` before the first commit that uses `git add -A`.

## Last checks

None run in this repo yet.
