# Vendored: EntitlementShield at fd5deca

- **Source:** `~/code/entitlement-shield`, commit `fd5deca638fec6e60b56452f5ab51ee23bea8d71` (2026-10-17),
  exported with `git archive`, so it holds tracked files only.
- **Included:** `backend/`, `config/`, `scripts/`, `pyproject.toml`, `Makefile`, `LICENSE` (MIT) and
  `docs/original-plan.md`. That one doc is included only because `pyproject.toml` names it as the
  readme.
- **Excluded:** `data/` (all worlds and labels), the rest of `docs/`, `reviews/`, `handoffs/`,
  `HANDOFF.md`, `graft/`, agent config and environment files.
- **Checked on 2026-10-21:** the code imports from this location, and `verified_manifest_hash()`
  returns `fe21a7b052bea913fcbb880037a85c022d0e2be9d855a426bd449f60172a2021` (`features-frozen-v2`).
  The end-to-end pipeline and the test suite have not been run from here; M0 does that.
- **Read-only.** Verify the file hashes from this folder with `shasum -a 256 -c VENDORED.sha256`.
