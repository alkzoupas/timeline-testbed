# Reuse map: vendored EntitlementShield (fd5deca)

- Paths are relative to `vendor/entitlement-shield/`, and line numbers are as of `fd5deca`.
- Import the library parts.
- The scripts hard-code Tier 4 paths, so copy the functions you need into `testbed/` and adapt them
  there.
- Never edit `vendor/`.

## Import as a library

| Need | Where |
|---|---|
| World data classes | `backend/app/services/generator.py:67-142`: `GenAccount`, `GenEvent`, `GenLinkage`, `GenCycle`, `GenLabel`, `GenSplit`, `GeneratedWorld` |
| The old family spec (as a pattern) | `generator.py:151-167` `FamilySpec`; `FAMILY_SPECS`; `PRESETS` at `:363` (`min`, `dev`, `dev_arm4`) |
| World frame | `generator.py:57`: `WORLD_END` = 2026-06-01T00:00Z |
| Event-generation helpers (as a pattern) | `generator.py:460-569` `_gen_events_for_account`; `:572-687` `_gen_cluster` |
| Deterministic bytes and artifacts | `generator.py:908` `canonical_bytes`; `:1010` `build_manifest`; `:1035` `write_artifacts` |
| Persisting to the two stores | `generator.py:925` `persist_world`. For the wiring (engines, `create_all_schemas`, `make_ground_truth_sink`, `record_world_receipts(tier="dev")`), see `scripts/seed.py` |
| Store engines | `app.db.engine.make_inference_engine(url)`, `app.db.ground_truth_store.make_ground_truth_engine(url)`, `app.db.schema.create_all_schemas` |
| World-pair check | `app.db.world_provenance.verify_world_pair` (`:245`). `WORLD_TIERS` is at `:28`; use tier `dev` |
| Detector | `app.services.detector.run_detector` (`:240`; session, as_of, run_id). Manifest check: `app.services.feature_manifest.verified_manifest_hash`. CLI pattern: `scripts/detector_run.py` |
| Label-free group membership (eval layer only) | `app.db.ground_truth_reader.load_split_groups` (`:50`) |
| The old split-scoped label reader | `app.services.evaluation.load_ground_truth` (`:109`); `assign_split` (`:148`) is a 60/20/20 hash. The new regimes don't use it |
| Ground-truth boundary test (as a pattern) | `backend/app/tests/test_ground_truth_boundary.py` |

## Copy and adapt

| Need | Where |
|---|---|
| Detecting on a store copy, and the speed-up | `scripts/tier4_arms_run.py:150` `memoized_signal_values`; `:177-217` `detect_base` |
| Units from a detected store | `scripts/p6_recall_pilot.py:321-342` `_open_units`, which reads `detection_evidence.payload_json` |
| Render | `scripts/p6_recall_pilot.py:137-247` `render_unit` (v1, v2, v3); `:250-286` `_history_v3` |
| Zero-shot runner (local OpenAI-compatible server, with caching) | `scripts/p6_recall_pilot.py:292-318` `_ask`; `:345-401` `run`; `config/recall-provider.local-qwen35-9b.yaml` |
| Nimble zero-shot (MLX) | `scripts/p6_3_export_renders.py`, `scripts/p6_3_nimble_run.py`, `scripts/p6_3_score.py` |
| The 64 features | `scripts/p6_4_features.py:95-203` `unit_features`; `:206-221` `render_check` |
| Models and CV | `scripts/p6_4_ceiling.py:64-69` `_models`; `:113-132` `at_volume`; `:158-223` `score` |
| LoRA feasibility on MLX | `scripts/p6_2_step0.py` |
| Shortcut audits (as patterns) | `scripts/tier4_audit.py` |

## Gotchas

- **Store URLs.** The default URLs are relative to the working directory (`data/synthetic/*.db`).
  Always pass explicit sqlite URLs.
- **Keep the vendored layout intact.** Config resolves from the vendored root
  (`Path(__file__).parents[3]`). The manifest check passed from the new location on 2026-10-21
  (`fe21a7b…a2021`).
- **The README file is there for the build.** The vendored `pyproject.toml` names
  `docs/original-plan.md` as its readme, so that file is included and `pip install -e
  vendor/entitlement-shield` can build.
- **Naming.** The vendored package's top-level module is `app`. Don't name anything in `testbed/`
  `app`.
- **Some vendored tests need the old `data/`** (for example `test_tier2.py`) and will fail or skip.
  Record the baseline in M0; don't fix them.
- **Don't tune the detector.** It was tuned on the old dev world, and here it's a baseline.
- **Receipts.** `verify_world_pair` needs receipts on both stores (`record_world_receipts`).
- **Don't use the hosted-API configs.** The vendored `config/` includes the old reviewer-provider
  files for hosted APIs (no paid calls).
- **Pass the render version explicitly.** `render_unit` defaults to `v1`, which is kept for old
  replays.
- **Detector speed.** The detector is the slow step at scale. Measure it in M0.
