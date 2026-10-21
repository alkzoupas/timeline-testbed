# Decisions

One line each: date · who · decision · link.

- 2026-10-21 · owner · A new project for an owned, mechanistic testbed (postmortem option A); EntitlementShield becomes the archive · `~/code/entitlement-shield/docs/postmortem-2026-10-21.md`
- 2026-10-21 · owner · Claims are method-level only, with no realism claims · SPEC §2–3
- 2026-10-21 · owner · Local compute only: no paid APIs or rentals; installs and downloads need the owner's go. The predecessor's Jev credit was a free grant and is not used · AGENTS.md
- 2026-10-21 · owner · Vendor EntitlementShield at `fd5deca`, read-only, without `data/` · `vendor/entitlement-shield/VENDORED.md`
- 2026-10-21 · owner · The predecessor's sealed Tier 4 split may be spent once, for the optional M6 transfer check, with the owner's go at that time · SPEC §8
- 2026-10-21 · owner · Tier 5 blind authoring is closed (codex-02 dropped); it's a follow-up only · old repo `docs/decisions.md` P6-9
- 2026-10-21 · proposed · A probe gate (G2) before any fine-tuning; no ambiguous class · SPEC §7.1, §3
- 2026-10-21 · owner · Roles: Opus 5.5 at xhigh coordinates; supervised Orca workers implement on Claude Sonnet 5.5, with Haiku 5.5 only for rudimentary groundwork; GPT-6.1 Sol reviews. Delegation starts at M1 · AGENTS.md "Delegation and reviews"
- 2026-10-21 · owner · Three cross-vendor review rounds: the M1 freeze, M4 (the G2 scorer and `prereg/M4.md`, before scoring) and the M6 pre-registration · AGENTS.md "Delegation and reviews"
- 2026-10-21 · owner · M0 spec review accepted in full; SPEC v0.2 adopts fixes 1–10. Fix 7 was corrected while writing it: the channel check uses a dedicated 200 + 800 world per T pair, because two pooled seeds would false-fail ~20% · `reports/M0-spec-review.md`, SPEC v0.2
- 2026-10-21 · owner · §2 claim reworded: "on these *k* held-out families", against the stronger of the F64 logistic and trees · SPEC §2
- 2026-10-21 · owner · Pair 6 redesigned as T (serial session operator); held-out folds {1,2,3} vs {4,5,6}, two T families per side · SPEC §5.2, §6.3
- 2026-10-21 · owner · The G2 probe gate (with v0.2's revisions) and no ambiguous class are confirmed · SPEC §7.1, §3
- 2026-10-21 · owner · Scale 6 × (40 + 160) at 20% abusive; *s* grid {0, .25, .5, .75, 1}; *s* = 0.75 the sole primary, 0.5 secondary · SPEC §5.7, §6.4, §12
- 2026-10-21 · owner · R2 on the cached Qwen 3.5 9B and on the R3 model; R3 on a Qwen 3.5 of about 4B, download decided at M3 · SPEC §7, §12
- 2026-10-21 · owner · Standing permissions confirmed, plus read-only use of `~/venvs/es-p6-train`; go for the M0 `.venv` install (deps only, vendored code on the path through a `.pth`) · AGENTS.md
- 2026-10-21 · owner · Tier 4 transfer check: default no, revisited at M6 only if G3 passed · SPEC §8
- 2026-10-21 · owner · One owner-written family after the M6 freeze, as a separate generator module; v1 families must regenerate byte-identically, so earlier worlds stay valid · SPEC §12
- 2026-10-21 · owner · No cross-vendor review of SPEC v0.2; the first one is the M1 freeze · AGENTS.md "Delegation and reviews"
- 2026-10-21 · owner · No cross-vendor review of M0; the M1 freeze review's scope adds `testbed/pipeline/detect.py` (the runtime patch of the vendored detector) and `testbed/pipeline/units.py` · AGENTS.md "Delegation and reviews", `reports/M0.md`
- 2026-10-21 · owner · Secret guard allowlists `tokenizer.json` under `~/.cache/huggingface/hub/`, for now (render sizing, R1–R3). Exact basename only: `tokenizer_config.json` and the cache's auth files stay blocked · `~/.claude/hooks/secret_guard.py`
