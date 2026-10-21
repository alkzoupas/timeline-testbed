# Lessons from EntitlementShield (one page)

The full postmortem is at `~/code/entitlement-shield/docs/postmortem-2026-10-21.md`. Each lesson
below is what to do, followed by the evidence.

1. **Decide the claim first, then build the data the claim needs.** Worlds were commissioned for one
   question and then used for another. Tier 4 was an appeals probe, reused for recall.
2. **Run the cheapest decisive experiment first.** A $0 supervised ceiling (P6.4, AUC 0.67) reframed
   eight weeks of work, and it ran last.
3. **Check the premise, and stop if it fails.** P6.0 found the detector's real problem was precision,
   and that the allowed pool held only 4 independent abusive groups. Four experiments then ran on that
   pool anyway.
4. **Count independent groups, not units.** 1,309 units came from 200 clusters and 22 abusive source
   groups.
5. **A target band gets you the band.** A world author asked to hit cluster AUC 0.80–0.92 did it with
   one uniform label shift on every latent variable.
6. **An acceptance test built from your own features selects worlds your features fit.**
7. **Run a feasibility smoke test at full size before any plan, pre-registration or review.** Each of
   these was found only after its plan existed:
   - Jev hit its token limits;
   - a local 27B model ran out of memory;
   - local LoRA failed at 4k tokens and above;
   - Gemini on the web couldn't execute code.
8. **Write budget constraints into the rules.** Plans kept depending on paid steps that the owner
   doesn't take.
9. **One question at a time, and a written reason for each pivot.** The question changed four times,
   and nothing carried over.
10. **Scale the process to the stakes.**
    - 9 cross-vendor review rounds in 9 days, 4 of them on a LinkedIn post.
    - About 206k words of prose against about 23k lines of code.
    - A STATUS file that became a diary.
11. **Analysis code needs known-answer tests.** Every published P6 number came from untested scripts.
    Freezing a script's hash stops edits, not bugs.
12. **Keep data out of indexes and commits.** A code indexer put a world generator into reviewers'
    context, and label files were committed.
13. **What worked, and should be kept:**
    - a separate label store, enforced by tests;
    - pre-registration before label reads;
    - honest reporting of nulls;
    - hash-checked regeneration and static review of generators, which caught a perfect shortcut and
      a single-shift world.
