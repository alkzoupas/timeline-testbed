# Kickoff prompt

Paste the quoted text into a new Claude Code session started in this folder on Opus 5.5 at xhigh
effort: `claude --model claude-opus-5-5 --effort xhigh`.

> You're the coordinator of a new project at `~/code/timeline-testbed`. Read AGENTS.md, SPEC.md,
> STATUS.md, DECISIONS.md and everything in `context/`. The predecessor repo is
> `~/code/entitlement-shield`. Don't read it except for paths named in the `context/` files, and never
> read its `data/` folder.
>
> **Step 1, no code: spec review.** Write `reports/M0-spec-review.md`, one page:
>
> - (a) The claim in SPEC §2, in your own words, and the result that would falsify it.
> - (b) Up to 10 problems, ambiguities or risks in SPEC.md, ranked, each with a proposed fix. Look
> especially at the family catalogue (§5.2), the channel check (§5.4), the render budget (§6.2) and
> the gates (§7.1).
> - (c) Anything you think won't work on this machine, and how you'd check it cheaply.
> - (d) Recommendations for the open items in SPEC §12.
>
> Then stop and ask me to decide.
>
> **Step 2, after my answers: M0 only** (SPEC §10), done yourself without workers. Installs need my
> go in that turn. When M0's exit criteria are met, report them with the actual command summaries
> and exit codes, update STATUS.md, and stop.