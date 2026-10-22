# Generator v1 design (coordinator, 2026-10-21)

This turns SPEC §5 into a buildable generator. SPEC wins where the two differ, so report a
conflict instead of resolving it in code. This file is part of what freezes at the end of M1
(DECISIONS gets its hash along with the code's). Numbers marked *(tune)* may change before the
freeze to pass the §5.6 checks. Record each change here.

## Frame (every family)

- `GENERATOR_VERSION = "tt-gen-v1"`. `WORLD_END` is the vendored constant. The window is
  `W0 = WORLD_END − 90 d` (2026-03-03, a Tuesday) to `WORLD_END`.
- Every timestamp is on a whole minute (seconds and microseconds are 0), and every one is
  `< WORLD_END`. Nothing happens before `W0` except an aged account's `created_at` and its signup
  event.
- Group size *k* is uniform on {2..6} for every family, by a balanced design: group *i* of a
  family has `k = 2 + i mod 5`. So every reported world holds exactly as many groups of each
  size per family (its counts are multiples of 5). *(M1 change: independent draws let a world's
  40 abusive groups differ in size from their twins by chance, and every count that follows size
  inherited it. On seed 1, pair 1 had group-size AUC 0.37 and failed the render-length rule at
  s = 0.)*
- **Cycles** anchor at `created_at`: cycle *c* is `[created + 30c d, created + 30(c+1) d)`, for
  every *c* whose start is `< WORLD_END` and whose end is `> W0`.
  - Each such cycle has one `CLAIM_CREDITS` of 25, at the later of the cycle start and `W0`, plus
    uniform{30..480} min, if that's `< WORLD_END`.
  - There's no consumption in a cycle before its claim.
  - `GenCycle`: `consumed` is the sum of that cycle's consume events, and `expired = 25 − consumed`
    if `cycle_end <= WORLD_END`, else 0. `consumed <= 25` always.
- **Signup** at `created_at`. **Verification:** if the level is not NONE, one `VERIFICATION`.
  With probability 0.8 it comes between signup and the first claim, otherwise 1–48 h after the
  first claim *(tune)*. Draw one `VERIFICATION_FAILED` before it with probability 0.1.
- **Consume events** have tier `HIGH`. Every other event type has tier `None`. Amounts are
  integers.
- **Logins** are drawn per account, independently of the mechanism: Poisson(4) *(tune)* logins
  over the account's in-window life, at the account's chronotype hour ± 2 h.
- **No consumption cap ever binds in T pairs.** Each account-cycle has at most 6 consume events,
  each of at most 4 credits. Truncation drops whole sessions, latest first, and happens in the
  twin's draw, before any mechanism.

## Nuisance sampler (one function, shared by all 12 families)

Per account:
- domain: CONSUMER .70, EDUCATION .10, ENTERPRISE .10, DISPOSABLE .10;
- level: NONE .10, EMAIL .40, PHONE .40, ENHANCED .10.

Per group:
- one edge value of type uniform{STABLE_DEVICE, PAYMENT_TOKEN, RECOVERY_CONTACT}, at
  reliability uniform{MEDIUM, HIGH}, observed once by every member at `created_at` +
  uniform{0..2880} min;
- with probability 0.4, a NETWORK_GROUP value at LOW reliability on every member;
- with probability 0.3, a BEHAVIORAL_TEMPLATE value at LOW reliability on every member.

Values are `sha256(version|seed|group|signal)[:32]`, unique per group (SPEC §5.5).

## Randomness, ids and order (SPEC §5.1(6–7))

- Two RNG streams per group, both `random.Random` seeded by a string:
  - `base = "{version}|{seed}|{family}|{i}|base"`;
  - `mech = "...|mech"`.
- The twin's process uses only `base`. A mechanism uses only `mech`, and only on top of the base
  draw. So at *s* = 0 an abusive group is its twin's process exactly, and worlds at different *s*
  share base draws.
- Account ids: `acct-` + `sha256(version|seed|family|i|j)[:16]`. Group ids (ground truth only):
  `grp-` + `sha256(version|seed|family|i)[:16]`.
- Event ids are assigned after an account's events are final. They're numbered in
  (timestamp, type) order, so id order equals time order.
- Linkage ids are `-l{n}`, cycle ids `-c{nn}`. World rows are sorted as the vendored
  `generate_world` sorts them.

## Families (SPEC §5.2). One sentence each says what *s* scales.

A **session** is a run of consume events inside one clock hour. Two sessions of one group never
share a (day, clock hour), so the oracles can recover sessions from the timeline.

**T-pair creation:** every account of pairs 2, 3, 4 and 6 is created in `[W0, W0 + 10 d)`. So no
signup falls in F-CYC-3's 60-day recency window (SPEC §5.1(9)).

**Collisions (M1 implementation notes):**
- Pair 3: on a day when several accounts have a session, each draws a distinct free hour.
- Pair 1: the rescale is a linear warp of session start instants, anchored at the claim. Warped
  sessions may leave the chronotype hour, and never share a (day, clock hour).

**Budget.** About 12 render tokens per event (`raw-v1`, measured). So the *(tune)* numbers aim
for at most ~120 in-window events per 6-account group (99th percentile).

**Tuned in M1 (constants in `families.py` and `generator.py`).** This table overrides the
numbers above. Measured on standard worlds, seeds 1–2, at s = 0 and 1: max render 1,677 tokens.

| Constant | Above | Now | Why |
|---|---|---|---|
| `LOGIN_RATE` | 4 | 2.0 | event budget |
| `LAB_ATTEND_P` (pair 2) | 0.75 | 0.6 | event budget. Co-attended meetings per account pair are ~4, not 6, and the s = 1 oracle AUC is still ~0.97 |
| `OFFICE_DAY_P` (pair 3) | 0.15 | 0.07 | event budget |
| `FAMILY_DAY_P` (pair 4) | 0.2 | 0.10 | event budget. 0.07 left the s = 1 oracle AUC at ~0.94, too near the 0.90 control |
| `SHARED_DAY_P` (pair 6) | 0.12 | 0.055 | event budget |
| `NEWCOMER_CYCLE_SESSIONS` (pair 5) | 1–3 | 1–2 | event budget |

**Further M1 implementation notes** (the generator worker's resolutions, accepted on review):
- T-pair sessions are eligible by clock hour: the whole hour follows the cycle's claim and ends
  inside the cycle. A mechanism that moves events within the hour can't change which sessions
  exist.
- Pair 1's first cycle has 4–5 sessions of 3–8 credits, summing to 25. Events at or after
  `WORLD_END` are dropped, which can only happen to the last account.
- Pair 5:
  - The main account paces its sessions (daily probability `min(1, 20 / days left)`). It reaches
    25 in every cycle with about two weeks left.
  - A rule-following newcomer keeps the twin's session count and event count per session, and
    copies the main account's amounts in order. Where the main account has too few session days
    left, the rest are its own sessions at its own hour.
  - Its sessions share the main account's (day, clock hour) by design: the exception to the
    session rule above. Pair 5's oracle compares exactly that.
- Linkage is observed at `max(created_at, W0)` + uniform{0..2880} min. The vendored clustering
  drops observations older than 365 days, and an aged main account would otherwise leave its
  group's cluster (coordinator fix on review).
- A cycle whose claim would fall at or after `min(cycle end, WORLD_END)` keeps its row (granted
  25, consumed 0) with no claim event. The cycle-end case covers an aged account whose
  overlapping cycle ends minutes after W0. Without this rule its claim landed in the next cycle,
  which then had two claims (found in M1).

**Measured channel tags (SPEC §5.4, M1, frozen as `MEASURED_TAGS`).** Pairs 2, 3, 4 and 6 measured
T: every one-sided upper bound ≤ 0.57. Pair 1, designed F, measured M (best F12 AUC 0.69): its
twin also takes turns, so F-CYC-2 and F-CYC-3 fire in both. Pair 5, designed M, measured F (0.91,
through F-CYC-3). The headings below keep the design tags. A world manifest records the measured
tags as `channel_tags` and the design tags as `designed_channel_tags`; the channel check reports
`frozen` and `matches_frozen` per pair (review round 1, finding 5).

### Pair 1: `rotator` vs `household_device` (F)

- **Twin.** Turn starts `T_1 ∈ W0 + [0, 10) d` and `T_{j+1} = T_j + uniform[6, 14) d`. Account
  *j* is created at `T_j`.
  - In its first cycle it consumes exactly 25 credits, in sessions on uniform random days within
    `X_j ~ uniform[4, 20) d` of its claim, at its own chronotype hour. The last event exhausts the
    cycle.
  - Later cycles: 0–2 sessions, at most 8 credits each.
- **Sentence:** *s* is the probability, per hand-off *j → j+1*, that account *j*'s first-cycle
  session days are rescaled so that its exhaustion lands uniform{60..600} min before `T_{j+1}`.
  Otherwise its span is the twin's. The event count, amounts and creation times are always the
  twin's.
- **Oracle:** the median over consecutive accounts (by creation) of |first claim of *j+1* −
  exhaustion of *j*|, in hours. A lower value is more abusive.

### Pair 2: `parallel_farm` vs `classroom_lab` (T)

- **Twin.** One weekly lab: a weekday and an hour `H ∈ 9..16`, fixed per group. At each meeting,
  each eligible account attends with probability 0.75 *(tune)*.
  - An attendee gets a distinct start offset from {0..20} min, *m* ∈ {1, 2} events, a gap of
    uniform{3..15} min and amounts of uniform{1..4}.
  - Target: at least 6 co-attended meetings per account pair on average.
- **Sentence:** *s* is the probability, per meeting, that the attendees' drawn start offsets are
  reassigned in sorted order to the group's fixed operator order. The order is a permutation drawn
  once per group from `mech`. Each account keeps its own events, gaps and amounts.
- **Oracle:** pairwise start-order concordance across co-attended meetings,
  `Σ_pairs |#a first − #b first| / Σ_pairs n_ab`, over pairs with `n_ab ≥ 2`.

### Pair 3: `template_operator` vs `office_card` (T)

- **Twin.** Each account, on each weekday with probability 0.15 *(tune)*, has one session at
  hour uniform{9..17}. It has exactly 2 consume events: start uniform{0..25} min, gap
  uniform{2..30} min, amounts uniform{1..4} each. At most 3 sessions per account-cycle.
- **Sentence:** *s* is the probability, per (account, session), that the session's (amount 1,
  amount 2, gap) is replaced by the group's template. The template is one draw from the twin's
  per-session distribution, using `mech`. Session time, start minute and count are the twin's.
- **Volume.** The template keeps the mean of total volume but not its spread: an account with *N*
  replayed sessions consumes `N·(a+b)` (variance 2.5·N², against the twin's 2.5·N). That spread
  is part of the mechanism, as pair 4's coupling is (owner decision, 2026-10-21; review round 1,
  finding 1). F12 can't see it (amounts stay under the rate limits and the grant); F64 partly can.
- **Oracle:** the fraction of cross-account session pairs whose (amount 1, amount 2, gap) are
  identical.

### Pair 4: `single_budget` vs `heavy_family` (T)

- **Twin.** Each account has its own hour `h_a ∈ 6..23`, at least 3 h from every other member's.
  The set is drawn uniformly from the valid sets, then shuffled over accounts. (8..22 fits only
  five accounts at a 3 h gap; changed in M1, before the freeze.)
  Each day is a family day with probability 0.2 *(tune)*.
  - On a family day, each eligible account is active with probability 0.8, in one session at
    `h_a`: *m* ∈ {1, 2} events, start uniform{0..30} min, gap uniform{2..25} min.
  - Every event of that account-day has amount `L = 1 + floor(4·Φ(Z_{a,d}))`.
  - In the twin, the `Z_{a,d}` are independent N(0, 1).
- **Sentence:** *s* is the strength of a negative exchangeable correlation, `ρ = −s/(n_d − 1)`,
  among the `Z_{a,d}` of the `n_d ≥ 2` accounts active on day *d*.
  - Construction: draw `ε` ~ N(0, I) from `base` (the same draws as the twin).
  - Let `t = s/(n_d − 1 + s)` and `γ = 1 − sqrt(1 − n_d·t)`.
  - Then `Z_a = (ε_a − γ·mean(ε)) / sqrt(1 − t)`.
  - At *s* = 0 this is the twin, and each account's marginal is unchanged at every *s*.
- **Oracle:** over days with at least 2 active accounts, take `x_{a,d}` = the amount per event,
  and `z` = *x* minus the account's own mean. The statistic is
  `Σ_d (Σ_a z)² / Σ_d Σ_a z²`, and a lower value is more abusive.

### Pair 5: `aged_expansion` vs `adds_relative` (M)

- **Twin.** The main account is created `WORLD_END − uniform[200, 400) d`. It reaches the 25 cap
  in every in-window cycle, in sessions at its own hour on its own days.
  - Each of the *k* − 1 newcomers is created at `uniform[W0 + 5 d, WORLD_END − 20 d)`.
  - A newcomer has its own hour, days and amounts, and 1–3 sessions per cycle *(tune)*.
- **Sentence:** *s* is the probability, per newcomer, that it follows the rule. If it does:
  - it's created uniform{60..1440} min after the main account's exhaustion nearest to its drawn
    creation time;
  - its sessions sit on main-account session days after its creation, at the main account's
    hour, with the main account's session amounts;
  - its session count per cycle is still the twin's draw.
- **Oracle:** for each newcomer, mean over its sessions of (main account active that day and in
  the same clock hour), plus 1 if its creation is within 24 h after a main-account exhaustion.
  Take the mean over newcomers.

### Pair 6: `serial_operator` vs `roommates_card` (T)

- **Twin.** Each day, with probability 0.12 *(tune)*, there's a shared session at an hour from
  18..22. Each eligible account joins with probability 0.7.
  - A joiner gets *m* ∈ {2, 3} consume events at distinct minutes, uniform over {0..55} for the
    whole session.
  - Amounts are uniform{1..4} if *m* = 2, or uniform{1..2} if *m* = 3, so an hour never exceeds 8.
- **Sentence:** *s* is the probability, per shared session (at least 2 joiners), that the
  session's sorted consume minutes are reassigned so each joiner's events are contiguous. The
  block order is random from `mech`, and each joiner keeps its event count and its amounts in
  order.
- **Oracle:** mean over shared sessions of (account switches among consume events − (joiners −
  1)). A lower value is more abusive.

"Eligible" means after the account's current-cycle claim and under its cycle cap.

## Worlds (SPEC §5.7, §5.4, §5.6)

- **`standard`:** 6 pairs × (40 abusive + 160 twin) = 1,200 groups.
- **`channel`:** pairs 2, 3, 4 and 6 × (200 + 800), at *s* = 1.
- **Generator-only sample:** *n* groups per family, in memory, not persisted.
- Labels: `GenLabel(cluster_id=<group id>, scenario_family=<family>, label)`, with notes `""`.
  Split rows are `GenSplit(account_id, <group id>)`.
- The world manifest carries the generator version, seed, *s*, scale, counts and the
  `canonical_bytes` checksum. Its `world_version` is the generator version alone, so the
  inference receipt never names *s*.
