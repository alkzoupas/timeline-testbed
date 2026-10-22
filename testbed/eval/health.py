"""World health checks, the generator-only sample and the channel check (SPEC §5.4, §5.5, §5.6).

These read labels, so they live in `testbed/eval/`. Rows are small dicts, `{"name", "passed", ...}`.
Each row's pass/fail rule is a pure function of scores, so a known-answer test can drive it with
synthetic arrays; the world and sample builders only gather the scores.

Conventions:

- Positive class = abusive family; negative = its legitimate twin. One unit per group, so a unit
  is a group in every AUC.
- An oracle that is undefined for a timeline ranks below every defined value (least abusive). The
  fill is pooled over both classes of a pair, so it adds no signal, and the counts are reported.
- The null rule's `tests` is every test in that row on that world or sample (SPEC §5.6). It is a
  permutation test at the Bonferroni level, seeded per item from the item key.
- The mechanism's own variable is excluded from a pair's nuisance balance (`MECHANISM_VARIABLES`).

CLI::

    python -m testbed.eval.health world DIR
    python -m testbed.eval.health gensample --seed 201 --n 2000 --out DIR
    python -m testbed.eval.health channel --channel DIR --standard DIR [DIR ...] --out FILE

Each exits 0 only if every applicable row passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from app.models.enums import EmailDomainType, Reliability, VerificationLevel
from app.services.generator import WORLD_END, canonical_bytes
from numpy.typing import ArrayLike

from testbed.eval.cv import oof_scores
from testbed.eval.labels import WorldLabels, assign_units, open_labels
from testbed.eval.metrics import NULL_SEED, auc, auc_upper_bound, null_rule
from testbed.eval.oracle import oracle
from testbed.features.f12 import F_IDS
from testbed.features.f64 import FEATURES as F64_NAMES
from testbed.pipeline.render import parse
from testbed.pipeline.timeline import WINDOW_START, UnitTimeline, load_timeline
from testbed.world.families import CHANNEL_TAGS, MEASURED_TAGS, PAIRS, T_PAIRS
from testbed.world.generator import (
    EDGE_SIGNALS,
    GENERATOR_VERSION,
    WorldSpec,
    generate_groups,
    generate_world,
    group_timeline,
)
from testbed.world.persist import Stores

Row = dict[str, Any]
Rescore = Callable[[str], Mapping[str, float]]

# Seeds are named constants (AGENTS.md): every random choice here replays.
BOOT_SEED = 20261021  # group bootstrap for the channel check's upper bounds
CV_SEED = 20261022  # fold assignment for the channel check's out-of-fold scores
PARITY_SEED = 20261023  # which units the F12 parity row rescores
N_BOOT = 2000
NULL_ALPHA = 0.05  # familywise level of a null-rule family (SPEC §5.6): a row, or a whole world
ORDER_TESTS = 3  # order coupling: id order, row order, creation order

POSITIVE_AUC_MIN = 0.90  # positive control (SPEC §5.6)
RENDER_AUC_MIN = 0.85  # render visibility at s = 1
RENDER_AUC_TOLERANCE = 0.05  # render visibility at s = 0.75, against the store oracle
LINKAGE_MIN = 0.95  # share of groups recovered as exactly one unit (SPEC §5.5)
TOKEN_BUDGET = 1792  # the coordinator's budget: 2,048 minus a 256-token prompt reserve
PARITY_PER_FAMILY = 10
PARITY_TOL = 1e-9  # float slack for "equal"; the two paths compute the same formula
CHANNEL_T_UPPER = 0.60  # SPEC §5.4: T needs every upper bound at or under this
CHANNEL_F_AUC = 0.80  # SPEC §5.4: F needs the better F12 model at or over this
FOLDS = 5

ABUSIVE = "abusive"
PAIR_OF_FAMILY = {family: pair for pair, names in PAIRS.items() for family in names}
RELIABILITY_ORDINAL = {r.name: float(i) for i, r in enumerate(Reliability)}
NO_EDGE = -1.0  # edge reliability of a group with no edge-forming value (the generator has none)

# Variables a pair's mechanism names, which its balance test must not include. Pair 5's mechanism
# makes a rule-following newcomer's creation follow the main account's exhaustion, so the creation
# time of the group's accounts differs between family and twin by design.
MECHANISM_VARIABLES: dict[int, frozenset[str]] = {
    5: frozenset({"created_mean_d", "created_sd_d"}),
}


# --- Nuisance variables (SPEC §5.1(3)) ------------------------------------------------------------


def nuisance(tl: UnitTimeline) -> dict[str, float]:
    """The matched-nuisance variables of one group's timeline, all floats.

    Account creation is in days from the window start (negative for an aged account), spread is
    the population standard deviation. Fractions are over the group's accounts. The edge value is
    the group's edge-forming shared value (device, payment token or recovery contact): its type
    one-hot and its reliability ordinal (low 0, medium 1, high 2; -1 if the group has none).
    `verified_before_first_claim_frac` counts an account if it has a verification event in the
    window before its first claim in the window.
    """
    n = len(tl.accounts)
    created = [(a.created_at - tl.window_start).total_seconds() / 86400 for a in tl.accounts]
    out: dict[str, float] = {
        "group_size": float(n),
        "created_mean_d": float(mean(created)),
        "created_sd_d": float(pstdev(created)),
    }
    for domain in EmailDomainType:
        out[f"domain_{domain.name.lower()}"] = (
            sum(a.email_domain_type == domain.name for a in tl.accounts) / n
        )
    for level in VerificationLevel:
        out[f"level_{level.name.lower()}"] = (
            sum(a.verification_level == level.name for a in tl.accounts) / n
        )
    edges = [s for s in tl.shared if s.signal_type in {x.name for x in EDGE_SIGNALS}]
    for signal in EDGE_SIGNALS:
        out[f"edge_{signal.name.lower()}"] = float(any(s.signal_type == signal.name for s in edges))
    out["edge_reliability"] = (
        float(mean(RELIABILITY_ORDINAL[s.reliability] for s in edges)) if edges else NO_EDGE
    )
    out["has_network"] = float(any(s.signal_type == "NETWORK_GROUP" for s in tl.shared))
    out["has_behavior"] = float(any(s.signal_type == "BEHAVIORAL_TEMPLATE" for s in tl.shared))

    kinds = Counter(e.event_type for e in tl.events)
    out["claims"] = float(kinds["CLAIM_CREDITS"])
    out["consume_events"] = float(kinds["CONSUME_CREDITS"])
    out["consumed_credits"] = float(
        sum(e.amount or 0 for e in tl.events if e.event_type == "CONSUME_CREDITS")
    )
    out["logins"] = float(kinds["LOGIN"])
    out["verifications"] = float(kinds["VERIFICATION"])
    out["failed_verifications"] = float(kinds["VERIFICATION_FAILED"])
    out["events"] = float(len(tl.events))

    first: dict[tuple[str, str], datetime] = {}
    for e in tl.events:
        if e.event_type in ("CLAIM_CREDITS", "VERIFICATION"):
            key = (e.alias, e.event_type)
            first[key] = min(e.at, first.get(key, e.at))
    out["verified_before_first_claim_frac"] = (
        sum(
            (a.alias, "VERIFICATION") in first
            and (a.alias, "CLAIM_CREDITS") in first
            and first[(a.alias, "VERIFICATION")] < first[(a.alias, "CLAIM_CREDITS")]
            for a in tl.accounts
        )
        / n
    )
    return out


# --- Scores and pass/fail rules (pure) ------------------------------------------------------------


def fill_undefined(
    pos: Sequence[float | None], neg: Sequence[float | None]
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Scores with each None placed below every defined value (least abusive), plus the counts.

    The fill is one below the smallest defined value over both classes, or 0 if none is defined.
    """
    defined = [v for v in (*pos, *neg) if v is not None]
    fill = min(defined) - 1.0 if defined else 0.0
    p = np.array([fill if v is None else v for v in pos], dtype=np.float64)
    n = np.array([fill if v is None else v for v in neg], dtype=np.float64)
    return p, n, sum(v is None for v in pos), sum(v is None for v in neg)


def _item_seed(key: str) -> int:
    """A permutation seed from `NULL_SEED` and the item key, so no result depends on item order."""
    digest = hashlib.sha256(f"{NULL_SEED}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def null_row(
    name: str, items: Mapping[str, tuple[ArrayLike, ArrayLike]], *, tests: int | None = None
) -> Row:
    """The null rule on each (positive, negative) score pair, one Bonferroni family of `tests`.

    Passes only if every item's two-sided test of AUC = 0.5 survives the adjusted level
    `alpha_adj`. `tests` defaults to the item count."""
    count = len(items) if tests is None else tests
    results: dict[str, Any] = {}
    alpha_adj = NULL_ALPHA / max(count, 1)
    for key, (pos, neg) in items.items():
        result = null_rule(pos, neg, tests=count, alpha=NULL_ALPHA, seed=_item_seed(key))
        alpha_adj = result.alpha_adj
        results[key] = {
            "auc": result.auc,
            "z": result.z,
            "p": result.p,
            "alpha_adj": result.alpha_adj,
            "method": result.method,
            "passed": result.passed,
            "n_pos": int(np.asarray(pos).size),
            "n_neg": int(np.asarray(neg).size),
        }
    failed = sorted(k for k, r in results.items() if not r["passed"])
    return {
        "name": name,
        "passed": not failed,
        "tests": count,
        "alpha_adj": alpha_adj,
        "failed": failed,
        "results": results,
    }


def positive_row(
    name: str, items: Mapping[str, tuple[ArrayLike, ArrayLike]], *, minimum: float
) -> Row:
    """Every item's AUC must reach `minimum` (positive control, render visibility at s = 1)."""
    results = {k: {"auc": auc(p, n), "passed": auc(p, n) >= minimum} for k, (p, n) in items.items()}
    failed = sorted(k for k, r in results.items() if not r["passed"])
    return {
        "name": name,
        "passed": not failed,
        "minimum": minimum,
        "failed": failed,
        "results": results,
    }


def order_coupling_row(
    positive: Sequence[bool],
    min_id: Sequence[str],
    first_row: Sequence[int],
    earliest_created: Sequence[float],
    *,
    tests: int = ORDER_TESTS,
) -> Row:
    """AUC of the label against the rank of each group's smallest account id, of its first row in
    `accounts`, and of its earliest `created_at`. Null rule, `tests` = 3 unless the world's family
    is passed."""
    y = np.asarray(positive, dtype=bool)
    ranks = {
        "min_account_id": _ranks(min_id),
        "first_row": _ranks(first_row),
        "earliest_created_at": _ranks(earliest_created),
    }
    items = {k: (v[y], v[~y]) for k, v in ranks.items()}
    return null_row("order_coupling", items, tests=tests)


def _ranks(values: Sequence[Any]) -> np.ndarray:
    """Dense ranks (ties share a rank). The AUC only reads the order, so any rank form does."""
    _, inverse = np.unique(np.asarray(values), return_inverse=True)
    return inverse.astype(np.float64)


# --- Per-pair samples: what the statistical rows read --------------------------------------------


@dataclass(frozen=True)
class ClassSample:
    """One class of one pair, reduced to what the rows read: nuisance dicts and oracle scores."""

    nuisance: tuple[dict[str, float], ...]
    oracle: tuple[float | None, ...]


@dataclass(frozen=True)
class PairSample:
    pair: int
    family: ClassSample
    twin: ClassSample


def summarize(pair: int, timelines: Iterable[UnitTimeline]) -> ClassSample:
    tls = list(timelines)
    return ClassSample(tuple(nuisance(t) for t in tls), tuple(oracle(pair, t) for t in tls))


def _oracle_items(
    samples: Mapping[int, PairSample],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, int]]]:
    """Per pair: (family, twin) oracle scores with undefined ranked last, and the None counts."""
    items: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    undefined: dict[str, dict[str, int]] = {}
    for pair, sample in sorted(samples.items()):
        p, n, p_none, n_none = fill_undefined(sample.family.oracle, sample.twin.oracle)
        items[f"pair{pair}"] = (p, n)
        undefined[f"pair{pair}"] = {PAIRS[pair][0]: p_none, PAIRS[pair][1]: n_none}
    return items, undefined


def _nuisance_items(samples: Mapping[int, PairSample]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(family, twin) values per pair and variable, minus the pair's mechanism variables."""
    items: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pair, sample in sorted(samples.items()):
        variables = sorted(sample.family.nuisance[0])
        for variable in variables:
            if variable in MECHANISM_VARIABLES.get(pair, frozenset()):
                continue
            items[f"pair{pair}/{variable}"] = (
                np.array([d[variable] for d in sample.family.nuisance]),
                np.array([d[variable] for d in sample.twin.nuisance]),
            )
    return items


def nuisance_row(
    samples: Mapping[int, PairSample], name: str = "nuisance_balance", *, tests: int | None = None
) -> Row:
    """Per pair and variable (minus the pair's mechanism variables), AUC family vs twin under the
    null rule. `tests` defaults to the number of (pair, variable) tests in this row."""
    row = null_row(name, _nuisance_items(samples), tests=tests)
    row["excluded"] = {
        f"pair{p}": sorted(v) for p, v in MECHANISM_VARIABLES.items() if p in samples
    }
    return row


def negative_row(
    samples: Mapping[int, PairSample], name: str = "negative_control", *, tests: int | None = None
) -> Row:
    """Oracle AUC per pair, family vs twin, under the null rule (`tests` defaults to one per
    pair)."""
    items, undefined = _oracle_items(samples)
    row = null_row(name, items, tests=tests)
    row["undefined"] = undefined
    return row


def positive_oracle_row(samples: Mapping[int, PairSample], name: str = "positive_control") -> Row:
    """Oracle AUC per pair, family vs twin, at least `POSITIVE_AUC_MIN`."""
    items, undefined = _oracle_items(samples)
    row = positive_row(name, items, minimum=POSITIVE_AUC_MIN)
    row["undefined"] = undefined
    return row


def undefined_row(samples: Mapping[int, PairSample], name: str = "oracle_undefined") -> Row:
    """Informational: how many timelines each family's oracle could not score."""
    _, undefined = _oracle_items(samples)
    return {"name": name, "passed": True, "undefined": undefined}


# --- Linkage, render visibility, render length, parity (pure rules) -------------------------------


def linkage_row(
    unit_members: Mapping[str, Sequence[str]], group_of: Mapping[str, str], mixed: int
) -> Row:
    """Share of groups whose accounts form exactly one unit with no other accounts, at least
    `LINKAGE_MIN`; and no mixed unit."""
    members_of: dict[str, set[str]] = defaultdict(set)
    for account, group in group_of.items():
        members_of[group].add(account)
    units_of: dict[str, set[str]] = defaultdict(set)
    for unit, members in unit_members.items():
        for account in members:
            units_of[group_of[account]].add(unit)
    recovered = sum(
        len(units_of[g]) == 1 and set(unit_members[next(iter(units_of[g]))]) == accounts
        for g, accounts in members_of.items()
    )
    share = recovered / len(members_of) if members_of else 0.0
    return {
        "name": "linkage",
        "passed": share >= LINKAGE_MIN and mixed == 0,
        "groups": len(members_of),
        "recovered": recovered,
        "share": share,
        "minimum": LINKAGE_MIN,
        "mixed": mixed,
    }


def render_visibility_row(
    s: float,
    render: Mapping[int, tuple[Sequence[float | None], Sequence[float | None]]],
    store: Mapping[int, tuple[Sequence[float | None], Sequence[float | None]]],
    differs: int,
) -> Row:
    """Per pair, the oracle AUC from the render's parse against the store's.

    `render` and `store` map pair -> (family scores, twin scores). At s = 1 the render AUC must
    reach `RENDER_AUC_MIN`; at s = 0.75 it must be within `RENDER_AUC_TOLERANCE` of the store's;
    otherwise the AUCs are only reported. `differs` is the number of units whose render oracle
    differs from the store oracle (expected 0, and required to be 0 here, as that is the
    information-parity fault this row exists to catch)."""
    results: dict[str, Any] = {}
    for pair in sorted(render):
        r_pos, r_neg, _, _ = fill_undefined(*render[pair])
        s_pos, s_neg, _, _ = fill_undefined(*store[pair])
        render_auc, store_auc = auc(r_pos, r_neg), auc(s_pos, s_neg)
        if s == 1.0:
            passed = render_auc >= RENDER_AUC_MIN
        elif s == 0.75:
            passed = abs(render_auc - store_auc) <= RENDER_AUC_TOLERANCE
        else:
            passed = True
        results[f"pair{pair}"] = {
            "render_auc": render_auc,
            "store_auc": store_auc,
            "passed": passed,
        }
    failed = sorted(k for k, r in results.items() if not r["passed"])
    rule = {
        1.0: f"render AUC >= {RENDER_AUC_MIN}",
        0.75: f"|render - store| <= {RENDER_AUC_TOLERANCE}",
    }
    return {
        "name": "render_visibility",
        "passed": not failed and differs == 0,
        "rule": rule.get(s, "reported only"),
        "differs": differs,
        "failed": failed,
        "results": results,
    }


def render_length_row(
    tokens: Mapping[int, tuple[Sequence[float], Sequence[float]]],
    max_tokens: int,
    *,
    tests: int | None = None,
) -> Row:
    """AUC of the token count, family vs twin, under the null rule (`tests` defaults to one per
    pair: 6 on a standard world), and the maximum token count over every unit at most
    `TOKEN_BUDGET`."""
    row = null_row("render_length", {f"pair{p}": v for p, v in tokens.items()}, tests=tests)
    row["max_tokens"] = max_tokens
    row["token_budget"] = TOKEN_BUDGET
    row["passed"] = bool(row["passed"]) and max_tokens <= TOKEN_BUDGET
    return row


def parity_row(
    sample: Mapping[str, list[str]],
    texts: Mapping[str, str],
    expected: Mapping[str, Mapping[str, float]],
    rescore: Rescore,
) -> Row:
    """Rescore each sampled unit's render and compare it with the unit's F12 values.

    `sample` maps family -> unit keys. A mismatch is any differing feature id set, any value off by
    more than `PARITY_TOL`, or a render the rescoring refuses. Passes with 0 mismatches."""
    mismatches: list[dict[str, Any]] = []
    checked = 0
    worst = 0.0
    for family, units in sorted(sample.items()):
        for unit in units:
            checked += 1
            try:
                got = rescore(texts[unit])
            except ValueError as error:
                mismatches.append({"family": family, "unit": unit, "error": str(error)})
                continue
            want = expected[unit]
            if set(got) != set(want):
                mismatches.append({"family": family, "unit": unit, "error": "feature ids differ"})
                continue
            diffs = {k: abs(got[k] - want[k]) for k in want}
            worst = max(worst, max(diffs.values(), default=0.0))
            off = sorted(k for k, d in diffs.items() if not d <= PARITY_TOL)
            if off:
                mismatches.append({"family": family, "unit": unit, "features": off})
    return {
        "name": "parity_f12",
        "passed": not mismatches,
        "checked": checked,
        "mismatches": mismatches,
        "max_abs_diff": worst,
    }


def parity_sample(
    family_of_unit: Mapping[str, str], per_family: int = PARITY_PER_FAMILY
) -> dict[str, list[str]]:
    """A seeded sample of up to `per_family` unit keys per family."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for unit, family in sorted(family_of_unit.items()):
        by_family[family].append(unit)
    return {
        family: sorted(
            random.Random(f"{PARITY_SEED}|{family}").sample(units, min(per_family, len(units)))
        )
        for family, units in sorted(by_family.items())
    }


# --- Mechanical validity (read-only SQL on the inference store) ----------------------------------


def _sql_ts(value: datetime) -> str:
    """The SQLite text form of an aware UTC instant (see `testbed.pipeline.timeline`)."""
    return f"{value.astimezone(UTC):%Y-%m-%d %H:%M:%S.%f}"


_SAME_CYCLE_CONSUMPTION = (
    "select count(*) from entitlement_cycles c where c.credits_consumed != coalesce("
    "(select sum(e.credits_amount) from account_events e where e.account_id = c.account_id "
    "and e.event_type = 'CONSUME_CREDITS' and e.timestamp >= c.cycle_start "
    "and e.timestamp < c.cycle_end), 0)"
)
# check name -> (sql, parameters: "end" = WORLD_END, "start" = the window start)
_MECHANICAL: dict[str, tuple[str, tuple[str, ...]]] = {
    "event_before_account_creation": (
        "select count(*) from account_events e join accounts a on a.account_id = e.account_id "
        "where e.timestamp < a.created_at",
        (),
    ),
    "linkage_before_account_creation": (
        "select count(*) from linkage_observations o join accounts a "
        "on a.account_id = o.account_id where o.timestamp < a.created_at",
        (),
    ),
    "account_not_before_world_end": (
        "select count(*) from accounts where created_at >= ?",
        ("end",),
    ),
    "event_not_before_world_end": (
        "select count(*) from account_events where timestamp >= ?",
        ("end",),
    ),
    "linkage_not_before_world_end": (
        "select count(*) from linkage_observations where timestamp >= ?",
        ("end",),
    ),
    "cycle_start_not_before_world_end": (
        "select count(*) from entitlement_cycles where cycle_start >= ?",
        ("end",),
    ),
    "cycle_consumed_over_granted": (
        "select count(*) from entitlement_cycles where credits_consumed > credits_granted",
        (),
    ),
    "cycle_consumed_not_event_sum": (_SAME_CYCLE_CONSUMPTION, ()),
    "cycle_outside_window": (
        "select count(*) from entitlement_cycles where cycle_end <= ? or cycle_start >= ?",
        ("start", "end"),
    ),
}
_UNIQUE_IDS = (
    ("accounts", "account_id"),
    ("account_events", "event_id"),
    ("linkage_observations", "observation_id"),
    ("entitlement_cycles", "cycle_id"),
)


def mechanical_row(conn: sqlite3.Connection) -> Row:
    """SPEC §5.6 mechanical validity: each check counts violations, and every count must be 0."""
    params = {"start": _sql_ts(WINDOW_START), "end": _sql_ts(WORLD_END)}
    violations: dict[str, int] = {}
    for name, (sql, keys) in _MECHANICAL.items():
        violations[name] = int(conn.execute(sql, [params[k] for k in keys]).fetchone()[0])
    for table, column in _UNIQUE_IDS:
        sql = f"select count(*) - count(distinct {column}) from {table}"
        violations[f"duplicate_{column}"] = int(conn.execute(sql).fetchone()[0])
    failed = sorted(k for k, v in violations.items() if v)
    return {"name": "mechanical", "passed": not failed, "failed": failed, "violations": violations}


def determinism_row(manifest: Mapping[str, Any]) -> Row:
    """Regenerate the world from the manifest's spec: the canonical bytes' sha256 must equal the
    manifest's checksum."""
    spec = WorldSpec(seed=int(manifest["seed"]), s=float(manifest["s"]), scale=manifest["scale"])
    regenerated = hashlib.sha256(canonical_bytes(generate_world(spec))).hexdigest()
    return {
        "name": "determinism",
        "passed": regenerated == manifest["checksum_sha256"],
        "generator_version": manifest.get("generator_version"),
        "current_generator_version": GENERATOR_VERSION,
        "manifest_checksum": manifest["checksum_sha256"],
        "regenerated_checksum": regenerated,
    }


# --- Worlds ---------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)


@dataclass(frozen=True)
class UnitRecord:
    """One unit assigned to one group, with what the statistical rows read."""

    unit: str
    family: str
    pair: int
    positive: bool
    nuisance: dict[str, float]
    store_oracle: float | None
    render_oracle: float | None
    tokens: int


def _order_coupling(
    conn: sqlite3.Connection, labels: WorldLabels, *, tests: int = ORDER_TESTS
) -> Row:
    """Per group: its smallest account id, its first row in `accounts` and its earliest creation."""
    smallest: dict[str, str] = {}
    first_row: dict[str, int] = {}
    earliest: dict[str, float] = {}
    for rowid, account_id, created in conn.execute(
        "select rowid, account_id, created_at from accounts"
    ):
        group = labels.group_of[account_id]
        stamp = datetime.fromisoformat(created.replace(" ", "T")).replace(tzinfo=UTC).timestamp()
        smallest[group] = min(account_id, smallest.get(group, account_id))
        first_row[group] = min(rowid, first_row.get(group, rowid))
        earliest[group] = min(stamp, earliest.get(group, stamp))
    groups = sorted(smallest)
    return order_coupling_row(
        [labels.label[g] == ABUSIVE for g in groups],
        [smallest[g] for g in groups],
        [first_row[g] for g in groups],
        [earliest[g] for g in groups],
        tests=tests,
    )


def _split(
    records: Sequence[UnitRecord], value: Callable[[UnitRecord], Any]
) -> dict[int, tuple[list[Any], list[Any]]]:
    """pair -> (family values, twin values) of `value` over the records."""
    out: dict[int, tuple[list[Any], list[Any]]] = {}
    for r in records:
        out.setdefault(r.pair, ([], []))[0 if r.positive else 1].append(value(r))
    return dict(sorted(out.items()))


def _pair_samples(records: Sequence[UnitRecord]) -> dict[int, PairSample]:
    def klass(rs: list[UnitRecord]) -> ClassSample:
        return ClassSample(tuple(r.nuisance for r in rs), tuple(r.store_oracle for r in rs))

    by_pair: dict[int, tuple[list[UnitRecord], list[UnitRecord]]] = {}
    for r in records:
        by_pair.setdefault(r.pair, ([], []))[0 if r.positive else 1].append(r)
    return {p: PairSample(p, klass(pos), klass(neg)) for p, (pos, neg) in sorted(by_pair.items())}


def world_family_size(
    samples: Mapping[int, PairSample],
    tokens: Mapping[int, tuple[Sequence[float], Sequence[float]]],
    *,
    negative: bool,
) -> int:
    """`T`, the number of null-rule items on a world: order coupling, nuisance balance, the
    negative control (s = 0 worlds only) and render length."""
    count = ORDER_TESTS + len(_nuisance_items(samples)) + len(tokens)
    return count + (len(_oracle_items(samples)[0]) if negative else 0)


def _in_world_family(rows: Sequence[Row], world_tests: int) -> None:
    """Mark each null-rule row as tested in the world family, with its own item count; the item
    counts must add up to the family."""
    if sum(len(r["results"]) for r in rows) != world_tests:
        raise ValueError(
            f"null-rule rows hold {[len(r['results']) for r in rows]}, not {world_tests}"
        )
    for row in rows:
        row["family"] = "world"
        row["row_items"] = len(row["results"])


def world_health(
    world: Path,
    *,
    rescore: Rescore | None = None,
    parity_per_family: int = PARITY_PER_FAMILY,
) -> Row:
    """Every applicable §5.6 row for one world directory; writes `world/health.json`.

    The world's null-rule rows (order coupling, nuisance balance, negative control at s = 0,
    render length) form one Bonferroni family: every item is tested at `NULL_ALPHA / T`, `T` being
    the world's total item count (owner decision, 2026-10-21, DECISIONS). The generator-only
    sample keeps per-row families. `rescore` is `testbed.pipeline.parity.rescore_from_render`
    unless a test passes another."""
    world = world.resolve()
    stores = Stores.under(world)
    labels = open_labels(stores)
    manifest = json.loads((world / "manifest.json").read_text())
    s = float(manifest["s"])
    units = _read_jsonl(world / "units.jsonl")
    texts = {r["unit"]: r["text"] for r in _read_jsonl(world / "renders.jsonl")}
    by_unit = {u["unit"]: u for u in units}
    assigned = assign_units(((u["unit"], u["members"]) for u in units), labels.group_of)

    records: list[UnitRecord] = []
    with closing(_ro(stores.inference_db)) as conn:
        for unit, group in assigned.group_of_unit.items():
            family = labels.family[group]
            pair = PAIR_OF_FAMILY[family]
            positive = labels.label[group] == ABUSIVE
            if positive != (family == PAIRS[pair][0]):
                raise ValueError(f"group {group}: label {labels.label[group]} vs family {family}")
            tl = load_timeline(conn, by_unit[unit]["members"])
            records.append(
                UnitRecord(
                    unit,
                    family,
                    pair,
                    positive,
                    nuisance(tl),
                    oracle(pair, tl),
                    oracle(pair, parse(texts[unit])),
                    int(by_unit[unit]["tokens"]),
                )
            )
        samples = _pair_samples(records)
        tokens = _split(records, lambda r: float(r.tokens))
        world_tests = world_family_size(samples, tokens, negative=s == 0.0)
        rows: list[Row] = [mechanical_row(conn), determinism_row(manifest)]
        null_rows = [_order_coupling(conn, labels, tests=world_tests)]

    null_rows.append(nuisance_row(samples, tests=world_tests))
    if s == 0.0:
        null_rows.append(negative_row(samples, tests=world_tests))
    rows.extend(null_rows)
    if s == 1.0:
        rows.append(positive_oracle_row(samples))
    rows.append(undefined_row(samples))
    rows.append(
        linkage_row({u["unit"]: u["members"] for u in units}, labels.group_of, len(assigned.mixed))
    )
    differs = sum(r.store_oracle != r.render_oracle for r in records)
    rows.append(
        render_visibility_row(
            s,
            _split(records, lambda r: r.render_oracle),
            _split(records, lambda r: r.store_oracle),
            differs,
        )
    )
    max_tokens = max((int(u["tokens"]) for u in units), default=0)
    null_rows.append(render_length_row(tokens, max_tokens, tests=world_tests))
    rows.append(null_rows[-1])
    _in_world_family(null_rows, world_tests)
    if rescore is None:
        from testbed.pipeline.parity import rescore_from_render

        rescore = rescore_from_render
    rows.append(
        parity_row(
            parity_sample({r.unit: r.family for r in records}, parity_per_family),
            texts,
            {u["unit"]: u["f12"] for u in units},
            rescore,
        )
    )
    report: Row = {
        "world": world.name,
        "manifest": manifest,
        "units": len(records),
        "mixed": len(assigned.mixed),
        "null_family": {
            "tests": world_tests,
            "alpha_adj": NULL_ALPHA / world_tests,
            "rows": [r["name"] for r in null_rows],
        },
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }
    _write_json(world / "health.json", report)
    return report


# --- Generator-only sample ------------------------------------------------------------------------


def gensample_health(seed: int, n: int, out: Path) -> Row:
    """SPEC §5.6 generator-only sample: `n` groups per family at s = 0 and s = 1, in memory.

    Twins ignore s (`draw_group`), so each twin sample is drawn once and used at both s.
    Rows: nuisance balance at each s (separate Bonferroni families), negative control at s = 0,
    positive control at s = 1. Writes `out/health.json`."""
    started = time.monotonic()
    samples: dict[float, dict[int, PairSample]] = {0.0: {}, 1.0: {}}
    for pair, (family, twin) in PAIRS.items():
        twin_sample = summarize(
            pair, (group_timeline(g) for g in generate_groups(twin, n, seed, 0.0))
        )
        for s in samples:
            samples[s][pair] = PairSample(
                pair,
                summarize(pair, (group_timeline(g) for g in generate_groups(family, n, seed, s))),
                twin_sample,
            )
        print(f"pair {pair} done at {time.monotonic() - started:.0f} s", file=sys.stderr)
    rows = [
        nuisance_row(samples[0.0], "nuisance_balance_s0"),
        nuisance_row(samples[1.0], "nuisance_balance_s1"),
        negative_row(samples[0.0]),
        positive_oracle_row(samples[1.0]),
        undefined_row(samples[0.0], "oracle_undefined_s0"),
        undefined_row(samples[1.0], "oracle_undefined_s1"),
    ]
    report: Row = {
        "seed": seed,
        "groups_per_side": n,
        "generator_version": GENERATOR_VERSION,
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
        "seconds": round(time.monotonic() - started, 1),
    }
    _write_json(out / "health.json", report)
    return report


# --- Channel check (SPEC §5.4) --------------------------------------------------------------------


def measured_tag(upper_bounds: Sequence[float], best_f12_auc: float) -> str:
    """T if every upper bound is at most `CHANNEL_T_UPPER`; F if the better F12 model reaches
    `CHANNEL_F_AUC`; otherwise M."""
    if all(u <= CHANNEL_T_UPPER for u in upper_bounds):
        return "T"
    if best_f12_auc >= CHANNEL_F_AUC:
        return "F"
    return "M"


@dataclass(frozen=True)
class ChannelUnit:
    group: str  # unique across pooled worlds
    family: str
    positive: bool
    f12: tuple[float, ...]
    f64: tuple[float, ...]
    detector: float


def _vector(
    value: Mapping[str, float] | Sequence[float], names: Sequence[str]
) -> tuple[float, ...]:
    """A units.jsonl feature entry as floats in `names` order (a dict, or an ordered list)."""
    if isinstance(value, Mapping):
        return tuple(float(value[name]) for name in names)
    if len(value) != len(names):
        raise ValueError(f"expected {len(names)} features, got {len(value)}")
    return tuple(float(v) for v in value)


def _channel_units(world: Path, index: int) -> list[ChannelUnit]:
    labels = open_labels(Stores.under(world))
    units = _read_jsonl(world / "units.jsonl")
    assigned = assign_units(((u["unit"], u["members"]) for u in units), labels.group_of)
    by_unit = {u["unit"]: u for u in units}
    return [
        ChannelUnit(
            f"{index}:{group}",
            labels.family[group],
            labels.label[group] == ABUSIVE,
            _vector(by_unit[unit]["f12"], F_IDS),
            _vector(by_unit[unit]["f64"], F64_NAMES),
            float(by_unit[unit]["detector_score"]),
        )
        for unit, group in assigned.group_of_unit.items()
    ]


def _bounded(pos: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    return {
        "auc": auc(pos, neg),
        "upper": auc_upper_bound(pos, neg, n_boot=N_BOOT, seed=BOOT_SEED),
    }


def channel_pair(pair: int, units: Sequence[ChannelUnit], *, folds: int = FOLDS) -> Row:
    """One pair's channel record from its units (family and twin only)."""
    y = np.array([u.positive for u in units], dtype=int)
    groups = [u.group for u in units]
    x12 = np.array([u.f12 for u in units])
    x64 = np.array([u.f64 for u in units])
    detector = np.array([u.detector for u in units])
    pos = y == 1

    def oof(x: np.ndarray, model: str) -> np.ndarray:
        return oof_scores(x, y, groups=groups, strata=y, model=model, folds=folds, seed=CV_SEED)

    models: dict[str, Any] = {}
    for model in ("logistic_l2", "hist_gradient_boosting"):
        scores = oof(x12, model)
        models[model] = _bounded(scores[pos], scores[~pos])
    detector_score = _bounded(detector[pos], detector[~pos])
    best = max(m["auc"] for m in models.values())
    uppers = [m["upper"] for m in models.values()] + [detector_score["upper"]]
    measured = measured_tag(uppers, best)
    f64_auc: dict[str, float] = {}
    for model in ("logistic_l2", "hist_gradient_boosting"):
        scores = oof(x64, model)
        f64_auc[model] = auc(scores[pos], scores[~pos])
    return {
        "family": PAIRS[pair][0],
        "twin": PAIRS[pair][1],
        "n_pos": int(pos.sum()),
        "n_neg": int((~pos).sum()),
        "designed": CHANNEL_TAGS[pair],
        "measured": measured,
        "matches": measured == CHANNEL_TAGS[pair],
        "frozen": MEASURED_TAGS[pair],
        "matches_frozen": measured == MEASURED_TAGS[pair],
        "f12_models": models,
        "detector": detector_score,
        "best_f12_auc": best,
        "per_feature_auc": {fid: auc(x12[pos, i], x12[~pos, i]) for i, fid in enumerate(F_IDS)},
        "f64_w90_auc": f64_auc,
    }


def channel_check(
    channel: Path,
    standard: Sequence[Path],
    out: Path,
    *,
    channel_scales: Sequence[str] = ("channel",),
    standard_scales: Sequence[str] = ("standard",),
    folds: int = FOLDS,
) -> Row:
    """SPEC §5.4: measure each pair's channel tag at s = 1 and compare it with the design tag.

    The T-designed pairs (2, 3, 4, 6) use the channel world; the others use the standard s = 1
    worlds pooled. Passes if every T-designed pair measures T (SPEC §5.4); F and M are measured
    and frozen as measured. Each record also compares the measurement with the frozen tag
    (`MEASURED_TAGS`). Writes `out`."""
    worlds = [(channel, channel_scales, 0)] + [
        (path, standard_scales, i + 1) for i, path in enumerate(standard)
    ]
    loaded: list[list[ChannelUnit]] = []
    for path, scales, index in worlds:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["s"] != 1.0 or manifest["scale"] not in scales:
            raise ValueError(f"{path}: expected s = 1 and scale in {list(scales)}, got {manifest}")
        loaded.append(_channel_units(path.resolve(), index))
    pairs: dict[str, Any] = {}
    for pair, names in sorted(PAIRS.items()):
        source = loaded[0] if pair in T_PAIRS else [u for w in loaded[1:] for u in w]
        units = [u for u in source if u.family in names]
        pairs[str(pair)] = channel_pair(pair, units, folds=folds)
        pairs[str(pair)]["worlds"] = "channel" if pair in T_PAIRS else "standard pooled"
    # SPEC §5.4: a T-designed pair must measure T; F and M are measured, and the measured tag is
    # the one frozen at M1. So only T-designed pairs can fail the check.
    report: Row = {
        "channel": str(channel),
        "standard": [str(p) for p in standard],
        "pairs": pairs,
        "measured_tags": {p: r["measured"] for p, r in pairs.items()},
        "passed": all(r["measured"] == "T" for r in pairs.values() if r["designed"] == "T"),
    }
    _write_json(out, report)
    return report


# --- CLI ------------------------------------------------------------------------------------------


def _summary(report: Row) -> str:
    rows = report.get("rows")
    if rows is None:
        rows = [
            {"name": f"pair {p}={r['measured']}", "passed": r["designed"] != "T" or r["matches"]}
            for p, r in report["pairs"].items()
        ]
    marks = " ".join(f"{r['name']}={'ok' if r['passed'] else 'FAIL'}" for r in rows)
    return f"{'passed' if report['passed'] else 'FAILED'}: {marks}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    world = sub.add_parser("world", help="health checks for one world directory")
    world.add_argument("dir", type=Path)
    sample = sub.add_parser("gensample", help="the generator-only sample")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--n", type=int, default=2000)
    sample.add_argument("--out", type=Path, required=True)
    channel = sub.add_parser("channel", help="the channel check")
    channel.add_argument("--channel", type=Path, required=True)
    channel.add_argument("--standard", type=Path, nargs="+", required=True)
    channel.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "world":
        report = world_health(args.dir)
    elif args.command == "gensample":
        report = gensample_health(args.seed, args.n, args.out)
    else:
        report = channel_check(args.channel, args.standard, args.out)
    print(_summary(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
