"""The frozen M4 prompt (SPEC §7 R1 and R2): one fixed wrapper around the `raw-v1` render.

R1 reads the answer `RISK_LEVEL: <1-10>` after it; R2 reads hidden states over the same prompt
(mean over the render span, and the final prompt token). The text is generic: it states the
setting (SPEC §2) and the answer format, and names no family, mechanism, group id, *s* or seed
(SPEC §9). Frozen by the coordinator before any M4 score; `PROMPT_SHA256` pins it, and a test
recomputes the hash, so an edit can't go unnoticed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

PROMPT_VERSION = "m4-v1"

BEFORE = (
    "You review accounts on StudioAI, a service that gives each person 25 free credits per "
    "30-day cycle. Abuse means one person running several accounts to collect extra free "
    "credits. Households, classrooms and offices legitimately share devices, payment cards and "
    "networks, so a shared resource alone is not evidence of abuse.\n\n"
    "The accounts below are linked by at least one shared resource. Here is their complete "
    "activity record:\n\n"
)
AFTER = (
    "\n\nJudge from the activity whether one person controls all of these accounts, or whether "
    "they are independent people who share resources. Answer with one line in exactly this "
    "format: RISK_LEVEL: <n>, where n is a whole number from 1 (clearly independent people) to "
    "10 (clearly one person running every account)."
)

# sha256 of f"{PROMPT_VERSION}\0{BEFORE}\0{AFTER}"
PROMPT_SHA256 = "0cfe33a7eb03557ca1686034105bd43d7ee5921dc040cf9a1cd9a22fa03945a7"


# Nimble's typed-score schema (R1's second model): the same setting and question in Nimble's form,
# where the render is the context and the schema carries the question.
SETTING = BEFORE.split("\n\n")[0]
QUESTION = AFTER.strip().split(" Answer with")[0]
NIMBLE_SCHEMA: dict[str, dict[str, object]] = {
    "risk_level": {
        "type": "enum",
        "description": f"{SETTING}\n\n{QUESTION}\n\nRate the risk from 1 to 10.",
        "choices": [str(n) for n in range(1, 11)],
        "choice_descriptions": {
            "1": "clearly independent people",
            "10": "clearly one person running every account",
        },
    }
}
# sha256 of json.dumps(NIMBLE_SCHEMA, sort_keys=True)
NIMBLE_SCHEMA_SHA256 = "a64a6abb6243262a2f1e0c6b2d566b5e45716abe8d5074f04a24ad7ed365e0a0"


def nimble_schema_sha256() -> str:
    return hashlib.sha256(json.dumps(NIMBLE_SCHEMA, sort_keys=True).encode()).hexdigest()


def prompt_sha256() -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}\0{BEFORE}\0{AFTER}".encode()).hexdigest()


def child_rows(renders: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """MLX child rows (`{"id", "before", "render", "after"}`) from `renders.jsonl` records
    (`{"unit", "text"}`), in input order."""
    return [
        {"id": r["unit"], "before": BEFORE, "render": r["text"], "after": AFTER} for r in renders
    ]
