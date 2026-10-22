"""The frozen M8 prompt, `m8-v1`, and the fixed Haiku route (context/m8-haiku-kickoff.md).

The text is the smoke's draft (`runs/m8-smoke/smoke.py`), verbatim. It reuses m4-v1's `SETTING`
and `QUESTION` (`prompt.py`), adds one bridge sentence and asks for a 0-100 probability through a
JSON schema. Instructions sit in the system prompt; the user message holds the examples (the
few-shot variant only) and then the unit, so one side's prefix is identical across its calls. The
text names no family, mechanism, group id, *s* or seed (SPEC §9).

Frozen by the coordinator before the M8 pilot. `PROMPT_SHA256` pins every text part, and a test
recomputes it. The route constants follow the AGENTS.md exception: isolated headless Claude Code on
the owner's subscription, no tools, safe mode, an explicit environment allowlist. Workers don't
edit this file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from testbed.models.prompt import QUESTION, SETTING

PROMPT_VERSION = "m8-v1"

SYSTEM = (
    f"{SETTING}\n\n"
    "Each request shows the complete activity record of accounts that are linked by at least one "
    f"shared resource. {QUESTION} Answer with the probability, as a whole number from 0 (clearly "
    "independent people) to 100 (clearly one person running every account), that one person "
    "controls all of the accounts."
)
UNIT_INTRO = (
    "The accounts below are linked by at least one shared resource. Here is their complete "
    "activity record:\n\n"
)
# An example's answer line: VERDICTS[0] if the example is abusive, VERDICTS[1] if legitimate.
VERDICTS = ("one person running every account", "independent people who share resources")
EXAMPLES_INTRO = "Labelled examples from other cases, for reference:\n\n"
EXAMPLE = "Example {i}. Activity record:\n\n{text}\n\nAnswer: {verdict}\n\n"
TARGET_LEAD = "Now the case to judge. "
SCHEMA = {
    "type": "object",
    "properties": {"probability": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["probability"],
    "additionalProperties": False,
}
SCHEMA_JSON = json.dumps(SCHEMA)  # the --json-schema argument, byte for byte

PROMPT_PARTS = (
    PROMPT_VERSION,
    SYSTEM,
    UNIT_INTRO,
    *VERDICTS,
    EXAMPLES_INTRO,
    EXAMPLE,
    TARGET_LEAD,
    SCHEMA_JSON,
)
# sha256 of "\0".join(PROMPT_PARTS)
PROMPT_SHA256 = "028db033f3c0fbe236f25f673c9c62a601e94c71556e78589da150914036bbf5"

# The route. The driver runs `argv()` with the user message on stdin, from an empty neutral temp
# dir, under `env_allowlist()`'s keys plus CLAUDE_CONFIG_DIR = ~/CONFIG_DIR_NAME. `--effort` is left
# at its default (thinking on), as in the smoke. Never inherit the full environment: an API key
# would bypass the subscription.
MODEL = "claude-haiku-5-5"
CONFIG_DIR_NAME = ".claude-haiku-scorer"
ROUTE_FLAGS = (
    "-p",
    "--model",
    MODEL,
    "--safe-mode",
    "--tools",
    "",
    "--strict-mcp-config",
    "--no-session-persistence",
    "--output-format",
    "json",
)
ENV_KEYS = ("HOME", "PATH", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TERM")


def prompt_sha256() -> str:
    return hashlib.sha256("\0".join(PROMPT_PARTS).encode()).hexdigest()


def argv() -> list[str]:
    """The scoring command, without the user message (it goes on stdin)."""
    return ["claude", *ROUTE_FLAGS, "--system-prompt", SYSTEM, "--json-schema", SCHEMA_JSON]


def zero_shot_user(render: str) -> str:
    return UNIT_INTRO + render


def few_shot_user(examples: Sequence[tuple[str, bool]], render: str) -> str:
    """The held-out few-shot message: each example is (its render, whether it is abusive), shown
    in the given order, then the unit."""
    if not examples:
        raise ValueError("few_shot_user needs at least one example")
    parts = [EXAMPLES_INTRO]
    for i, (text, abusive) in enumerate(examples, 1):
        parts.append(EXAMPLE.format(i=i, text=text, verdict=VERDICTS[0 if abusive else 1]))
    parts.append(TARGET_LEAD + UNIT_INTRO + render)
    return "".join(parts)
