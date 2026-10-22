"""Known-answer tests for the frozen M8 prompt `m8-v1` and the Haiku route constants.
Coordinator-owned, like the module."""

from __future__ import annotations

import hashlib
import json

import pytest

from testbed.models import haiku_prompt as hp
from testbed.models import prompt
from testbed.world.families import PAIRS

# The smoke's draft system prompt (reports/M8.md, runs/m8-smoke/out/*.meta.json).
SMOKE_SYSTEM_SHA256 = "0f6af5efd171431357144fc6f1157459fb23ed97d352434d29e9368054b3b4ec"


def test_prompt_hash_is_pinned() -> None:
    assert hp.prompt_sha256() == hp.PROMPT_SHA256
    assert hp.PROMPT_VERSION == "m8-v1"


def test_system_is_the_smoke_draft_and_reuses_m4_text() -> None:
    assert hashlib.sha256(hp.SYSTEM.encode()).hexdigest() == SMOKE_SYSTEM_SHA256
    assert hp.SYSTEM.startswith(prompt.SETTING + "\n\n")
    assert prompt.QUESTION in hp.SYSTEM


def test_prompt_names_no_family() -> None:
    text = " ".join(hp.PROMPT_PARTS).lower()
    for names in PAIRS.values():
        for family in names:
            assert family.lower() not in text
            assert family.replace("_", " ").lower() not in text


def test_zero_shot_user_is_intro_then_render() -> None:
    assert hp.zero_shot_user("R") == (
        "The accounts below are linked by at least one shared resource. Here is their complete "
        "activity record:\n\nR"
    )


def test_few_shot_user_known_answer() -> None:
    got = hp.few_shot_user([("A {x}", True), ("B", False)], "T")
    assert got == (
        "Labelled examples from other cases, for reference:\n\n"
        "Example 1. Activity record:\n\nA {x}\n\nAnswer: one person running every account\n\n"
        "Example 2. Activity record:\n\nB\n\n"
        "Answer: independent people who share resources\n\n"
        "Now the case to judge. The accounts below are linked by at least one shared resource. "
        "Here is their complete activity record:\n\nT"
    )


def test_few_shot_user_needs_examples() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        hp.few_shot_user([], "T")


def test_schema_is_an_integer_probability() -> None:
    assert json.loads(hp.SCHEMA_JSON) == hp.SCHEMA
    field = hp.SCHEMA["properties"]["probability"]  # type: ignore[index]
    assert field == {"type": "integer", "minimum": 0, "maximum": 100}
    assert hp.SCHEMA["required"] == ["probability"]


def test_argv_is_the_isolated_route() -> None:
    cmd = hp.argv()
    assert cmd == [
        "claude",
        "-p",
        "--model",
        "claude-haiku-5-5",
        "--safe-mode",
        "--tools",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--system-prompt",
        hp.SYSTEM,
        "--json-schema",
        hp.SCHEMA_JSON,
    ]
    assert "--effort" not in cmd


def test_env_allowlist_carries_no_credentials() -> None:
    assert hp.ENV_KEYS == ("HOME", "PATH", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TERM")
    assert hp.CONFIG_DIR_NAME == ".claude-haiku-scorer"
