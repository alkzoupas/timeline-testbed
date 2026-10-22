"""Known-answer tests for the M8 driver (testbed/models/haiku_run.py).

`claude` is never run: a fake executable first on the child's PATH answers `--version`, `auth
status` and each call from a plan file, and logs what it saw. The home directory and the temp dir
are moved under pytest's tmp_path."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from testbed.models import haiku_prompt as hp
from testbed.models import haiku_run as hr

VERSION = "9.9.9 (fake)"
FAKE = """#!@PY@
import hashlib, json, os, sys

state = "@STATE@"
plan = json.load(open(state + "/plan.json"))
with open(state + "/env.log", "a") as f:
    f.write(json.dumps(sorted(os.environ)) + "\\n")
args = sys.argv[1:]
if args == ["--version"]:
    print(plan["version"])
    sys.exit(0)
if args[:2] == ["auth", "status"]:
    info = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "configDirectory": os.environ.get("CLAUDE_CONFIG_DIR"),
        "email": "someone@example.com",
    }
    info.update(plan.get("auth", {}))
    print(json.dumps(info))
    sys.exit(0)
user = sys.stdin.read()
digest = hashlib.sha256(user.encode()).hexdigest()
n = 0
while True:
    try:
        os.close(os.open(f"{state}/n-{digest}-{n}", os.O_CREAT | os.O_EXCL))
        break
    except FileExistsError:
        n += 1
with open(state + "/calls.log", "a") as f:
    f.write(json.dumps({"user": user, "n": n, "cwd": os.getcwd(), "args": args}) + "\\n")
specs = plan["calls"].get(user, [])
spec = specs[n] if n < len(specs) else plan["default"]
model = "claude-haiku-5-5"
body = {"type": "result", "is_error": False, "modelUsage": {model: {}}}
if spec == "error":
    sys.stderr.write("boom: usage limit")
    body.update(is_error=True, result="You've hit your limit", api_error_status=429)
    print(json.dumps(body))
    sys.exit(1)
if spec == "badjson":
    sys.stdout.write("not json")
elif spec == "other_model":
    body.update(modelUsage={"claude-other": {}}, structured_output={"probability": 5})
    sys.stdout.write(json.dumps(body))
elif spec in ("unparsed", "unparsed_long"):
    text = "You've hit your limit" if spec == "unparsed" else "x" * 1000 + "THE-END"
    body.update(result=text, usage={"input_tokens": 10, "output_tokens": 5})
    sys.stdout.write(json.dumps(body))
else:
    if spec == "bump":  # another session's auto-update: --version answers differently from now on
        plan["version"] = "9.9.10 (fake)"
        json.dump(plan, open(state + "/plan.json", "w"))
        spec = 55
    body.update(structured_output={"probability": spec})
    body.update(usage={"input_tokens": 10, "output_tokens": 5})
    sys.stdout.write(json.dumps(body))
"""


def ok(p: object = 80, **extra: Any) -> str:
    return json.dumps(
        {
            "type": "result",
            "is_error": False,
            "modelUsage": {hp.MODEL: {}},
            "structured_output": {"probability": p},
            **extra,
        }
    )


# --- classify -------------------------------------------------------------------------------------


def body(**override: Any) -> str:
    data: dict[str, Any] = {
        "type": "result",
        "is_error": False,
        "modelUsage": {hp.MODEL: {}},
        "structured_output": {"probability": 42},
        "usage": {"input_tokens": 2, "output_tokens": 7},
    }
    data.update(override)
    return json.dumps({k: v for k, v in data.items() if v is not Ellipsis})


@pytest.mark.parametrize("p", [0, 1, 50, 99, 100])
def test_classify_ok(p: int) -> None:
    assert hr.classify(0, body(structured_output={"probability": p})) == ("ok", p)


@pytest.mark.parametrize(
    "stdout",
    [
        body(structured_output=Ellipsis),
        body(structured_output={}),
        body(structured_output={"probability": 101}),
        body(structured_output={"probability": -1}),
        body(structured_output={"probability": True}),
        body(structured_output={"probability": "80"}),
        body(structured_output={"probability": 80.5}),
        body(structured_output={"probability": None}),
        body(structured_output=[80]),
    ],
)
def test_classify_unparsed(stdout: str) -> None:
    assert hr.classify(0, stdout) == ("unparsed", None)


@pytest.mark.parametrize(
    ("rc", "stdout"),
    [
        (1, body()),
        (-1, body()),
        (0, "not json"),
        (0, ""),
        (0, "[1, 2]"),
        (0, body(is_error=True)),
        (0, body(is_error=Ellipsis)),
        (0, body(type="assistant")),
        (0, body(modelUsage={"claude-sonnet-5-5": {}})),
        (0, body(modelUsage={hp.MODEL: {}, "claude-sonnet-5-5": {}})),
        (0, body(modelUsage={})),
        (0, body(modelUsage=Ellipsis)),
        # an API error is an error even when the rest of the response looks like an answer
        (0, body(api_error_status=429)),
        (0, body(api_error_status=529, structured_output=Ellipsis)),
        # unparsed needs a model that answered: output tokens that are an int > 0
        (0, body(structured_output=Ellipsis, usage=Ellipsis)),
        (0, body(structured_output=Ellipsis, usage={})),
        (0, body(structured_output=Ellipsis, usage={"input_tokens": 2})),
        (0, body(structured_output=Ellipsis, usage={"output_tokens": 0})),
        (0, body(structured_output=Ellipsis, usage={"output_tokens": -3})),
        (0, body(structured_output=Ellipsis, usage={"output_tokens": True})),
        (0, body(structured_output=Ellipsis, usage={"output_tokens": "7"})),
        (0, body(structured_output=Ellipsis, usage={"output_tokens": 7.0})),
        (0, body(structured_output=Ellipsis, usage=[7])),
        (0, body(structured_output={"probability": 101}, usage={"output_tokens": 0})),
    ],
)
def test_classify_error(rc: int, stdout: str) -> None:
    assert hr.classify(rc, stdout) == ("error", None)


def test_classify_api_error_status_none_is_fine_and_ok_needs_no_usage() -> None:
    assert hr.classify(0, body(api_error_status=None)) == ("ok", 42)
    assert hr.classify(0, body(usage=Ellipsis)) == ("ok", 42)  # ok is decided by the probability
    assert hr.classify(0, body(structured_output=Ellipsis, usage={"output_tokens": 1})) == (
        "unparsed",
        None,
    )


def test_result_tail_keeps_the_last_300_chars() -> None:
    assert hr.RESULT_TAIL == 300
    assert hr.result_tail(body(result="a" * 1000 + "END")) == "a" * 297 + "END"
    assert hr.result_tail(body(result="short")) == "short"
    assert hr.result_tail(body(result=5)) == "" and hr.result_tail(body()) == ""
    assert hr.result_tail("not json") == "" and hr.result_tail("[1]") == ""


def test_is_done() -> None:
    assert hr.is_done(["ok"]) and hr.is_done(["unparsed", "error", "ok"])
    assert not hr.is_done([]) and not hr.is_done(["error"]) and not hr.is_done(["unparsed"] * 2)
    assert hr.is_done(["unparsed"] * hr.MAX_ATTEMPTS)
    assert not hr.is_done(["error", "unparsed", "error", "unparsed"])  # errors don't count


# --- The fake route -------------------------------------------------------------------------------


class Fake:
    """A fake `claude` on a private PATH, a fake home and a neutral temp dir. As on a real install,
    `bin/claude` is a symlink to `versions/<name>/claude`; `flip` re-points it."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.home = root / "fakehome"
        self.state = root / "state"
        self.tmpdir = root / "scratch"
        self.bindir = root / "bin"
        for d in (self.home, self.state, self.tmpdir, self.bindir):
            d.mkdir()
        self.install("v1")
        self.flip("v1")
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setenv("PATH", str(self.bindir))
        monkeypatch.setattr(tempfile, "tempdir", str(self.tmpdir))
        self.out = root / "raw"
        self.jobs = root / "jobs.jsonl"
        self.plan: dict[str, Any] = {"version": VERSION, "calls": {}, "default": 80}
        self.write_plan()

    def install(self, name: str, text: str | None = None) -> Path:
        """`versions/<name>/claude`: the fake script, or `text` (a different binary)."""
        script = self.root / "versions" / name / "claude"
        script.parent.mkdir(parents=True)
        script.write_text(
            text or FAKE.replace("@PY@", sys.executable).replace("@STATE@", str(self.state))
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return script

    def flip(self, name: str) -> None:
        """Point `bin/claude` at `versions/<name>/claude`, atomically, as an auto-update does."""
        tmp = self.bindir / "claude.tmp"
        tmp.symlink_to(self.root / "versions" / name / "claude")
        os.replace(tmp, self.bindir / "claude")

    def write_plan(self) -> None:
        (self.state / "plan.json").write_text(json.dumps(self.plan))

    def script(self, key: str, *specs: object) -> None:
        self.plan["calls"][f"user {key}"] = list(specs)
        self.write_plan()

    def write_jobs(self, keys: Sequence[str], **override: Any) -> None:
        rows = [self.row(key, **override) for key in keys]
        self.jobs.write_text("".join(json.dumps(r) + "\n" for r in rows))

    @staticmethod
    def row(key: str, **override: Any) -> dict[str, Any]:
        user = f"user {key}"
        row = {
            "key": key,
            "world": "w",
            "unit": key.split(".")[0],
            "variant": "zero_shot",
            "draw": 1,
            "prompt_sha256": hp.PROMPT_SHA256,
            "user": user,
            "user_sha256": hr.sha256_text(user),
        }
        return row | override

    def calls(self) -> list[dict[str, Any]]:
        path = self.state / "calls.log"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def main(self, *extra: str) -> int:
        return hr.main(["--jobs", str(self.jobs), "--out", str(self.out), *extra])

    def run_json(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.out / "run.json").read_text())
        return data


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    lowered = str(tmp_path).lower()
    if any(w in lowered for w in (*hr.CWD_WORDS, "fakehome")):
        pytest.skip("pytest's tmp path itself contains a word the cwd check refuses")
    return Fake(tmp_path, monkeypatch)


KEYS = [f"u{i}.zero_shot.d1" for i in range(6)]


# --- Environment and route ------------------------------------------------------------------------


def test_child_env_is_the_allowlist_plus_the_config_dir(
    fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.setenv(name, "planted")
    monkeypatch.setenv("TERM", "xterm")
    env = hr.child_env()
    assert set(env) <= {*hp.ENV_KEYS, "CLAUDE_CONFIG_DIR"}
    assert env["CLAUDE_CONFIG_DIR"] == str(fake.home / ".claude-haiku-scorer")
    assert env["HOME"] == str(fake.home) and env["TERM"] == "xterm"
    assert not {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"} & set(env)
    assert hr.ENV_NAMES == (*hp.ENV_KEYS, "CLAUDE_CONFIG_DIR")


def test_only_the_allowlist_reaches_the_child(fake: Fake, monkeypatch: pytest.MonkeyPatch) -> None:
    planted = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    )
    for name in planted:
        monkeypatch.setenv(name, "planted")
    fake.write_jobs(KEYS[:2])
    assert fake.main() == 0
    logs = [json.loads(x) for x in (fake.state / "env.log").read_text().splitlines()]
    assert len(logs) == 2 + 2 + 1  # --version, auth status, two calls, --version again
    for names in logs:
        assert not set(planted) & set(names)
        assert "CLAUDE_CONFIG_DIR" in names
    assert all(set(names) & {"HOME", "PATH"} == {"HOME", "PATH"} for names in logs)
    assert fake.run_json()["env_keys"] == list(hr.ENV_NAMES)  # names only
    assert "planted" not in (fake.out / "run.json").read_text()


def test_each_call_uses_the_frozen_argv_stdin_and_one_empty_cwd(fake: Fake) -> None:
    fake.write_jobs(KEYS[:3])
    assert fake.main() == 0
    calls = fake.calls()
    assert len(calls) == 3
    assert {c["user"] for c in calls} == {f"user {k}" for k in KEYS[:3]}
    assert all(c["args"] == hp.argv()[1:] for c in calls)
    cwd = Path(fake.run_json()["cwd"])
    assert cwd.parent == fake.tmpdir and not any(cwd.iterdir())
    assert {os.path.realpath(c["cwd"]) for c in calls} == {os.path.realpath(cwd)}
    assert fake.run_json()["cwd_empty_after"] is True


# --- Preflight ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("auth", "word"),
    [
        ({"authMethod": "api_key"}, "subscription_auth"),
        ({"authMethod": None}, "subscription_auth"),
        ({"configDirectory": "/elsewhere"}, "config_dir_is_scorer"),
        ({"loggedIn": False}, "logged_in"),
    ],
)
def test_preflight_refuses_a_wrong_login(
    fake: Fake, capsys: pytest.CaptureFixture[str], auth: dict[str, Any], word: str
) -> None:
    fake.plan["auth"] = auth
    fake.write_plan()
    fake.write_jobs(KEYS[:2])
    assert fake.main() == 2
    assert word in capsys.readouterr().err
    assert fake.calls() == [] and not (fake.out / "run.json").exists()


def test_run_json_keeps_only_the_three_auth_booleans(fake: Fake) -> None:
    fake.write_jobs(KEYS[:1])
    assert fake.main() == 0
    assert fake.run_json()["preflight"] == {
        "logged_in": True,
        "subscription_auth": True,
        "config_dir_is_scorer": True,
    }
    assert "someone@example.com" not in (fake.out / "run.json").read_text()


@pytest.mark.parametrize("word", [*hr.CWD_WORDS, "fakehome"])
def test_check_cwd_refuses_a_path_with_a_forbidden_word(
    fake: Fake, tmp_path: Path, word: str
) -> None:
    bad = tmp_path / f"x{word.upper()}y"
    bad.mkdir()
    with pytest.raises(hr.Refusal, match="contains"):
        hr.check_cwd(bad)
    good = tmp_path / "plain"
    good.mkdir()
    hr.check_cwd(good)
    (good / "f").write_text("x")
    with pytest.raises(hr.Refusal, match="not empty"):
        hr.check_cwd(good)


def test_a_temp_dir_with_a_forbidden_word_is_refused(
    fake: Fake, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = fake.root / "timeline-scratch"
    bad.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(bad))
    fake.write_jobs(KEYS[:1])
    assert fake.main() == 2
    assert "cwd checks" in capsys.readouterr().err
    assert fake.calls() == []
    assert not any(bad.iterdir())  # the refused dirs were removed again


def test_a_non_empty_recorded_cwd_is_refused_and_a_missing_one_replaced(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs(KEYS[:2])
    assert fake.main("--max-calls", "0") == 0
    cwd = Path(fake.run_json()["cwd"])
    (cwd / "stray").write_text("x")
    assert fake.main() == 2
    assert "not empty" in capsys.readouterr().err and fake.calls() == []
    (cwd / "stray").unlink()
    cwd.rmdir()
    assert fake.main() == 0
    assert Path(fake.run_json()["cwd"]).is_dir() and Path(fake.run_json()["cwd"]) != cwd


# --- Running --------------------------------------------------------------------------------------


def test_a_run_writes_raw_then_meta_per_attempt_and_resumes_without_redoing(fake: Fake) -> None:
    fake.write_jobs(KEYS)
    assert fake.main() == 0
    calls = fake.out / "calls"
    for key in KEYS:
        raw, meta = calls / f"{key}.a1.json", calls / f"{key}.a1.meta.json"
        assert json.loads(raw.read_text())["structured_output"] == {"probability": 80}
        m = json.loads(meta.read_text())
        assert m["key"] == key and m["attempt"] == 1 and m["status"] == "ok"
        assert m["probability"] == 80 and m["rc"] == 0 and m["models"] == [hp.MODEL]
        assert m["cli_version"] == VERSION and m["user_sha256"] == hr.sha256_text(f"user {key}")
        assert set(m) == {
            "key",
            "attempt",
            "status",
            "probability",
            "rc",
            "started",
            "wall_s",
            "stderr_tail",
            "user_sha256",
            "cli_version",
            "models",
        }
    assert len(fake.calls()) == 6
    before = {p.name: p.read_text() for p in calls.iterdir()}
    assert fake.main() == 0
    assert len(fake.calls()) == 6  # nothing redone
    assert {p.name: p.read_text() for p in calls.iterdir()} == before  # nothing rewritten
    assert not list(calls.glob("*.tmp"))


def statuses(fake: Fake, key: str) -> list[str]:
    files = sorted((fake.out / "calls").glob(f"{key}.a*.meta.json"))
    return [json.loads(f.read_text())["status"] for f in files]


def test_unparsed_is_retried_up_to_max_attempts(fake: Fake) -> None:
    fake.script(KEYS[0], "unparsed", "unparsed", 77)  # ok on the third attempt
    fake.script(KEYS[1], "unparsed", 33)
    fake.script(KEYS[2], "unparsed", "unparsed", 44)  # an ok resets the streak each time
    fake.write_jobs(KEYS[:4])
    assert fake.main("--workers", "1") == 0
    assert statuses(fake, KEYS[0]) == ["unparsed", "unparsed", "ok"]
    assert statuses(fake, KEYS[1]) == ["unparsed", "ok"]
    assert statuses(fake, KEYS[2]) == ["unparsed", "unparsed", "ok"]
    assert statuses(fake, KEYS[3]) == ["ok"]
    assert len(fake.calls()) == 3 + 2 + 3 + 1


def test_a_key_is_done_unparsed_after_max_attempts_and_not_retried(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.script(KEYS[0], "unparsed", "unparsed", "unparsed", 77)  # DONE unparsed at three
    fake.write_jobs(KEYS[:3])
    # Three unparsed in a row is also the streak breaker's trigger: the run stops, exit 3.
    assert fake.main("--workers", "1") == 3
    assert statuses(fake, KEYS[0]) == ["unparsed"] * hr.MAX_ATTEMPTS == ["unparsed"] * 3
    assert len(fake.calls()) == 3
    assert "remaining 2" in capsys.readouterr().out  # key 0 is DONE: two keys remain
    assert fake.main("--workers", "1") == 0  # the rerun resumes with the next keys
    assert statuses(fake, KEYS[0]) == ["unparsed"] * 3  # not retried
    assert len(fake.calls()) == 3 + 2


def test_three_unparsed_in_a_row_stop_the_run_like_an_error(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.plan["default"] = "unparsed"  # a limit that comes back looking like a result, forever
    fake.write_plan()
    fake.write_jobs(KEYS)
    assert fake.main("--workers", "1") == 3
    out = capsys.readouterr().out
    assert len(fake.calls()) == hr.UNPARSED_STREAK == 3  # no new call after the third
    assert statuses(fake, KEYS[0]) == ["unparsed"] * 3  # committed, counted toward MAX_ATTEMPTS
    assert "STOPPED after 3 unparsed attempts in a row" in out
    assert out.count("You've hit your limit") == 3
    assert "errors this run 0" in out
    # A rerun resumes: key 0 is DONE (unparsed), key 1 starts and trips the breaker again.
    assert fake.main("--workers", "1") == 3
    assert len(fake.calls()) == 6 and statuses(fake, KEYS[1]) == ["unparsed"] * 3
    # Once the route works again the rest finishes; the unparsed keys are not retried.
    fake.plan["default"] = 80
    fake.write_plan()
    assert fake.main("--workers", "1") == 0
    assert len(fake.calls()) == 6 + 4
    assert statuses(fake, KEYS[0]) == ["unparsed"] * 3 and statuses(fake, KEYS[5]) == ["ok"]


def test_the_streak_stops_in_flight_calls_committed_with_several_workers(fake: Fake) -> None:
    fake.plan["default"] = "unparsed"
    fake.write_plan()
    fake.write_jobs(KEYS)
    assert fake.main("--workers", "3") == 3
    metas = list((fake.out / "calls").glob("*.meta.json"))
    raws = [p for p in (fake.out / "calls").glob("*.json") if ".meta." not in p.name]
    assert len(fake.calls()) == len(metas) == len(raws) >= 3  # whatever started was committed
    assert len(fake.calls()) < 3 * len(KEYS)  # but the run did not carry on to the end


def test_the_streak_tails_are_cut_to_300_chars(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.script(KEYS[0], "unparsed_long", "unparsed_long", "unparsed_long")
    fake.write_jobs(KEYS[:2])
    assert fake.main("--workers", "1") == 3
    out = capsys.readouterr().out
    assert out.count("THE-END") == 3 and "x" * 301 not in out


def test_an_ok_resets_the_streak(fake: Fake) -> None:
    for key in KEYS[:4]:
        fake.script(key, "unparsed", "unparsed", 60)
    fake.write_jobs(KEYS[:4])
    assert fake.main("--workers", "1") == 0  # u u ok, u u ok, ...: never three in a row
    assert len(fake.calls()) == 12


def test_the_first_error_stops_new_calls_and_a_rerun_resumes_it(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.script(KEYS[2], "error", 61)
    fake.write_jobs(KEYS)
    assert fake.main("--workers", "1") == 3
    out = capsys.readouterr().out
    assert (
        "rc=1" in out and "api_error_status=" in out and "429" in out and "boom: usage limit" in out
    )
    assert "remaining 4" in out and "errors this run 1" in out
    assert len(fake.calls()) == 3  # two ok, the error; the other three were never started
    calls = fake.out / "calls"
    failed = json.loads((calls / f"{KEYS[2]}.a1.meta.json").read_text())
    assert failed["status"] == "error" and failed["rc"] == 1 and "boom" in failed["stderr_tail"]
    assert fake.main("--workers", "1") == 0
    resumed = json.loads((calls / f"{KEYS[2]}.a2.meta.json").read_text())
    assert resumed["status"] == "ok" and resumed["probability"] == 61 and resumed["attempt"] == 2
    assert len(fake.calls()) == 3 + 1 + 3  # the retry and the three untouched keys
    assert not (calls / f"{KEYS[0]}.a2.json").exists()


@pytest.mark.parametrize("spec", ["badjson", "other_model"])
def test_a_bad_answer_is_an_error_not_a_score(fake: Fake, spec: str) -> None:
    fake.script(KEYS[0], spec)
    fake.write_jobs(KEYS[:1])
    assert fake.main("--workers", "1") == 3
    meta = json.loads((fake.out / "calls" / f"{KEYS[0]}.a1.meta.json").read_text())
    assert meta["status"] == "error" and meta["probability"] is None


def test_errors_do_not_count_toward_max_attempts(fake: Fake) -> None:
    fake.script(KEYS[0], "error", "unparsed", "unparsed", "error", "unparsed", 9)
    fake.write_jobs(KEYS[:1])
    codes = [fake.main("--workers", "1") for _ in range(3)]
    assert codes == [3, 3, 0]  # the two errors stop a run each; the third unparsed makes it DONE
    metas = sorted((fake.out / "calls").glob("*.meta.json"))
    statuses = [json.loads(p.read_text())["status"] for p in metas]
    assert statuses == ["error", "unparsed", "unparsed", "error", "unparsed"]
    assert len(fake.calls()) == 5  # the 9 is never reached
    assert fake.main() == 0 and len(fake.calls()) == 5


def test_max_calls_caps_the_subprocess_calls(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs(KEYS)
    assert fake.main("--workers", "1", "--max-calls", "2") == 0
    out = capsys.readouterr().out
    assert len(fake.calls()) == 2 and "remaining 4" in out and "stopped at --max-calls" in out
    assert fake.main("--workers", "2", "--max-calls", "3") == 0
    assert len(fake.calls()) == 5
    assert fake.main() == 0 and len(fake.calls()) == 6


def test_an_attempt_file_without_its_meta_is_ignored_and_never_overwritten(fake: Fake) -> None:
    fake.write_jobs(KEYS[:1])
    calls = fake.out / "calls"
    calls.mkdir(parents=True)
    (calls / f"{KEYS[0]}.a1.json").write_text("half-written")  # a crash before the meta
    assert fake.main() == 0
    assert (calls / f"{KEYS[0]}.a1.json").read_text() == "half-written"
    assert json.loads((calls / f"{KEYS[0]}.a2.meta.json").read_text())["attempt"] == 2
    assert not (calls / f"{KEYS[0]}.a1.meta.json").exists()


# --- Binding --------------------------------------------------------------------------------------


def test_a_rerun_with_a_changed_binding_is_refused(
    fake: Fake, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs(KEYS[:2])
    assert fake.main() == 0
    capsys.readouterr()
    n_calls = len(fake.calls())

    fake.plan["version"] = "9.9.10 (fake)"  # the CLI version
    fake.write_plan()
    assert fake.main() == 2 and "cli_version" in capsys.readouterr().err
    fake.plan["version"] = VERSION
    fake.write_plan()

    with monkeypatch.context() as m:  # the route
        m.setattr(hp, "ROUTE_FLAGS", (*hp.ROUTE_FLAGS, "--verbose"))
        assert fake.main() == 2 and "route" in capsys.readouterr().err

    fake.write_jobs(KEYS[:3])  # the job file
    assert fake.main() == 2 and "jobs_sha256" in capsys.readouterr().err
    assert len(fake.calls()) == n_calls


def test_the_calls_keep_the_binary_resolved_at_preflight(
    fake: Fake, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = fake.root / "second-binary-ran"
    fake.install("v2", f"#!/bin/sh\ntouch {marker}\necho 1.0.0\nexit 1\n")
    real_preflight = hr.preflight

    def flipping(*args: Any, **kwargs: Any) -> Any:
        result = real_preflight(*args, **kwargs)
        fake.flip("v2")  # an auto-update lands right after the preflight
        return result

    monkeypatch.setattr(hr, "preflight", flipping)
    fake.write_jobs(KEYS[:3])
    assert fake.main("--workers", "1") == 0
    assert (fake.bindir / "claude").resolve().parent.name == "v2"  # PATH now leads to the new one
    assert not marker.exists()  # but nothing ran it
    assert len(fake.calls()) == 3
    run = fake.run_json()
    first = os.path.realpath(fake.root / "versions" / "v1" / "claude")
    assert run["cli_path"] == first and run["cli_version_after"] == VERSION
    assert f"cli_path {first}" in capsys.readouterr().out


def test_a_changed_resolved_cli_path_on_rerun_is_refused(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs(KEYS[:2])
    assert fake.main() == 0
    capsys.readouterr()
    fake.install("v2")  # the same version string, a different file
    fake.flip("v2")
    assert fake.main() == 2 and "cli_path" in capsys.readouterr().err
    assert len(fake.calls()) == 2
    fake.flip("v1")  # back to the bound path: resumes
    assert fake.main() == 0


def test_a_cli_not_on_the_childs_path_is_refused(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    (fake.bindir / "claude").unlink()
    fake.write_jobs(KEYS[:1])
    assert fake.main() == 2 and "not on the child's PATH" in capsys.readouterr().err
    assert fake.calls() == [] and not (fake.out / "run.json").exists()


def test_a_cli_version_that_changes_during_the_run_is_exit_3(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.script(KEYS[1], "bump")  # this call's binary starts reporting another version
    fake.write_jobs(KEYS[:3])
    assert fake.main("--workers", "1") == 3
    out = capsys.readouterr().out
    assert "CLI VERSION CHANGED" in out and VERSION in out and "9.9.10 (fake)" in out
    assert len(fake.calls()) == 3  # the calls were made and committed
    assert len(list((fake.out / "calls").glob("*.meta.json"))) == 3
    run = fake.run_json()
    assert run["cli_version"] == VERSION and run["cli_version_after"] == "9.9.10 (fake)"
    assert fake.main() == 2 and "cli_version" in capsys.readouterr().err  # and a rerun is refused


def test_cli_version_after_is_recorded_only_when_calls_were_made(fake: Fake) -> None:
    fake.write_jobs(KEYS[:2])
    assert fake.main("--max-calls", "0") == 0
    assert "cli_version_after" not in fake.run_json()
    assert fake.main() == 0
    assert fake.run_json()["cli_version_after"] == VERSION
    assert fake.main() == 0  # nothing left to call: the recorded value stays
    assert fake.run_json()["cli_version_after"] == VERSION


def test_a_failing_final_version_check_is_exit_3(
    fake: Fake, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(hr, "cli_version_of", lambda *args: "")  # `--version` failed at the end
    fake.write_jobs(KEYS[:1])
    assert fake.main() == 3
    assert "CLI VERSION CHANGED" in capsys.readouterr().out
    assert fake.run_json()["cli_version_after"] == ""


def test_check_binding_names_the_first_differing_field() -> None:
    current = hr.binding("a" * 64, "/opt/claude/1.0", VERSION, Path("/x"))
    recorded = dict(current)
    hr.check_binding(recorded, {**current, "cwd": "/elsewhere"})  # the cwd may change
    for field, value in (
        ("prompt_version", "m8-v0"),
        ("prompt_sha256", "0" * 64),
        ("jobs_sha256", "0" * 64),
        ("route", ["-p"]),
        ("model", "claude-other"),
        ("cli_path", "/opt/claude/2.0"),
        ("cli_version", "1.0"),
        ("env_keys", ["HOME"]),
        ("max_attempts", 9),
        ("call_timeout_s", 1),
    ):
        with pytest.raises(hr.Refusal, match=field):
            hr.check_binding({**recorded, field: value}, current)


def test_a_changed_prompt_hash_is_refused(
    fake: Fake, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs(KEYS[:2])
    assert fake.main() == 0
    monkeypatch.setattr(hp, "PROMPT_SHA256", "f" * 64)
    monkeypatch.setattr(hp, "prompt_sha256", lambda: "f" * 64)
    assert fake.main() == 2
    assert "prompt_sha256" in capsys.readouterr().err


# --- Job checks -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "word"),
    [
        ({"user_sha256": "0" * 64}, "user_sha256"),
        ({"prompt_sha256": "0" * 64}, "prompt_sha256"),
        ({"positive": True}, "fields"),  # a label field
        ({"key": "../escape.zero_shot.d1"}, "filename-safe"),
        ({"key": ".hidden"}, "filename-safe"),
    ],
)
def test_a_bad_job_row_is_refused(
    fake: Fake, capsys: pytest.CaptureFixture[str], override: dict[str, Any], word: str
) -> None:
    fake.write_jobs(KEYS[:2])
    rows = [json.loads(x) for x in fake.jobs.read_text().splitlines()]
    rows[1] = {**rows[1], **override}
    fake.jobs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert fake.main() == 2
    assert word in capsys.readouterr().err
    assert fake.calls() == []


def test_duplicate_keys_and_an_empty_file_are_refused(
    fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs([KEYS[0], KEYS[0]])
    assert fake.main() == 2 and "duplicate key" in capsys.readouterr().err
    fake.jobs.write_text("")
    assert fake.main() == 2 and "no jobs" in capsys.readouterr().err


def test_a_changed_prompt_text_is_refused_before_any_call(
    fake: Fake, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.write_jobs(KEYS[:1])
    monkeypatch.setattr(hp, "PROMPT_PARTS", (*hp.PROMPT_PARTS, "changed"))
    assert fake.main() == 2 and "differs from PROMPT_SHA256" in capsys.readouterr().err
    assert fake.calls() == []


def test_the_driver_never_mentions_the_eval_package() -> None:
    source = Path(hr.__file__).read_text()
    assert "testbed.eval" not in source
    assert "ground_truth" not in source
    assert set(hr.JOB_FIELDS).isdisjoint({"family", "pair", "side", "positive", "label", "group"})


def test_models_helpers() -> None:
    assert hr.models_of(ok()) == [hp.MODEL]
    assert hr.models_of("nope") == [] and hr.models_of("[1]") == []
    assert hr.tail("x" * 5000) == "x" * hr.STDERR_TAIL and hr.tail(None) == ""
    detail = hr.error_detail(1, json.dumps({"result": "limit", "api_error_status": 429}), "oops\n")
    assert "rc=1" in detail and "limit" in detail and "429" in detail and "oops" in detail
    assert hr.percentile_line([]) == "wall p50/max: n/a"
    assert hr.percentile_line([1.0, 3.0, 5.0]) == "wall p50/max: 3.0/5.0 s"
