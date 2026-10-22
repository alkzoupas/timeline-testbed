"""M8 driver: the Claude Haiku calls of a job file (context/m8-haiku-kickoff.md "The route").

    python -m testbed.models.haiku_run --jobs runs/m8-pilot/jobs.jsonl --out runs/m8-pilot/raw \
        [--workers 5] [--max-calls N]

Label-free: it reads only the job file (`key`, `unit`, `user`, ... and no label field) and runs
`haiku_prompt.argv()` once per call with the row's `user` on stdin. The child gets an explicit
environment, never ours: `haiku_prompt.ENV_KEYS` plus `CLAUDE_CONFIG_DIR` (an API key in our
environment would bypass the subscription). It runs in one empty temp directory whose path names
nothing; a path with a project, tool or home-directory word in it is refused.

Preflight: the CLI is resolved once, to `cli_path` (the real path of `claude` on the child's PATH),
and every call, the preflight's and the run's, execs that path, so another Claude Code session's
auto-update (the `claude` symlink flipping to a new version) can't switch binaries mid-run.
`claude auth status` must report the subscription login on the scorer's config dir, and
`claude --version` is recorded. `<out>/run.json` binds the prompt hash, the job file's sha256, the
route flags, the model, the CLI path and the CLI version; a rerun must match all of them (the cwd
may change). An invocation that made calls ends with `--version` again; a different answer (or none)
is recorded as `cli_version_after` and exits 3.

Each attempt writes `calls/<key>.a<N>.json` (the raw stdout, verbatim) and then
`calls/<key>.a<N>.meta.json`, the commit marker, via a temp file and `os.replace`. Nothing is
overwritten; an attempt file without its meta is ignored, and the next attempt number is above
every file present. `classify` sorts an attempt into "ok", "unparsed" (the model answered, but not
with a probability) or "error" (anything else). A key is DONE with an "ok" attempt or after
`MAX_ATTEMPTS` "unparsed" ones; "error" attempts don't count toward that. The first error stops
new calls, in-flight ones finish, and the run exits 3; a rerun resumes. So do `UNPARSED_STREAK`
"unparsed" attempts in a row (completion order, one invocation): a usage limit or route fault that
looks like a normal result must not turn the rest of the job into 50s. Those attempts stay
committed and count toward `MAX_ATTEMPTS`; the streak's `result` texts are printed. Exit 2 is a
refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from testbed.models import haiku_prompt as hp

DEFAULT_WORKERS = 5
CALL_TIMEOUT_S = 900
PREFLIGHT_TIMEOUT_S = 60
MAX_ATTEMPTS = 3  # the first try plus two retries of an "unparsed" answer
UNPARSED_STREAK = 3  # "unparsed" attempts in a row that stop the run like an error
RESULT_TAIL = 300  # chars of a `result` text printed in a report line
STDERR_TAIL = 4000
OK, UNPARSED, ERROR = "ok", "unparsed", "error"
TIMEOUT_RC = -1  # the recorded rc of a call that hit CALL_TIMEOUT_S
SPAWN_RC = -2  # the recorded rc of a call whose process could not be started
EXIT_REFUSED = 2
EXIT_ERROR = 3
# A job row's exact fields: the driver's input carries no label.
JOB_FIELDS = ("key", "world", "unit", "variant", "draw", "prompt_sha256", "user", "user_sha256")
KEY_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")  # filename-safe
CWD_WORDS = ("timeline", "testbed", "claude", "haiku", "m8", "abuse")  # not in the cwd path
CWD_TRIES = 20
ENV_NAMES = (*hp.ENV_KEYS, "CLAUDE_CONFIG_DIR")  # the allowlist, recorded by name only
ATTEMPT_FILE = re.compile(r"^(?P<key>.+)\.a(?P<n>[0-9]+)(?P<meta>\.meta)?\.json$")


class Refusal(Exception):
    """A check failed before any call: exit 2."""


# --- Pure core ------------------------------------------------------------------------------------


def classify(rc: int, stdout: str) -> tuple[str, int | None]:
    """(status, probability) of one attempt: "ok", "unparsed" or "error" (see the module doc).

    "error" also covers an `api_error_status`, and a response with no output tokens (no answer was
    made, whatever else it says): "unparsed" needs `usage.output_tokens` > 0."""
    if rc != 0:
        return ERROR, None
    try:
        data = json.loads(stdout)
    except ValueError:
        return ERROR, None
    if (
        not isinstance(data, dict)
        or data.get("type") != "result"
        or data.get("is_error") is not False
        or data.get("api_error_status") is not None
    ):
        return ERROR, None
    models = data.get("modelUsage")
    if not isinstance(models, dict) or set(models) != {hp.MODEL}:
        return ERROR, None
    structured = data.get("structured_output")
    value = structured.get("probability") if isinstance(structured, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100:
        return OK, value
    usage = data.get("usage")
    produced = usage.get("output_tokens") if isinstance(usage, dict) else None
    if isinstance(produced, int) and not isinstance(produced, bool) and produced > 0:
        return UNPARSED, None
    return ERROR, None


def is_done(statuses: Sequence[str]) -> bool:
    """A key is DONE with an "ok" attempt or `MAX_ATTEMPTS` "unparsed" ones."""
    return OK in statuses or statuses.count(UNPARSED) >= MAX_ATTEMPTS


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config_dir() -> Path:
    return Path.home() / hp.CONFIG_DIR_NAME


def child_env() -> dict[str, str]:
    """The child's whole environment: the allowlisted keys that are set, and the scorer's config
    dir. Nothing else is inherited."""
    env = {k: os.environ[k] for k in hp.ENV_KEYS if k in os.environ}
    env["CLAUDE_CONFIG_DIR"] = str(config_dir())
    return env


def check_cwd(path: Path) -> None:
    """Raise `Refusal` unless `path` is an empty directory whose lowercased path has no word of
    `CWD_WORDS` and not the home directory's name."""
    words = (*CWD_WORDS, Path.home().name.lower())
    lowered = str(path).lower()
    for word in words:
        if word and word in lowered:
            raise Refusal(f"cwd {path} contains {word!r}; it must name nothing")
    if not path.is_dir():
        raise Refusal(f"cwd {path} is not a directory")
    if any(path.iterdir()):
        raise Refusal(f"cwd {path} is not empty")


def read_jobs(path: Path) -> tuple[list[dict[str, Any]], str]:
    """(job rows in file order, sha256 of the file's bytes), after the per-row checks."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows:
        raise Refusal(f"{path}: no jobs")
    if hp.prompt_sha256() != hp.PROMPT_SHA256:
        raise Refusal("haiku_prompt.py text differs from PROMPT_SHA256")
    keys: set[str] = set()
    for i, row in enumerate(rows):
        if set(row) != set(JOB_FIELDS):
            raise Refusal(f"job {i}: fields {sorted(row)} != {sorted(JOB_FIELDS)}")
        key = row["key"]
        if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
            raise Refusal(f"job {i}: key {key!r} is not filename-safe")
        if key in keys:
            raise Refusal(f"job {i}: duplicate key {key}")
        keys.add(key)
        if row["prompt_sha256"] != hp.PROMPT_SHA256:
            raise Refusal(f"{key}: prompt_sha256 is not the frozen {hp.PROMPT_VERSION} hash")
        if row["user_sha256"] != sha256_text(row["user"]):
            raise Refusal(f"{key}: user_sha256 does not match its user message")
    return rows, hashlib.sha256(raw).hexdigest()


def tail(text: str | None) -> str:
    return (text or "")[-STDERR_TAIL:]


def models_of(stdout: str) -> list[str]:
    """The sorted `modelUsage` keys of an attempt's stdout, or [] if it has none."""
    try:
        usage = json.loads(stdout).get("modelUsage")
    except ValueError, AttributeError:
        return []
    return sorted(usage) if isinstance(usage, dict) else []


def error_detail(rc: int, stdout: str, stderr: str) -> str:
    """One line for an errored attempt: rc, the result and api_error_status if present, stderr."""
    parts = [f"rc={rc}"]
    try:
        data = json.loads(stdout)
    except ValueError:
        data = None
    if isinstance(data, dict):
        for field in ("result", "api_error_status"):
            if data.get(field) is not None:
                parts.append(f"{field}={str(data[field])[:300]!r}")
    if stderr.strip():
        parts.append(f"stderr tail={stderr.strip()[-300:]!r}")
    return " ".join(parts)


def result_tail(stdout: str) -> str:
    """The last `RESULT_TAIL` chars of an attempt's `result` text, "" if it has none."""
    try:
        text = json.loads(stdout).get("result")
    except ValueError, AttributeError:
        return ""
    return text[-RESULT_TAIL:] if isinstance(text, str) else ""


# --- Claude ---------------------------------------------------------------------------------------


def claude(
    args: Sequence[str], user: str | None, cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        input=user,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=dict(env),
        timeout=timeout,
    )


def resolve_cli(env: Mapping[str, str]) -> str:
    """The real path of `claude` on the child's PATH, or `Refusal`. Resolved once per run."""
    found = shutil.which(hp.argv()[0], path=env.get("PATH", ""))
    if found is None:
        raise Refusal(f"{hp.argv()[0]} is not on the child's PATH")
    return os.path.realpath(found)


def cli_version_of(cli_path: str, cwd: Path, env: Mapping[str, str]) -> str:
    """`<cli_path> --version`, stripped; "" if it failed or printed nothing."""
    try:
        done = claude([cli_path, "--version"], None, cwd, env, PREFLIGHT_TIMEOUT_S)
    except OSError, subprocess.TimeoutExpired:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def preflight(cli_path: str, cwd: Path, env: Mapping[str, str]) -> tuple[str, dict[str, bool]]:
    """(the CLI version, the three auth checks as booleans); raise `Refusal` unless the login is
    the subscription's on the scorer's config dir. Nothing else of `auth status` is kept."""
    try:
        version = claude([cli_path, "--version"], None, cwd, env, PREFLIGHT_TIMEOUT_S)
        status = claude([cli_path, "auth", "status"], None, cwd, env, PREFLIGHT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Refusal(f"preflight could not run claude: {type(e).__name__}") from e
    if version.returncode != 0 or not version.stdout.strip():
        raise Refusal(f"claude --version failed (rc {version.returncode})")
    try:
        info = json.loads(status.stdout)
    except ValueError:
        info = None
    if not isinstance(info, dict):
        raise Refusal(f"claude auth status gave no JSON (rc {status.returncode})")
    checks = {
        "logged_in": info.get("loggedIn") is True,
        "subscription_auth": info.get("authMethod") == "claude.ai",
        "config_dir_is_scorer": info.get("configDirectory") == str(config_dir()),
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise Refusal(f"auth preflight failed: {', '.join(failed)}")
    return version.stdout.strip(), checks


# --- The attempt store ----------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Store:
    """`<out>/calls/`: per key, the attempts that have a raw file and the committed ones."""

    def __init__(self, out: Path) -> None:
        self.dir = out / "calls"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.highest: dict[str, int] = defaultdict(int)
        self.statuses: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for path in sorted(self.dir.iterdir()):
            m = ATTEMPT_FILE.match(path.name)
            if m is None:
                continue
            key, n = m["key"], int(m["n"])
            self.highest[key] = max(self.highest[key], n)
            if m["meta"]:
                self.statuses[key].append((n, json.loads(path.read_text())["status"]))

    def status_list(self, key: str) -> list[str]:
        return [status for _, status in sorted(self.statuses[key])]

    def done(self, key: str) -> bool:
        return is_done(self.status_list(key))

    def next_attempt(self, key: str) -> int:
        return self.highest[key] + 1

    def commit(self, meta: dict[str, Any], stdout: str) -> None:
        """Write the raw stdout, then the meta last. The raw file is created, never overwritten."""
        key, n = meta["key"], meta["attempt"]
        raw = self.dir / f"{key}.a{n}.json"
        with raw.open("x", encoding="utf-8", newline="") as f:
            f.write(stdout)
        self.highest[key] = n
        write_atomic(self.dir / f"{key}.a{n}.meta.json", json.dumps(meta, sort_keys=True) + "\n")
        self.statuses[key].append((n, meta["status"]))

    def metas(self) -> list[dict[str, Any]]:
        return [
            json.loads(p.read_text())
            for p in sorted(self.dir.iterdir())
            if (m := ATTEMPT_FILE.match(p.name)) and m["meta"]
        ]


# --- The run --------------------------------------------------------------------------------------


def binding(jobs_sha256: str, cli_path: str, cli_version: str, cwd: Path) -> dict[str, Any]:
    return {
        "prompt_version": hp.PROMPT_VERSION,
        "prompt_sha256": hp.PROMPT_SHA256,
        "jobs_sha256": jobs_sha256,
        "route": list(hp.ROUTE_FLAGS),
        "model": hp.MODEL,
        "cli_path": cli_path,
        "cli_version": cli_version,
        "cwd": str(cwd),
        "env_keys": list(ENV_NAMES),
        "max_attempts": MAX_ATTEMPTS,
        "call_timeout_s": CALL_TIMEOUT_S,
    }


def check_binding(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Every field but the cwd must equal the first run's."""
    for field in current:
        if field != "cwd" and recorded.get(field) != current[field]:
            raise Refusal(f"run.json {field} differs from the first run's; refusing to resume")


def choose_cwd(recorded: Mapping[str, Any] | None) -> Path:
    """The recorded cwd if it still exists, else a fresh `tempfile.mkdtemp()` dir whose path passes
    `check_cwd` (a failing empty one is removed and another is tried)."""
    if recorded and Path(recorded["cwd"]).is_dir():
        return Path(recorded["cwd"])
    for _ in range(CWD_TRIES):
        path = Path(tempfile.mkdtemp())
        try:
            check_cwd(path)
            return path
        except Refusal:
            path.rmdir()
    raise Refusal(f"no temp dir under {tempfile.gettempdir()} passes the cwd checks")


def percentile_line(values: Sequence[float]) -> str:
    if not values:
        return "wall p50/max: n/a"
    return f"wall p50/max: {statistics.median(values):.1f}/{max(values):.1f} s"


class Run:
    """The state shared by the worker threads of one invocation."""

    def __init__(
        self, store: Store, cli_path: str, cwd: Path, env: Mapping[str, str], max_calls: int | None
    ):
        self.store, self.cli_path, self.cwd, self.env = store, cli_path, cwd, env
        self.max_calls = max_calls
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.calls = 0
        self.capped = False
        self.errors: list[str] = []
        self.streak: list[str] = []  # the `result` tails of the current run of "unparsed" attempts
        self.tripped: list[str] | None = None  # the streak that stopped the run, if one did

    def acquire(self) -> bool:
        """Claim one subprocess call; False once an error was seen or `max_calls` is used."""
        with self.lock:
            if self.stop.is_set():
                return False
            if self.max_calls is not None and self.calls >= self.max_calls:
                self.capped = True
                return False
            self.calls += 1
            return True

    def attempt(self, job: Mapping[str, Any], cli_version: str) -> str:
        """One call of `job`; commits its files and returns the status."""
        key = job["key"]
        n = self.store.next_attempt(key)
        started, t0 = time.time(), time.monotonic()
        try:
            args = [self.cli_path, *hp.argv()[1:]]
            done = claude(args, job["user"], self.cwd, self.env, CALL_TIMEOUT_S)
            rc, stdout, stderr = done.returncode, done.stdout or "", done.stderr or ""
        except subprocess.TimeoutExpired as e:
            rc, stdout, stderr = TIMEOUT_RC, _text(e.stdout), f"timeout after {CALL_TIMEOUT_S} s"
        except OSError as e:
            rc, stdout, stderr = SPAWN_RC, "", f"could not start claude: {type(e).__name__}"
        wall = time.monotonic() - t0
        status, probability = classify(rc, stdout)
        meta = {
            "key": key,
            "attempt": n,
            "status": status,
            "probability": probability,
            "rc": rc,
            "started": started,
            "wall_s": round(wall, 3),
            "stderr_tail": tail(stderr),
            "user_sha256": job["user_sha256"],
            "cli_version": cli_version,
            "models": models_of(stdout),
        }
        self.store.commit(meta, stdout)
        shown = probability if status == OK else status
        print(f"{key} a{n}: {shown} ({wall:.1f} s)", flush=True)
        with self.lock:
            if status == ERROR:
                self.errors.append(f"{key} a{n}: {error_detail(rc, stdout, stderr)}")
                self.stop.set()
            elif status == UNPARSED:
                self.streak.append(result_tail(stdout))
                if len(self.streak) >= UNPARSED_STREAK and self.tripped is None:
                    self.tripped = list(self.streak)
                    self.stop.set()
            else:
                self.streak.clear()
        return status

    def work(self, job: Mapping[str, Any], cli_version: str) -> None:
        while not self.store.done(job["key"]):
            if not self.acquire():
                return
            if self.attempt(job, cli_version) == ERROR:
                return


def _text(value: bytes | str | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def summary(store: Store, jobs: Sequence[Mapping[str, Any]], run: Run) -> str:
    done = [store.status_list(j["key"]) for j in jobs if store.done(j["key"])]
    ok = sum(OK in statuses for statuses in done)
    walls = [m["wall_s"] for m in store.metas()]
    return (
        f"done ok {ok}, done unparsed {len(done) - ok}, remaining {len(jobs) - len(done)}, "
        f"errors this run {len(run.errors)}, calls this run {run.calls}; {percentile_line(walls)}"
    )


def run_jobs(jobs_path: Path, out: Path, workers: int, max_calls: int | None) -> int:
    rows, jobs_sha256 = read_jobs(jobs_path)
    out.mkdir(parents=True, exist_ok=True)
    run_json = out / "run.json"
    recorded = json.loads(run_json.read_text()) if run_json.exists() else None
    cwd = choose_cwd(recorded)
    check_cwd(cwd)
    env = child_env()
    cli_path = resolve_cli(env)
    cli_version, auth = preflight(cli_path, cwd, env)
    current = binding(jobs_sha256, cli_path, cli_version, cwd)
    if recorded is not None:
        check_binding(recorded, current)
    state = {**(recorded or {}), **current, "preflight": auth}
    write_atomic(run_json, json.dumps(state, indent=1, sort_keys=True) + "\n")

    store = Store(out)
    run = Run(store, cli_path, cwd, env, max_calls)
    pending = [job for job in rows if not store.done(job["key"])]
    print(f"{len(rows)} jobs, {len(pending)} pending, {workers} workers; claude {cli_version}")
    print(f"cli_path {cli_path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run.work, job, cli_version) for job in pending]
        failure: Exception | None = None
        for future in futures:
            try:
                future.result()
            except Exception as e:  # a write failed: let in-flight calls finish, then raise
                run.stop.set()
                failure = failure or e
    if failure is not None:
        raise failure
    state["cwd_empty_after"] = not any(cwd.iterdir())
    version_after = cli_version_of(cli_path, cwd, env) if run.calls else cli_version
    if run.calls:
        state["cli_version_after"] = version_after
    write_atomic(run_json, json.dumps(state, indent=1, sort_keys=True) + "\n")
    print(summary(store, rows, run))
    if not state["cwd_empty_after"]:
        print(f"WARNING: {cwd} is not empty after the run")
    failed = False
    if version_after != cli_version:
        print(f"CLI VERSION CHANGED during the run: bound {cli_version!r}, now {version_after!r}")
        failed = True
    if run.errors:
        print(f"STOPPED at the first error ({len(run.errors)} in flight or recorded):")
        for line in run.errors:
            print(f"  {line}")
        failed = True
    if run.tripped is not None:
        print(f"STOPPED after {UNPARSED_STREAK} unparsed attempts in a row; their result tails:")
        for text in run.tripped:
            print(f"  {text!r}")
        failed = True
    if failed:
        return EXIT_ERROR
    if run.capped:
        print("stopped at --max-calls")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.max_calls is not None and args.max_calls < 0):
        parser.error("--workers must be at least 1 and --max-calls at least 0")
    try:
        return run_jobs(args.jobs, args.out, args.workers, args.max_calls)
    except Refusal as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
