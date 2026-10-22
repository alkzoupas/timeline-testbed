"""Known-answer tests for the Nimble R1 runner (testbed/models/nimble_run.py) and the pure parts
of its child (testbed/models/nimble_child.py), on synthetic renders and a fake child. No model is
loaded and the Nimble venv is never started."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from testbed.models import nimble_child, nimble_run, prompt

PACKAGES = {"mlx": "0.1", "mlx-lm": "0.2", "transformers": "5.0"}


def good_provenance(**override: Any) -> dict[str, Any]:
    prov: dict[str, Any] = {
        "nimble_commit": nimble_child.NIMBLE_COMMIT,
        "adapter_repo": nimble_child.ADAPTER_REPO,
        "adapter_revision": nimble_child.ADAPTER_REVISION,
        "adapter_sha256": nimble_child.ADAPTER_SHA256,
        "contract_sha256": "c" * 64,
        "max_input_tokens": nimble_child.MAX_INPUT_TOKENS,
        "temperature": nimble_child.TEMPERATURE,
        "mode": nimble_child.MODE,
        "packages": dict(PACKAGES),
        "nimble_schema_sha256": prompt.NIMBLE_SCHEMA_SHA256,
        "child_sha256": nimble_run.child_sha256(),
    }
    return prov | override


def fake_logits(unit: str) -> list[float]:
    rng = np.random.default_rng(zlib.crc32(unit.encode()))
    return [float(x) for x in rng.normal(scale=2.0, size=10)]


# --- The child's pure helpers ---------------------------------------------------------------------


def test_summarize_known_answer_and_refusals() -> None:
    uniform = nimble_child.summarize([0.0] * 10, nimble_child.LEVELS)
    assert uniform["p"] == pytest.approx([0.1] * 10)
    assert uniform["expected"] == pytest.approx(5.5) and uniform["argmax"] == 1  # first on a tie
    # logits ln(1), ln(3) on levels 2 and 10: p = 1/4 and 3/4 over those two (the rest -inf-ish)
    logits = [-50.0] * 10
    logits[1], logits[9] = math.log(1.0), math.log(3.0)
    two = nimble_child.summarize(logits, nimble_child.LEVELS)
    assert two["expected"] == pytest.approx(2 * 0.25 + 10 * 0.75)
    assert two["argmax"] == 10 and sum(two["p"]) == pytest.approx(1.0)
    ints = nimble_child.summarize([0.0] * 10, list(range(1, 11)))  # int choices are accepted
    assert ints["expected"] == pytest.approx(5.5)
    with pytest.raises(ValueError, match="choices must be the levels"):
        nimble_child.summarize([0.0] * 10, list(reversed(nimble_child.LEVELS)))
    with pytest.raises(ValueError, match="finite logits"):
        nimble_child.summarize([0.0] * 9 + [math.nan], nimble_child.LEVELS)
    with pytest.raises(ValueError, match="finite logits"):
        nimble_child.summarize([0.0] * 9, nimble_child.LEVELS)


def test_softmax_is_shift_invariant_and_does_not_overflow() -> None:
    a = nimble_child.softmax([1.0, 2.0, 3.0])
    b = nimble_child.softmax([1001.0, 1002.0, 1003.0])
    assert a == pytest.approx(b) and sum(a) == pytest.approx(1.0)
    assert nimble_child.softmax([0.0, math.log(3.0)])[1] == pytest.approx(0.75)


def test_the_child_refuses_a_schema_that_is_not_the_frozen_one() -> None:
    text = nimble_run.schema_text()
    assert nimble_child.check_schema_text(text, prompt.NIMBLE_SCHEMA_SHA256) == json.loads(text)
    with pytest.raises(SystemExit, match="differs from --schema-sha256"):
        nimble_child.check_schema_text(text + " ", prompt.NIMBLE_SCHEMA_SHA256)
    other = copy.deepcopy(prompt.NIMBLE_SCHEMA)
    other["risk_level"]["choices"] = ["1", "2"]
    other_text = json.dumps(other, sort_keys=True)
    with pytest.raises(SystemExit, match="one risk_level field"):
        nimble_child.check_schema_text(other_text, hashlib.sha256(other_text.encode()).hexdigest())


def test_the_child_refuses_a_merged_model_from_another_adapter() -> None:
    ready = {
        "model": nimble_child.ADAPTER_REPO,
        "revision": nimble_child.ADAPTER_REVISION,
        "adapter_sha256": nimble_child.ADAPTER_SHA256,
    }
    nimble_child.check_ready(ready)
    with pytest.raises(SystemExit, match="another adapter release"):
        nimble_child.check_ready(ready | {"revision": "0" * 40})
    with pytest.raises(SystemExit, match="another adapter$"):
        nimble_child.check_ready(ready | {"adapter_sha256": "0" * 64})
    nimble_child.check_contract_base(
        {"model": nimble_child.BASE_REPO, "revision": nimble_child.BASE_REVISION}
    )
    with pytest.raises(SystemExit, match="another base"):
        nimble_child.check_contract_base({"model": nimble_child.BASE_REPO, "revision": "x"})


def test_the_child_reads_only_clean_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    def write(rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    write([{"id": "a", "context": "x"}, {"id": "b", "context": "y"}])
    rows, rows_sha256 = nimble_child.read_rows(path)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows_sha256 == nimble_run.sha256_text(nimble_run.rows_text(rows))  # the bytes as read
    write([{"id": "a", "context": "x", "label": 1}])
    with pytest.raises(SystemExit, match="id/context"):
        nimble_child.read_rows(path)
    write([{"id": "a", "context": "  "}])
    with pytest.raises(SystemExit, match="nonempty"):
        nimble_child.read_rows(path)
    write([{"id": "a", "context": "x"}, {"id": "a", "context": "y"}])
    with pytest.raises(SystemExit, match="duplicate"):
        nimble_child.read_rows(path)


def test_the_child_hashes_its_own_source() -> None:
    source = Path(nimble_child.__file__).read_bytes()
    assert nimble_child.CHILD_SHA256 == hashlib.sha256(source).hexdigest()
    assert nimble_run.child_sha256() == nimble_child.CHILD_SHA256


def test_the_child_parser_takes_the_two_commands() -> None:
    args = nimble_child.build_parser().parse_args(
        ["score", "--in", "r", "--schema", "s", "--schema-sha256", "h", "--out", "o"]
    )
    assert (args.command, args.fn) == ("score", nimble_child.cmd_score)
    assert (
        nimble_child.build_parser()
        .parse_args(["count", "--in", "r", "--schema", "s", "--schema-sha256", "h", "--out", "o"])
        .fn
        is nimble_child.cmd_count
    )


# --- Token stats and the checks -------------------------------------------------------------------


def test_token_stats_known_answer() -> None:
    n = [1000] * 8 + [2048, 3000]  # p90 interpolates between the 9th and 10th values
    stats = nimble_run.token_stats(n)
    assert stats["max"] == 3000 and stats["mean"] == pytest.approx(1304.8)
    assert stats["p90"] == pytest.approx(2048 + 0.1 * (3000 - 2048))
    assert stats["n_over_2048"] == 1  # 2,048 itself fits the cap
    assert stats["n_over_max_input"] == 0
    assert nimble_run.token_stats([9000, 2049, 10])["n_over_max_input"] == 1
    assert nimble_run.token_stats([9000, 2049, 10])["n_over_2048"] == 2
    with pytest.raises(ValueError, match="no prompts"):
        nimble_run.token_stats([])


def count_result(rows: Sequence[dict[str, str]], n_prompt: int = 900) -> dict[str, Any]:
    return {
        "command": "count",
        "provenance": good_provenance(),
        "rows_sha256": nimble_run.sha256_text(nimble_run.rows_text(rows)),
        "rows": [{"id": r["id"], "n_prompt": n_prompt} for r in rows],
    }


def score_result(rows: Sequence[dict[str, str]], n_prompt: int = 900) -> dict[str, Any]:
    out = []
    for r in rows:
        s = nimble_child.summarize(fake_logits(r["id"]), nimble_child.LEVELS)
        out.append({"id": r["id"], "n_prompt": n_prompt, **s, "seconds": 4.0})
    return {
        "command": "score",
        "provenance": good_provenance(),
        "rows_sha256": nimble_run.sha256_text(nimble_run.rows_text(rows)),
        "rows": out,
        "peak_memory_gib": 21.5,
        "steady_tokens": n_prompt * (len(rows) - 1),
        "steady_seconds": 4.0 * (len(rows) - 1),
    }


ROWS = [{"id": f"u{i}", "context": f"render {i}"} for i in range(3)]


@pytest.mark.parametrize(
    "key",
    [
        "nimble_commit",
        "adapter_repo",
        "adapter_revision",
        "adapter_sha256",
        "nimble_schema_sha256",
        "max_input_tokens",
        "temperature",
        "mode",
        "child_sha256",
    ],
)
def test_check_provenance_refuses_each_pin(key: str) -> None:
    nimble_run.check_provenance(good_provenance())
    with pytest.raises(RuntimeError, match=f"{key} is "):
        nimble_run.check_provenance(good_provenance(**{key: "other"}))
    broken = good_provenance()
    del broken[key]
    with pytest.raises(RuntimeError, match=f"{key} is None"):
        nimble_run.check_provenance(broken)
    with pytest.raises(RuntimeError, match="no contract_sha256"):
        nimble_run.check_provenance(good_provenance(contract_sha256=""))


def test_check_score_accepts_a_good_result_and_refuses_each_defect() -> None:
    nimble_run.check_score(score_result(ROWS), ROWS)

    def broken(edit: Any) -> dict[str, Any]:
        result = score_result(ROWS)
        edit(result)
        return result

    cases = {
        "wrong command": (lambda r: r.update(command="count"), "result is of 'count'"),
        "stale pin": (lambda r: r["provenance"].update(adapter_sha256="x"), "adapter_sha256"),
        "other rows": (lambda r: r.update(rows_sha256="0" * 64), "other rows"),
        "swapped ids": (lambda r: r["rows"].reverse(), "row ids differ"),
        "short p": (lambda r: r["rows"][0]["p"].pop(), "does not have 10 finite"),
        "nan": (lambda r: r["rows"][1]["p"].__setitem__(0, float("nan")), "10 finite"),
        "null": (lambda r: r["rows"][1]["p"].__setitem__(0, None), "10 finite"),
        "p not normalized": (lambda r: r["rows"][0]["p"].__setitem__(0, 0.9), "sum to"),
        "wrong expected": (lambda r: r["rows"][2].update(expected=5.0), "expected"),
        "wrong argmax": (lambda r: r["rows"][0].update(argmax=11), "argmax"),
        "no tokens": (lambda r: r["rows"][0].update(n_prompt=0), "n_prompt"),
    }
    for edit, message in cases.values():
        with pytest.raises(RuntimeError, match=message):
            nimble_run.check_score(broken(edit), ROWS)


def test_check_fits_is_closed_at_the_child_limit() -> None:
    limit = nimble_child.MAX_INPUT_TOKENS
    nimble_run.check_fits([{"id": "a", "n_prompt": limit}])
    with pytest.raises(ValueError, match=f"1 prompts exceed {limit}"):
        nimble_run.check_fits([{"id": "a", "n_prompt": limit}, {"id": "b", "n_prompt": limit + 1}])


# --- run_child, with a stand-in for the Nimble venv's python --------------------------------------

FAKE_CHILD = """\
import hashlib, json, os, sys
argv = sys.argv[1:]
opts = dict(zip(argv[1::2], argv[2::2], strict=False))
log = {"argv": argv, "env": sorted(os.environ), "isolated": sys.flags.isolated}
log["schema_sha256"] = hashlib.sha256(open(opts["--schema"], "rb").read()).hexdigest()
log["rows"] = open(opts["--in"]).read().splitlines()
open(opts["--out"], "w").write(json.dumps(log))
"""


@pytest.fixture
def fake_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "fake_child.py"
    script.write_text(FAKE_CHILD)
    monkeypatch.setattr(nimble_run, "VENV_PYTHON", Path(sys.executable))
    monkeypatch.setattr(nimble_run, "CHILD_SCRIPT", script)
    return script


def test_run_child_passes_the_pinned_schema_and_an_offline_env(
    fake_python: Path, tmp_path: Path
) -> None:
    result = nimble_run.run_child("count", ROWS, work_dir=tmp_path / "w", tag="t0")
    assert result["isolated"] == 1  # python -I
    assert result["argv"][0] == "count"
    assert result["schema_sha256"] == prompt.NIMBLE_SCHEMA_SHA256
    assert "HF_HUB_OFFLINE" in result["env"]
    assert [json.loads(x)["id"] for x in result["rows"]] == ["u0", "u1", "u2"]
    assert all(set(json.loads(x)) == {"id", "context"} for x in result["rows"])
    assert (tmp_path / "w" / "t0.rows.jsonl").exists()  # kept for the record
    scratch = nimble_run.run_child("score", ROWS)  # no work_dir: a temp dir, removed afterwards
    assert scratch["argv"][0] == "score"


def test_run_child_reports_a_failed_child(
    fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nimble_run.run_child("count", ROWS, work_dir=tmp_path / "w")  # the stand-in itself works
    monkeypatch.setattr(nimble_run, "CHILD_SCRIPT", tmp_path / "missing.py")
    with pytest.raises(RuntimeError, match=r"nimble_child count failed \(exit [1-9]"):
        nimble_run.run_child("count", ROWS, work_dir=tmp_path / "w2")


# --- run ------------------------------------------------------------------------------------------


class FakeNimble:
    """Stands in for `nimble_run.count` and `nimble_run.score`; writes the chunk json the way
    `run_child` does, and can fail on a chosen score call."""

    def __init__(self, fail_on_call: int | None = None, n_prompt: int = 900) -> None:
        self.calls: list[list[str]] = []
        self.counts = 0
        self.fail_on_call = fail_on_call
        self.n_prompt = n_prompt

    def count(self, rows: Sequence[Any], **kw: Any) -> dict[str, Any]:
        self.counts += 1
        return count_result(list(rows), self.n_prompt)

    def score(self, rows: Sequence[Any], **kw: Any) -> dict[str, Any]:
        self.calls.append([r["id"] for r in rows])
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("child died")
        result = score_result(list(rows), self.n_prompt)
        nimble_run.check_score(result, rows)
        (Path(kw["work_dir"]) / f"{kw['tag']}.json").write_text(json.dumps(result))
        return result


def install(monkeypatch: pytest.MonkeyPatch, child: FakeNimble) -> None:
    monkeypatch.setattr(nimble_run, "count", child.count)
    monkeypatch.setattr(nimble_run, "score", child.score)


@pytest.fixture
def small_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    world = tmp_path / "w1"
    world.mkdir()
    manifest = {"checksum_sha256": "sum-1", "s": 0.75, "generator_version": "g1"}
    (world / "manifest.json").write_text(json.dumps(manifest))
    renders = [{"unit": f"u{i}", "text": f"render {i}"} for i in range(5)]
    (world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
    monkeypatch.setattr(nimble_run, "CHUNK", 2)
    return world


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_run_writes_one_row_per_unit_in_render_order_with_provenance(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeNimble()
    install(monkeypatch, child)
    out = tmp_path / "r1"
    record = nimble_run.run(small_world, out)
    assert child.counts == 1  # one count over every row, before any forward pass
    assert child.calls == [["u0", "u1"], ["u2", "u3"], ["u4"]]
    rows = read_rows(out / "nimble" / "w1.jsonl")
    assert [r["id"] for r in rows] == [f"u{i}" for i in range(5)]
    assert all(set(r) == set(nimble_run.ROW_FIELDS) for r in rows)
    for r in rows:  # the score is the expectation of the softmax over the ten candidates
        assert len(r["p"]) == 10 and sum(r["p"]) == pytest.approx(1.0)
        assert r["expected"] == pytest.approx(sum(n * p for n, p in enumerate(r["p"], 1)))
        assert r["argmax"] == 1 + int(np.argmax(r["p"])) and r["n_prompt"] == 900
        want = nimble_child.summarize(fake_logits(r["id"]), nimble_child.LEVELS)
        assert r["p"] == pytest.approx(want["p"])
    saved = json.loads((out / "nimble" / "w1.json").read_text())
    assert saved == record
    assert saved["nimble_schema_sha256"] == prompt.NIMBLE_SCHEMA_SHA256
    assert saved["adapter_revision"] == nimble_child.ADAPTER_REVISION
    assert saved["adapter_sha256"] == nimble_child.ADAPTER_SHA256
    assert saved["nimble_commit"] == nimble_child.NIMBLE_COMMIT
    assert saved["world_checksum_sha256"] == "sum-1" and saved["model"] == "nimble"
    assert saved["limit"] is None and saved["n_rows"] == 5 and saved["count_only"] is False
    assert saved["chunks"] == 3 and saved["chunks_reused"] == 0
    assert saved["n_prompt"] == {
        "max": 900,
        "mean": 900.0,
        "p90": 900.0,
        "n_over_2048": 0,
        "n_over_max_input": 0,
    }
    assert saved["steady_tokens_per_s"] == pytest.approx(225.0)  # 900 tokens per 4 s
    assert saved["steady_seconds_per_unit"] == pytest.approx(4.0)
    assert saved["mean_seconds_per_unit"] == pytest.approx(4.0)
    assert saved["peak_memory_gib"] == 21.5
    assert saved["n_distinct_expected"] == len({r["expected"] for r in rows})
    assert not (out / "nimble" / "w1.parts").exists()
    with pytest.raises(FileExistsError):
        nimble_run.run(small_world, out)


def test_run_resumes_from_the_chunks_already_written(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "r1"
    install(monkeypatch, FakeNimble(fail_on_call=3))
    with pytest.raises(RuntimeError, match="child died"):
        nimble_run.run(small_world, out)
    parts = out / "nimble" / "w1.parts"
    assert sorted(p.name for p in parts.glob("chunk*.json")) == ["chunk000.json", "chunk001.json"]
    assert not (out / "nimble" / "w1.jsonl").exists()
    (parts / "chunk001.json").write_text("{not json")  # a damaged chunk is redone
    resumed = FakeNimble()
    install(monkeypatch, resumed)
    record = nimble_run.run(small_world, out)
    assert resumed.calls == [["u2", "u3"], ["u4"]]
    assert record["chunks_reused"] == 1 and record["n_rows"] == 5


@pytest.mark.parametrize(
    "stale",
    [
        lambda r: r["provenance"].update(adapter_revision="old"),  # another adapter release
        lambda r: r["provenance"].update(nimble_schema_sha256="0" * 64),  # another schema
        lambda r: r.update(rows_sha256="0" * 64),  # another world's renders under the same ids
        lambda r: r["provenance"].update(child_sha256="0" * 64),  # nimble_child.py edited since
        lambda r: r["provenance"].pop("child_sha256"),  # a chunk that predates the binding
        lambda r: r["rows"][0]["p"].__setitem__(0, 0.5),  # an edited score
    ],
)
def test_run_redoes_a_chunk_that_fails_its_checks(
    small_world: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale: Any,
) -> None:
    out = tmp_path / "r1"
    install(monkeypatch, FakeNimble(fail_on_call=2))
    with pytest.raises(RuntimeError, match="child died"):
        nimble_run.run(small_world, out)
    chunk = out / "nimble" / "w1.parts" / "chunk000.json"
    result = json.loads(chunk.read_text())
    stale(result)
    chunk.write_text(json.dumps(result))
    resumed = FakeNimble()
    install(monkeypatch, resumed)
    record = nimble_run.run(small_world, out)
    assert resumed.calls[0] == ["u0", "u1"] and record["chunks_reused"] == 0


def test_run_redoes_a_chunk_made_from_other_render_text_under_the_same_ids(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "r1"
    install(monkeypatch, FakeNimble(fail_on_call=2))
    with pytest.raises(RuntimeError, match="child died"):
        nimble_run.run(small_world, out)
    renders = [{"unit": f"u{i}", "text": f"edited {i}"} for i in range(5)]
    (small_world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
    resumed = FakeNimble()
    install(monkeypatch, resumed)
    record = nimble_run.run(small_world, out)
    assert resumed.calls[0] == ["u0", "u1"] and record["chunks_reused"] == 0
    want = [{"id": f"u{i}", "context": f"edited {i}"} for i in range(5)]
    assert record["rows_sha256"] == nimble_run.sha256_text(nimble_run.rows_text(want))


def test_run_records_the_child_and_rows_hashes(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeNimble())
    record = nimble_run.run(small_world, tmp_path / "r1")
    independent = "".join(
        json.dumps({"id": f"u{i}", "context": f"render {i}"}) + "\n" for i in range(5)
    )
    assert record["rows_sha256"] == hashlib.sha256(independent.encode()).hexdigest()
    assert (
        record["child_sha256"] == hashlib.sha256(nimble_run.CHILD_SCRIPT.read_bytes()).hexdigest()
    )
    assert json.loads((tmp_path / "r1" / "nimble" / "w1.json").read_text()) == record


def test_run_refuses_chunks_scored_by_different_child_scripts(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "nimble_child.py"
    script.write_text("# v1\n")
    monkeypatch.setattr(nimble_run, "CHILD_SCRIPT", script)

    class Editing(FakeNimble):
        def score(self, rows: Sequence[Any], **kw: Any) -> dict[str, Any]:
            if len(self.calls) == 1:
                script.write_text("# v2\n")  # edited between chunks 0 and 1
            return super().score(rows, **kw)

    install(monkeypatch, Editing())
    with pytest.raises(RuntimeError, match="different child scripts"):
        nimble_run.run(small_world, tmp_path / "r1")
    assert not (tmp_path / "r1" / "nimble" / "w1.json").exists()


def test_run_refuses_chunks_scored_under_different_packages(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "r1"
    install(monkeypatch, FakeNimble(fail_on_call=2))
    with pytest.raises(RuntimeError, match="child died"):
        nimble_run.run(small_world, out)
    chunk = out / "nimble" / "w1.parts" / "chunk000.json"
    result = json.loads(chunk.read_text())
    result["provenance"]["packages"]["mlx"] = "0.9"
    chunk.write_text(json.dumps(result))
    install(monkeypatch, FakeNimble())
    with pytest.raises(RuntimeError, match="different packages or settings"):
        nimble_run.run(small_world, out)


def test_run_limit_scores_only_the_first_rows(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeNimble())
    record = nimble_run.run(small_world, tmp_path / "r1", limit=3)
    assert record["n_rows"] == 3 and record["limit"] == 3
    assert [r["id"] for r in read_rows(tmp_path / "r1" / "nimble" / "w1.jsonl")] == [
        "u0",
        "u1",
        "u2",
    ]
    with pytest.raises(ValueError, match="at least 1"):
        nimble_run.run(small_world, tmp_path / "r2", limit=0)


def test_run_fails_closed_on_a_prompt_over_the_child_limit_before_any_forward_pass(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = nimble_child.MAX_INPUT_TOKENS
    child = FakeNimble(n_prompt=limit + 1)
    install(monkeypatch, child)
    with pytest.raises(ValueError, match=f"exceed {limit}"):
        nimble_run.run(small_world, tmp_path / "r1")
    assert child.calls == []
    ok = FakeNimble(n_prompt=limit)
    install(monkeypatch, ok)
    assert nimble_run.run(small_world, tmp_path / "r2")["n_prompt"]["max"] == limit


def test_run_does_not_truncate_to_the_spec_cap_but_reports_the_rows_over_it(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeNimble(n_prompt=2049)
    install(monkeypatch, child)
    record = nimble_run.run(small_world, tmp_path / "r1")
    assert record["n_prompt"]["n_over_2048"] == 5 and len(child.calls) == 3


def test_count_only_writes_only_the_token_counts(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeNimble(n_prompt=2049)
    install(monkeypatch, child)
    out = tmp_path / "r1"
    record = nimble_run.run(small_world, out, count_only=True)
    assert child.calls == [] and child.counts == 1  # no forward pass
    assert sorted(p.name for p in (out / "nimble").iterdir()) == ["w1.counts.json"]
    saved = json.loads((out / "nimble" / "w1.counts.json").read_text())
    assert saved == record and saved["count_only"] is True
    assert saved["rows"] == [{"id": f"u{i}", "n_prompt": 2049} for i in range(5)]
    assert saved["n_prompt"]["n_over_2048"] == 5 and saved["n_prompt"]["max"] == 2049
    assert saved["nimble_schema_sha256"] == prompt.NIMBLE_SCHEMA_SHA256
    with pytest.raises(FileExistsError):
        nimble_run.run(small_world, out, count_only=True)
    # counting is not refused for a prompt over the child limit: that is the point of the count
    big = FakeNimble(n_prompt=nimble_child.MAX_INPUT_TOKENS + 1)
    install(monkeypatch, big)
    record = nimble_run.run(small_world, tmp_path / "r2", count_only=True)
    assert record["n_prompt"]["n_over_max_input"] == 5


def test_run_refuses_duplicate_unit_ids(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renders = [{"unit": "u0", "text": "a"}, {"unit": "u0", "text": "b"}]
    (small_world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
    child = FakeNimble()
    install(monkeypatch, child)
    with pytest.raises(ValueError, match="duplicate unit ids"):
        nimble_run.run(small_world, tmp_path / "r1")
    assert child.counts == 0


def test_run_refuses_a_schema_that_differs_from_its_pin(
    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeNimble()
    install(monkeypatch, child)
    monkeypatch.setitem(prompt.NIMBLE_SCHEMA["risk_level"], "description", "edited")
    with pytest.raises(RuntimeError, match="differs from NIMBLE_SCHEMA_SHA256"):
        nimble_run.run(small_world, tmp_path / "r1")
    assert child.counts == 0


def test_the_cli_prints_the_prompt_length_summary(
    small_world: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install(monkeypatch, FakeNimble(n_prompt=3000))
    code = nimble_run.main(
        ["--world", str(small_world), "--out", str(tmp_path / "o"), "--count-only"]
    )
    assert code == 0
    assert "max 3000, mean 3000, p90 3000, over 2048: 5" in capsys.readouterr().out
