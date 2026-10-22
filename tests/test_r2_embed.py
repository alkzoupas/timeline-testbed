"""Known-answer tests for the frozen M4 prompt and the R2 embedding driver. No model is loaded:
the child calls are replaced by fakes that write what the real child writes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from testbed.models import local_llm, mlx_child, prompt, r2_embed
from testbed.world.families import PAIRS

# --------------------------------------------------------------------------- prompt


def test_prompt_hash_is_pinned() -> None:
    assert prompt.prompt_sha256() == prompt.PROMPT_SHA256


def test_prompt_names_no_family_and_holds_no_sentinel() -> None:
    text = (prompt.BEFORE + prompt.AFTER).lower()
    for names in PAIRS.values():
        for family in names:
            assert family.lower() not in text
            assert family.replace("_", " ").lower() not in text
    assert mlx_child.SENTINEL not in text
    assert "RISK_LEVEL: <n>" in prompt.AFTER


def test_nimble_schema_hash_is_pinned_and_asks_the_same_question() -> None:
    assert prompt.nimble_schema_sha256() == prompt.NIMBLE_SCHEMA_SHA256
    field = prompt.NIMBLE_SCHEMA["risk_level"]
    assert field["choices"] == [str(n) for n in range(1, 11)]
    description = str(field["description"])
    assert description.startswith(prompt.BEFORE.split("\n\n")[0])
    assert "RISK_LEVEL" not in description and prompt.QUESTION in prompt.AFTER


def test_child_rows_wrap_each_render_in_order() -> None:
    rows = prompt.child_rows([{"unit": "u2", "text": "R2"}, {"unit": "u1", "text": "R1"}])
    assert rows == [
        {"id": "u2", "before": prompt.BEFORE, "render": "R2", "after": prompt.AFTER},
        {"id": "u1", "before": prompt.BEFORE, "render": "R1", "after": prompt.AFTER},
    ]


# --------------------------------------------------------------------------- pure helpers


def test_chunk_ranges() -> None:
    assert r2_embed.chunk_ranges(7, 3) == [(0, 3), (3, 6), (6, 7)]
    assert r2_embed.chunk_ranges(6, 3) == [(0, 3), (3, 6)]
    assert r2_embed.chunk_ranges(1, 300) == [(0, 1)]
    with pytest.raises(ValueError):
        r2_embed.chunk_ranges(0, 3)


def test_merge_parts_concatenates_in_order_and_checks_keys() -> None:
    a = {"ids": np.array(["x"]), "mean_final": np.ones((1, 2), np.float32)}
    b = {"ids": np.array(["y", "z"]), "mean_final": np.zeros((2, 2), np.float32)}
    merged = r2_embed.merge_parts([a, b])
    assert merged["ids"].tolist() == ["x", "y", "z"]
    assert merged["mean_final"].tolist() == [[1, 1], [0, 0], [0, 0]]
    with pytest.raises(ValueError, match="keys"):
        r2_embed.merge_parts([a, {"ids": np.array(["q"])}])


def test_check_fits_reserves_the_answer() -> None:
    limit = r2_embed.TOKEN_LIMIT - r2_embed.ANSWER_RESERVE
    assert (
        r2_embed.check_fits([{"id": "a", "n_prompt": limit}, {"id": "b", "n_prompt": 5}]) == limit
    )
    with pytest.raises(ValueError, match="exceed"):
        r2_embed.check_fits([{"id": "a", "n_prompt": limit + 1}])


# --------------------------------------------------------------------------- run, with fakes

HIDDEN = 3
NAMES = [x.strip() for x in r2_embed.LAYERS.split(",")]
OLD_CHILD, NEW_CHILD = "a" * 64, "b" * 64


def _vec(text: str) -> float:
    """What the fake child makes of a render: a number that depends on its text, not its id."""
    return float(int(hashlib.sha256(text.encode()).hexdigest()[:6], 16))


def _rows_sha256(texts: Sequence[str], start: int = 0) -> str:
    """The rows text of units u<start>.. written out by hand, hashed independently of the code."""
    lines = [
        json.dumps(
            {"id": f"u{start + i:02d}", "before": prompt.BEFORE, "render": t, "after": prompt.AFTER}
        )
        for i, t in enumerate(texts)
    ]
    return hashlib.sha256("".join(line + "\n" for line in lines).encode()).hexdigest()


def _world(tmp_path: Path, n: int) -> Path:
    world = tmp_path / "w-s0.75-seed9"
    world.mkdir()
    manifest = {"checksum_sha256": "c" * 64, "generator_version": "tt-gen-v1"}
    (world / "manifest.json").write_text(json.dumps(manifest))
    lines = [json.dumps({"unit": f"u{i:02d}", "text": f"render {i}"}) for i in range(n)]
    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")
    return world


class _FakeChild:
    def __init__(self, n_prompt: int = 100) -> None:
        self.n_prompt = n_prompt
        self.child_sha256 = (
            OLD_CHILD  # the "current" mlx_child.py hash, and what a new chunk reports
        )
        self.switch_child_at_call: int | None = None  # edit the child script before this embed call
        self.embed_calls: list[str] = []
        self.fail_at_call: int | None = (
            None  # raise inside this embed call, as a killed child would
        )

    def count(self, model_key: str, rows: Sequence[dict[str, str]], **kw: Any) -> dict[str, Any]:
        assert all(r["before"] == prompt.BEFORE and r["after"] == prompt.AFTER for r in rows)
        return {"rows": [{"id": r["id"], "n_prompt": self.n_prompt} for r in rows]}

    def embed(
        self, model_key: str, rows: Sequence[dict[str, str]], layers: str, work_dir: Path, tag: str
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        assert layers == r2_embed.LAYERS
        for suffix in (".npz", ".json", ".done.json"):  # a child never writes beside stale files
            assert not (work_dir / f"{tag}{suffix}").exists(), f"{tag}{suffix} was not cleared"
        if self.fail_at_call == len(self.embed_calls):
            raise RuntimeError("child killed")
        if self.switch_child_at_call == len(self.embed_calls):
            self.child_sha256 = NEW_CHILD
        self.embed_calls.append(tag)
        ids = [r["id"] for r in rows]
        arrays: dict[str, np.ndarray] = {"ids": np.array(ids)}
        for name in NAMES:
            value = np.array([[_vec(r["render"])] * HIDDEN for r in rows], dtype=np.float32)
            arrays[f"mean_{name}"] = value
            arrays[f"last_{name}"] = -value
        np.savez(work_dir / f"{tag}.npz", **dict[str, Any](arrays))
        result = {
            "model_dir": str(local_llm.model_dir(model_key)),
            "child_sha256": self.child_sha256,
            "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
            "layers": NAMES,
            "hidden_size": HIDDEN,
            "rows": [{"id": i, "n_prompt": self.n_prompt} for i in ids],
            "steady_tokens": self.n_prompt * (len(ids) - 1),
            "steady_seconds": 1.0,
            "peak_memory_gib": 1.5,
        }
        (work_dir / f"{tag}.json").write_text(json.dumps(result))
        return result, arrays


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeChild:
    child = _FakeChild()
    monkeypatch.setattr(local_llm, "count", child.count)
    monkeypatch.setattr(local_llm, "embed", child.embed)
    monkeypatch.setattr(local_llm, "model_revision", lambda key: "rev-" + key)
    monkeypatch.setattr(local_llm, "model_dir", lambda key: Path("/models") / ("snap-" + key))
    monkeypatch.setattr(local_llm, "child_sha256", lambda: child.child_sha256)
    monkeypatch.setattr(r2_embed, "CHUNK", 2)
    return child


def _stale_chunk(tmp_path: Path, fake: _FakeChild, world: Path, texts: Sequence[str]) -> Path:
    """Leave chunk000 of `world` in the parts directory, made by the fake from `texts`."""
    work = tmp_path / "out" / "9b" / f"{world.name}.parts"
    work.mkdir(parents=True)
    rows = prompt.child_rows([{"unit": f"u{i:02d}", "text": t} for i, t in enumerate(texts)])
    fake.embed("9b", rows, r2_embed.LAYERS, work, "chunk000")
    fake.embed_calls.clear()
    _commit(work, "chunk000")
    return work


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(work: Path, tag: str) -> None:
    """The commit marker a finished chunk has: the hashes of its npz and json, taken by hand."""
    marker = {
        "tag": tag,
        "npz_sha256": _sha256_of(work / f"{tag}.npz"),
        "json_sha256": _sha256_of(work / f"{tag}.json"),
    }
    (work / f"{tag}.done.json").write_text(json.dumps(marker))


def _replace_npz(work: Path, tag: str) -> None:
    """Overwrite the chunk's npz with another run's vectors: finite, same ids, keys and shapes."""
    with np.load(work / f"{tag}.npz") as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    for key in arrays:
        if key != "ids":
            arrays[key] = arrays[key] + np.float32(7)
    np.savez(work / f"{tag}.npz", **dict[str, Any](arrays))


def _final_means(tmp_path: Path, world: Path) -> list[float]:
    with np.load(tmp_path / "out" / "9b" / f"{world.name}.npz") as data:
        return [float(x) for x in data["mean_final"][:, 0]]


def test_run_merges_chunks_in_render_order(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 5)
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk000", "chunk001", "chunk002"]
    with np.load(tmp_path / "out" / "9b" / f"{world.name}.npz") as data:
        assert data["ids"].tolist() == [f"u{i:02d}" for i in range(5)]
        assert data["mean_final"][:, 0].tolist() == [_vec(f"render {i}") for i in range(5)]
        assert data["last_16"][:, 0].tolist() == [-_vec(f"render {i}") for i in range(5)]
    assert record["prompt_sha256"] == prompt.PROMPT_SHA256
    assert record["model_revision"] == "rev-9b"
    assert record["world_checksum_sha256"] == "c" * 64
    assert record["n_rows"] == 5 and record["chunks"] == 3 and record["chunks_reused"] == 0
    assert not (tmp_path / "out" / "9b" / f"{world.name}.parts").exists()
    saved = json.loads((tmp_path / "out" / "9b" / f"{world.name}.json").read_text())
    assert saved["n_prompt"]["max"] == 100
    assert saved == record


def test_run_records_the_child_rows_and_npz_hashes(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 5)
    record = r2_embed.run(world, "9b", tmp_path / "out")
    npz = tmp_path / "out" / "9b" / f"{world.name}.npz"
    assert record["child_sha256"] == OLD_CHILD
    assert record["rows_sha256"] == _rows_sha256([f"render {i}" for i in range(5)])
    assert record["npz_sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()


def test_run_reuses_a_chunk_made_from_the_same_rows(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 5)
    _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk001", "chunk002"]
    assert record["chunks_reused"] == 1
    assert _final_means(tmp_path, world) == [_vec(f"render {i}") for i in range(5)]


def test_run_recomputes_a_chunk_whose_npz_was_replaced(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    _replace_npz(work, "chunk000")  # json and marker are untouched and still valid on their own
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk000", "chunk001", "chunk002"]
    assert record["chunks_reused"] == 0
    assert _final_means(tmp_path, world) == [_vec(f"render {i}") for i in range(5)]


def test_run_recomputes_a_valid_json_beside_another_runs_npz_without_a_marker(
    tmp_path: Path, fake: _FakeChild
) -> None:
    """An overwrite killed after the child wrote the npz and before its json (or marker)."""
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    (work / "chunk000.done.json").unlink()
    _replace_npz(work, "chunk000")
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk000", "chunk001", "chunk002"]
    assert record["chunks_reused"] == 0
    assert _final_means(tmp_path, world) == [_vec(f"render {i}") for i in range(5)]


@pytest.mark.parametrize("field", ["npz_sha256", "json_sha256"])
def test_run_recomputes_a_chunk_whose_marker_names_other_hashes(
    tmp_path: Path, fake: _FakeChild, field: str
) -> None:
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    marker = json.loads((work / "chunk000.done.json").read_text())
    marker[field] = "0" * 64
    (work / "chunk000.done.json").write_text(json.dumps(marker))
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls[0] == "chunk000" and record["chunks_reused"] == 0


@pytest.mark.parametrize("damage", ["garbage", "empty", "other_tag", "no_json_hash"])
def test_run_recomputes_a_chunk_whose_marker_is_unreadable(
    tmp_path: Path, fake: _FakeChild, damage: str
) -> None:
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    done = work / "chunk000.done.json"
    marker = json.loads(done.read_text())
    if damage == "garbage":
        done.write_text("{not json")
    elif damage == "empty":
        done.write_text("")
    elif damage == "other_tag":
        done.write_text(json.dumps({**marker, "tag": "chunk001"}))
    else:
        del marker["json_sha256"]
        done.write_text(json.dumps(marker))
    r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls[0] == "chunk000"


def test_run_commits_each_chunk_with_the_hashes_of_its_files(
    tmp_path: Path, fake: _FakeChild
) -> None:
    fake.fail_at_call = 2  # chunks 0 and 1 finish, the third child is killed
    world = _world(tmp_path, 5)
    with pytest.raises(RuntimeError, match="child killed"):
        r2_embed.run(world, "9b", tmp_path / "out")
    work = tmp_path / "out" / "9b" / f"{world.name}.parts"
    for tag in ("chunk000", "chunk001"):
        marker = json.loads((work / f"{tag}.done.json").read_text())
        assert marker == {
            "tag": tag,
            "npz_sha256": _sha256_of(work / f"{tag}.npz"),
            "json_sha256": _sha256_of(work / f"{tag}.json"),
        }
    assert not (work / "chunk002.done.json").exists()
    assert not list(work.glob("*.tmp"))
    fake.fail_at_call = None
    fake.embed_calls.clear()
    record = r2_embed.run(world, "9b", tmp_path / "out")  # the rerun reuses what was committed
    assert fake.embed_calls == ["chunk002"] and record["chunks_reused"] == 2


def test_run_removes_stale_chunk_files_before_the_child_runs(
    tmp_path: Path, fake: _FakeChild
) -> None:
    """The fake child asserts that none of the chunk's files exist when it starts."""
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["", ""])  # not this world's text: not reusable
    (work / "chunk000.done.json.tmp").write_text("half")
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls[0] == "chunk000" and record["chunks_reused"] == 0


def test_run_recomputes_a_chunk_whose_render_text_changed(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 5)
    _stale_chunk(tmp_path, fake, world, ["", ""])  # same ids and shapes, other render text
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk000", "chunk001", "chunk002"]
    assert record["chunks_reused"] == 0
    assert _final_means(tmp_path, world) == [_vec(f"render {i}") for i in range(5)]
    assert record["rows_sha256"] == _rows_sha256([f"render {i}" for i in range(5)])


def test_run_recomputes_a_chunk_made_by_another_child(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 5)
    _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    fake.child_sha256 = NEW_CHILD  # mlx_child.py was edited since
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk000", "chunk001", "chunk002"]
    assert record["chunks_reused"] == 0 and record["child_sha256"] == NEW_CHILD


@pytest.mark.parametrize(
    "stale", [{"layers": ["8", "final"]}, {"model_dir": "/models/other-snapshot"}]
)
def test_run_recomputes_a_chunk_made_for_other_layers_or_another_snapshot(
    tmp_path: Path, fake: _FakeChild, stale: dict[str, Any]
) -> None:
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    saved = json.loads((work / "chunk000.json").read_text())
    (work / "chunk000.json").write_text(json.dumps({**saved, **stale}))
    record = r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls == ["chunk000", "chunk001", "chunk002"]
    assert record["chunks_reused"] == 0


@pytest.mark.parametrize("field", ["child_sha256", "rows_sha256", "model_dir", "layers"])
def test_run_recomputes_a_chunk_whose_result_lacks_a_binding_field(
    tmp_path: Path, fake: _FakeChild, field: str
) -> None:
    world = _world(tmp_path, 5)
    work = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
    saved = json.loads((work / "chunk000.json").read_text())
    del saved[field]
    (work / "chunk000.json").write_text(json.dumps(saved))
    r2_embed.run(world, "9b", tmp_path / "out")
    assert fake.embed_calls[0] == "chunk000"


def test_run_refuses_chunks_made_by_different_children(tmp_path: Path, fake: _FakeChild) -> None:
    fake.switch_child_at_call = 1  # mlx_child.py is edited while the run is going
    with pytest.raises(RuntimeError, match="different child script"):
        r2_embed.run(_world(tmp_path, 5), "9b", tmp_path / "out")
    assert not list((tmp_path / "out" / "9b").glob("*.json"))  # nothing was merged


def test_run_refuses_an_existing_output(tmp_path: Path, fake: _FakeChild) -> None:
    world = _world(tmp_path, 3)
    r2_embed.run(world, "9b", tmp_path / "out")
    with pytest.raises(FileExistsError):
        r2_embed.run(world, "9b", tmp_path / "out")


def test_run_refuses_prompts_that_do_not_fit(tmp_path: Path, fake: _FakeChild) -> None:
    fake.n_prompt = r2_embed.TOKEN_LIMIT
    with pytest.raises(ValueError, match="exceed"):
        r2_embed.run(_world(tmp_path, 3), "9b", tmp_path / "out")
    assert fake.embed_calls == []
