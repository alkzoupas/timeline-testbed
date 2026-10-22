"""Known-answer tests for the local-LLM wrapper and the MLX child's pure helpers.

The real model is never loaded here. The integration test runs the child's `selftest-tiny` in the
`es-p6-train` venv (a tiny random-weight model, no files) and skips when that venv is missing.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from testbed.models import local_llm, mlx_child

# --------------------------------------------------------------------------- prompt pieces


def _encode(s: str) -> list[int]:
    return [ord(c) for c in s]


def test_assemble_prompt_span_and_independence() -> None:
    templated = "<u>HEAD" + mlx_child.SENTINEL + "TAIL</u>"
    p = mlx_child.assemble_prompt(_encode, templated, "render")
    assert p.ids == _encode("<u>HEAD") + _encode("render") + _encode("TAIL</u>")
    assert p.span == (len("<u>HEAD"), len("<u>HEAD") + len("render"))
    assert p.ids[p.span[0] : p.span[1]] == _encode("render")
    # The render's ids do not depend on the prompt around it.
    other = mlx_child.assemble_prompt(_encode, "x" + mlx_child.SENTINEL + "yy", "render")
    assert other.ids[other.span[0] : other.span[1]] == _encode("render")


@pytest.mark.parametrize("text", ["no sentinel", mlx_child.SENTINEL * 2])
def test_split_template_needs_exactly_one_sentinel(text: str) -> None:
    with pytest.raises(ValueError, match="sentinel"):
        mlx_child.split_template(text)


class _FakeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> str:
        assert kw == {"add_generation_prompt": True, "enable_thinking": False, "tokenize": False}
        assert [m["role"] for m in messages] == ["user"]
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return _encode(text)


def test_row_prompt_builds_one_user_message() -> None:
    row = {"id": "u", "before": "B ", "render": "RR", "after": " A"}
    p = mlx_child.row_prompt(_FakeTokenizer(), row)
    head = "<|im_start|>user\nB "
    assert p.span == (len(head), len(head) + 2)
    assert p.ids[: p.span[0]] == _encode(head)
    assert p.ids[-len(" A<|im_end|>\n") :] == _encode(" A<|im_end|>\n")


# --------------------------------------------------------------------------- numpy helpers


def test_mean_pool_over_span_on_known_array() -> None:
    states = np.arange(12, dtype=np.float32).reshape(4, 3)  # rows 0..3
    # Rows 1 and 2: mean of [3,4,5] and [6,7,8].
    np.testing.assert_array_equal(mlx_child.mean_pool(states, (1, 3)), [4.5, 5.5, 6.5])
    assert mlx_child.mean_pool(states, (1, 3)).dtype == np.float32
    np.testing.assert_array_equal(mlx_child.mean_pool(states, (3, 4)), states[3])


@pytest.mark.parametrize("span", [(2, 2), (3, 2), (-1, 2), (0, 5)])
def test_mean_pool_rejects_bad_spans(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="span"):
        mlx_child.mean_pool(np.zeros((4, 3), dtype=np.float32), span)


def test_pad_batch_right_pads_and_reports_lengths() -> None:
    ids, lengths = mlx_child.pad_batch([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert lengths.tolist() == [3, 1]


def test_parse_layers() -> None:
    assert mlx_child.parse_layers("8,16, 24,final", 32) == [8, 16, 24, "final"]
    with pytest.raises(ValueError, match="outside"):
        mlx_child.parse_layers("32", 32)


def test_hand_masked_ce_uniform_logits_is_log_vocab() -> None:
    logits = np.zeros((1, 6, 5))
    ids = np.array([[1, 2, 3, 4, 0, 1]])
    ce = mlx_child.hand_masked_ce(logits, ids, np.array([3]), n_ans=2)
    np.testing.assert_allclose(ce, [np.log(5)])


def test_hand_masked_ce_picks_the_answer_positions_only() -> None:
    logits = np.full((1, 4, 3), -100.0)
    logits[0, 1, 2] = 100.0  # position 1 predicts token 2 (answer token at index 2)
    logits[0, 2, 0] = 100.0  # position 2 predicts token 0 (answer token at index 3)
    ids = np.array([[1, 1, 2, 0]])
    ce = mlx_child.hand_masked_ce(logits, ids, np.array([2]), n_ans=2)
    np.testing.assert_allclose(ce, [0.0], atol=1e-9)


# --------------------------------------------------------------------------- snapshots


def _snapshot(hub: Path, cache: str, name: str) -> Path:
    path = hub / cache / "snapshots" / name
    path.mkdir(parents=True)
    return path


def test_snapshot_dir_resolution(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="found 0"):
        local_llm.snapshot_dir("models--x", "abc", hub=tmp_path)  # no snapshots at all
    only = _snapshot(tmp_path, "models--x", "abcdef")
    assert local_llm.snapshot_dir("models--x", "abc", hub=tmp_path) == only
    _snapshot(tmp_path, "models--x", "zzz")  # a wrong revision next to it is ignored
    assert local_llm.snapshot_dir("models--x", "abc", hub=tmp_path) == only
    with pytest.raises(FileNotFoundError, match="found 0"):
        local_llm.snapshot_dir("models--x", "abd", hub=tmp_path)
    _snapshot(tmp_path, "models--x", "abc999")
    with pytest.raises(FileNotFoundError, match="found 2"):
        local_llm.snapshot_dir("models--x", "abc", hub=tmp_path)


def test_model_dir_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        local_llm.model_dir("70b")


# --------------------------------------------------------------------------- run_child


def _fake_python(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-python"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_run_child_raises_with_stderr_tail_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_llm, "VENV_PYTHON", _fake_python(tmp_path, "echo boom >&2\nexit 3\n"))
    with pytest.raises(RuntimeError, match=r"exit 3\): boom"):
        local_llm.run_child("count", rows=[], work_dir=tmp_path / "w")


def test_run_child_parses_json_and_sets_only_the_offline_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Finds the argument after --out and writes a JSON file there; also records argv[1] and the env.
    body = (
        'while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done\n'
        'printf \'{"hf": "%s", "home": "%s"}\' "$HF_HUB_OFFLINE" "$HOME" > "$out"\n'
    )
    monkeypatch.setattr(local_llm, "VENV_PYTHON", _fake_python(tmp_path, body))
    run = local_llm.run_child(
        "score",
        rows=[{"id": "a", "before": "", "render": "r", "after": ""}],
        work_dir=tmp_path / "w",
    )
    assert run.result == {"hf": "1", "home": ""}  # HOME is not passed through
    assert run.npz is None
    assert json.loads((tmp_path / "w" / "score.rows.jsonl").read_text())["id"] == "a"


def test_run_child_embed_needs_a_work_dir() -> None:
    with pytest.raises(ValueError, match="work_dir"):
        local_llm.run_child("embed")


# --------------------------------------------------------------------------- embed checks


def _write_npz(path: Path, ids: list[str], **arrays: np.ndarray) -> None:
    payload: dict[str, Any] = {"ids": np.array(ids), **arrays}
    np.savez(path, **payload)


def test_check_embeddings_accepts_a_good_file(tmp_path: Path) -> None:
    f = tmp_path / "e.npz"
    a = np.ones((2, 3), dtype=np.float32)
    _write_npz(f, ["u1", "u2"], mean_8=a, last_8=a, mean_final=a, last_final=a)
    got = local_llm.check_embeddings(f, ["u1", "u2"], ["8", "final"], hidden=3)
    assert got["mean_8"].shape == (2, 3)


@pytest.mark.parametrize(
    ("ids", "arrays", "hidden", "message"),
    [
        (["u2", "u1"], {}, None, "ids differ"),
        (["u1", "u2"], {"drop": "last_8"}, None, "keys"),
        (["u1", "u2"], {"nan": True}, None, "non-finite"),
        (["u1", "u2"], {}, 4, "width"),
    ],
)
def test_check_embeddings_rejects(
    tmp_path: Path, ids: list[str], arrays: dict[str, Any], hidden: int | None, message: str
) -> None:
    f = tmp_path / "e.npz"
    a = np.ones((2, 3), dtype=np.float32)
    mean = a.copy()
    if arrays.get("nan"):
        mean[0, 0] = np.nan
    parts = {"mean_8": mean, "last_8": a}
    parts.pop(arrays.get("drop", ""), None)
    _write_npz(f, ["u1", "u2"], **parts)
    with pytest.raises(RuntimeError, match=message):
        local_llm.check_embeddings(f, ids, ["8"], hidden=hidden)


# --------------------------------------------------------------------------- real child

TINY_PYTHON = local_llm.VENV_PYTHON


@pytest.mark.skipif(
    not TINY_PYTHON.exists(), reason=f"{TINY_PYTHON} is missing: the MLX venv is not installed"
)
def test_selftest_tiny_in_the_mlx_venv() -> None:
    result = local_llm.selftest_tiny()
    assert len(result["forward"]) == len(result["lora"]) == 2  # untied and tied embeddings
    for f in result["forward"]:
        assert f["final_vs_model_max_abs_diff"] <= 1e-5
        assert f["pool_vs_numpy_max_abs_diff"] <= 1e-5
        assert f["head_vs_model_logits_max_abs_diff"] <= 1e-5
        assert f["cached_vs_full_max_abs_diff"] <= 1e-3
        assert f["full_vs_hand_max_abs_diff"] <= 1e-4
        assert f["finite"]
        assert f["feature_dtypes"] == ["float32"]
    for lo in result["lora"]:
        assert lo["loss_vs_hand_abs_diff"] <= 1e-4
        assert lo["row_loss_vs_hand_max_abs_diff"] <= 1e-4
        assert lo["padded_row_vs_alone_abs_diff"] <= 1e-4
        assert lo["long_row_vs_alone_abs_diff"] <= 1e-4
        assert lo["grads_finite"]
        assert lo["grad_tensors_nonzero"] > 0
        assert lo["loss_changed"]
