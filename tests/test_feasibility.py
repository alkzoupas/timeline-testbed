"""Known-answer tests for the M3 feasibility CLI: statistics, selection, projections and the run
itself with `local_llm` faked, so no model is loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from testbed.models import feasibility as fz


def test_describe_on_one_to_hundred() -> None:
    d = fz.describe(list(range(1, 101)))
    assert d["n"] == 100
    assert d["mean"] == pytest.approx(50.5)
    assert d["p50"] == pytest.approx(50.5)
    assert d["p90"] == pytest.approx(90.1)  # linear interpolation: 1 + 0.9 * 99
    assert d["p99"] == pytest.approx(99.01)
    assert d["max"] == 100
    assert d["sum"] == 5050


def test_nearest_units_ties_go_to_the_smaller_id() -> None:
    n = {"b": 100, "a": 120, "c": 80, "d": 500}
    assert fz.nearest_units(n, 100) == ["b"]
    assert fz.nearest_units(n, 100, 3) == ["b", "a", "c"]  # a and c are both 20 away
    assert fz.nearest_units(n, 110) == ["a"]  # a is 10 away, b is 10 away: a < b


def test_sample_with_longest_is_seeded_and_ends_with_the_longest() -> None:
    n = {f"u{i:02d}": i for i in range(30)}
    a, b = fz.sample_with_longest(n, 10, seed=3), fz.sample_with_longest(n, 10, seed=3)
    assert a == b
    assert a[-1] == "u29"
    assert len(a) == 11 and len(set(a)) == 11
    assert fz.sample_with_longest(n, 10, seed=4) != a


def test_embed_layers_quarter_half_three_quarter_final() -> None:
    assert fz.embed_layers(32) == "8,16,24,final"
    assert fz.embed_layers(40) == "10,20,30,final"


def test_embed_hours() -> None:
    assert fz.embed_hours(7_200_000, 1000) == pytest.approx(2.0)  # 7,200 s
    assert fz.embed_hours(2_400_000, 300) == pytest.approx(2.2222, abs=1e-4)


def test_tokens_per_epoch_and_lora_epoch_hours() -> None:
    # 1,000 prompt tokens over 4 units plus 2 answer tokens each; half trained on; two worlds.
    assert fz.tokens_per_epoch(1000, 4, 2, 0.5, 2) == pytest.approx(1008)
    assert fz.lora_epoch_hours(3_600_000, 500) == pytest.approx(2.0)  # 7,200 s
    assert fz.lora_epoch_hours(fz.tokens_per_epoch(1_800_000, 1200, 2, 0.5, 2), 25) == (
        pytest.approx((1_800_000 + 2400) * 1.0 / 25 / 3600)
    )


def _step(k: int, warmup: bool, tokens: int, seconds: float, **kw: Any) -> dict[str, Any]:
    return {
        "step": k,
        "warmup": warmup,
        "tokens": tokens,
        "seconds": seconds,
        "tokens_per_s": tokens / seconds,
        "loss": 1.0,
        "loss_finite": True,
        "grads_finite": True,
        "grad_tensors_nonzero": 10,
        "peak_memory_gib": 5.0 + k,
        **kw,
    }


def _bench(batch: int, steps: list[dict[str, Any]], error: Any = None) -> dict[str, Any]:
    return {"batch_size": batch, "n_prompt": [1000], "steps": steps, "error": error}


def test_bench_summary_excludes_warmup_from_the_rate() -> None:
    steps = [_step(0, True, 1000, 50.0), _step(1, False, 1000, 10.0), _step(2, False, 1000, 30.0)]
    s = fz.bench_summary(_bench(1, steps))
    assert s["ok"]
    assert s["tokens_per_s"] == pytest.approx(2000 / 40.0)
    assert s["warmup_seconds"] == [50.0]
    assert s["steady_seconds"] == [10.0, 30.0]
    assert s["peak_memory_gib"] == 7.0


def test_bench_summary_marks_errors_and_nonfinite_as_not_ok() -> None:
    err = {"step": 1, "tokens": 9000, "error": "RuntimeError: [metal::malloc] Resource limit"}
    s = fz.bench_summary(_bench(1, [_step(0, True, 9000, 5.0)], err))
    assert not s["ok"] and s["tokens_per_s"] is None and s["error"] == err
    bad = _step(1, False, 1000, 10.0, grads_finite=False)
    assert not fz.bench_summary(_bench(1, [_step(0, True, 1000, 5.0), bad]))["ok"]


def test_best_batch_takes_the_fastest_clean_median_bench() -> None:
    benches: dict[str, dict[str, Any]] = {
        "median-b1": {"ok": True, "tokens_per_s": 30.0},
        "median-b4": {"ok": True, "tokens_per_s": 42.0},
        "p90-b1": {"ok": True, "tokens_per_s": 99.0},  # not a median bench
    }
    assert fz.best_batch(benches) == ("median-b4", 42.0)
    benches["median-b4"] = {"ok": False, "tokens_per_s": 42.0}
    assert fz.best_batch(benches) == ("median-b1", 30.0)
    benches["median-b1"] = {"ok": False, "tokens_per_s": None}
    assert fz.best_batch(benches) is None


# --------------------------------------------------------------------------- out dir, rows


def test_check_out_dir_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    fz.check_out_dir(tmp_path / "new")  # absent is fine
    fz.check_out_dir(tmp_path)  # empty is fine
    (tmp_path / "x").write_text("")
    with pytest.raises(SystemExit, match="non-empty"):
        fz.check_out_dir(tmp_path)


def test_main_refuses_a_non_empty_out_dir_before_doing_anything(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.json").write_text("{}")
    with pytest.raises(SystemExit, match="non-empty"):
        fz.main(["--world", str(tmp_path / "no-such-world"), "--model", "9b", "--out", str(out)])


def _world(root: Path, n: int) -> Path:
    world = root / "world"
    world.mkdir()
    with (world / "renders.jsonl").open("w") as f:
        for i in range(n):
            f.write(json.dumps({"unit": f"u{i:02d}", "text": f"render {i}"}) + "\n")
    return world


def test_load_rows_wraps_renders_in_the_draft_prompt(tmp_path: Path) -> None:
    rows = fz.load_rows(_world(tmp_path, 2))
    assert rows[1] == {
        "id": "u01",
        "before": fz.DRAFT_BEFORE,
        "render": "render 1",
        "after": fz.DRAFT_AFTER,
    }


# --------------------------------------------------------------------------- the run, faked


def test_run_with_a_fake_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    n_units = 20
    world = _world(tmp_path, n_units)
    n_prompt = {f"u{i:02d}": 100 * (i + 1) for i in range(n_units)}  # 100 .. 2000
    monkeypatch.setattr(fz, "EMBED_SAMPLE", 5)
    monkeypatch.setattr(fz, "SCORE_UNITS", 4)
    monkeypatch.setattr(fz, "n_layers", lambda model: 32)
    monkeypatch.setattr(fz, "machine", lambda: {"hw.memsize": "1", "machdep.cpu.brand_string": "M"})
    monkeypatch.setattr(fz.local_llm, "model_revision", lambda key: "abcdef0123456789")
    calls: dict[str, Any] = {}

    def count(model: str, rows: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        template = {"ab": {"A": {"ids": [32, 248046]}, "B": {"ids": [33, 248046]}}}
        return {
            "rows": [
                {
                    "id": r["id"],
                    "n_prompt": n_prompt[r["id"]],
                    "render_tokens": n_prompt[r["id"]] - 50,
                }
                for r in rows
            ],
            "template": template,
        }

    def embed(model: str, rows: list[dict[str, str]], layers: str, work_dir: Path, tag: str) -> Any:
        calls["embed"] = ([r["id"] for r in rows], layers)
        return (
            {
                "steady_tokens_per_s": 500.0,
                "first_row_seconds": 4.0,
                "peak_memory_gib": 7.5,
                "wall_seconds": 20.0,
                "mlx": "0.32.3",
                "mlx_lm": "0.32.0",
            },
            {},
        )

    def score(model: str, rows: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        calls["score"] = [r["id"] for r in rows]
        return {
            "max_abs_diff": {"ab": 0.001, "level": 0.12},
            "peak_memory_gib": 9.0,
            "rows": [
                {"seconds_cached": 2.0, "seconds_full": 8.0, "ab": {"mass_cached": 0.99}}
                for _ in rows
            ],
        }

    def lora_bench(
        model: str, rows: list[dict[str, str]], steps: int, warmup: int, batch: int, **kw: Any
    ) -> dict[str, Any]:
        calls.setdefault("lora", {})[kw["tag"]] = ([r["id"] for r in rows], batch)
        if kw["tag"] == "lora-longest-b1":
            err = {"step": 0, "tokens": 2002, "error": "RuntimeError: Resource limit"}
            return {**_bench(1, [], err), "n_prompt": [2000]}
        tokens = sum(n_prompt[r["id"]] + 2 for r in rows)
        rate = 20.0 if batch == 1 else 40.0
        steps_out = [_step(k, k < warmup, tokens, tokens / rate) for k in range(steps + warmup)]
        return _bench(batch, steps_out)

    for name, fn in (
        ("count", count),
        ("embed", embed),
        ("score", score),
        ("lora_bench", lora_bench),
    ):
        monkeypatch.setattr(fz.local_llm, name, fn)

    out = tmp_path / "out"
    s = fz.run(world, "9b", out, seed=0, train_worlds=2)

    assert (out / "summary.json").exists() and (out / "summary.md").exists()
    # Step 1: 100..2000 step 100 -> 20 prompts; mean 1050, median 1050, sum 21,000; 0 over 2,048.
    assert s["count"]["n_prompt"]["sum"] == 21_000
    assert s["count"]["n_prompt"]["p50"] == pytest.approx(1050)
    assert s["count"]["over_limit"] == 0
    # Step 2: 5 sampled units plus the longest (last), at layers 8,16,24,final.
    assert calls["embed"][0][-1] == "u19" and len(calls["embed"][0]) == 6
    assert calls["embed"][1] == "8,16,24,final"
    assert calls["score"][-1] == "u19" and len(calls["score"]) == 4
    # Step 4: median 1050 is 50 from u09 (1000) and u10 (1100): u09. p90 is 1810: u17 (1800).
    assert calls["lora"]["lora-median-b1"] == (["u09"], 1)
    assert calls["lora"]["lora-p90-b1"] == (["u17"], 1)
    assert calls["lora"]["lora-longest-b1"] == (["u19"], 1)
    assert calls["lora"]["lora-median-b4"] == (["u09", "u10", "u08", "u11"], 4)
    # Projections, by hand.
    p = s["projections"]
    assert p["embed_hours_per_world"] == pytest.approx(21_000 / 500.0 / 3600)
    assert p["lora_tokens_per_epoch"] == pytest.approx((21_000 + 20 * 2) * 0.5 * 2)
    assert p["lora_best_median_bench"] == "median-b4"
    assert p["lora_epoch_hours_median"] == pytest.approx(p["lora_tokens_per_epoch"] / 40.0 / 3600)
    assert p["lora_epoch_hours_p90_bound"] == pytest.approx(
        p["lora_tokens_per_epoch"] / 20.0 / 3600
    )
    assert not s["lora"]["longest-b1"]["ok"]
    md = (out / "summary.md").read_text()
    assert "Resource limit" in md and "abcdef012345" in md and "DRAFT" in md
    assert np.isfinite(
        json.loads((out / "summary.json").read_text())["embed"]["steady_tokens_per_s"]
    )
