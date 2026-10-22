"""MLX child for the local-LLM runs (SPEC §7 R1-R3, M3): run in the `es-p6-train` venv, not `.venv`.

    ~/venvs/es-p6-train/bin/python -I testbed/models/mlx_child.py <command> ...

Commands (every one writes a JSON result to `--out`; `embed` also writes an `.npz`):

- `count`: prompt length and render span per row.
- `embed`: the R2 hidden-state extractor (mean over the render span, and the last position, of
  chosen layers).
- `score`: candidate log-probs from one shared prefill with cloned caches, against a plain full
  forward pass of each `prompt + candidate`.
- `lora-bench`: LoRA step throughput on a FIXED dummy answer. Nothing is saved; no label is read.
- `selftest-tiny`: the same id-level core on a tiny random-weight model, for the tests.

This file runs outside the package (`-I`), so it imports nothing from `testbed` and never opens a
database. MLX is imported only inside functions, so `.venv` can import the pure helpers (numpy
only) for tests and type checks. It stays Python 3.12-compatible.

A row is `{"id", "before", "render", "after"}`. The prompt is the chat template applied to
`before + SENTINEL + after`, split on the sentinel, and the three pieces are encoded separately:
`ids = enc(head) + enc(render) + enc(tail)`. The render's ids are then independent of the prompt.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

SENTINEL = "@@RENDER-SENTINEL@@"
MEMORY_LIMIT_GIB = 40  # of 48 GiB unified memory; keeps the machine out of swap
LORA = {"rank": 16, "scale": 2.0, "dropout": 0.0}
LR = 1e-4
DUMMY_TARGET = "A"  # a fixed answer for the throughput bench; never a label
LEVELS = tuple(str(n) for n in range(1, 11))
END_TOKEN = "<|im_end|>"
SEED = 20261021
PAD_ID = 0  # padding sits after every answer position; the loss gathers answers only


class Prompt(NamedTuple):
    ids: list[int]
    span: tuple[int, int]  # the render's [start, end) in ids


# --------------------------------------------------------------------------- pure helpers


def split_template(text: str) -> tuple[str, str]:
    """Split a rendered chat template on the sentinel, which must occur exactly once."""
    parts = text.split(SENTINEL)
    if len(parts) != 2:
        raise ValueError(f"the sentinel occurs {len(parts) - 1} times in the template output")
    return parts[0], parts[1]


def assemble_prompt(encode: Callable[[str], list[int]], templated: str, render: str) -> Prompt:
    """Encode head, render and tail separately and join them; return the ids and render span."""
    head, tail = split_template(templated)
    head_ids, render_ids, tail_ids = encode(head), encode(render), encode(tail)
    start = len(head_ids)
    return Prompt(head_ids + render_ids + tail_ids, (start, start + len(render_ids)))


def row_prompt(tokenizer: Any, row: dict[str, Any]) -> Prompt:
    messages = [{"role": "user", "content": row["before"] + SENTINEL + row["after"]}]
    templated = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    return assemble_prompt(
        lambda s: list(tokenizer.encode(s, add_special_tokens=False)), templated, row["render"]
    )


def mean_pool(states: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    """Mean of `states[start:end]` over positions, as float32. `states` is (T, hidden)."""
    start, end = span
    if not 0 <= start < end <= states.shape[0]:
        raise ValueError(f"span {span} does not fit {states.shape[0]} positions")
    return states[start:end].astype(np.float32).mean(axis=0, dtype=np.float32)


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int = PAD_ID) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad sequences to a (B, T) int32 array; also return the true lengths."""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    out = np.full((len(seqs), int(lengths.max())), pad_id, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lengths


def parse_layers(spec: str, n_layers: int) -> list[int | str]:
    """`8,16,final` -> [8, 16, "final"]; layer indices are 0-based decoder outputs."""
    out: list[int | str] = []
    for part in spec.split(","):
        part = part.strip()
        if part == "final":
            out.append("final")
            continue
        index = int(part)
        if not 0 <= index < n_layers:
            raise ValueError(f"layer {index} is outside 0..{n_layers - 1}")
        out.append(index)
    return out


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def hand_masked_ce(
    logits: np.ndarray, ids: np.ndarray, starts: np.ndarray, n_ans: int
) -> np.ndarray:
    """Per-row mean CE over each row's answer tokens, from full logits (B, T, V), in numpy."""
    out = []
    for b in range(ids.shape[0]):
        lp = log_softmax_np(logits[b])
        pos = int(starts[b]) - 1 + np.arange(n_ans)
        out.append(-lp[pos, ids[b, pos + 1]].mean())
    return np.array(out)


def gib(n_bytes: float) -> float:
    return round(n_bytes / 2**30, 3)


# --------------------------------------------------------------------------- MLX helpers


def init_mlx() -> Any:
    import mlx.core as mx

    mx.set_memory_limit(MEMORY_LIMIT_GIB * 2**30)
    mx.reset_peak_memory()
    return mx


def text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def head(tm: Any, hidden: Any) -> Any:
    """The output head applied to hidden states (already past the final norm)."""
    if tm.args.tie_word_embeddings:
        return tm.model.embed_tokens.as_linear(hidden)
    return tm.lm_head(hidden)


def forward_states(mx: Any, model: Any, ids: Sequence[int], wanted: set[int]) -> dict[Any, Any]:
    """One forward pass, batch 1, no cache, layer by layer.

    Returns {i: (1, T, H) output of decoder layer i} for i in `wanted`, plus "final" (after the
    final norm). It runs embed_tokens, each layer with the model's own mask choice, then norm: the
    same ops as `language_model.model(ids)`, with no patching of mlx-lm.
    """
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    inner = text_model(model).model
    h = inner.embed_tokens(mx.array(list(ids))[None])
    attn_mask, ssm_mask = create_attention_mask(h, None), create_ssm_mask(h, None)
    out: dict[Any, Any] = {}
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=ssm_mask if layer.is_linear else attn_mask, cache=None)
        if i in wanted:
            out[i] = h
    out["final"] = inner.norm(h)
    return out


def embed_ids(
    mx: Any, model: Any, ids: Sequence[int], span: tuple[int, int], layers: Sequence[int | str]
) -> dict[str, np.ndarray]:
    """`mean_<L>` (mean over the render span) and `last_<L>` (final prompt token), float32."""
    states = forward_states(mx, model, ids, {x for x in layers if isinstance(x, int)})
    out: dict[str, np.ndarray] = {}
    for name in layers:
        full = np.array(states[name][0].astype(mx.float32))
        out[f"mean_{name}"] = mean_pool(full, span)
        out[f"last_{name}"] = full[-1].astype(np.float32)
    return out


def clone_cache(cache: list[Any]) -> list[Any]:
    """Independent copy of a prompt cache, so each candidate continues the same state."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    out = []
    for c in cache:
        if isinstance(c, KVCache):
            n: Any = KVCache()
            n.keys = c.keys[..., : c.offset, :]
            n.values = c.values[..., : c.offset, :]
            n.offset = c.offset
        elif isinstance(c, ArraysCache):
            n = ArraysCache(size=len(c.cache))
            n.cache = list(c.cache)
            n.left_padding, n.lengths = c.left_padding, c.lengths
        else:
            raise TypeError(f"unsupported cache type {type(c).__name__}")
        out.append(n)
    return out


def _log_softmax(mx: Any, logits: Any) -> Any:
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def cached_logprobs(
    mx: Any, model: Any, prompt: Sequence[int], conts: dict[str, list[int]]
) -> dict[str, float]:
    """log P(whole continuation | prompt) per candidate, from one prefill with cloned caches."""
    tm = text_model(model)
    cache = tm.make_cache()
    first = _log_softmax(mx, head(tm, tm.model(mx.array(list(prompt))[None], cache=cache)[:, -1]))
    mx.eval(first)
    out = {}
    for name, cont in conts.items():
        total = first[0, cont[0]]
        if len(cont) > 1:
            branch = clone_cache(cache)
            hidden = tm.model(mx.array(cont[:-1])[None], cache=branch)
            lp = _log_softmax(mx, head(tm, hidden))
            total = total + mx.take_along_axis(lp[0], mx.array(cont[1:])[:, None], axis=-1).sum()
        out[name] = float(total.item())
    return out


def full_logprob(mx: Any, model: Any, prompt: Sequence[int], cont: Sequence[int]) -> float:
    """log P(continuation | prompt) from one plain forward pass of `prompt + continuation`."""
    tm = text_model(model)
    ids = mx.array(list(prompt) + list(cont))
    hidden = tm.model(ids[None, :-1])[:, len(prompt) - 1 :, :]
    lp = _log_softmax(mx, head(tm, hidden))
    return float(mx.take_along_axis(lp[0], ids[len(prompt) :][:, None], axis=-1).sum().item())


def make_row_ce(mx: Any, nn: Any) -> Callable[..., Any]:
    def row_ce(model: Any, ids: Any, starts: Any, n_ans: int) -> Any:
        """Per-row mean CE over the answer tokens only. Padding is never gathered.

        `ids` is (B, T) right-padded, `starts` (B,) the first answer index of each row.
        """
        tm = text_model(model)
        hidden = tm.model(ids[:, :-1])
        pos = (starts[:, None] - 1) + mx.arange(n_ans)[None, :]
        picked = mx.take_along_axis(hidden, pos[:, :, None], axis=1)
        logits = head(tm, picked).astype(mx.float32)
        targets = mx.take_along_axis(ids, pos + 1, axis=1)
        return nn.losses.cross_entropy(logits, targets, reduction="none").mean(axis=1)

    return row_ce


def make_loss(mx: Any, nn: Any) -> Callable[..., Any]:
    row_ce = make_row_ce(mx, nn)

    def loss_fn(model: Any, ids: Any, starts: Any, n_ans: int) -> Any:
        return row_ce(model, ids, starts, n_ans).mean()

    return loss_fn


_LORA_READY = False


def lora_runtime() -> None:
    """Install mlx-lm-lora's memory-safe recurrences and per-layer checkpointing, once."""
    global _LORA_READY
    if _LORA_READY:
        return
    from mlx_lm_lora.recurrent_patch import enable_memory_safe_recurrences

    enable_memory_safe_recurrences()
    _LORA_READY = True


_CKPT_READY = False


def lora_wrap(model: Any) -> int:
    """Freeze, then LoRA (rank 16, scale 2.0, dropout 0) on all layers, with checkpointing."""
    global _CKPT_READY
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    tm = text_model(model)
    model.freeze()
    linear_to_lora_layers(tm, len(tm.layers), dict(LORA))
    if not _CKPT_READY:  # patches the layer class, so once per process
        grad_checkpoint(tm.layers[0])
        _CKPT_READY = True
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


def grad_stats(mx: Any, grads: Any) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    flat = [g for _, g in tree_flatten(grads)]
    finite = [mx.isfinite(g).all() for g in flat]
    nonzero = [(g != 0).any() for g in flat]
    mx.eval(finite, nonzero)
    return {
        "grads_finite": all(bool(f.item()) for f in finite),
        "grad_tensors": len(flat),
        "grad_tensors_nonzero": sum(int(n.item()) for n in nonzero),
    }


def train_step(
    mx: Any,
    model: Any,
    value_and_grad: Any,
    optimizer: Any,
    batch: tuple[np.ndarray, np.ndarray],
    n_ans: int,
) -> dict[str, Any]:
    """One timed LoRA step on a right-padded batch: loss, grads, Adam update."""
    ids, starts = mx.array(batch[0]), mx.array(batch[1])
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    loss, grads = value_and_grad(model, ids, starts, n_ans)
    mx.eval(loss, grads)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    seconds = time.perf_counter() - t0
    return {
        "loss": float(loss.item()),
        "loss_finite": bool(np.isfinite(loss.item())),
        "seconds": seconds,
        "peak_memory_gib": gib(mx.get_peak_memory()),
        **grad_stats(mx, grads),
    }


def make_batches(
    prompts: Sequence[Prompt], target: Sequence[int], batch_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rows in input order, `batch_size` per batch, each `prompt + target`, right-padded."""
    batches = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        ids, _ = pad_batch([p.ids + list(target) for p in chunk])
        batches.append((ids, np.array([len(p.ids) for p in chunk], dtype=np.int32)))
    return batches


# --------------------------------------------------------------------------- tokenizer side


def continuation_ids(tokenizer: Any, content: str) -> list[int]:
    """The template's own assistant continuation for `content`, up to and including <|im_end|>.

    The full conversation, minus the generation prompt, keeps whatever the template emits.
    """
    user = {"role": "user", "content": "x"}
    kw = {"enable_thinking": False, "tokenize": False}
    gen = tokenizer.apply_chat_template([user], add_generation_prompt=True, **kw)
    full = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": content}], add_generation_prompt=False, **kw
    )
    if not full.startswith(gen):
        raise SystemExit("chat template: the generation prompt is not a prefix of the full turn")
    ids = list(tokenizer.encode(full[len(gen) :], add_special_tokens=False))
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    if end not in ids:
        raise SystemExit(f"chat template: no {END_TOKEN} after {content!r}")
    return ids[: ids.index(end) + 1]


def candidate_sets(tokenizer: Any) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    """Terminated candidates: A/B, and RISK_LEVEL 1-10. Plus a report of what the template emits."""
    sets = {
        "ab": {x: continuation_ids(tokenizer, x) for x in ("A", "B")},
        "level": {n: continuation_ids(tokenizer, f"RISK_LEVEL: {n}") for n in LEVELS},
    }
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    report: dict[str, Any] = {
        name: {
            key: {"ids": ids, "pieces": [tokenizer.decode([t]) for t in ids]}
            for key, ids in group.items()
        }
        for name, group in sets.items()
    }
    report["ab_single_token_plus_end"] = all(
        len(ids) == 2 and ids[1] == end for ids in sets["ab"].values()
    )
    return sets, report


# --------------------------------------------------------------------------- commands


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        if set(r) != {"id", "before", "render", "after"}:
            raise SystemExit(f"row keys must be id/before/render/after, got {sorted(r)}")
    return rows


def meta(mx: Any, command: str, model_dir: str | None, t0: float) -> dict[str, Any]:
    return {
        "command": command,
        "model_dir": model_dir,
        "peak_memory_gib": gib(mx.get_peak_memory()),
        "mlx": md.version("mlx"),
        "mlx_lm": md.version("mlx-lm"),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n")


def cmd_count(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    mx = init_mlx()
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(args.model_dir)
    out = []
    for row in read_rows(args.inp):
        p = row_prompt(tokenizer, row)
        out.append(
            {
                "id": row["id"],
                "n_prompt": len(p.ids),
                "span": list(p.span),
                "render_tokens": p.span[1] - p.span[0],
            }
        )
    _, template = candidate_sets(tokenizer)
    write_json(
        args.out, {**meta(mx, "count", args.model_dir, t0), "rows": out, "template": template}
    )


def load_model(model_dir: str) -> tuple[Any, Any]:
    from mlx_lm import load

    model, tokenizer = load(model_dir)
    return model, tokenizer


def cmd_embed(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    mx = init_mlx()
    model, tokenizer = load_model(args.model_dir)
    tm = text_model(model)
    layers = parse_layers(args.layers, len(tm.layers))
    model.eval()
    rows = read_rows(args.inp)
    arrays: dict[str, list[np.ndarray]] = {}
    per_row = []
    for row in rows:
        p = row_prompt(tokenizer, row)
        t1 = time.perf_counter()
        feats = embed_ids(mx, model, p.ids, p.span, layers)  # numpy conversion forces the eval
        seconds = time.perf_counter() - t1
        for key, value in feats.items():
            arrays.setdefault(key, []).append(value)
        per_row.append(
            {"id": row["id"], "n_prompt": len(p.ids), "span": list(p.span), "seconds": seconds}
        )
    payload: dict[str, Any] = {k: np.stack(v).astype(np.float32) for k, v in arrays.items()}
    payload["ids"] = np.array([r["id"] for r in rows])
    np.savez(args.out_npz, **payload)
    rest = per_row[1:]
    steady_tokens = sum(r["n_prompt"] for r in rest)
    steady_seconds = sum(r["seconds"] for r in rest)
    write_json(
        args.out,
        {
            **meta(mx, "embed", args.model_dir, t0),
            "layers": [str(x) for x in layers],
            "hidden_size": int(tm.args.hidden_size),
            "n_layers": len(tm.layers),
            "rows": per_row,
            "first_row_seconds": per_row[0]["seconds"] if per_row else None,
            "steady_tokens": steady_tokens,
            "steady_seconds": steady_seconds,
            "steady_tokens_per_s": steady_tokens / steady_seconds if steady_seconds else None,
        },
    )


def _mass(lps: dict[str, float]) -> float:
    return float(sum(np.exp(v) for v in lps.values()))


def cmd_score(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    mx = init_mlx()
    model, tokenizer = load_model(args.model_dir)
    model.eval()
    sets, template = candidate_sets(tokenizer)
    flat = {f"{g}:{k}": v for g, group in sets.items() for k, v in group.items()}
    out = []
    for row in read_rows(args.inp):
        p = row_prompt(tokenizer, row)
        t1 = time.perf_counter()
        cached = cached_logprobs(mx, model, p.ids, flat)
        t_cached = time.perf_counter() - t1
        t1 = time.perf_counter()
        full = {name: full_logprob(mx, model, p.ids, cont) for name, cont in flat.items()}
        t_full = time.perf_counter() - t1
        entry: dict[str, Any] = {
            "id": row["id"],
            "n_prompt": len(p.ids),
            "seconds_cached": t_cached,
            "seconds_full": t_full,
        }
        for group in sets:
            keys = [k for k in flat if k.startswith(f"{group}:")]
            c = {k.split(":", 1)[1]: cached[k] for k in keys}
            f = {k.split(":", 1)[1]: full[k] for k in keys}
            entry[group] = {
                "cached": c,
                "full": f,
                "max_abs_diff": max(abs(c[k] - f[k]) for k in c),
                "mass_cached": _mass(c),
                "mass_full": _mass(f),
            }
        out.append(entry)
    write_json(
        args.out,
        {
            **meta(mx, "score", args.model_dir, t0),
            "rows": out,
            "max_abs_diff": {g: max(r[g]["max_abs_diff"] for r in out) for g in sets},
            "template": template,
        },
    )


def cmd_lora_bench(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    mx = init_mlx()
    import mlx.nn as nn
    import mlx.optimizers as optim

    lora_runtime()
    model, tokenizer = load_model(args.model_dir)
    n_trainable = lora_wrap(model)
    target = continuation_ids(tokenizer, DUMMY_TARGET)
    prompts = [row_prompt(tokenizer, r) for r in read_rows(args.inp)]
    batches = make_batches(prompts, target, args.batch_size)
    value_and_grad = nn.value_and_grad(model, make_loss(mx, nn))
    optimizer = optim.Adam(learning_rate=LR)
    model.train()
    steps: list[dict[str, Any]] = []
    error = None
    for k in range(args.warmup + args.steps):
        batch = batches[k % len(batches)]  # cycles when there are fewer batches than steps
        tokens = int((batch[1] + len(target)).sum())  # non-pad tokens: prompts plus answers
        try:
            rec = train_step(mx, model, value_and_grad, optimizer, batch, len(target))
        except Exception as exc:  # a Metal failure at one length must not hide the others
            error = {
                "step": k,
                "tokens": tokens,
                "padded_shape": list(batch[0].shape),
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            }
            mx.clear_cache()
            break
        rec.update(
            step=k,
            warmup=k < args.warmup,
            tokens=tokens,
            tokens_per_s=tokens / rec["seconds"],
        )
        steps.append(rec)
    write_json(
        args.out,
        {
            **meta(mx, "lora-bench", args.model_dir, t0),
            "target": {"text": DUMMY_TARGET, "ids": target, "is_label": False},
            "note": "FIXED dummy answer; no label is read. No adapter is saved.",
            "lora": {**LORA, "lr": LR, "trainable_parameters": n_trainable},
            "batch_size": args.batch_size,
            "n_rows": len(prompts),
            "n_prompt": [len(p.ids) for p in prompts],
            "steps": steps,
            "error": error,
        },
    )


# --------------------------------------------------------------------------- selftest


def tiny_model(mx: Any, tie: bool) -> Any:
    from mlx_lm.models import qwen3_5

    text = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 300,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "tie_word_embeddings": tie,
        "max_position_embeddings": 512,
    }
    mx.random.seed(SEED)
    model = qwen3_5.Model(
        qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": text})
    )
    mx.eval(model.parameters())
    return model


def _tiny_ids() -> np.ndarray:
    return np.random.RandomState(0).randint(5, 295, size=40)


def selftest_forward(mx: Any, tie: bool) -> dict[str, Any]:
    """Layer loop against the model's own forward; mean-pool; cached against full scoring."""
    model = tiny_model(mx, tie)
    model.eval()
    ids, span = [int(x) for x in _tiny_ids()], (10, 30)
    own = np.array(text_model(model).model(mx.array(ids)[None])[0])
    states = forward_states(mx, model, ids, {1, 2})
    final = np.array(states["final"][0])
    feats = embed_ids(mx, model, ids, span, [1, 2, "final"])
    ref = np.array(states[2][0].astype(mx.float32))
    full_logits = np.array(model(mx.array(ids)[None])[0])
    tm = text_model(model)
    head_logits = np.array(head(tm, mx.array(own)[None])[0])
    conts = {"x": [7, 8], "y": [9, 10, 11], "z": [12, 13]}
    cached = cached_logprobs(mx, model, ids, conts)
    full = {k: full_logprob(mx, model, ids, v) for k, v in conts.items()}
    hand_full = {}  # numpy log-softmax of the model's own full logits on prompt + candidate
    for k, v in conts.items():
        seq = ids + v
        lg = log_softmax_np(np.array(model(mx.array(seq)[None])[0]))
        hand_full[k] = float(sum(lg[len(ids) - 1 + i, t] for i, t in enumerate(v)))
    return {
        "tie_word_embeddings": tie,
        "final_vs_model_max_abs_diff": float(np.abs(final - own).max()),
        "pool_vs_numpy_max_abs_diff": float(
            max(
                np.abs(feats["mean_2"] - ref[span[0] : span[1]].mean(axis=0)).max(),
                np.abs(feats["last_2"] - ref[-1]).max(),
            )
        ),
        "head_vs_model_logits_max_abs_diff": float(np.abs(head_logits - full_logits).max()),
        "cached_vs_full_max_abs_diff": max(abs(cached[k] - full[k]) for k in conts),
        "full_vs_hand_max_abs_diff": max(abs(full[k] - hand_full[k]) for k in conts),
        "feature_shapes": {k: list(v.shape) for k, v in feats.items()},
        "feature_dtypes": sorted({str(v.dtype) for v in feats.values()}),
        "finite": bool(all(np.isfinite(v).all() for v in feats.values())),
    }


def selftest_lora(mx: Any, tie: bool) -> dict[str, Any]:
    """One LoRA step's loss against a numpy masked CE; a padded row against the row alone."""
    import mlx.nn as nn
    import mlx.optimizers as optim

    model = tiny_model(mx, tie)
    model.eval()
    target = [7, 8]
    rng = np.random.RandomState(1)
    prompts = [rng.randint(5, 295, size=n).tolist() for n in (30, 18)]
    ids, lengths = pad_batch([p + target for p in prompts])
    starts = np.array([len(p) for p in prompts], dtype=np.int32)
    assert ids.shape == (2, 32) and lengths.tolist() == [32, 20]

    full_logits = np.array(model(mx.array(ids))).astype(np.float64)
    hand = hand_masked_ce(full_logits, ids, starts, len(target))

    lora_runtime()
    n_trainable = lora_wrap(model)
    model.train()
    row_ce = make_row_ce(mx, nn)
    loss_fn = make_loss(mx, nn)
    batch_rows = np.array(row_ce(model, mx.array(ids), mx.array(starts), len(target)))
    alone, alone_ids = [], []
    for p in prompts:
        one, _ = pad_batch([p + target])
        alone_ids.append(one)
        alone.append(float(row_ce(model, mx.array(one), mx.array([len(p)]), len(target))[0]))
    value_and_grad = nn.value_and_grad(model, loss_fn)
    step = train_step(
        mx, model, value_and_grad, optim.Adam(learning_rate=LR), (ids, starts), len(target)
    )
    after = float(loss_fn(model, mx.array(ids), mx.array(starts), len(target)).item())
    return {
        "tie_word_embeddings": tie,
        "trainable_parameters": n_trainable,
        "step_loss": step["loss"],
        "hand_ce_mean": float(hand.mean()),
        "loss_vs_hand_abs_diff": abs(step["loss"] - float(hand.mean())),
        "row_loss_vs_hand_max_abs_diff": float(np.abs(batch_rows - hand).max()),
        "padded_row_vs_alone_abs_diff": abs(float(batch_rows[1]) - alone[1]),
        "long_row_vs_alone_abs_diff": abs(float(batch_rows[0]) - alone[0]),
        "grads_finite": step["grads_finite"],
        "grad_tensors_nonzero": step["grad_tensors_nonzero"],
        "loss_after_step": after,
        "loss_changed": abs(after - step["loss"]) > 1e-7,
    }


def cmd_selftest_tiny(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    mx = init_mlx()
    # Order matters: the LoRA wrap patches the layer class, so every plain forward goes first.
    forward = [selftest_forward(mx, tie) for tie in (False, True)]
    lora = [selftest_lora(mx, tie) for tie in (False, True)]
    write_json(
        args.out,
        {
            **meta(mx, "selftest-tiny", None, t0),
            "forward": forward,
            "lora": lora,
            "note": "random weights; no timing claims",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], None], model: bool = True) -> Any:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if model:
            p.add_argument("--model-dir", required=True)
            p.add_argument("--in", dest="inp", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        return p

    add("count", cmd_count)
    p = add("embed", cmd_embed)
    p.add_argument("--layers", required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    add("score", cmd_score)
    p = add("lora-bench", cmd_lora_bench)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    add("selftest-tiny", cmd_selftest_tiny, model=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
