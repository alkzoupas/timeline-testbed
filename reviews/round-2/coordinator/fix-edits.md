# Fix worker edits, from its session transcript (parsed, not executed)

## 2026-10-22T06:18:26.401Z `testbed/models/mlx_child.py`

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
 import argparse
+import hashlib
 import importlib.metadata as md
```

```diff
--- before
+++ after
@@ -1 +1,2 @@
 PAD_ID = 0  # padding sits after every answer position; the loss gathers answers only
+CHILD_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # this source, at startup
```

```diff
--- before
+++ after
@@ -1,12 +1,18 @@
-def read_rows(path: Path) -> list[dict[str, Any]]:
-    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
+def read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
+    """The rows and the sha256 of the file's bytes as read."""
+    raw = path.read_bytes()
+    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
     for r in rows:
         if set(r) != {"id", "before", "render", "after"}:
             raise SystemExit(f"row keys must be id/before/render/after, got {sorted(r)}")
-    return rows
+    return rows, hashlib.sha256(raw).hexdigest()
 
 
-def meta(mx: Any, command: str, model_dir: str | None, t0: float) -> dict[str, Any]:
+def meta(
+    mx: Any, command: str, model_dir: str | None, t0: float, rows_sha256: str | None = None
+) -> dict[str, Any]:
     return {
         "command": command,
         "model_dir": model_dir,
+        "child_sha256": CHILD_SHA256,
+        "rows_sha256": rows_sha256,
```

```diff
--- before
+++ after
@@ -1,4 +1,5 @@
     out = []
-    for row in read_rows(args.inp):
+    rows, rows_sha256 = read_rows(args.inp)
+    for row in rows:
         p = row_prompt(tokenizer, row)
         out.append(
@@ -12,4 +13,5 @@
     _, template = candidate_sets(tokenizer)
     write_json(
-        args.out, {**meta(mx, "count", args.model_dir, t0), "rows": out, "template": template}
+        args.out,
+        {**meta(mx, "count", args.model_dir, t0, rows_sha256), "rows": out, "template": template},
     )
```

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-    rows = read_rows(args.inp)
+    rows, rows_sha256 = read_rows(args.inp)
     arrays:
```

```diff
--- before
+++ after
@@ -1 +1 @@
-**meta(mx, "embed", args.model_dir, t0),
+**meta(mx, "embed", args.model_dir, t0, rows_sha256),
```

```diff
--- before
+++ after
@@ -1,4 +1,5 @@
     out = []
-    for row in read_rows(args.inp):
+    rows, rows_sha256 = read_rows(args.inp)
+    for row in rows:
         p = row_prompt(tokenizer, row)
         t1 = time.perf_counter()
```

```diff
--- before
+++ after
@@ -1 +1 @@
-**meta(mx, "score", args.model_dir, t0),
+**meta(mx, "score", args.model_dir, t0, rows_sha256),
```

```diff
--- before
+++ after
@@ -1,4 +1,5 @@
     out = []
-    for row in read_rows(args.inp):
+    rows, rows_sha256 = read_rows(args.inp)
+    for row in rows:
         p = row_prompt(tokenizer, row)
         entry = zeroshot_row
```

```diff
--- before
+++ after
@@ -1 +1 @@
-**meta(mx, "zeroshot", args.model_dir, t0),
+**meta(mx, "zeroshot", args.model_dir, t0, rows_sha256),
```

```diff
--- before
+++ after
@@ -1 +1,2 @@
-    prompts = [row_prompt(tokenizer, r) for r in read_rows(args.inp)]
+    rows, rows_sha256 = read_rows(args.inp)
+    prompts = [row_prompt(tokenizer, r) for r in rows]
```

```diff
--- before
+++ after
@@ -1 +1 @@
-**meta(mx, "lora-bench", args.model_dir, t0),
+**meta(mx, "lora-bench", args.model_dir, t0, rows_sha256),
```

## 2026-10-22T06:18:46.124Z `testbed/models/nimble_child.py`

```diff
--- before
+++ after
@@ -2,3 +2,6 @@
 
 
+CHILD_SHA256 = sha256_file(Path(__file__))  # this source, at startup
+
+
 def check_schema_text
```

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
-def read_rows(path: Path) -> list[dict[str, Any]]:
-    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
+def read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
+    """The rows and the sha256 of the file's bytes as read."""
+    raw = path.read_bytes()
+    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
```

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
         raise SystemExit("duplicate row ids")
-    return rows
+    return rows, hashlib.sha256(raw).hexdigest()
```

```diff
--- before
+++ after
@@ -1,7 +1,11 @@
 def _provenance(
-    provenance: dict[str, Any], args: argparse.Namespace, command: str, t0: float
+    provenance: dict[str, Any], args: argparse.Namespace, command: str, t0: float, rows_sha256: str
 ) -> dict[str, Any]:
     return {
         "command": command,
-        "provenance": {**provenance, "nimble_schema_sha256": args.schema_sha256},
-        "rows_sha256": sha256_file(args.inp),
+        "provenance": {
+            **provenance,
+            "nimble_schema_sha256": args.schema_sha256,
+            "child_sha256": CHILD_SHA256,
+        },
+        "rows_sha256": rows_sha256,
```

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
     out = []
-    for row in read_rows(args.inp):
+    rows, rows_sha256 = read_rows(args.inp)
+    for row in rows:
         prepared
```

```diff
--- before
+++ after
@@ -1 +1 @@
-{**_provenance(provenance, args, "count", t0), "rows": out}
+{**_provenance(provenance, args, "count", t0, rows_sha256), "rows": out}
```

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-    rows = read_rows(args.inp)
+    rows, rows_sha256 = read_rows(args.inp)
     schema = load_schema(args)
```

```diff
--- before
+++ after
@@ -1 +1 @@
-**_provenance(provenance, args, "score", t0),
+**_provenance(provenance, args, "score", t0, rows_sha256),
```

## 2026-10-22T06:18:56.048Z `testbed/models/local_llm.py`

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
+import hashlib
 import json
 import subprocess
```

```diff
--- before
+++ after
@@ -1,2 +1,25 @@
+def sha256_text(text: str) -> str:
+    return hashlib.sha256(text.encode()).hexdigest()
+
+
+def rows_text(rows: Sequence[Row]) -> str:
+    """The rows file the child reads: one JSON object per row. Every rows hash is of this text."""
+    return "".join(json.dumps(dict(r)) + "\n" for r in rows)
+
+
+def child_sha256() -> str:
+    """The sha256 of the child script as it is now."""
+    return hashlib.sha256(CHILD_SCRIPT.read_bytes()).hexdigest()
+
+
+def check_child_identity(command: str, result: Mapping[str, Any], rows: Sequence[Row] | None) -> None:
+    """Raise `RuntimeError` unless the result names this `CHILD_SCRIPT` (as it is now) and, when
+    the child was given `rows`, exactly the rows text that was written for it."""
+    if result.get("child_sha256") != child_sha256():
+        raise RuntimeError(f"mlx_child {command}: child_sha256 differs from {CHILD_SCRIPT.name}")
+    if rows is not None and result.get("rows_sha256") != sha256_text(rows_text(rows)):
+        raise RuntimeError(f"mlx_child {command}: the child read other rows than were written")
+
+
 @dataclass(frozen=True)
 class ChildRun:
```

```diff
--- before
+++ after
@@ -1 +1 @@
-        rows_file.write_text("".join(json.dumps(dict(r)) + "\n" for r in rows))
+        rows_file.write_bytes(rows_text(rows).encode())
```

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
     result = json.loads(out_json.read_text())
+    check_child_identity(command, result, rows)
     return ChildRun(
```

## 2026-10-22T06:19:19.478Z `tests/test_local_llm.py`

```diff
--- before
+++ after
@@ -1,17 +1,83 @@
+ROW_A = {"id": "a", "before": "", "render": "r", "after": ""}
+
+
+def _identity(rows: list[dict[str, str]], **override: Any) -> dict[str, Any]:
+    """What a faithful child reports: its own script's hash and the hash of the rows it read."""
+    out = {
+        "child_sha256": local_llm.child_sha256(),
+        "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
+    }
+    out.update(override)
+    return {k: v for k, v in out.items() if v is not None}
+
+
+def test_rows_text_is_what_the_child_reads_and_hashes(tmp_path: Path) -> None:
+    rows = [ROW_A, {**ROW_A, "id": "b", "render": "é\n\"q\""}]
+    text = local_llm.rows_text(rows)
+    assert text == "".join(json.dumps(r) + "\n" for r in rows)
+    assert local_llm.sha256_text(text) == hashlib.sha256(text.encode()).hexdigest()
+    assert local_llm.child_sha256() == hashlib.sha256(local_llm.CHILD_SCRIPT.read_bytes()).hexdigest()
+
+
 def test_run_child_parses_json_and_sets_only_the_offline_env(
     tmp_path: Path, monkeypatch: pytest.MonkeyPatch
 ) -> None:
     # Finds the argument after --out and writes a JSON file there; also records argv[1] and the env.
+    ident = _identity([ROW_A])
     body = (
         'while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done\n'
-        'printf \'{"hf": "%s", "home": "%s"}\' "$HF_HUB_OFFLINE" "$HOME" > "$out"\n'
+        'printf \'{"hf": "%s", "home": "%s", "child_sha256": "'
+        + ident["child_sha256"]
+        + '", "rows_sha256": "'
+        + ident["rows_sha256"]
+        + '"}\' "$HF_HUB_OFFLINE" "$HOME" > "$out"\n'
     )
     monkeypatch.setattr(local_llm, "VENV_PYTHON", _fake_python(tmp_path, body))
-    run = local_llm.run_child(
-        "score",
-        rows=[{"id": "a", "before": "", "render": "r", "after": ""}],
-        work_dir=tmp_path / "w",
-    )
-    assert run.result == {"hf": "1", "home": ""}  # HOME is not passed through
+    run = local_llm.run_child("score", rows=[ROW_A], work_dir=tmp_path / "w")
+    assert run.result == {"hf": "1", "home": "", **ident}  # HOME is not passed through
     assert run.npz is None
     assert json.loads((tmp_path / "w" / "score.rows.jsonl").read_text())["id"] == "a"
+
+
+def _run_with_result(
+    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: dict[str, Any], rows: Any = (ROW_A,)
+) -> Any:
+    canned = tmp_path / "canned.json"
+    canned.write_text(json.dumps(result))
+    body = (
+        'while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done\n'
+        f'cp "{canned}" "$out"\n'
+    )
+    monkeypatch.setattr(local_llm, "VENV_PYTHON", _fake_python(tmp_path, body))
+    return local_llm.run_child("score", rows=list(rows), work_dir=tmp_path / "w")
+
+
+@pytest.mark.parametrize(
+    ("override", "message"),
+    [
+        ({"child_sha256": "0" * 64}, "child_sha256 differs"),
+        ({"child_sha256": None}, "child_sha256 differs"),  # a missing field fails closed
+        ({"rows_sha256": "0" * 64}, "other rows"),
+        ({"rows_sha256": None}, "other rows"),
+    ],
+)
+def test_run_child_rejects_a_result_from_another_child_or_other_rows(
+    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, Any], message: str
+) -> None:
+    result = _identity([ROW_A], **override)
+    with pytest.raises(RuntimeError, match=message):
+        _run_with_result(tmp_path, monkeypatch, result)
+
+
+def test_run_child_rejects_a_child_edited_while_it_ran(
+    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    script = tmp_path / "mlx_child.py"
+    script.write_text("# before\n")
+    monkeypatch.setattr(local_llm, "CHILD_SCRIPT", script)
+    result = _identity([ROW_A])  # the hash of the script as the child read it
+    script.write_text("# after\n")  # edited before the parent checks
+    with pytest.raises(RuntimeError, match="child_sha256 differs"):
+        _run_with_result(tmp_path, monkeypatch, result)
+    script.write_text("# before\n")
+    assert _run_with_result(tmp_path, monkeypatch, result).result == result  # the same bytes pass
```

```diff
--- before
+++ after
@@ -1,3 +1,3 @@
 def _zeroshot_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: Any) -> Any:
     canned = tmp_path / "canned.json"
-    canned.write_text(json.dumps(result))
+    canned.write_text(json.dumps({**_identity([ROW_A]), **result}))
```

```diff
--- before
+++ after
@@ -1,4 +1,4 @@
     return local_llm.zeroshot(
         "9b",
-        [{"id": "a", "before": "", "render": "r", "after": ""}],
+        [ROW_A],
         check_full=True,
```

```diff
--- before
+++ after
@@ -1 +1,2 @@
+import hashlib
 import json
```

## 2026-10-22T06:19:53.401Z `testbed/models/r2_embed.py`

```diff
--- before
+++ after
@@ -1,3 +1,5 @@
 tokens, checked with the model's own tokenizer before any forward pass. The MLX child's `embed`
 then runs in chunks of `CHUNK` rows, so a crash loses at most one chunk: a rerun reuses every
-chunk already written and checked.
+chunk already written and checked. A chunk is reused only if its result names the current child
+script, the exact rows text of the chunk, the current model snapshot and the current layers
+(`check_chunk`); otherwise it is recomputed.
```

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-`LAYERS`, plus `ids`, in render order) and `<world>.json` (provenance and throughput). The chunk
-directory
+`LAYERS`, plus `ids`, in render order) and `<world>.json` (provenance, throughput, and the sha256 of
+the child script, of the rows text and of the npz). The chunk directory
```

```diff
--- before
+++ after
@@ -1,9 +1,32 @@
-def _load_part(work: Path, tag: str, ids: Sequence[str]) -> dict[str, np.ndarray] | None:
-    """A finished chunk's arrays, or None if it is missing or fails the checks."""
+def layer_names() -> list[str]:
+    return [x.strip() for x in LAYERS.split(",")]
+
+
+def check_chunk(
+    result: Mapping[str, Any], rows: Sequence[local_llm.Row], model: str, tag: str
+) -> None:
+    """Raise `RuntimeError` unless a chunk's child result was made from exactly `rows` (the rows
+    text, by hash), by the current child script, from the current `model` snapshot and for the
+    current `LAYERS`."""
+    local_llm.check_child_identity(f"embed {tag}", result, rows)
+    if result.get("model_dir") != str(local_llm.model_dir(model)):
+        raise RuntimeError(f"{tag}: made from {result.get('model_dir')!r}, not the current snapshot")
+    if result.get("layers") != layer_names():
+        raise RuntimeError(f"{tag}: made for layers {result.get('layers')!r}, not {layer_names()}")
+
+
+def _load_part(
+    work: Path, tag: str, rows: Sequence[local_llm.Row], model: str
+) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
+    """A finished chunk's (child result, arrays), or None if it is missing, fails the structural
+    checks or was not made from these rows, this child, this snapshot and these layers."""
     npz, js = work / f"{tag}.npz", work / f"{tag}.json"
     if not (npz.exists() and js.exists()):
         return None
     try:
-        return local_llm.check_embeddings(npz, ids, [x.strip() for x in LAYERS.split(",")])
-    except RuntimeError, ValueError, KeyError, OSError:
+        result = json.loads(js.read_text())
+        check_chunk(result, rows, model, tag)
+        arrays = local_llm.check_embeddings(npz, [r["id"] for r in rows], layer_names())
+    except RuntimeError, ValueError, KeyError, TypeError, OSError:
         return None
+    return dict(result), arrays
```

```diff
--- before
+++ after
@@ -1,7 +1,10 @@
-        part = _load_part(work, tag, ids[a:b])
-        if part is None:
+        loaded = _load_part(work, tag, rows[a:b], model)
+        if loaded is None:
             result, part = local_llm.embed(model, rows[a:b], LAYERS, work, tag=tag)
+            check_chunk(result, rows[a:b], model, tag)
         else:
-            result = json.loads((work / f"{tag}.json").read_text())
+            result, part = loaded
             reused += 1
+        if results and result["child_sha256"] != results[0]["child_sha256"]:
+            raise RuntimeError(f"{tag}: made by a different child script than chunk000")
         parts.append(part)
```

```diff
--- before
+++ after
@@ -2,2 +2,3 @@
     if merged["ids"].tolist() != ids:
         raise RuntimeError("merged ids differ from the render order")
+    np.savez(final_npz, **dict[str, Any](merged))
```

```diff
--- before
+++ after
@@ -1,2 +1,5 @@
         "prompt_sha256": prompt.PROMPT_SHA256,
+        "child_sha256": results[0]["child_sha256"],  # every chunk agrees (checked above)
+        "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
+        "npz_sha256": hashlib.sha256(final_npz.read_bytes()).hexdigest(),
         "layers": results[0]["layers"],
```

```diff
--- before
+++ after
@@ -1,2 +1 @@
-    np.savez(final_npz, **dict[str, Any](merged))
     final_json.write_text(
```

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
 import argparse
+import hashlib
 import json
```

## 2026-10-22T06:20:03.682Z `testbed/models/r2_embed.py, testbed/models/r1_run.py`

```diff
--- before
+++ after
@@ -1,10 +1,16 @@
+def check_source(
+    result: Mapping[str, Any], rows: Sequence[local_llm.Row], model: str, tag: str
+) -> None:
+    """Raise `RuntimeError` unless a chunk's child result was made from exactly `rows` (the rows
+    text, by hash), by the current child script and from the current `model` snapshot."""
+    local_llm.check_child_identity(tag, result, rows)
+    if result.get("model_dir") != str(local_llm.model_dir(model)):
+        raise RuntimeError(f"{tag}: made from {result.get('model_dir')!r}, not the current snapshot")
+
+
 def check_chunk(
     result: Mapping[str, Any], rows: Sequence[local_llm.Row], model: str, tag: str
 ) -> None:
-    """Raise `RuntimeError` unless a chunk's child result was made from exactly `rows` (the rows
-    text, by hash), by the current child script, from the current `model` snapshot and for the
-    current `LAYERS`."""
-    local_llm.check_child_identity(f"embed {tag}", result, rows)
-    if result.get("model_dir") != str(local_llm.model_dir(model)):
-        raise RuntimeError(f"{tag}: made from {result.get('model_dir')!r}, not the current snapshot")
+    """`check_source`, and the result must be for the current `LAYERS`."""
+    check_source(result, rows, model, tag)
     if result.get("layers")
```

```diff
--- before
+++ after
@@ -1,6 +1,7 @@
 def _load_part(
-    work: Path, tag: str, rows: Sequence[Mapping[str, str]], check_full: bool
+    work: Path, tag: str, rows: Sequence[Mapping[str, str]], check_full: bool, model: str
 ) -> dict[str, Any] | None:
-    """A finished chunk's child result, or None if it is missing or fails the checks."""
+    """A finished chunk's child result, or None if it is missing, fails the structural checks or
+    was not made from these rows, this child script and this model snapshot."""
     path = work / f"{tag}.json"
     if not path.exists():
@@ -8,3 +9,4 @@
     try:
         result = json.loads(path.read_text())
+        r2_embed.check_source(result, rows, model, f"zeroshot {tag}")
         local_llm.check_zeroshot(result, rows, check_full)
```

```diff
--- before
+++ after
@@ -1,6 +1,9 @@
-        result = _load_part(work, tag, rows[a:b], check_full)
+        result = _load_part(work, tag, rows[a:b], check_full, model)
         if result is None:
             result = local_llm.zeroshot(model, rows[a:b], check_full, work_dir=work, tag=tag)
+            r2_embed.check_source(result, rows[a:b], model, f"zeroshot {tag}")
         else:
             reused += 1
+        if results and result["child_sha256"] != results[0]["child_sha256"]:
+            raise RuntimeError(f"{tag}: made by a different child script than chunk000")
         results.append(result)
```

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
         "prompt_sha256": prompt.PROMPT_SHA256,
+        "child_sha256": results[0]["child_sha256"],  # every chunk agrees (checked above)
+        "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
         "head_dtype": "float32",
```

## 2026-10-22T06:20:11.592Z `testbed/models/r2_embed.py`

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-    check_source(result, rows, model, tag)
+    check_source(result, rows, model, f"embed {tag}")
     if result.get("layers")
```

## 2026-10-22T06:20:27.765Z `testbed/models/nimble_run.py`

```diff
--- before
+++ after
@@ -1 +1,6 @@
+def child_sha256() -> str:
+    """The sha256 of the Nimble child script as it is now."""
+    return hashlib.sha256(CHILD_SCRIPT.read_bytes()).hexdigest()
+
+
 def schema_text() -> str:
```

```diff
--- before
+++ after
@@ -1,3 +1,5 @@
     """Raise `RuntimeError` unless a child result's provenance names the pinned Nimble commit,
-    adapter release and weights, the frozen schema, and the fixed scoring settings."""
+    adapter release and weights, the frozen schema, the fixed scoring settings and this child
+    script as it is now."""
     expected = {
+        "child_sha256": child_sha256(),
```

```diff
--- before
+++ after
@@ -1 +1,7 @@
-    keys = ("nimble_commit", "adapter_revision", "adapter_sha256", "contract_sha256")
+    keys = (
+        "nimble_commit",
+        "adapter_revision",
+        "adapter_sha256",
+        "contract_sha256",
+        "child_sha256",
+    )
```

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
-    """A finished chunk's child result, or None if it is missing or fails the checks."""
+    """A finished chunk's child result, or None if it is missing, or fails the checks: the pinned
+    provenance with this child script, and the sha256 of exactly these rows' text."""
     path = work / f"{tag}.json"
```

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
         "model": "nimble",
         **_provenance_record(counted["provenance"]),
+        "rows_sha256": counted["rows_sha256"],  # checked against the rows text in `count`
         "limit": limit,
```

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
     if len(provenance) != 1:
         raise RuntimeError("chunks were scored under different packages or settings")
+    if results[0]["provenance"]["child_sha256"] != base["child_sha256"]:
+        raise RuntimeError("chunks were scored by a different child script than the count")
```

## 2026-10-22T06:20:36.991Z `testbed/models/nimble_run.py`

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
         **base,
+        "child_sha256": results[0]["provenance"]["child_sha256"],  # every chunk agrees
         "count_only": False,
         "chunks": len(results),
```

## 2026-10-22T06:20:56.165Z `testbed/eval/r2.py, testbed/eval/r1.py`

```diff

```

```diff
--- before
+++ after
@@ -1 +1,3 @@
-is checked against the frozen prompt, the world's checksum and the model revision before use.
+is checked before use against the frozen prompt, the world's checksum and the model revision, the
+sha256 of the world's rows text (prompt and renders as they are now), of the current MLX child
+script and of the npz file.
```

```diff
--- before
+++ after
@@ -1,17 +1,58 @@
-def check_provenance(meta: Mapping[str, Any], world_dir: Path, model: str) -> None:
-    """Raise unless the embedding's json matches the frozen prompt, the world and the model."""
+def world_rows_sha256(world_dir: Path, limit: int | None = None) -> str:
+    """The sha256 of the rows text the MLX child reads for `world_dir` (the first `limit` renders
+    if given), as `r2_embed` and `r1_run` write it."""
+    renders = [json.loads(x) for x in (world_dir / "renders.jsonl").read_text().splitlines() if x]
+    rows = prompt.child_rows(renders[:limit] if limit is not None else renders)
+    return local_llm.sha256_text(local_llm.rows_text(rows))
+
+
+def _require(meta: Mapping[str, Any], field: str, world_dir: Path, what: str) -> Any:
+    if not meta.get(field):
+        raise ValueError(f"{world_dir.name}: {what} has no {field}")
+    return meta[field]
+
+
+def check_provenance(
+    meta: Mapping[str, Any], world_dir: Path, model: str, *, limit: int | None = None, what: str
+) -> None:
+    """Raise unless the run json `meta` matches the frozen prompt, the world, the model, the
+    world's rows text (the first `limit` renders for a limited run) and the current MLX child
+    script. A missing field is a refusal."""
     if meta.get("prompt_sha256") != prompt.PROMPT_SHA256:
-        raise ValueError(f"{world_dir.name}: embedding was made with a different prompt")
+        raise ValueError(f"{world_dir.name}: {what} was made with a different prompt")
     checksum = json.loads((world_dir / "manifest.json").read_text())["checksum_sha256"]
     if meta.get("world_checksum_sha256") != checksum:
-        raise ValueError(f"{world_dir.name}: embedding is of a different world (checksum differs)")
+        raise ValueError(f"{world_dir.name}: {what} is of a different world (checksum differs)")
     if meta.get("model_revision") != local_llm.model_revision(model):
-        raise ValueError(f"{world_dir.name}: embedding is from a different {model} revision")
+        raise ValueError(f"{world_dir.name}: {what} is from a different {model} revision")
+    if _require(meta, "rows_sha256", world_dir, what) != world_rows_sha256(world_dir, limit):
+        raise ValueError(f"{world_dir.name}: {what} rows_sha256 differs from the world's rows now")
+    if _require(meta, "child_sha256", world_dir, what) != local_llm.child_sha256():
+        raise ValueError(f"{world_dir.name}: {what} was made by a different mlx_child.py")
+
+
+def check_npz_sha256(meta: Mapping[str, Any], npz: Path, world_dir: Path) -> None:
+    """Raise unless the embedding json's `npz_sha256` is the hash of `npz` as it is now."""
+    want = _require(meta, "npz_sha256", world_dir, "embedding")
+    if hashlib.sha256(npz.read_bytes()).hexdigest() != want:
+        raise ValueError(f"{world_dir.name}: embedding npz_sha256 differs from {npz.name}")
+
+
+PROVENANCE_FIELDS = (
+    "prompt_sha256",
+    "model_revision",
+    "world_checksum_sha256",
+    "layers",
+    "rows_sha256",
+    "child_sha256",
+    "npz_sha256",
+)
 
 
 def embedding_provenance(embed_dir: Path, model: str, world_dir: Path) -> dict[str, Any]:
     """The embedding json's fields that identify what it was computed from."""
-    meta = json.loads((embed_dir / model / f"{world_dir.resolve().name}.json").read_text())
-    return {
-        k: meta[k] for k in ("prompt_sha256", "model_revision", "world_checksum_sha256", "layers")
-    }
+    world_dir = world_dir.resolve()
+    meta = json.loads((embed_dir / model / f"{world_dir.name}.json").read_text())
+    for field in PROVENANCE_FIELDS:
+        _require(meta, field, world_dir, "embedding")
+    return {k: meta[k] for k in PROVENANCE_FIELDS}
```

```diff
--- before
+++ after
@@ -1,3 +1,5 @@
     Raises if the embedding json's prompt hash, world checksum or model revision disagree with
-    the frozen prompt, the world's manifest or the local snapshot, if a unit has no row, if the ids
-    repeat, or if a value is not finite."""
+    the frozen prompt, the world's manifest or the local snapshot; if its rows hash differs from
+    the world's rows text now, its child hash from the current `mlx_child.py` or its npz hash from
+    the npz file; if any of these fields is missing; if a unit has no row, if the ids repeat, or
+    if a value is not finite."""
```

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
-    check_provenance(meta, world_dir, model)
+    check_provenance(meta, world_dir, model, what="embedding")
+    check_npz_sha256(meta, Path(f"{stem}.npz"), world_dir)
     with np.load(Path(f"{stem}.npz"), allow_pickle=False) as data:
```

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
 import argparse
+import hashlib
 import json
```

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-        r2.check_provenance(meta, world_dir, model)
+        r2.check_provenance(meta, world_dir, model, limit=meta.get("limit"), what="R1 run")
         if meta.get("head_dtype")
```

```diff
--- before
+++ after
@@ -1,2 +1,6 @@
     if meta.get("adapter_revision") != nimble_child.ADAPTER_REVISION:
         raise ValueError(f"{world_dir.name}: Nimble run is from a different adapter revision")
+    if not meta.get("child_sha256"):
+        raise ValueError(f"{world_dir.name}: Nimble run has no child_sha256")
+    if meta["child_sha256"] != nimble_run.child_sha256():
+        raise ValueError(f"{world_dir.name}: Nimble run was made by a different nimble_child.py")
```

```diff
--- before
+++ after
@@ -1 +1 @@
-from testbed.models import nimble_child, prompt
+from testbed.models import nimble_child, nimble_run, prompt
```

```diff
--- before
+++ after
@@ -1,3 +1,5 @@
-    prompt, the world's manifest or the local snapshot, if the head was not float32, if a unit has
-    no row, if the ids repeat, or if a score is not finite. For `nimble` the checks are the schema
-    hash, world checksum and pinned adapter instead (`check_nimble_provenance`)."""
+    prompt, the world's manifest or the local snapshot, if its rows hash differs from the world's
+    rows text now (the first `limit` renders for a limited run) or its child hash from the current
+    `mlx_child.py`, if the head was not float32, if a unit has no row, if the ids repeat, or if a
+    score is not finite. For `nimble` the checks are the schema hash, world checksum, pinned
+    adapter and the current `nimble_child.py` hash instead (`check_nimble_provenance`)."""
```

## 2026-10-22T06:21:37.447Z `tests/test_r2_embed.py`

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
+import hashlib
 import json
 from collections
```

UNPARSED part: the script verbatim:

```python
from pathlib import Path
p = Path("tests/test_r2_embed.py")
s = p.read_text()
start = s.index("HIDDEN = 3")
end = s.index("def test_run_refuses_an_existing_output")
new = '''HIDDEN = 3
NAMES = [x.strip() for x in r2_embed.LAYERS.split(",")]
OLD_CHILD, NEW_CHILD = "a" * 64, "b" * 64


def _vec(text: str) -> float:
    """What the fake child makes of a render: a number that depends on its text, not its id."""
    return float(int(hashlib.sha256(text.encode()).hexdigest()[:6], 16))


def _rows_sha256(texts: Sequence[str], start: int = 0) -> str:
    """The rows text of units u<start>.. written out by hand, hashed independently of the code."""
    lines = [
        json.dumps({"id": f"u{start + i:02d}", "before": prompt.BEFORE, "render": t, "after": prompt.AFTER})
        for i, t in enumerate(texts)
    ]
    return hashlib.sha256("".join(line + "\\n" for line in lines).encode()).hexdigest()


def _world(tmp_path: Path, n: int) -> Path:
    world = tmp_path / "w-s0.75-seed9"
    world.mkdir()
    manifest = {"checksum_sha256": "c" * 64, "generator_version": "tt-gen-v1"}
    (world / "manifest.json").write_text(json.dumps(manifest))
    lines = [json.dumps({"unit": f"u{i:02d}", "text": f"render {i}"}) for i in range(n)]
    (world / "renders.jsonl").write_text("\\n".join(lines) + "\\n")
    return world


class _FakeChild:
    def __init__(self, n_prompt: int = 100) -> None:
        self.n_prompt = n_prompt
        self.child_sha256 = OLD_CHILD  # the "current" mlx_child.py hash, and what a new chunk reports
        self.switch_child_at_call: int | None = None  # edit the child script before this embed call
        self.embed_calls: list[str] = []

    def count(self, model_key: str, rows: Sequence[dict[str, str]], **kw: Any) -> dict[str, Any]:
        assert all(r["before"] == prompt.BEFORE and r["after"] == prompt.AFTER for r in rows)
        return {"rows": [{"id": r["id"], "n_prompt": self.n_prompt} for r in rows]}

    def embed(
        self, model_key: str, rows: Sequence[dict[str, str]], layers: str, work_dir: Path, tag: str
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        assert layers == r2_embed.LAYERS
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


def _stale_chunk(
    tmp_path: Path, fake: _FakeChild, world: Path, texts: Sequence[str]
) -> tuple[Path, int]:
    """Leave chunk000 of `world` in the parts directory, made by the fake from `texts` (u00, u01)."""
    work = tmp_path / "out" / "9b" / f"{world.name}.parts"
    work.mkdir(parents=True)
    rows = prompt.child_rows([{"unit": f"u{i:02d}", "text": t} for i, t in enumerate(texts)])
    fake.embed("9b", rows, r2_embed.LAYERS, work, "chunk000")
    fake.embed_calls.clear()
    return work, 1


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
    work, _ = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
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
    work, _ = _stale_chunk(tmp_path, fake, world, ["render 0", "render 1"])
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


'''
s = s[:start] + new + s[end:]
s = s.replace("import json\nfrom collections", "import hashlib\nimport json\nfrom collections", 1)
p.write_text(s)
```

## 2026-10-22T06:21:42.451Z `tests/test_r2_embed.py`

```diff
--- before
+++ after
@@ -1 +1 @@
-work, _ = _stale_chunk
+work = _stale_chunk
```

```diff
--- before
+++ after
@@ -1 +1 @@
-    return work, 1
+    return work
```

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-) -> tuple[Path, int]:
+) -> Path:
     """Leave chunk000
```

## 2026-10-22T06:22:02.494Z `tests/test_r2.py`

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
+import hashlib
 import json
 import zlib
```

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
 REVISION = "rev-test"
+CHILD = "c" * 64  # the "current" mlx_child.py hash
 DIM = 6
 
@@ -6,2 +7,3 @@
 def _revision(monkeypatch: pytest.MonkeyPatch) -> None:
     monkeypatch.setattr(local_llm, "model_revision", lambda key: REVISION)
+    monkeypatch.setattr(local_llm, "child_sha256", lambda: CHILD)
```

```diff
--- before
+++ after
@@ -3,3 +3,19 @@
     path.mkdir(parents=True)
     (path / "manifest.json").write_text(json.dumps({"checksum_sha256": checksum, "s": 0.75}))
+    write_renders(path, [f"render {i} of {name}" for i in range(3)])
     return path
+
+
+def write_renders(world: Path, texts: Sequence[str]) -> None:
+    lines = [json.dumps({"unit": f"{world.name}-r{i}", "text": t}) for i, t in enumerate(texts)]
+    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")
+
+
+def rows_sha256_of(world: Path) -> str:
+    """The hash of the rows text `r2_embed` would write for `world`, built here by hand."""
+    out = ""
+    for line in (world / "renders.jsonl").read_text().splitlines():
+        r = json.loads(line)
+        row = {"id": r["unit"], "before": prompt.BEFORE, "render": r["text"], "after": prompt.AFTER}
+        out += json.dumps(row) + "\n"
+    return hashlib.sha256(out.encode()).hexdigest()
```

```diff
--- before
+++ after
@@ -1,4 +1,9 @@
     revision: str = REVISION,
+    rows_sha: str | None = None,
+    child_sha: str = CHILD,
+    npz_sha: str | None = None,
+    drop: Sequence[str] = (),
 ) -> None:
     """Write `<embed>/<model>/<world>.npz` + `.json` as `r2_embed` does (rows in render order,
-    here the reverse of `units` if `reverse`, plus ids of units that aren't scored)."""
+    here the reverse of `units` if `reverse`, plus ids of units that aren't scored). `rows_sha`
+    and `npz_sha` default to the true hashes; `drop` leaves fields out of the json."""
```

```diff
--- before
+++ after
@@ -1,11 +1,13 @@
     np.savez(dest / f"{world.name}.npz", **arrays)
-    (dest / f"{world.name}.json").write_text(
-        json.dumps(
-            {
-                "prompt_sha256": prompt_sha,
-                "model_revision": revision,
-                "world_checksum_sha256": checksum,
-                "layers": ["16", "final"],
-            }
-        )
-    )
+    meta = {
+        "prompt_sha256": prompt_sha,
+        "model_revision": revision,
+        "world_checksum_sha256": checksum,
+        "layers": ["16", "final"],
+        "rows_sha256": rows_sha or rows_sha256_of(world),
+        "child_sha256": child_sha,
+        "npz_sha256": npz_sha or hashlib.sha256((dest / f"{world.name}.npz").read_bytes()).hexdigest(),
+    }
+    for field in drop:
+        del meta[field]
+    (dest / f"{world.name}.json").write_text(json.dumps(meta))
```

```diff
--- before
+++ after
@@ -1,3 +1,9 @@
         ({"revision": "rev-old"}, "different 9b revision"),
+        ({"rows_sha": "0" * 64}, "rows_sha256 differs"),
+        ({"child_sha": "0" * 64}, "different mlx_child.py"),
+        ({"npz_sha": "0" * 64}, "npz_sha256 differs"),
+        ({"drop": ["rows_sha256"]}, "no rows_sha256"),
+        ({"drop": ["child_sha256"]}, "no child_sha256"),
+        ({"drop": ["npz_sha256"]}, "no npz_sha256"),
     ],
 )
```

```diff
--- before
+++ after
@@ -1 +1,24 @@
+def test_load_features_refuses_rows_changed_after_embedding(tmp_path: Path) -> None:
+    world = make_world(tmp_path, "w1")
+    units = make_units("w1")
+    write_embedding(tmp_path / "e", "9b", world, units, {"mean_final": planted(units, signal=False)})
+    r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")  # fine as embedded
+    write_renders(world, ["render 0 of w1", "edited", "render 2 of w1"])  # same ids, other text
+    with pytest.raises(ValueError, match="w1: embedding rows_sha256 differs"):
+        r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")
+
+
+def test_load_features_refuses_a_replaced_npz(tmp_path: Path) -> None:
+    world = make_world(tmp_path, "w1")
+    units = make_units("w1")
+    write_embedding(tmp_path / "e", "9b", world, units, {"mean_final": planted(units, signal=False)})
+    npz = tmp_path / "e" / "9b" / "w1.npz"
+    with np.load(npz) as data:
+        arrays = {k: np.array(data[k]) for k in data.files}
+    arrays["mean_final"] = arrays["mean_final"] + 1.0  # same ids and shapes, other values
+    np.savez(npz, **arrays)
+    with pytest.raises(ValueError, match="w1: embedding npz_sha256 differs"):
+        r2.load_features(tmp_path / "e", "9b", world, units, "mean_final")
+
+
 def test_load_features_refuses_missing_duplicate_and_non_finite_rows(
```

```diff
--- before
+++ after
@@ -1 +1,6 @@
     assert result["embedding"]["model_revision"] == REVISION
+    assert result["embedding"]["child_sha256"] == CHILD
+    assert result["embedding"]["rows_sha256"] == rows_sha256_of(tune_world)
+    assert result["embedding"]["npz_sha256"] == hashlib.sha256(
+        (tmp_path / "e" / "9b" / f"{tune_world.name}.npz").read_bytes()
+    ).hexdigest()
```

## 2026-10-22T06:22:07.025Z `tests/test_r2.py`

```diff
--- before
+++ after
@@ -1,3 +1,5 @@
     world, embed = tuning_setup
     result = r2.tune(world, embed, "9b")
+    tune_world = world
+    tmp_path = embed.parent
     grid =
```

```diff
--- before
+++ after
@@ -1 +1 @@
-rows_sha256_of(tune_world)
+rows_sha256_of(world)
```

```diff
--- before
+++ after
@@ -1,2 +0,0 @@
-    tune_world = world
-    tmp_path = embed.parent
```

```diff
--- before
+++ after
@@ -1,3 +1,3 @@
     assert result["embedding"]["npz_sha256"] == hashlib.sha256(
-        (tmp_path / "e" / "9b" / f"{tune_world.name}.npz").read_bytes()
+        (embed / "9b" / f"{world.name}.npz").read_bytes()
     ).hexdigest()
```

## 2026-10-22T06:22:54.310Z `testbed/models/nimble_run.py, tests/test_nimble_run.py`

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
+    if len({res["provenance"]["child_sha256"] for res in results}) != 1:
+        raise RuntimeError("chunks were scored by different child scripts")
     if len(provenance) != 1:
         raise RuntimeError("chunks were scored under different packages or settings")
```

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
         "nimble_schema_sha256": prompt.NIMBLE_SCHEMA_SHA256,
+        "child_sha256": nimble_run.child_sha256(),
     }
     return prov | override
```

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
     write([{"id": "a", "context": "x"}, {"id": "b", "context": "y"}])
-    assert [r["id"] for r in nimble_child.read_rows(path)] == ["a", "b"]
+    rows, rows_sha256 = nimble_child.read_rows(path)
+    assert [r["id"] for r in rows] == ["a", "b"]
+    assert rows_sha256 == nimble_run.sha256_text(nimble_run.rows_text(rows))  # the bytes as read
```

```diff
--- before
+++ after
@@ -1 +1,7 @@
+def test_the_child_hashes_its_own_source() -> None:
+    source = Path(nimble_child.__file__).read_bytes()
+    assert nimble_child.CHILD_SHA256 == hashlib.sha256(source).hexdigest()
+    assert nimble_run.child_sha256() == nimble_child.CHILD_SHA256
+
+
 def test_the_child_parser_takes_the_two_commands() -> None:
```

```diff
--- before
+++ after
@@ -2,4 +2,5 @@
         "temperature",
         "mode",
+        "child_sha256",
     ],
 )
```

```diff
--- before
+++ after
@@ -1 +1,3 @@
         lambda r: r.update(rows_sha256="0" * 64),  # another world's renders under the same ids
+        lambda r: r["provenance"].update(child_sha256="0" * 64),  # nimble_child.py edited since
+        lambda r: r["provenance"].pop("child_sha256"),  # a chunk that predates the binding
```

```diff
--- before
+++ after
@@ -1 +1,50 @@
+def test_run_redoes_a_chunk_made_from_other_render_text_under_the_same_ids(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    out = tmp_path / "r1"
+    install(monkeypatch, FakeNimble(fail_on_call=2))
+    with pytest.raises(RuntimeError, match="child died"):
+        nimble_run.run(small_world, out)
+    renders = [{"unit": f"u{i}", "text": f"edited {i}"} for i in range(5)]
+    (small_world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
+    resumed = FakeNimble()
+    install(monkeypatch, resumed)
+    record = nimble_run.run(small_world, out)
+    assert resumed.calls[0] == ["u0", "u1"] and record["chunks_reused"] == 0
+    want = [{"id": f"u{i}", "context": f"edited {i}"} for i in range(5)]
+    assert record["rows_sha256"] == nimble_run.sha256_text(nimble_run.rows_text(want))
+
+
+def test_run_records_the_child_and_rows_hashes(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    install(monkeypatch, FakeNimble())
+    record = nimble_run.run(small_world, tmp_path / "r1")
+    independent = "".join(
+        json.dumps({"id": f"u{i}", "context": f"render {i}"}) + "\n" for i in range(5)
+    )
+    assert record["rows_sha256"] == hashlib.sha256(independent.encode()).hexdigest()
+    assert record["child_sha256"] == hashlib.sha256(nimble_run.CHILD_SCRIPT.read_bytes()).hexdigest()
+    assert json.loads((tmp_path / "r1" / "nimble" / "w1.json").read_text()) == record
+
+
+def test_run_refuses_chunks_scored_by_different_child_scripts(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    script = tmp_path / "nimble_child.py"
+    script.write_text("# v1\n")
+    monkeypatch.setattr(nimble_run, "CHILD_SCRIPT", script)
+
+    class Editing(FakeNimble):
+        def score(self, rows: Sequence[Any], **kw: Any) -> dict[str, Any]:
+            if len(self.calls) == 1:
+                script.write_text("# v2\n")  # edited between chunks 0 and 1
+            return super().score(rows, **kw)
+
+    install(monkeypatch, Editing())
+    with pytest.raises(RuntimeError, match="different child scripts"):
+        nimble_run.run(small_world, tmp_path / "r1")
+    assert not (tmp_path / "r1" / "nimble" / "w1.json").exists()
+
+
 def test_run_refuses_chunks_scored_under_different_packages(
```

## 2026-10-22T06:23:38.650Z `tests/test_r1.py`

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
+import hashlib
 import json
 import zlib
```

```diff
--- before
+++ after
@@ -1 +1 @@
-from testbed.models import local_llm, mlx_child, nimble_child, prompt, r1_run
+from testbed.models import local_llm, mlx_child, nimble_child, nimble_run, prompt, r1_run
```

```diff
--- before
+++ after
@@ -1,3 +1,5 @@
 REVISION = "rev-test"
+CHILD = "c" * 64  # the "current" mlx_child.py hash
+OLD_CHILD = "d" * 64
 
 
@@ -5,2 +7,4 @@
 def _revision(monkeypatch: pytest.MonkeyPatch) -> None:
     monkeypatch.setattr(local_llm, "model_revision", lambda key: REVISION)
+    monkeypatch.setattr(local_llm, "model_dir", lambda key: Path("/models") / f"snap-{key}")
+    monkeypatch.setattr(local_llm, "child_sha256", lambda: CHILD)
```

```diff
--- before
+++ after
@@ -1,5 +1,21 @@
     (path / "manifest.json").write_text(json.dumps(manifest))
+    write_renders(path, [f"render {i} of {name}" for i in range(3)])
     return path
 
 
+def write_renders(world: Path, texts: Sequence[str], unit: str = "r") -> None:
+    lines = [json.dumps({"unit": f"{unit}{i}", "text": t}) for i, t in enumerate(texts)]
+    (world / "renders.jsonl").write_text("\n".join(lines) + "\n")
+
+
+def rows_sha256_of(world: Path, limit: int | None = None) -> str:
+    """The hash of the rows text `r1_run` would write for `world`, built here by hand."""
+    out = ""
+    for line in (world / "renders.jsonl").read_text().splitlines()[:limit]:
+        r = json.loads(line)
+        row = {"id": r["unit"], "before": prompt.BEFORE, "render": r["text"], "after": prompt.AFTER}
+        out += json.dumps(row) + "\n"
+    return hashlib.sha256(out.encode()).hexdigest()
+
+
 def level_row
```

```diff
--- before
+++ after
@@ -1,3 +1,7 @@
     head_dtype: str = "float32",
+    rows_sha: str | None = None,
+    child_sha: str = CHILD,
+    drop: Sequence[str] = (),
+    limit: int | None = None,
 ) -> None:
     order = list(range(len(units)))[::-1] if reverse else list(range(len(units)))
```

```diff
--- before
+++ after
@@ -1,12 +1,13 @@
-    (dest / f"{world.name}.json").write_text(
-        json.dumps(
-            {
-                "prompt_sha256": prompt_sha,
-                "model_revision": revision,
-                "world_checksum_sha256": checksum,
-                "head_dtype": head_dtype,
-                "limit": None,
-                "agreement": {"n_distinct_expected": len(set(expected.tolist()))},
-            }
-        )
-    )
+    meta = {
+        "prompt_sha256": prompt_sha,
+        "model_revision": revision,
+        "world_checksum_sha256": checksum,
+        "head_dtype": head_dtype,
+        "rows_sha256": rows_sha or rows_sha256_of(world, limit),
+        "child_sha256": child_sha,
+        "limit": limit,
+        "agreement": {"n_distinct_expected": len(set(expected.tolist()))},
+    }
+    for field in drop:
+        del meta[field]
+    (dest / f"{world.name}.json").write_text(json.dumps(meta))
```

```diff
--- before
+++ after
@@ -3,4 +3,6 @@
         self.fail_on_call = fail_on_call
         self.n_prompt = n_prompt
+        self.child_sha256 = CHILD  # what a chunk made now reports
+        self.switch_child_at_call: int | None = None  # edit the child before this zeroshot call
 
     def count(self, model: str
```

```diff
--- before
+++ after
@@ -1,2 +1,4 @@
+        if self.switch_child_at_call == len(self.calls):
+            self.child_sha256 = OLD_CHILD
         self.calls.append([r["id"] for r in rows])
         if self.fail_on_call == len(self.calls):
```

```diff
--- before
+++ after
@@ -1,3 +1,6 @@
         result = {
+            "model_dir": str(local_llm.model_dir(model)),
+            "child_sha256": self.child_sha256,
+            "rows_sha256": local_llm.sha256_text(local_llm.rows_text(rows)),
             "head_dtype": "float32",
             "check_full": check_full,
```

```diff
--- before
+++ after
@@ -1 +1,91 @@
+def _crashed_run(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> tuple[Path, Path]:
+    """Run until the third zeroshot call dies: chunks 0 and 1 are left in the parts directory."""
+    out = tmp_path / "r1"
+    install(monkeypatch, FakeChild(fail_on_call=3))
+    with pytest.raises(RuntimeError, match="child died"):
+        r1_run.run(small_world, "9b", out)
+    return out, out / "9b" / "w1.parts"
+
+
+def test_run_reuses_chunks_made_from_the_same_rows_child_and_snapshot(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    out, _ = _crashed_run(small_world, tmp_path, monkeypatch)
+    resumed = FakeChild()
+    install(monkeypatch, resumed)
+    record = r1_run.run(small_world, "9b", out)
+    assert resumed.calls == [["u4"]] and record["chunks_reused"] == 2
+
+
+def test_run_redoes_every_chunk_when_the_render_text_changed_under_the_same_ids(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    out, _ = _crashed_run(small_world, tmp_path, monkeypatch)
+    renders = [{"unit": f"u{i}", "text": f"edited {i}"} for i in range(5)]
+    (small_world / "renders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in renders))
+    resumed = FakeChild()
+    install(monkeypatch, resumed)
+    record = r1_run.run(small_world, "9b", out)
+    assert resumed.calls == [["u0", "u1"], ["u2", "u3"], ["u4"]] and record["chunks_reused"] == 0
+    assert record["rows_sha256"] == rows_sha256_of(small_world)
+
+
+@pytest.mark.parametrize(
+    "stale",
+    [
+        {"child_sha256": OLD_CHILD},  # mlx_child.py was edited since
+        {"model_dir": "/models/other-snapshot"},
+        {"rows_sha256": "0" * 64},
+    ],
+)
+def test_run_redoes_a_chunk_that_names_another_child_snapshot_or_rows(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: dict[str, str]
+) -> None:
+    out, parts = _crashed_run(small_world, tmp_path, monkeypatch)
+    saved = json.loads((parts / "chunk000.json").read_text())
+    (parts / "chunk000.json").write_text(json.dumps({**saved, **stale}))
+    resumed = FakeChild()
+    install(monkeypatch, resumed)
+    record = r1_run.run(small_world, "9b", out)
+    assert resumed.calls == [["u0", "u1"], ["u4"]] and record["chunks_reused"] == 1
+
+
+@pytest.mark.parametrize("field", ["child_sha256", "rows_sha256", "model_dir"])
+def test_run_redoes_a_chunk_that_lacks_a_binding_field(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
+) -> None:
+    out, parts = _crashed_run(small_world, tmp_path, monkeypatch)
+    saved = json.loads((parts / "chunk001.json").read_text())
+    del saved[field]
+    (parts / "chunk001.json").write_text(json.dumps(saved))
+    resumed = FakeChild()
+    install(monkeypatch, resumed)
+    r1_run.run(small_world, "9b", out)
+    assert resumed.calls == [["u2", "u3"], ["u4"]]
+
+
+def test_run_records_the_child_and_rows_hashes(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    install(monkeypatch, FakeChild())
+    record = r1_run.run(small_world, "9b", tmp_path / "r1")
+    assert record["child_sha256"] == CHILD
+    assert record["rows_sha256"] == rows_sha256_of(small_world)
+    limited = r1_run.run(small_world, "9b", tmp_path / "r2", limit=3)
+    assert limited["rows_sha256"] == rows_sha256_of(small_world, 3)
+
+
+def test_run_refuses_chunks_made_by_different_children(
+    small_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    child = FakeChild()
+    child.switch_child_at_call = 1  # edited while the run is going
+    install(monkeypatch, child)
+    with pytest.raises(RuntimeError, match="different child script"):
+        r1_run.run(small_world, "9b", tmp_path / "r1")
+    assert not (tmp_path / "r1" / "9b" / "w1.json").exists()
+
+
 def test_run_limit_and_check_full(
```

```diff
--- before
+++ after
@@ -1,3 +1,7 @@
         ({"head_dtype": "bfloat16"}, "head dtype"),
+        ({"rows_sha": "0" * 64}, "rows_sha256 differs"),
+        ({"child_sha": "0" * 64}, "different mlx_child.py"),
+        ({"drop": ["rows_sha256"]}, "no rows_sha256"),
+        ({"drop": ["child_sha256"]}, "no child_sha256"),
     ],
 )
```

```diff
--- before
+++ after
@@ -1 +1,21 @@
+def test_load_r1_refuses_rows_changed_after_the_run(tmp_path: Path) -> None:
+    world = make_world(tmp_path, "w1")
+    units = make_units("w1")
+    write_r1(tmp_path / "r", "9b", world, units, np.ones(len(units)))
+    r1.load_r1(tmp_path / "r", "9b", world, units)  # fine as run
+    write_renders(world, ["render 0 of w1", "edited", "render 2 of w1"])  # same ids, other text
+    with pytest.raises(ValueError, match="w1: R1 run rows_sha256 differs"):
+        r1.load_r1(tmp_path / "r", "9b", world, units)
+
+
+def test_load_r1_checks_a_limited_runs_rows_against_the_first_renders(tmp_path: Path) -> None:
+    world = make_world(tmp_path, "w1")
+    units = make_units("w1")[:2]
+    write_r1(tmp_path / "r", "9b", world, units, np.ones(2), limit=2)
+    r1.load_r1(tmp_path / "r", "9b", world, units)
+    write_r1(tmp_path / "r", "9b", world, units, np.ones(2), limit=2, rows_sha=rows_sha256_of(world))
+    with pytest.raises(ValueError, match="rows_sha256 differs"):  # the hash of all 3 renders
+        r1.load_r1(tmp_path / "r", "9b", world, units)
+
+
 def test_load_r1_refuses_missing_duplicate_and_non_finite_scores(
```

```diff
--- before
+++ after
@@ -1,3 +1,4 @@
         "mode": nimble_child.MODE,
+        "child_sha256": nimble_run.child_sha256(),
         "limit": None,
         "n_distinct_expected"
```

```diff
--- before
+++ after
@@ -1,3 +1,6 @@
         ({"mode": "independent"}, "mode"),
+        ({"child_sha256": "0" * 64}, "different nimble_child.py"),
+        ({"child_sha256": None}, "no child_sha256"),
+        ({"child_sha256": ""}, "no child_sha256"),
     ],
 )
```

## 2026-10-22T06:23:54.199Z `tests/test_r1.py`

```diff
--- before
+++ after
@@ -1,2 +1,3 @@
     child.switch_child_at_call = 1  # edited while the run is going
+    monkeypatch.setattr(local_llm, "child_sha256", lambda: child.child_sha256)
     install(monkeypatch, child)
```

## 2026-10-22T06:24:08.350Z `tests/test_local_llm.py`

UNPARSED part: the script verbatim:

```python
from pathlib import Path
p = Path("tests/test_local_llm.py")
s = p.read_text()
old = "def test_run_child_parses_json_and_sets_only_the_offline_env("
assert s.count(old)==1
s = s.replace(old, '''def test_the_child_hashes_its_own_source_and_the_rows_bytes_as_read(tmp_path: Path) -> None:
    assert mlx_child.CHILD_SHA256 == local_llm.child_sha256()
    path = tmp_path / "rows.jsonl"
    path.write_bytes(local_llm.rows_text([ROW_A, {**ROW_A, "id": "b"}]).encode())
    rows, rows_sha256 = mlx_child.read_rows(path)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows_sha256 == local_llm.sha256_text(local_llm.rows_text(rows))
    path.write_text(json.dumps({**ROW_A, "label": 1}) + "\\n")
    with pytest.raises(SystemExit, match="row keys"):
        mlx_child.read_rows(path)


''' + old, 1)
p.write_text(s)
```

## 2026-10-22T06:24:08.350Z shell command touching files (verbatim)

```
python3 - <<'EOF'
from pathlib import Path
p = Path("tests/test_local_llm.py")
s = p.read_text()
old = "def test_run_child_parses_json_and_sets_only_the_offline_env("
assert s.count(old)==1
s = s.replace(old, '''def test_the_child_hashes_its_own_source_and_the_rows_bytes_as_read(tmp_path: Path) -> None:
    assert mlx_child.CHILD_SHA256 == local_llm.child_sha256()
    path = tmp_path / "rows.jsonl"
    path.write_bytes(local_llm.rows_text([ROW_A, {**ROW_A, "id": "b"}]).encode())
    rows, rows_sha256 = mlx_child.read_rows(path)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows_sha256 == local_llm.sha256_text(local_llm.rows_text(rows))
    path.write_text(json.dumps({**ROW_A, "label": 1}) + "\\n")
    with pytest.raises(SystemExit, match="row keys"):
        mlx_child.read_rows(path)


''' + old, 1)
p.write_text(s)
EOF
.venv/bin/ruff format testbed tests 2>&1 | tail -3; .venv/bin/ruff format --check testbed tests; echo "format exit $?"; git status --short | head -30
```

## 2026-10-22T06:26:46.340Z `testbed/models/r2_embed.py, tests/test_r2_embed.py, tests/test_r2.py`

```diff
--- before
+++ after
@@ -1 +1,2 @@
-"""Leave chunk000 of `world` in the parts directory, made by the fake from `texts` (u00, u01)."""
+the child script, of the rows text and of the npz). The chunk directory `<world>.parts/` is
+removed once the merged file is written.
```

```diff
--- before
+++ after
@@ -1 +1 @@
-"""Leave chunk000 of `world` in the parts directory, made by the fake from `texts` (u00, u01)."""
+"""Leave chunk000 of `world` in the parts directory, made by the fake from `texts`."""
```

UNPARSED part: the script verbatim:

```python
from pathlib import Path
p = Path("testbed/models/r2_embed.py"); s = p.read_text()
old="the child script, of the rows text and of the npz). The chunk directory `<world>.parts/` is removed once the merged file is written."
assert old in s
s = s.replace(old, "the child script, of the rows text and of the npz). The chunk directory `<world>.parts/` is\nremoved once the merged file is written.")
p.write_text(s)
p = Path("tests/test_r2_embed.py"); s = p.read_text()
old='"""Leave chunk000 of `world` in the parts directory, made by the fake from `texts` (u00, u01)."""'
assert old in s
s = s.replace(old, '"""Leave chunk000 of `world` in the parts directory, made by the fake from `texts`."""')
p.write_text(s)
p = Path("tests/test_r2.py"); s = p.read_text()
print(s[s.index("def test_load_features_refuses_a_replaced_npz"):][:700])
```

## 2026-10-22T06:26:51.096Z `tests/test_r2.py`

```diff
--- before
+++ after
@@ -1,2 +1,2 @@
-        arrays = {k: np.array(data[k]) for k in data.files}
+        arrays: dict[str, Any] = {k: np.array(data[k]) for k in data.files}
     arrays["mean_final"] = arrays
```

## 2026-10-22T06:26:51.096Z shell command touching files (verbatim)

```
python3 - <<'EOF'
from pathlib import Path
p = Path("tests/test_r2.py"); s = p.read_text()
s = s.replace("        arrays = {k: np.array(data[k]) for k in data.files}\n    arrays[\"mean_final\"] = arrays", "        arrays: dict[str, Any] = {k: np.array(data[k]) for k in data.files}\n    arrays[\"mean_final\"] = arrays")
p.write_text(s)
EOF
.venv/bin/ruff format testbed tests | tail -1; make lint > $TMPDIR/lint.txt 2>&1; echo "lint exit $?"; tail -2 $TMPDIR/lint.txt; make typecheck > $TMPDIR/ty.txt 2>&1; echo "typecheck exit $?"; tail -2 $TMPDIR/ty.txt
```

## 2026-10-22T06:26:58.352Z `testbed/models/r2_embed.py`

```diff
--- before
+++ after
@@ -1 +1 @@
-    local_llm.check_child_identity(tag, result, rows)
+    local_llm.check_child_identity(tag, result, None)  # MUTATION: no rows comparison
```

## 2026-10-22T06:26:58.352Z shell command touching files (verbatim)

```
cp testbed/models/r2_embed.py $TMPDIR/r2_embed.orig && python3 - <<'EOF'
from pathlib import Path
p = Path("testbed/models/r2_embed.py"); s = p.read_text()
old = "    local_llm.check_child_identity(tag, result, rows)\n"
assert s.count(old) == 1
p.write_text(s.replace(old, "    local_llm.check_child_identity(tag, result, None)  # MUTATION: no rows comparison\n"))
EOF
.venv/bin/python -m pytest -q tests/test_r2_embed.py > $TMPDIR/mut.txt 2>&1; echo "mutant exit $?"; grep -E "^FAILED|passed|failed" $TMPDIR/mut.txt; cp $TMPDIR/r2_embed.orig testbed/models/r2_embed.py; .venv/bin/python -m pytest -q tests/test_r2_embed.py 2>&1 | tail -2; diff $TMPDIR/r2_embed.orig testbed/models/r2_embed.py && echo restored
```

