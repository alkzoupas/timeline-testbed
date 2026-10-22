"""Token counts with the Qwen 3.5 9B tokenizer (SPEC §6.2: render length is checked per model).

The counting runs out of process in the existing `es-p6-train` venv, which has `tokenizers`. The
main venv never imports it. The child reads only the cached `tokenizer.json`: no
`tokenizer_config.json`, no `AutoTokenizer`, no network (`HF_HUB_OFFLINE=1`). Texts go in as JSON
on stdin and counts come back as JSON on stdout.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

VENV_PYTHON = Path.home() / "venvs" / "es-p6-train" / "bin" / "python"
_MODEL_CACHE = "models--mlx-community--Qwen3.5-9B-MLX-4bit"
_SNAPSHOTS = Path.home() / ".cache" / "huggingface" / "hub" / _MODEL_CACHE / "snapshots"

# Runs in the child interpreter. `add_special_tokens=False`: count the text itself.
_CHILD = """
import json, sys
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(sys.argv[1])
texts = json.load(sys.stdin)
counts = [len(tokenizer.encode(t, add_special_tokens=False).ids) for t in texts]
json.dump(counts, sys.stdout)
"""


def tokenizer_file() -> Path:
    """The one cached `tokenizer.json`, found by globbing the snapshot revision."""
    found = sorted(_SNAPSHOTS.glob("*/tokenizer.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one cached tokenizer.json, found {len(found)}")
    return found[0]


def count_tokens(texts: Sequence[str]) -> list[int]:
    """Qwen 3.5 9B token count of each text, in order, without special tokens."""
    if not texts:
        return []
    done = subprocess.run(
        [str(VENV_PYTHON), "-I", "-c", _CHILD, str(tokenizer_file())],
        input=json.dumps(list(texts)),
        capture_output=True,
        text=True,
        env={"HF_HUB_OFFLINE": "1"},
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"token counting failed (exit {done.returncode}): {done.stderr[-500:]}")
    counts = json.loads(done.stdout)
    if not isinstance(counts, list) or len(counts) != len(texts):
        raise RuntimeError("token counting returned an unexpected result")
    return [int(c) for c in counts]
