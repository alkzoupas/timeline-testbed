"""The ground-truth boundary (SPEC §9, AGENTS.md invariants). Coordinator-owned; never delegated.

Patterned on the vendored `backend/app/tests/test_ground_truth_boundary.py`:
  1. STATIC: only `testbed/eval/labels.py` and `stores.py` import a ground-truth reader, name the
     sealed stores or touch a store's ground-truth path, and each of their openers calls
     `guard_stores` first; `testbed/world/persist.py` may import exactly the engine factory and
     the write-only sink; code outside `testbed/eval/` reaches only its label-free modules;
     label-carrying generator types stay inside `testbed/world/` and `testbed/eval/`. Imports are
     resolved to full names (relative and literal `import_module` ones included), as are
     attribute chains, calls (through import aliases) and dotted string literals; an unreadable
     dynamic import fails closed. The scan catches ordinary spellings, not constructed strings or
     reflection.
  2. RUNTIME: the inference store has no ground-truth tables and can't read them.
  3. SEAL: `open_labels` and `verify_pair` refuse a `confirm-*` world unless `prereg/M6.md` seals
     the artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.db.engine import make_inference_engine
from app.db.ground_truth_store import GroundTruth
from app.services.generator import build_manifest, generate_world
from sqlalchemy import inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from testbed.eval.labels import SealError, assign_units, check_seal, open_labels
from testbed.eval.stores import verify_pair
from testbed.world.persist import Stores, persist

REPO = Path(__file__).resolve().parents[1]
TESTBED = REPO / "testbed"
EVAL = TESTBED / "eval"
WORLD = TESTBED / "world"
PERSIST = WORLD / "persist.py"
# The guarded ground-truth openers: the only modules that may read ground truth.
GT_MODULES = {EVAL / "labels.py", EVAL / "stores.py"}
GUARDED_OPENERS = {"testbed/eval/labels.py:open_labels", "testbed/eval/stores.py:verify_pair"}

# The only ground-truth imports persist.py may make: the engine factory and the write-only sink.
PERSIST_GT_IMPORTS = {
    "app.db.ground_truth_store.make_ground_truth_engine",
    "app.db.ground_truth_writer.make_ground_truth_sink",
}
# Pure, label-free eval modules that other packages may import (labels arrive as arguments).
SHARED_EVAL_MODULES = {"testbed.eval.metrics", "testbed.eval.cv"}
SEALED_LITERALS = ("ground_truth", "split_group", "world_evaluation_receipt")
GT_PATH_ATTRIBUTES = {"ground_truth_db", "ground_truth_url"}
GT_FACTORY = "make_ground_truth_engine"
DYNAMIC = "<dynamic import>"  # an import whose target the scan can't read: always a violation
# Label-carrying generator surface: only testbed/world/ and testbed/eval/ may touch it.
LABEL_NAMES = {
    "GenLabel",
    "GenSplit",
    "GroundTruthLabel",
    "LinkageFixture",
    "generate_groups",
    "GroupDraw",
}
LABEL_ATTRIBUTES = {"labels", "split_groups", "fixtures"}
# A path for the inline specimens below; the file itself is never read.
SPECIMEN = TESTBED / "pipeline" / "specimen.py"


def _modules() -> Iterator[Path]:
    for path in sorted(TESTBED.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _rel(path: Path) -> Path:
    return path.relative_to(REPO)


def _module_name(path: Path) -> str:
    """Dotted name of a module under the repo (`testbed/eval/__init__.py` -> `testbed.eval`)."""
    parts = _rel(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _package(path: Path) -> str:
    """The package a module's relative imports resolve against (`__package__`)."""
    name = _module_name(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _resolve(node: ast.ImportFrom, path: Path) -> str:
    """The absolute module of a `from` import; a relative one resolves against `path`'s package."""
    if node.level == 0:
        return node.module or ""
    package = _package(path).split(".")
    base = package[: max(0, len(package) - node.level + 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _dotted(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    """`a.b.c` for a name or an attribute chain rooted in a name, its import alias expanded."""
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    return ".".join([aliases.get(expr.id, expr.id), *reversed(parts)])


def _dynamic_target(node: ast.Call, path: Path) -> str:
    """The module an `import_module` call loads, resolved as the interpreter would: a literal name,
    relative to a literal package or to `__package__`. Anything else is `DYNAMIC`."""
    kwargs = {k.arg: k.value for k in node.keywords}
    name = node.args[0] if node.args else kwargs.get("name")
    if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
        return DYNAMIC
    if not name.value.startswith("."):
        return name.value
    given = node.args[1] if len(node.args) > 1 else kwargs.get("package")
    if isinstance(given, ast.Constant) and isinstance(given.value, str):
        package = given.value
    elif given is None or (isinstance(given, ast.Name) and given.id == "__package__"):
        package = _package(path)
    else:
        return DYNAMIC
    try:
        return importlib.util.resolve_name(name.value, package)
    except ImportError, ValueError:
        return DYNAMIC


def _import_targets(tree: ast.Module, path: Path) -> tuple[set[str], dict[str, str]]:
    """Fully qualified import targets (`from m import x` gives `m.x`) and each bound alias. A
    dynamic import adds what it loads (`_dynamic_target`); `__import__` is always `DYNAMIC`."""
    targets: set[str] = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
                if alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(node, path)
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                targets.add(name)
                aliases[alias.asname or alias.name] = name
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = _dotted(node.func, aliases)
            if called == "importlib.import_module":
                targets.add(_dynamic_target(node, path))
            elif called is not None and called.rpartition(".")[2] == "__import__":
                targets.add(DYNAMIC)
    return targets, aliases


def _docstrings(tree: ast.Module) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _literals(tree: ast.Module) -> list[str]:
    skip = _docstrings(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


def _references(tree: ast.Module) -> list[ast.expr]:
    """Standalone names and outermost attribute chains."""
    attributes = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    inner = {id(n.value) for n in attributes}
    out: list[ast.expr] = [n for n in attributes if id(n) not in inner]
    out.extend(n for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in inner)
    return out


def _targets(tree: ast.Module, path: Path) -> set[str]:
    """Every full name the module reaches: import targets (dynamic ones included), outermost
    attribute chains rooted in a name (an import alias expanded), and dotted `testbed.` literals."""
    targets, aliases = _import_targets(tree, path)
    for node in _references(tree):
        if isinstance(node, ast.Attribute) and (dotted := _dotted(node, aliases)) is not None:
            targets.add(dotted)
    targets.update(s for s in _literals(tree) if s.startswith("testbed."))
    return targets


def _eval_violations(tree: ast.Module, path: Path) -> set[str]:
    """`testbed.eval` targets other than the shared label-free modules, and unreadable dynamic
    imports (fail closed)."""
    return {
        t
        for t in _targets(tree, path)
        if t == DYNAMIC
        or (
            (t == "testbed.eval" or t.startswith("testbed.eval."))
            and not any(t == m or t.startswith(f"{m}.") for m in SHARED_EVAL_MODULES)
        )
    }


def _gt_imports(tree: ast.Module, path: Path) -> set[str]:
    return {t for t in _import_targets(tree, path)[0] if "ground_truth" in t or t == DYNAMIC}


def _outside_eval() -> list[Path]:
    return [p for p in _modules() if not p.is_relative_to(EVAL)]


def _guard_report(tree: ast.Module, path: Path) -> tuple[set[str], list[str]]:
    """(guarded openers, violations) for one module. An opener is a function that calls the
    ground-truth engine factory; it must call the real `guard_stores` first. Calls resolve through
    import aliases, and the factory may only be called, never rebound or passed on. `persist` is
    exempt: it refuses a store that already exists (tested below)."""
    _, aliases = _import_targets(tree, path)
    guards = {"testbed.eval.labels.guard_stores"} | (
        {"guard_stores"} if path == EVAL / "labels.py" else set()
    )

    def factory(expr: ast.expr) -> bool:
        dotted = _dotted(expr, aliases)
        return dotted is not None and dotted.rpartition(".")[2] == GT_FACTORY

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    called = {id(n.func) for n in calls}
    violations = [
        f"{_rel(path)}:{n.lineno} uses {GT_FACTORY} without calling it"
        for n in _references(tree)
        if factory(n) and id(n) not in called
    ]
    engines = {id(n) for n in calls if factory(n.func)}
    guarded: set[str] = set()
    inside: set[int] = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        made = [n for n in ast.walk(func) if isinstance(n, ast.Call) and id(n) in engines]
        if not made:
            continue
        inside.update(id(n) for n in made)
        where = f"{_rel(path)}:{func.name}"
        if (path, func.name) == (PERSIST, "persist"):
            continue
        guard_lines = [
            n.lineno
            for n in ast.walk(func)
            if isinstance(n, ast.Call) and _dotted(n.func, aliases) in guards
        ]
        if guard_lines and min(guard_lines) < min(n.lineno for n in made):
            guarded.add(where)
        else:
            violations.append(f"{where} builds an engine without guard_stores first")
    if not engines <= inside:
        violations.append(f"{_rel(path)} builds an engine outside a function")
    return guarded, violations


# ---- 1. STATIC ------------------------------------------------------------------------------


def test_scan_covers_the_packages() -> None:
    scanned = {p.relative_to(TESTBED).parts[0] for p in _modules()}
    assert {"world", "pipeline", "features", "models", "eval"} <= scanned
    assert PERSIST in set(_modules()) and GT_MODULES <= set(_modules())


def test_ground_truth_imports_stay_in_the_guarded_modules() -> None:
    for path in _modules():
        gt = _gt_imports(_parse(path), path)
        if path == PERSIST:
            assert gt <= PERSIST_GT_IMPORTS, f"persist.py imports {sorted(gt - PERSIST_GT_IMPORTS)}"
        elif path not in GT_MODULES:
            assert not gt, f"{_rel(path)} imports {sorted(gt)}"


def test_eval_imports_from_outside_are_label_free() -> None:
    for path in _outside_eval():
        found = _eval_violations(_parse(path), path)
        assert not found, f"{_rel(path)} reaches {sorted(found)}"


def test_sealed_names_appear_only_in_the_guarded_modules() -> None:
    for path in _modules():
        if path in GT_MODULES:
            continue
        for literal in _literals(_parse(path)):
            hits = [s for s in SEALED_LITERALS if s in literal.lower()]
            if path == PERSIST and literal == "ground_truth.db":
                continue  # Stores.under names the file it creates for the write-only sink
            assert not hits, f"{_rel(path)} has the literal {literal!r}"


def test_ground_truth_paths_stay_in_the_guarded_modules() -> None:
    for path in _modules():
        if path in GT_MODULES or path == PERSIST:
            continue
        attributes = {n.attr for n in ast.walk(_parse(path)) if isinstance(n, ast.Attribute)}
        found = attributes & GT_PATH_ATTRIBUTES
        assert not found, f"{_rel(path)} reads {sorted(found)}"


def test_every_ground_truth_engine_is_guarded() -> None:
    guarded: set[str] = set()
    for path in _modules():
        found, violations = _guard_report(_parse(path), path)
        assert not violations, violations
        guarded |= found
    assert guarded == GUARDED_OPENERS


def test_label_surface_stays_in_world_and_eval() -> None:
    for path in _outside_eval():
        if path.is_relative_to(WORLD):
            continue
        tree = _parse(path)
        names = {t.rpartition(".")[2] for t in _import_targets(tree, path)[0]}
        assert not names & LABEL_NAMES, f"{_rel(path)} imports {names & LABEL_NAMES}"
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not attributes & LABEL_ATTRIBUTES, (
            f"{_rel(path)} reads {sorted(attributes & LABEL_ATTRIBUTES)}"
        )


@pytest.mark.parametrize(
    "source",
    [
        "from ..eval.labels import open_labels",
        "from ..eval import labels",
        "from .. import eval",
        "from ..eval import *",
        "from testbed import eval",
        "from testbed.eval import labels",
        "from testbed.eval import metrics, health",
        "from testbed.eval.labels import open_labels",
        "import testbed.eval.labels",
        "import testbed.eval.stores as s",
        "import testbed\ntestbed.eval.labels.open_labels",
        "import testbed as t\nt.eval.labels.open_labels",
        "import testbed.eval.metrics\ntestbed.eval.labels.open_labels",
        "import importlib\nimportlib.import_module('testbed.eval.labels')",
        "from importlib import import_module\nimport_module('..eval.labels', __package__)",
        "import importlib as il\nil.import_module('..eval.stores', package=__package__)",
        "import importlib\nimportlib.import_module('.labels', 'testbed.eval')",
        "from importlib import import_module\nimport_module('..eval.labels')",
        "import importlib\nimportlib.import_module(name)",
        "from importlib import import_module\nimport_module('..eval.labels', package)",
        "__import__('testbed.eval.labels')",
    ],
)
def test_eval_guard_catches_label_access(source: str) -> None:
    assert _eval_violations(ast.parse(source), SPECIMEN)


@pytest.mark.parametrize(
    "source",
    [
        "from ..eval.metrics import auc",
        "from ..eval import metrics, cv",
        "from testbed.eval import metrics",
        "from testbed.eval.cv import oof_scores",
        "import testbed.eval.metrics\ntestbed.eval.metrics.auc",
        "from ..features import f12",
        "from . import render",
        "from importlib import import_module\nimport_module('..eval.metrics', __package__)",
        "import importlib\nimportlib.import_module('testbed.features.f12')",
    ],
)
def test_eval_guard_allows_the_shared_modules(source: str) -> None:
    assert not _eval_violations(ast.parse(source), SPECIMEN)


GT_IMPORT = "from app.db.ground_truth_store import make_ground_truth_engine"
GUARD_IMPORT = "from testbed.eval.labels import guard_stores"


@pytest.mark.parametrize(
    "source",
    [
        # an aliased factory, then a module-aliased one
        f"{GT_IMPORT} as make_gt\ndef f(s):\n    return make_gt(s.ground_truth_url)",
        "import app.db.ground_truth_store as g\n"
        "def f(s):\n    return g.make_ground_truth_engine(s.ground_truth_url)",
        # rebound, passed on, and built at module level
        f"{GT_IMPORT}\nfactory = make_ground_truth_engine",
        f"{GT_IMPORT}\ndef f(s, build=make_ground_truth_engine):\n    return build(s)",
        f"{GT_IMPORT}\nENGINE = make_ground_truth_engine('sqlite://')",
        # the guard after the engine, and a local stand-in for the guard
        f"{GT_IMPORT}\n{GUARD_IMPORT}\n"
        "def f(s):\n    e = make_ground_truth_engine(s.ground_truth_url)\n    guard_stores(s)",
        f"{GT_IMPORT}\ndef guard_stores(s):\n    pass\n"
        "def f(s):\n    guard_stores(s)\n    return make_ground_truth_engine(s.ground_truth_url)",
    ],
)
def test_guard_scan_catches_unguarded_engines(source: str) -> None:
    assert _guard_report(ast.parse(source), SPECIMEN)[1]


def test_guard_scan_follows_aliases_to_a_guarded_engine() -> None:
    source = (
        f"{GT_IMPORT} as make_gt\nfrom testbed.eval import labels\n"
        "def f(s):\n    labels.guard_stores(s)\n    return make_gt(s.ground_truth_url)"
    )
    assert _guard_report(ast.parse(source), SPECIMEN) == ({"testbed/pipeline/specimen.py:f"}, [])


@pytest.mark.parametrize(
    "source",
    [
        "from app.db import ground_truth_reader",
        "import app.db.ground_truth_reader as r",
        "from app.db.ground_truth_reader import load_labels",
        "from app.db.ground_truth_store import make_ground_truth_engine",
        "import importlib\nimportlib.import_module('app.db.ground_truth_reader')",
        "__import__('app.db.' + name)",
    ],
)
def test_ground_truth_guard_catches_imports(source: str) -> None:
    assert _gt_imports(ast.parse(source), SPECIMEN)


def _from(source: str) -> ast.ImportFrom:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ImportFrom)
    return node


def test_relative_imports_resolve_against_the_package() -> None:
    init = EVAL / "__init__.py"
    assert _resolve(_from("from ..eval import x"), SPECIMEN) == "testbed.eval"
    assert _resolve(_from("from . import x"), SPECIMEN) == "testbed.pipeline"
    assert _resolve(_from("from . import x"), init) == "testbed.eval"
    assert _resolve(_from("from .labels import x"), init) == "testbed.eval.labels"


# ---- 2. RUNTIME -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> Stores:
    world = generate_world(1, "min")
    stores = Stores.under(tmp_path_factory.mktemp("boundary"))
    persist(world, build_manifest(world, 1, "min"), stores)
    return stores


def test_inference_store_has_no_ground_truth(stores: Stores) -> None:
    engine = make_inference_engine(stores.inference_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert {"ground_truth", "split_groups", "world_evaluation_receipt"}.isdisjoint(tables)
        with Session(engine, future=True) as session, pytest.raises(OperationalError):
            session.scalars(select(GroundTruth)).all()
    finally:
        engine.dispose()


def test_open_labels_reads_a_dev_world(stores: Stores) -> None:
    labels = open_labels(stores)
    assert set(labels.label) == set(labels.family) == set(labels.group_of.values())
    assert set(labels.label.values()) <= {"abusive", "legitimate", "ambiguous"}


def test_open_labels_refuses_missing_stores(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_labels(Stores.under(tmp_path))
    assert not (tmp_path / "ground_truth.db").exists()


def test_verify_pair_refuses_missing_stores(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_pair(Stores.under(tmp_path))
    assert not (tmp_path / "ground_truth.db").exists()


def test_persist_refuses_an_existing_store(stores: Stores) -> None:
    world = generate_world(1, "min")
    with pytest.raises(FileExistsError):
        persist(world, build_manifest(world, 1, "min"), stores)


# ---- 3. SEAL --------------------------------------------------------------------------------


def _confirm_world(root: Path) -> Stores:
    world = generate_world(2, "min")
    stores = Stores.under(root / "runs" / "confirm-test")
    stores.inference_db.parent.mkdir(parents=True)
    persist(world, build_manifest(world, 2, "min"), stores)
    return stores


def _write_seal(root: Path, body: str) -> None:
    (root / "prereg").mkdir(exist_ok=True)
    (root / "prereg" / "M6.md").write_text(f"# M6\n\n```seal\n{body}```\n")


def test_repo_has_no_m6_seal_yet() -> None:
    assert not (REPO / "prereg" / "M6.md").exists()
    with pytest.raises(SealError):
        check_seal(REPO / "prereg" / "M6.md", REPO)


def test_confirmation_world_is_sealed(tmp_path: Path) -> None:
    stores = _confirm_world(tmp_path)
    with pytest.raises(SealError):
        open_labels(stores, repo_root=tmp_path)  # no prereg/M6.md

    artifact = tmp_path / "frozen.txt"
    artifact.write_text("frozen\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    _write_seal(tmp_path, "")
    with pytest.raises(SealError, match="no seal block"):
        open_labels(stores, repo_root=tmp_path)
    _write_seal(tmp_path, f"{digest} frozen.txt\n")  # one space: malformed
    with pytest.raises(SealError, match="malformed"):
        open_labels(stores, repo_root=tmp_path)
    _write_seal(tmp_path, f"{digest}  ../outside.txt\n")
    with pytest.raises(SealError, match="missing"):
        open_labels(stores, repo_root=tmp_path)
    _write_seal(tmp_path, f"{'0' * 64}  frozen.txt\n")
    with pytest.raises(SealError, match="changed"):
        open_labels(stores, repo_root=tmp_path)

    _write_seal(tmp_path, f"{digest}  frozen.txt\n")
    assert open_labels(stores, repo_root=tmp_path).label

    artifact.write_text("edited after the freeze\n")
    with pytest.raises(SealError, match="changed"):
        open_labels(stores, repo_root=tmp_path)


def test_verify_pair_is_sealed(tmp_path: Path) -> None:
    stores = _confirm_world(tmp_path)
    with pytest.raises(SealError):
        verify_pair(stores, repo_root=tmp_path)  # no prereg/M6.md

    artifact = tmp_path / "frozen.txt"
    artifact.write_text("frozen\n")
    _write_seal(tmp_path, f"{'0' * 64}  frozen.txt\n")
    with pytest.raises(SealError, match="changed"):
        verify_pair(stores, repo_root=tmp_path)
    _write_seal(tmp_path, f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  frozen.txt\n")
    assert verify_pair(stores, repo_root=tmp_path).world_id


def test_assign_units_excludes_mixed_clusters() -> None:
    group_of = {"a": "g1", "b": "g1", "c": "g2"}
    result = assign_units([("u1", ["a", "b"]), ("u2", ["b", "c"]), ("u3", ["c"])], group_of)
    assert result.group_of_unit == {"u1": "g1", "u3": "g2"}
    assert result.mixed == ("u2",)
