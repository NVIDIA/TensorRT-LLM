# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routing modules and the target directories they import must not drift apart.

Everything here is read off disk as text -- no target module is imported, so
this runs anywhere, including a CI machine with no GPU and no built
extensions. That is deliberate: the failure mode being guarded against is a
rename or a move, and those are visible without executing anything.

The paths it walks are the *package's*, resolved from the imported module
rather than from this file, because the tests live under tests/ and the tree
they guard lives under tensorrt_llm/.

What is *not* guarded here is whether a target is correct; that is what its
gate records are for.
"""

from __future__ import annotations

import ast
import re
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest
import yaml

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2
from tensorrt_llm._torch._experimental.modeling_v2._router_index import registered_routes

# The package, not this file: these paths address the tree under test, and this
# test lives in tests/ while that tree lives in tensorrt_llm/.
pytestmark = pytest.mark.cpu_only

_ROOT = Path(_modeling_v2.__file__).resolve().parent
_ARCHS = sorted(registered_routes())


def _routing_path(arch: str) -> Path:
    """The routing.py whose ``route`` is registered for ``arch``."""
    return Path(sys.modules[registered_routes()[arch].__module__].__file__).resolve()


# ``route`` hands back the class itself, imported from the target directory
# right where it is returned. That import line is the one place a family's
# routing module names its targets, so it is what these tests read.
_TARGET_IMPORT = re.compile(
    r"^\s*from \.(\w+)\.modeling import (?:(\w+)|\(\s*(\w+),?\s*\))$", re.MULTILINE
)


def _routed_targets(arch: str):
    """(target directory, class name) for every target ``route`` can return."""
    source = _routing_path(arch).read_text()
    family_dir = _routing_path(arch).parent
    return [
        (family_dir / target, plain or wrapped)
        for target, plain, wrapped in _TARGET_IMPORT.findall(source)
    ]


def test_no_routing_module_is_orphaned():
    """A routing.py that models/__init__.py does not import, or whose ``route``
    carries no ``@routes``, would never be consulted."""
    registered = {_routing_path(arch) for arch in _ARCHS}
    on_disk = set((_ROOT / "models").glob("*/routing.py"))
    assert on_disk == registered, (
        f"routing modules on disk with no registered route: "
        f"{sorted(p.relative_to(_ROOT) for p in on_disk - registered)}"
    )


@pytest.mark.parametrize("arch", _ARCHS)
def test_no_target_directory_is_orphaned(arch):
    """A target directory ``route`` never imports from is dead code: nothing
    else can reach it."""
    family_dir = _routing_path(arch).parent
    on_disk = {p.parent for p in family_dir.glob("*/modeling.py")}
    routed = {target_dir for target_dir, _ in _routed_targets(arch)}
    assert on_disk == routed, (
        f"{arch}: target directories route() never imports: "
        f"{sorted(p.name for p in on_disk - routed)}; imports with no directory: "
        f"{sorted(p.name for p in routed - on_disk)}"
    )


@pytest.mark.parametrize("arch", _ARCHS)
def test_target_identity_matches_its_path(arch):
    """Identity is the directory name: <checkpoint>__<gpu arch>__<parallel>,
    and the class name encodes the same triple."""
    for target_dir, name in _routed_targets(arch):
        segments = target_dir.name.split("__")
        assert len(segments) == 3, (
            f"{arch}: {target_dir.name!r} is not a <checkpoint>__<sm>__<parallel> directory name"
        )

        def camel(segment: str) -> str:
            return "".join(w.capitalize() for w in segment.split("_"))

        for segment in segments:
            assert camel(segment).lower() in name.lower(), (
                f"{arch}: {name} does not carry path segment {segment!r}"
            )


def test_every_target_ships_both_products():
    """modeling.py and weights.py travel together.

    The forward and the weights it expects. A target missing either is not a
    target.

    Two products, not the four the out-of-tree tree carried. The per-target
    ``smoke.py`` and ``configs/`` went with the move: the boot-and-generate
    check is ``examples/llm-api/quickstart_advanced.py``, the knob variants are
    LLM API arguments the accuracy gates pass directly, and what a target was
    measured to do is recorded where every other model records it --
    ``tests/integration/defs/accuracy/references/``, read by the gates CI runs
    rather than by a reader.
    """
    for arch in _ARCHS:
        for target_dir, name in _routed_targets(arch):
            for product in ("modeling.py", "weights.py"):
                assert (target_dir / product).is_file(), f"{name}: missing {product}"


def _is_entry_module(module: str, target_dir: Path) -> bool:
    family = f"{_modeling_v2.__name__}.models.{target_dir.parent.name}"
    roots = (
        f"{_modeling_v2.__name__}.catalog",
        f"{_modeling_v2.__name__}._native",
        f"{family}.ops",
        f"{family}.{target_dir.name}.ops",
    )
    return any(module == root or module.startswith(root + ".") for root in roots)


def test_targets_do_not_share_files():
    for arch in _ARCHS:
        for target_dir, name in _routed_targets(arch):
            own_target = (
                f"{_modeling_v2.__name__}.models.{target_dir.parent.name}.{target_dir.name}"
            )
            family_kernels = f"{_modeling_v2.__name__}.models.{target_dir.parent.name}.kernels"
            for source_file in target_dir.rglob("*.py"):
                package = ".".join(
                    (_modeling_v2.__name__, *source_file.parent.relative_to(_ROOT).parts)
                )
                for node in ast.walk(ast.parse(source_file.read_text())):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                        relative = False
                    elif isinstance(node, ast.ImportFrom):
                        module = resolve_name("." * node.level + (node.module or ""), package)
                        modules = [f"{module}.{alias.name}" for alias in node.names]
                        relative = node.level > 0
                    else:
                        continue
                    for module in modules:
                        if not relative and not module.startswith(
                            f"{_modeling_v2.__name__}.models."
                        ):
                            continue
                        assert (
                            module == own_target
                            or module.startswith(own_target + ".")
                            or _is_entry_module(module, target_dir)
                            or module == family_kernels
                            or module.startswith(family_kernels + ".")
                        ), (
                            f"{name}: {source_file.relative_to(target_dir)} imports non-owned model code {module}"
                        )


@pytest.mark.parametrize(
    "source_file, statement, accepted",
    [
        ("modeling.py", "from .ops.attention.entry import op", True),
        ("modeling.py", "from ..ops.attention.entry import op", True),
        ("modeling.py", "from ..kernels.kernel import op", True),
        ("modeling.py", "from ....catalog.attention.entry import op", True),
        ("__init__.py", "from ...._native import load_target_ops", True),
        ("modeling.py", "from ..sibling.ops.attention.entry import op", False),
        ("modeling.py", "from ...other.ops.attention.entry import op", False),
        ("modeling.py", "from ..helper import op", False),
        ("modeling.py", "from {package}.models.family.ops.attention.entry import op", True),
        ("modeling.py", "from {package}.models.family.target.ops.attention.entry import op", True),
        ("modeling.py", "import {package}.models.family.sibling.ops.attention.entry", False),
        ("modeling.py", "from {package}.models.other.ops.attention.entry import op", False),
        ("modeling.py", "from {package}.models.family.ops_extra.entry import op", False),
        (
            "ops/attention/entry.py",
            "from {package}.models.family.sibling.ops.attention.entry import op",
            False,
        ),
    ],
)
def test_target_import_isolation(tmp_path, monkeypatch, source_file, statement, accepted):
    target_dir = tmp_path / "models/family/target"
    source = target_dir / source_file
    source.parent.mkdir(parents=True)
    source.write_text(statement.format(package=_modeling_v2.__name__))
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_ARCHS", ["family"])
    monkeypatch.setattr(
        sys.modules[__name__], "_routed_targets", lambda arch: [(target_dir, "Target")]
    )
    if accepted:
        test_targets_do_not_share_files()
    else:
        with pytest.raises(AssertionError, match="non-owned model code"):
            test_targets_do_not_share_files()


@pytest.mark.parametrize(
    "module, accepted",
    [
        ("catalog.attention.entry", True),
        ("models.family.ops.attention.entry", True),
        ("models.family.target.ops.attention.entry", True),
        ("models.family.sibling.ops.attention.entry", False),
        ("models.other.ops.attention.entry", False),
        ("models.family.ops_extra.attention.entry", False),
        ("models.family.helper", False),
        ("models.family.kernels.kernel", False),
    ],
)
def test_generic_call_import_scope(tmp_path, monkeypatch, module, accepted):
    target_dir = tmp_path / "models/family/target"
    target_dir.mkdir(parents=True)
    (target_dir / "modeling.py").write_text(
        f"UNCERTIFIED_GENERIC_CALLS = ()\nfrom {_modeling_v2.__name__}.{module} import op\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_ARCHS", ["family"])
    monkeypatch.setattr(
        sys.modules[__name__], "_routed_targets", lambda arch: [(target_dir, "Target")]
    )
    if accepted:
        test_uncertified_generic_calls_name_every_stock_import()
    else:
        with pytest.raises(AssertionError, match="imported but not in UNCERTIFIED_GENERIC_CALLS"):
            test_uncertified_generic_calls_name_every_stock_import()


#: tensorrt_llm imports that compute nothing, so a target's generic path need not declare them: types and configs
#: it reads, enums it passes, the metadata it is handed and the views it takes of it, the graph-capture context it
#: asks about and its logger.
_NON_COMPUTE_IMPORTS = frozenset(
    {
        "AllReduceStrategy",
        "AttentionMetadata",
        "AuxStreamType",
        "KimiK3HybridCacheManager",
        "Mapping",
        "ModelConfig",
        "MoESchedulerKind",
        "QuantAlgo",
        "QuantConfig",
        "SiTuActivation",
        "is_in_breakable_cuda_graph",
        "k3_mla_decode_view",
        "logger",
    }
)


def _declared_tuple(tree: ast.Module, name: str):
    """The string tuple assigned to ``name`` at module level, or None when the module does not assign it."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    return None


def test_uncertified_generic_calls_name_every_stock_import():
    """A target whose generic path runs stock code (``UNCERTIFIED_GENERIC_CALLS``) declares every tensorrt_llm
    name it imports outside the catalog, module level or function local, unless the name computes nothing; and it
    declares nothing it does not import, so the list cannot go stale as entries replace stock calls."""
    checked = 0
    for arch in _ARCHS:
        for target_dir, name in _routed_targets(arch):
            tree = ast.parse((target_dir / "modeling.py").read_text())
            declared = _declared_tuple(tree, "UNCERTIFIED_GENERIC_CALLS")
            if declared is None:
                continue
            checked += 1
            imported = set()
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module is not None
                    and node.module.split(".")[0] == "tensorrt_llm"
                    and not _is_entry_module(node.module, target_dir)
                ):
                    imported.update(
                        f"{node.module}.{alias.name}"
                        for alias in node.names
                        if alias.name not in _NON_COMPUTE_IMPORTS
                    )
            assert imported <= set(declared), (
                f"{name}: imported but not in UNCERTIFIED_GENERIC_CALLS: {sorted(imported - set(declared))}"
            )
            assert set(declared) <= imported, (
                f"{name}: UNCERTIFIED_GENERIC_CALLS names what it does not import: "
                f"{sorted(set(declared) - imported)}"
            )
    assert checked, "no target declares UNCERTIFIED_GENERIC_CALLS"


# `CELLS = certified_cells(op, TARGET_CELLS)`; the test function's own name
# `test_certified_cells(` must not count as driving the cells.
_DRIVES_TARGET_CELLS = re.compile(r"^CELLS\s*=\s*certified_cells\(\w+,\s*TARGET_CELLS\)", re.M)


def test_every_entry_with_cells_has_a_test_that_drives_them():
    """A cell list nothing executes is worse than no cell list.

    It reads as coverage. `mla_rope_append_paged_kv_assign_q` shipped three
    cells with no driver for exactly one commit, and nothing went red -- the
    entry's own test had been wired to the entry's `compare` and not to its
    `CELLS`, so the suite passed and the three configurations had never run.

    Checked by reading rather than importing, like the rest of this file: an
    entry declares `Cell(` in its source, and some test under this tree has to
    name both the entry and `.CELLS` -- or assign
    `CELLS = certified_cells(op, TARGET_CELLS)`, which drives the entry's cells
    and its targets'. That is weaker than proving the cells were
    driven, and it is what a check that runs anywhere can say.
    """
    tests = list(Path(__file__).resolve().parent.rglob("*.py"))
    sources = {t: t.read_text() for t in tests}

    undriven = []
    for entry in sorted(_entry_files()):
        if "Cell(" not in entry.read_text():
            continue
        if not any(
            entry.stem in src and (".CELLS" in src or _DRIVES_TARGET_CELLS.search(src))
            for src in sources.values()
        ):
            undriven.append(str(entry.relative_to(_ROOT)))

    assert not undriven, (
        f"{len(undriven)} entry(ies) declare cells that no test drives: "
        f"{', '.join(undriven)}. Parametrize the entry's test off its CELLS."
    )


def _target_cells_tests() -> list[Path]:
    unit_root = Path(__file__).resolve().parent
    return sorted(
        p for p in unit_root.glob("*/test_modeling_v2_*.py") if "TARGET_CELLS" in p.read_text()
    )


@pytest.mark.parametrize(
    "test", _target_cells_tests(), ids=lambda p: str(p.relative_to(p.parents[1]))
)
def test_target_cells_feed_entries_the_target_calls(test):
    """An entry test's `TARGET_CELLS` hold the shapes each target calls the entry with.

    The test sits at `<area>/test_modeling_v2_<entry>.py`, mirroring
    `catalog/<area>/<entry>.py`. Each key is a target directory under
    `models/`, and that target's own code imports the entry: a cell for an
    entry the target never calls certifies nothing it runs.
    """
    area, entry = test.parent.name, test.stem.removeprefix("test_modeling_v2_")
    assert (_ROOT / "catalog" / area / f"{entry}.py").is_file(), (
        f"{test.name}: declares TARGET_CELLS but mirrors no catalog/{area}/{entry}.py"
    )
    text = test.read_text()
    assert _DRIVES_TARGET_CELLS.search(text), (
        f"{test.name}: declares TARGET_CELLS but never passes them to certified_cells"
    )
    assert re.search(r'parametrize\(\s*"cell",\s*CELLS\b', text), (
        f"{test.name}: test_certified_cells does not parametrize over CELLS"
    )
    targets = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TARGET_CELLS":
            targets = [ast.literal_eval(k) for k in node.value.keys]
    assert targets, f"{test.name}: TARGET_CELLS is not a dict literal keyed by target directory"
    for target in targets:
        target_dir = _ROOT / "models" / target
        assert (target_dir / "modeling.py").is_file(), (
            f"{test.name}: {target!r} is not a target directory under models/"
        )
        used = "".join(
            p.read_text() for p in target_dir.rglob("*.py") if "__pycache__" not in p.parts
        )
        assert re.search(rf"catalog\.{area}\.{entry} import", used), (
            f"{test.name}: declares cells for {target}, which never imports {area}/{entry}"
        )


_TESTS_ROOT = Path(__file__).resolve().parents[3]


def test_every_target_has_the_accuracy_gate_of_its_name():
    """A target's gate is `Test<TargetClass>` in `accuracy/modeling_v2/test_<family>.py`.

    CBTS reaches a target's gate through that name alone, so a gate renamed
    away from its target, or a target with none, is one a change to it would
    never run.
    """
    routed = {
        (_routing_path(arch).parent.name, f"Test{name}")
        for arch in _ARCHS
        for _, name in _routed_targets(arch)
    }
    gates = set()
    for path in (_TESTS_ROOT / "integration/defs/accuracy/modeling_v2").glob("test_*.py"):
        family = path.stem.removeprefix("test_")
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("TestModelingV2"):
                gates.add((family, node.name))
    assert routed == gates, (
        f"targets with no gate of their name: {sorted(routed - gates)}; "
        f"gates naming no routed target: {sorted(gates - routed)}"
    )


# Unit tests that need hardware no CI stage provides, by path under
# `unittest/`, with what they wait for. They are the only tests no list may
# reach, and none of them may be listed.
_NO_CI_STAGE = {
    "unittest/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__tp16/comm": (
        "4 ranks on sm_107; CI has only the single-GPU GR100 stage (auto:gr100-x1)"
    ),
}


def test_every_unit_test_is_in_a_test_list():
    """The modeling_v2 unit tests are listed per file or per family directory.

    Lists that name files do not pick up a new file by themselves, and a test
    no list reaches never runs in CI. The other way round, an entry naming a
    path with no test at it fails its CI stage: pytest finds nothing to run.
    A test under `_NO_CI_STAGE` is exempt until a stage can run it.
    """
    listed = set()
    for path in (_TESTS_ROOT / "integration/test_lists/test-db").glob("*.yml"):
        for blocks in (yaml.safe_load(path.read_text()) or {}).values():
            for block in blocks if isinstance(blocks, list) else ():
                for entry in block.get("tests") or ():
                    listed.add(entry.split()[0].split("::")[0])

    unit_root = Path(__file__).resolve().parent
    unlisted = []
    for test in sorted(unit_root.rglob("test_*.py")):
        rel = test.relative_to(_TESTS_ROOT / "unittest")
        candidates = {f"unittest/{p.as_posix()}" for p in (rel, *rel.parents)}
        if not candidates & (listed | _NO_CI_STAGE.keys()):
            unlisted.append(rel.as_posix())
    assert not unlisted, f"unit tests no test-db list reaches: {unlisted}"

    # An exemption outlives its reason once its tests move, vanish, or a list
    # starts running them.
    stale = sorted(
        path
        for path in _NO_CI_STAGE
        if not any((_TESTS_ROOT / path).rglob("test_*.py"))
        or any(e == path or e.startswith(f"{path}/") for e in listed)
    )
    assert not stale, f"_NO_CI_STAGE entries with no test or now listed: {stale}"

    # A directory left behind with only `__pycache__` in it exists on disk but
    # holds no test, so require a test file at or under each entry.
    empty = sorted(
        entry
        for entry in listed
        if entry.startswith("unittest/_torch/modeling_v2/")
        and not ((_TESTS_ROOT / entry).is_file() or any((_TESTS_ROOT / entry).rglob("test_*.py")))
    )
    assert not empty, f"test-db entries with no modeling_v2 unit test at them: {empty}"


# The phase a step runs at is resolved once, at the dispatcher, and a target is
# what runs after that resolution. These two gates are what keep that true:
# without them a later edit can put `if md.num_contexts:` back inside a decode
# body and nothing goes red.
_PHASE_FIELDS = ("num_contexts", "num_ctx_tokens")


def _core_modules() -> list[Path]:
    return [
        target_dir / "modeling.py"
        for arch in _ARCHS
        for target_dir, _ in _routed_targets(arch)
        if target_dir.parent.name != "kimi_k3_vl"
    ]


def test_a_decode_target_never_reads_the_phase_back():
    """Decode is routed to only when there are no context rows. Reading the
    count back inside it is not a redundancy, it is the design being violated:
    the value is a per-capture constant, so a branch on it inside a captured
    graph is frozen at whatever the capturing batch happened to carry.

    Prefill is deliberately exempt -- it holds the mixed batch and genuinely
    needs `num_ctx_tokens` to split it. The asymmetry is the design, not an
    oversight.

    Scoped to the `DecodeTarget` class node rather than inverted over the
    whole module: that inversion existed only while a decode body could be
    inherited from a shared base, where `DecodeTarget`'s own node held just
    the literal-args `step_args` call and a gate keyed on the class name saw
    nothing of the forward that actually ran. Every target in this tree is
    now a complete, independent class -- `DecodeTarget.forward` is not
    inherited from anywhere -- so the class node is once again the target's
    whole body, and the direct form catches a violation planted anywhere in
    it.

    Read by parsing rather than by importing, like everything else in this
    file: no GPU, no built extensions.
    """
    offenders = []
    for path in _core_modules():
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "DecodeTarget":
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in _PHASE_FIELDS:
                    offenders.append(f"{path.relative_to(_ROOT)}:{inner.lineno} .{inner.attr}")
    assert not offenders, "a decode target reads the phase it was routed on: " + ", ".join(
        offenders
    )


def test_decode_step_args_state_the_zero_rather_than_reading_it():
    """The one place the gpt_oss default target's prefill and decode targets differ.

    Decode's routing already guarantees no context rows, so it says so instead
    of reading the field back. That is what makes the source gate in
    test_modeling_v2_claims.py meaningful on this target rather than vacuous.
    """
    source = (_ROOT / "models/gpt_oss/gpt_oss_120b__sm_10x__default/modeling.py").read_text()
    decode = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget"
    )
    literals = {
        kw.arg: kw.value.value
        for call in ast.walk(decode)
        if isinstance(call, ast.Call)
        for kw in call.keywords
        if kw.arg in ("num_contexts", "num_ctx_tokens") and isinstance(kw.value, ast.Constant)
    }
    assert literals == {"num_contexts": 0, "num_ctx_tokens": 0}


def test_every_core_ships_both_targets():
    missing = []
    for path in _core_modules():
        if not path.is_file():
            continue
        classes = {
            n.name for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)
        }
        for required in ("PrefillTarget", "DecodeTarget"):
            if required not in classes:
                missing.append(f"{path.relative_to(_ROOT)}: {required}")
    assert not missing, "core modules missing a target class: " + ", ".join(missing)


# An entry computes a pure function of its arguments. These two gates are what
# keep that true; without them an entry can quietly start reading instance
# state, and CELLS stops covering what the target actually runs.


def _entry_files() -> list[Path]:
    """Every entry module: the shared catalog and each target's own ``ops/``."""
    shared = (_ROOT / "catalog").glob("*/*.py")
    family = (_ROOT / "models").glob("*/ops/*/*.py")
    owned = (_ROOT / "models").glob("*/*/ops/*/*.py")
    return sorted(p for p in (*shared, *family, *owned) if p.name != "__init__.py")


def test_entry_discovery_covers_family_level_ops(tmp_path, monkeypatch):
    entries = {
        tmp_path / "catalog/attention/shared.py",
        tmp_path / "models/family/ops/attention/family.py",
        tmp_path / "models/family/target/ops/attention/target.py",
    }
    for path in entries:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        (path.parent / "__init__.py").touch()
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    assert set(_entry_files()) == entries


def _catalog_entries() -> list[tuple[Path, ast.ClassDef]]:
    """Every OpWrapper subclass, shared or owned, as (path, class node)."""
    found = []
    for path in _entry_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "OpWrapper" for b in node.bases
            ):
                found.append((path, node))
    return found


def test_call_reads_no_instance_state():
    """`__call__` takes every argument explicitly.

    The moment it reads any attribute of `self`, the arguments a cell drives it
    with stop being the whole input -- and `CELLS` silently narrows to whatever
    that instance happened to hold.
    """
    offenders = []
    for path, cls in _catalog_entries():
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef) or node.name != "__call__":
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self"
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{inner.lineno} self.{inner.attr}")
    assert not offenders, (
        "__call__ must take every argument explicitly; these read instance state: "
        + ", ".join(offenders)
    )


def test_is_valid_accepts_everything_call_accepts():
    """`is_valid` must accept every argument `__call__` accepts.

    Not parameter-for-parameter mirroring -- an earlier version of this gate
    demanded that, and it was wrong: it demanded a property the tree does not
    have and does not need. `is_valid` may narrow to only the parameters it
    actually inspects, as long as it also carries a `**kwargs` absorber.
    `validating()` forwards the call's full argument set to `is_valid`
    verbatim, so the absorber is what keeps that forward from
    raising `TypeError` on the arguments the guard does not name --
    `thop_attention.is_valid` checks three of `__call__`'s ~115 parameters
    and absorbs the rest in `**unused_kwargs`; that is not a gap the gate
    missed, it is the mechanism the tree chose so a 115-parameter signature
    does not have to be restated to be validated. An entry with no `is_valid`
    of its own inherits the base class's `(*args, **kwargs)` no-op, which
    already accepts everything, so it is exempt from this check the same way.

    `reference` is not checked here at all, and was wrong to check before.
    It is driven by a cell's own explicit argument list, never by the forward
    that reaches `is_valid`, so it never receives an argument it did
    not declare -- there is nothing for it to absorb and nothing to mirror.
    `thop_attention.reference` takes 16 of `__call__`'s ~115 parameters on
    purpose: a 115-parameter reference implementation would be unmaintainable,
    and the cell driving it only ever supplies those 16. Eight entries narrow
    `reference` this way; all eight are correct.

    The `self`-is-first-parameter check is kept as-is: an entry once shipped
    `__call__` and `is_valid` both missing `self`, and a mirror check that
    assumed the first parameter was `self` dropped a real argument from both
    and reported a match.
    """
    problems = []
    for path, cls in _catalog_entries():
        funcs = {
            n.name: n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("__call__", "reference", "is_valid")
        }
        call = funcs.get("__call__")
        if call is None:
            continue
        for name, fn in funcs.items():
            params = [a.arg for a in fn.args.args]
            if not params or params[0] != "self":
                problems.append(
                    f"{path.relative_to(_ROOT)}: {cls.name}.{name} first param is not self"
                )

        is_valid = funcs.get("is_valid")
        if is_valid is None or is_valid.args.kwarg is not None:
            continue  # inherits the base no-op, or absorbs the remainder itself
        call_names = {a.arg for a in call.args.args + call.args.kwonlyargs} - {"self"}
        valid_names = {a.arg for a in is_valid.args.args + is_valid.args.kwonlyargs} - {"self"}
        missing = sorted(call_names - valid_names)
        if missing:
            problems.append(
                f"{path.relative_to(_ROOT)}: {cls.name}.is_valid does not accept "
                f"{missing} that __call__ accepts, and has no ** absorber"
            )
    assert not problems, "; ".join(problems)
