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
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2
from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_ROUTERS,
    routing_module,
)

# The package, not this file: these paths address the tree under test, and this
# test lives in tests/ while that tree lives in tensorrt_llm/.
pytestmark = pytest.mark.cpu_only

_ROOT = Path(_modeling_v2.__file__).resolve().parent
_ARCHS = sorted(MODELING_V2_ROUTERS)


def _module_path(dotted: str) -> Path:
    """Resolve a package-relative dotted module name to its file."""
    return _ROOT / (dotted.replace(".", "/") + ".py")


# ``route`` hands back the class itself, imported from the target directory
# right where it is returned. That import line is the one place a family's
# routing module names its targets, so it is what these tests read.
_TARGET_IMPORT = re.compile(r"^\s*from \.(\w+)\.modeling import (\w+)$", re.MULTILINE)


def _routed_targets(arch: str):
    """(target directory, class name) for every target ``route`` can return."""
    source = _module_path(MODELING_V2_ROUTERS[arch]).read_text()
    family_dir = _module_path(MODELING_V2_ROUTERS[arch]).parent
    return [(family_dir / target, name) for target, name in _TARGET_IMPORT.findall(source)]


def test_every_routed_architecture_has_an_importable_routing_module():
    for arch in _ARCHS:
        assert routing_module(arch) is not None, arch


def test_no_routing_module_is_orphaned():
    """A routing.py the index does not name would never be consulted."""
    indexed = {_module_path(m) for m in MODELING_V2_ROUTERS.values()}
    on_disk = set((_ROOT / "models").glob("*/routing.py"))
    assert on_disk == indexed, (
        f"routing modules on disk but not in MODELING_V2_ROUTERS: "
        f"{sorted(p.relative_to(_ROOT) for p in on_disk - indexed)}"
    )


@pytest.mark.parametrize("arch", _ARCHS)
def test_no_target_directory_is_orphaned(arch):
    """A target directory ``route`` never imports from is dead code: nothing
    else can reach it."""
    family_dir = _module_path(MODELING_V2_ROUTERS[arch]).parent
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


_PACKAGE = _modeling_v2.__name__


def _dotted(root: Path, package: str, path: Path) -> str:
    """The dotted module name of ``path``, a file of the tree rooted at ``root``."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([package, *parts])


def _module_file(root: Path, package: str, dotted: str):
    """The file defining module ``dotted`` inside the tree, or None (outside it, or a name)."""
    if dotted != package and not dotted.startswith(package + "."):
        return None
    rel = dotted[len(package) :].lstrip(".").replace(".", "/")
    for candidate in (root / (rel + ".py"), root / rel / "__init__.py"):
        if rel and candidate.is_file():
            return candidate
    return None


def _imported_modules(root: Path, package: str, path: Path) -> list[tuple[str, int]]:
    """Every module an import in ``path`` names, as absolute dotted names, with its line.

    Absolute and relative ``import`` / ``from ... import`` statements at any depth (a
    function-local import executes all the same); an alias changes nothing about what is
    imported. ``from P import n`` names the module ``P.n`` when that is a module of the tree
    (a submodule imported through its package) and ``P`` otherwise (a name re-exported by P).
    """
    module = _dotted(root, package, path)
    here = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here.split(".")[: len(here.split(".")) - (node.level - 1)]
                source = ".".join(base + ([node.module] if node.module else []))
            else:
                source = node.module or ""
            for alias in node.names:
                submodule = f"{source}.{alias.name}"
                named = submodule if _module_file(root, package, submodule) else source
                found.append((named, node.lineno))
    return found


def _isolation_violations(root: Path, package: str, target_dir: Path) -> list[str]:
    """Imports in a target's production closure that leave what it may use.

    A target may import its own package, the catalog, the router index, its family's
    routing module and the packages on its own path (their ``__init__``). The closure
    follows every module of the target and of the catalog it reaches, so a forbidden
    module cannot hide behind an alias, a relative form, a function-local import or a
    re-export through the catalog.
    """
    target = _dotted(root, package, target_dir / "__init__.py")
    family = target.rsplit(".", 1)[0]
    catalog = f"{package}.catalog"
    allowed_exact = {
        package,
        f"{package}.models",
        family,
        f"{package}._router_index",
        f"{family}.routing",
    }

    def allowed(dotted: str) -> bool:
        return (
            dotted in allowed_exact
            or dotted == target
            or dotted.startswith(target + ".")
            or dotted == catalog
            or dotted.startswith(catalog + ".")
        )

    pending = sorted(target_dir.rglob("*.py"))
    seen = set(pending)
    violations = []
    while pending:
        path = pending.pop()
        for dotted, line in _imported_modules(root, package, path):
            file = _module_file(root, package, dotted)
            if file is None:
                if dotted == package or dotted.startswith(package + "."):
                    violations.append(
                        f"{path.relative_to(root)}:{line} imports {dotted}, no module"
                    )
                continue  # outside this tree: torch, the runtime, the native extension
            if not allowed(dotted):
                violations.append(f"{path.relative_to(root)}:{line} imports {dotted}")
                continue
            if file not in seen and (dotted.startswith(target) or dotted.startswith(catalog)):
                seen.add(file)
                pending.append(file)
    return sorted(violations)


def test_targets_do_not_share_files():
    """Isolation is the property being demonstrated; measure it.

    Targets are allowed to import the catalog and nothing else of each other's -- nor of
    the earlier single-purpose components of their family, whose code a target owns
    itself when it needs it. A shared helper between two targets is the first step back
    to the abstraction this package exists to avoid. Checked over every routed target's
    whole import closure, with no exemption for any target.
    """
    violations = []
    for arch in _ARCHS:
        for target_dir, _ in _routed_targets(arch):
            violations += _isolation_violations(_ROOT, _PACKAGE, target_dir)
    assert not violations, "targets reach outside their directory and the catalog: " + (
        "; ".join(violations)
    )


def _write_tree(root: Path, files: dict) -> None:
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


_FIXTURE_TREE = {
    "__init__.py": "",
    "_router_index.py": "",
    "catalog/__init__.py": "",
    "catalog/norm/__init__.py": "",
    "catalog/norm/rmsnorm.py": "",
    "models/__init__.py": "",
    "models/fam/__init__.py": "",
    "models/fam/routing.py": "",
    "models/fam/old_component/__init__.py": "",
    "models/fam/old_component/model.py": "",
    "models/fam/target__sm_100__tp1/__init__.py": "",
    "models/fam/target__sm_100__tp1/helper.py": "",
    "models/fam/target__sm_100__tp1/modeling.py": "",
}


@pytest.mark.parametrize(
    "source",
    [
        "from pkg.models.fam.old_component.model import Block\n",
        "from ..old_component.model import Block\n",
        "import pkg.models.fam.old_component.model as m\n",
        "from ..old_component import model\n",
        "from pkg.models.fam import old_component\n",
        "def build():\n    from ..old_component.model import Block\n    return Block\n",
    ],
    ids=["absolute", "relative", "aliased", "package-submodule", "package", "function-local"],
)
def test_isolation_gate_rejects_every_import_form(tmp_path, source):
    _write_tree(tmp_path, {**_FIXTURE_TREE, "models/fam/target__sm_100__tp1/modeling.py": source})
    target = tmp_path / "models/fam/target__sm_100__tp1"
    assert _isolation_violations(tmp_path, "pkg", target), source


@pytest.mark.parametrize(
    "files",
    [
        # Through the target's own helper module.
        {"models/fam/target__sm_100__tp1/helper.py": "from ..old_component.model import Block\n"},
        # Through the catalog: an entry re-exporting a component.
        {
            "catalog/norm/rmsnorm.py": "from pkg.models.fam.old_component.model import Block\n",
            "models/fam/target__sm_100__tp1/helper.py": "from ....catalog.norm.rmsnorm import Block\n",
        },
    ],
    ids=["own-helper", "catalog-re-export"],
)
def test_isolation_gate_follows_the_closure(tmp_path, files):
    _write_tree(
        tmp_path,
        {
            **_FIXTURE_TREE,
            **files,
            "models/fam/target__sm_100__tp1/modeling.py": "from . import helper\n",
        },
    )
    assert _isolation_violations(tmp_path, "pkg", tmp_path / "models/fam/target__sm_100__tp1")


def test_isolation_gate_accepts_the_allowed_imports(tmp_path):
    source = (
        "from . import helper\n"
        "from .helper import name\n"
        "from pkg.models.fam.target__sm_100__tp1 import helper as h\n"
        "from pkg.catalog.norm.rmsnorm import rmsnorm\n"
        "from ....catalog.norm import rmsnorm as r\n"
        "from pkg._router_index import NoTarget\n"
        "from ..routing import LIMIT\n"
        "import torch\n"
    )
    _write_tree(tmp_path, {**_FIXTURE_TREE, "models/fam/target__sm_100__tp1/modeling.py": source})
    assert _isolation_violations(tmp_path, "pkg", tmp_path / "models/fam/target__sm_100__tp1") == []


def test_every_entry_with_cells_has_a_test_that_drives_them():
    """A cell list nothing executes is worse than no cell list.

    It reads as coverage. `mla_rope_append_paged_kv_assign_q` shipped three
    cells with no driver for exactly one commit, and nothing went red -- the
    entry's own test had been wired to the entry's `compare` and not to its
    `CELLS`, so the suite passed and the three configurations had never run.

    Checked by reading rather than importing, like the rest of this file: an
    entry declares `Cell(` in its source, and some test under this tree has to
    name both the entry and `.CELLS`. That is weaker than proving the cells were
    driven, and it is what a check that runs anywhere can say.
    """
    tests = list(Path(__file__).resolve().parent.rglob("*.py"))
    sources = {t: t.read_text() for t in tests}

    undriven = []
    for entry in sorted(_entry_files()):
        if "Cell(" not in entry.read_text():
            continue
        if not any(entry.stem in src and ".CELLS" in src for src in sources.values()):
            undriven.append(str(entry.relative_to(_ROOT)))

    assert not undriven, (
        f"{len(undriven)} entry(ies) declare cells that no test drives: "
        f"{', '.join(undriven)}. Parametrize the entry's test off its CELLS."
    )


# The phase a step runs at is resolved once, at the dispatcher, and a target is
# what runs after that resolution. These two gates are what keep that true:
# without them a later edit can put `if md.num_contexts:` back inside a decode
# body and nothing goes red.
_PHASE_FIELDS = ("num_contexts", "num_ctx_tokens")


def _core_modules() -> list[Path]:
    """Every target module ``route`` can reach."""
    return [
        target_dir / "modeling.py" for arch in _ARCHS for target_dir, _ in _routed_targets(arch)
    ]


def _decode_phase_reads(source: str) -> list[str]:
    """Phase reads in what a module's ``DecodeTarget`` executes: ``Class.method:line .field``.

    Starts at ``DecodeTarget.forward`` and follows, with a visited set, every ``self.m(...)``
    and ``super().m(...)`` call through the class's bases defined in the module (so an
    inherited or delegated body such as ``_Target._run`` is read where it is written) and
    every call of a module-level function of the module. Calls into other modules (the
    component the target drives) are not followed.
    """
    tree = ast.parse(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "DecodeTarget" not in classes:
        return []

    def mro(name: str) -> list:
        order, todo = [], [name]
        while todo:
            current = todo.pop(0)
            if current in classes and current not in order:
                order.append(current)
                todo += [b.id for b in classes[current].bases if isinstance(b, ast.Name)]
        return order

    def method(cls_name: str, name: str, start: int = 0):
        for owner in mro(cls_name)[start:]:
            for node in classes[owner].body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return owner, node
        return None

    reads, visited = [], set()
    pending = [method("DecodeTarget", "forward")]
    while pending:
        item = pending.pop()
        if item is None or (item[0], item[1].name) in visited:
            continue
        visited.add((item[0], item[1].name))
        owner, fn = item
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and node.attr in _PHASE_FIELDS:
                reads.append(f"{owner}.{fn.name}:{node.lineno} .{node.attr}")
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
                if callee.value.id == "self":
                    pending.append(method("DecodeTarget", callee.attr))
            elif isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Call):
                inner = callee.value.func
                if isinstance(inner, ast.Name) and inner.id == "super":
                    pending.append(method(owner, callee.attr, start=1))
            elif isinstance(callee, ast.Name) and callee.id in functions:
                pending.append(("<module>", functions[callee.id]))
    return sorted(set(reads))


def test_a_decode_target_never_reads_the_phase_back():
    """Decode is routed to only when there are no context rows. Reading the
    count back inside it is not a redundancy, it is the design being violated:
    the value is a per-capture constant, so a branch on it inside a captured
    graph is frozen at whatever the capturing batch happened to carry.

    Prefill is deliberately exempt -- it holds the mixed batch and genuinely
    needs `num_ctx_tokens` to split it. The asymmetry is the design, not an
    oversight.

    Read over what a `DecodeTarget` executes, not only its class node: its own
    methods, the bodies it inherits from or delegates to in its module (a shared
    `_Target._run`) and the module's helper functions it calls.

    Read by parsing rather than by importing, like everything else in this
    file: no GPU, no built extensions.
    """
    offenders = []
    for path in _core_modules():
        if path.is_file():
            offenders += [
                f"{path.relative_to(_ROOT)} {r}" for r in _decode_phase_reads(path.read_text())
            ]
    assert not offenders, "a decode target reads the phase it was routed on: " + ", ".join(
        offenders
    )


_PHASE_FIXTURE_BASE = """
class _Target:
    def _run(self, md, rows):
        return self.model(md, rows)

    def _rows(self, md):
        return None


class PrefillTarget(_Target):
    def forward(self, md):
        return self._run(md, md.num_ctx_tokens)
"""


@pytest.mark.parametrize(
    "decode",
    [
        # Direct: the decode body branches on the phase.
        "class DecodeTarget(_Target):\n"
        "    def forward(self, md):\n"
        "        return self._run(md, md.num_contexts)\n",
        # Inherited: the decode target runs a base forward that reads it.
        "class _Base(_Target):\n"
        "    def forward(self, md):\n"
        "        return self._run(md, md.num_ctx_tokens)\n"
        "class DecodeTarget(_Base):\n"
        "    pass\n",
        # Delegated: the decode forward hands off to a shared body that reads it.
        "class DecodeTarget(_Target):\n"
        "    def forward(self, md):\n"
        "        return self._delegate(md)\n"
        "    def _delegate(self, md):\n"
        "        return super()._shared(md)\n",
        # Same-module helper.
        "def _split(md):\n"
        "    return md.num_contexts\n"
        "class DecodeTarget(_Target):\n"
        "    def forward(self, md):\n"
        "        return self._run(md, _split(md))\n",
    ],
    ids=["direct", "inherited", "delegated", "helper"],
)
def test_decode_phase_gate_catches_executed_bodies(decode):
    shared = "\n\nclass _Shared:\n    def _shared(self, md):\n        return md.num_contexts\n"
    source = _PHASE_FIXTURE_BASE + shared + "\n\n" + decode
    if "super()._shared" in decode:
        source = source.replace("class _Target:", "class _Target(_Shared):")
    assert _decode_phase_reads(source), decode


def test_decode_phase_gate_follows_the_shared_run_body():
    source = (
        _PHASE_FIXTURE_BASE.replace(
            "return self.model(md, rows)", "return self.model(md, rows or md.num_contexts)"
        )
        + "\n\nclass DecodeTarget(_Target):\n    def forward(self, md):\n        return self._run(md, None)\n"
    )
    assert _decode_phase_reads(source) == ["_Target._run:4 .num_contexts"]


def test_decode_phase_gate_accepts_the_prefill_split():
    source = _PHASE_FIXTURE_BASE + (
        "\n\nclass DecodeTarget(_Target):\n"
        "    def forward(self, md):\n"
        "        return self._run(md, None)\n"
    )
    assert _decode_phase_reads(source) == []


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
    """A core with one target routes half its steps into a KeyError.

    Names, not a table: the gate above finds the decode body by class name, so
    the name is load-bearing and is pinned here rather than left to convention.
    """
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
    owned = (_ROOT / "models").glob("*/*/ops/*/*.py")
    return sorted(p for p in (*shared, *owned) if p.name != "__init__.py")


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
