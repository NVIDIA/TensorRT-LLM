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


def test_targets_do_not_share_files():
    """Isolation is the property being demonstrated; measure it.

    Targets are allowed to import the catalog and nothing else of each
    other's. A shared helper between two targets is the first step back to
    the abstraction this package exists to avoid.
    """
    for arch in _ARCHS:
        for target_dir, name in _routed_targets(arch):
            for source_file in target_dir.glob("*.py"):
                text = source_file.read_text()
                for match in re.finditer(r"^from (\.+)([\w.]*) import", text, re.MULTILINE):
                    dots, tail = match.group(1), match.group(2)
                    if len(dots) == 1:
                        continue  # sibling within the target
                    assert tail.startswith("catalog"), (
                        f"{name}: {source_file.name} reaches outside its own "
                        f"directory for {tail!r}; targets may import the "
                        f"catalog and nothing else"
                    )


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
    catalog = _ROOT / "catalog"
    tests = list(Path(__file__).resolve().parent.rglob("*.py"))
    sources = {t: t.read_text() for t in tests}

    undriven = []
    for entry in sorted(catalog.glob("*/*.py")):
        if entry.name == "__init__.py":
            continue
        if "Cell(" not in entry.read_text():
            continue
        if not any(entry.stem in src and ".CELLS" in src for src in sources.values()):
            undriven.append(f"{entry.parent.name}/{entry.stem}")

    assert not undriven, (
        f"{len(undriven)} entry(ies) declare cells that no test drives: "
        f"{', '.join(undriven)}. Parametrize the entry's test off its CELLS."
    )


# The phase a step runs at is resolved once, at the dispatcher, and a target is
# what runs after that resolution. These two gates are what keep that true:
# without them a later edit can put `if md.num_contexts:` back inside a decode
# body and nothing goes red.
_PHASE_FIELDS = ("num_contexts", "num_ctx_tokens")


# Cores not yet on the target layer. An entry is deleted when that core is
# migrated, so finishing the migration is a visible diff rather than a gate
# that quietly stopped covering anything.
_NOT_YET_ON_THE_TARGET_LAYER = frozenset({"r1_0528_nvfp4__sm_103__dep4"})


def _core_modules() -> list[Path]:
    """Every target module the routing tables can reach and this layer covers."""
    modules = []
    for arch in _ARCHS:
        routing = routing_module(arch)
        for dotted in routing.TARGET_MODULES.values():
            path = _module_path(dotted)
            if path.parent.name in _NOT_YET_ON_THE_TARGET_LAYER:
                continue
            modules.append(path)
    return modules


def test_a_decode_target_never_reads_the_phase_back():
    """Decode is routed to only when there are no context rows. Reading the
    count back inside it is not a redundancy, it is the design being violated:
    the value is a per-capture constant, so a branch on it inside a captured
    graph is frozen at whatever the capturing batch happened to carry.

    Prefill is deliberately exempt -- it holds the mixed batch and genuinely
    needs `num_ctx_tokens` to split it. The asymmetry is the design (2.2), not
    an oversight.

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
