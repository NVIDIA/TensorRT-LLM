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
_TARGET_IMPORT = re.compile(
    r"^\s*from \.(\w+)\.modeling import (?:(\w+)|\(\s*(\w+),?\s*\))$", re.MULTILINE
)


def _routed_targets(arch: str):
    """(target directory, class name) for every target ``route`` can return."""
    source = _module_path(MODELING_V2_ROUTERS[arch]).read_text()
    family_dir = _module_path(MODELING_V2_ROUTERS[arch]).parent
    return [
        (family_dir / target, plain or wrapped)
        for target, plain, wrapped in _TARGET_IMPORT.findall(source)
    ]


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


#: tensorrt_llm imports that compute nothing, so a target's generic path need not declare them: types and configs
#: it reads, enums it passes, the metadata it is handed and the views it takes of it, the graph-capture context it
#: asks about and its logger.
_NON_COMPUTE_IMPORTS = frozenset(
    {
        "AllReduceStrategy",
        "AttentionMetadata",
        "AuxStreamType",
        "MambaHybridCacheManagerV2",
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
                    and ".modeling_v2.catalog" not in node.module
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
