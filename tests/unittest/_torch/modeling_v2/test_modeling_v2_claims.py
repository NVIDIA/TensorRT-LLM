# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routing tables and the targets they name must not drift apart.

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

import re
from pathlib import Path

import pytest

import tensorrt_llm._torch.modeling_v2 as _modeling_v2
from tensorrt_llm._torch.modeling_v2._router_index import MODELING_V2_ROUTERS, routing_module

# The package, not this file: these paths address the tree under test, and this
# test lives in tests/ while that tree lives in tensorrt_llm/.
_ROOT = Path(_modeling_v2.__file__).resolve().parent
_ARCHS = sorted(MODELING_V2_ROUTERS)


def _module_path(dotted: str) -> Path:
    """Resolve a package-relative dotted module name to its file."""
    return _ROOT / (dotted.replace(".", "/") + ".py")


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
def test_targets_table_and_target_modules_agree(arch):
    """Every name the tree can return must have a module that registers it."""
    routing = routing_module(arch)
    produced = set(routing._TARGETS.values())
    declared = set(routing.TARGET_MODULES)
    assert produced == declared, (
        f"{arch}: _TARGETS yields {sorted(produced - declared)} with no "
        f"TARGET_MODULES entry; TARGET_MODULES declares "
        f"{sorted(declared - produced)} the tree cannot return"
    )


@pytest.mark.parametrize("arch", _ARCHS)
def test_each_target_module_registers_its_own_name(arch):
    """The synthetic name is only a registry key -- the module must fill it.

    Nothing else can: the built-in static index does not carry modeling_v2
    names, so a target whose decorator says something different resolves to
    None and the engine reports an unknown architecture.
    """
    routing = routing_module(arch)
    for name, dotted in routing.TARGET_MODULES.items():
        path = _module_path(dotted)
        assert path.is_file(), f"{arch}: {name} -> missing {path}"
        source = path.read_text()
        assert f'@register_auto_model("{name}")' in source, (
            f"{arch}: {path.relative_to(_ROOT)} does not register {name!r}"
        )


@pytest.mark.parametrize("arch", _ARCHS)
def test_target_identity_matches_its_path(arch):
    """Identity is the path: <checkpoint>/<gpu arch>/<parallel>.

    The class name encodes the same triple, and the routing module's ``_SM``
    has to be the arch segment those targets actually live under -- a target
    moved to a new SM directory without its routing constant following is the
    one drift that would still route, and route wrong.
    """
    routing = routing_module(arch)
    major, minor = routing._SM
    expected_segment = f"sm_{major}{minor}"

    for name, dotted in routing.TARGET_MODULES.items():
        parts = dotted.split(".")
        assert parts[-1] == "modeling", dotted
        parallel, sm_segment, checkpoint = parts[-2], parts[-3], parts[-4]

        assert sm_segment == expected_segment, (
            f"{arch}: {name} lives under {sm_segment} but its routing module "
            f"only ever matches {expected_segment}"
        )

        def camel(segment: str) -> str:
            return "".join(w.capitalize() for w in segment.split("_"))

        for segment in (checkpoint, sm_segment, parallel):
            assert camel(segment).lower() in name.lower(), (
                f"{arch}: {name} does not carry path segment {segment!r}"
            )

        assert (checkpoint, parallel) in routing._TARGETS, (
            f"{arch}: no _TARGETS entry keyed ({checkpoint!r}, {parallel!r})"
        )
        assert routing._TARGETS[(checkpoint, parallel)] == name


@pytest.mark.parametrize("arch", _ARCHS)
def test_checkpoint_fingerprints_are_distinct(arch):
    """Two checkpoints sharing a fingerprint would route to one target."""
    routing = routing_module(arch)
    names = list(routing._CHECKPOINTS.values())
    assert len(names) == len(set(names)), f"{arch}: duplicate checkpoint names in _CHECKPOINTS"
    assert set(names) == {c for c, _ in routing._TARGETS}, (
        f"{arch}: _CHECKPOINTS and _TARGETS name different checkpoints"
    )


# Context fields a routing tree may not branch on yet, and what has to happen
# before it can. Both would otherwise decide silently on a value that is not
# the deployment's -- the failure ``require`` exists to prevent. Delete a row
# once its prerequisite is met.
_UNREADABLE_CONTEXT_FIELDS = {
    "is_disagg": (
        "no caller sets it, so it reads False in every deployment, "
        "disaggregated or not; plumb it onto ModelConfig first"
    ),
    "spec_config": (
        "explain.py has no flag for a speculative config, so it would report "
        "the wrong target for every drafting configuration; give explain a "
        "way to name one first"
    ),
}


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize("field", sorted(_UNREADABLE_CONTEXT_FIELDS))
def test_no_routing_module_reads_an_unplumbed_dimension(arch, field):
    """A routing tree may only read what both the engine and explain can fill.

    ``explain`` replays the same tree to answer "why did I not get the target
    I expected". A criterion it cannot evaluate makes that answer wrong on
    exactly the configurations someone would ask about -- so the set of
    readable fields is bounded by the weaker of the two callers, not the
    engine alone.
    """
    source = _module_path(MODELING_V2_ROUTERS[arch]).read_text()
    assert field not in source, (
        f"{arch}: routing reads ctx.{field}, but {_UNREADABLE_CONTEXT_FIELDS[field]}"
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
        routing = routing_module(arch)
        for name, dotted in routing.TARGET_MODULES.items():
            target_dir = _module_path(dotted).parent
            for product in ("modeling.py", "weights.py"):
                assert (target_dir / product).is_file(), f"{name}: missing {product}"


def test_targets_do_not_share_files():
    """Isolation is the property being demonstrated; measure it.

    Targets are allowed to import the catalog and nothing else of each
    other's. A shared helper between two targets is the first step back to
    the abstraction this package exists to avoid.
    """
    for arch in _ARCHS:
        routing = routing_module(arch)
        for name, dotted in routing.TARGET_MODULES.items():
            target_dir = _module_path(dotted).parent
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
