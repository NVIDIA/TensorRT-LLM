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
    """Identity is the directory name: <checkpoint>__<gpu arch>__<parallel>.

    The class name encodes the same triple, and the routing module's ``_SM``
    has to be the arch segment those targets actually live under -- a target
    moved to a new SM without its routing constant following is the one drift
    that would still route, and route wrong.
    """
    routing = routing_module(arch)
    major, minor = routing._SM
    expected_segment = f"sm_{major}{minor}"

    for name, dotted in routing.TARGET_MODULES.items():
        parts = dotted.split(".")
        assert parts[-1] == "modeling", dotted
        segments = parts[-2].split("__")
        assert len(segments) == 3, (
            f"{arch}: {parts[-2]!r} is not a <checkpoint>__<sm>__<parallel> directory name"
        )
        checkpoint, sm_segment, parallel = segments

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


def _phase_field_offenders(path: Path) -> list[str]:
    """Attribute reads of `_PHASE_FIELDS` anywhere in `path`, outside the two
    exempt regions computed from this module's own AST."""
    tree = ast.parse(path.read_text())
    allowed_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PrefillTarget":
            allowed_ranges.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.FunctionDef) and node.name == "_check_step_contract":
            allowed_ranges.append((node.lineno, node.end_lineno))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in _PHASE_FIELDS:
            continue
        if any(lo <= node.lineno <= hi for lo, hi in allowed_ranges):
            continue
        offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno} .{node.attr}")
    return offenders


def test_a_decode_target_never_reads_the_phase_back():
    """An attribute read of `num_contexts` or `num_ctx_tokens` anywhere in a
    core's module is a violation of the phase split, with exactly two named
    exemptions:

    `PrefillTarget` -- it holds the mixed batch and genuinely needs
    `num_ctx_tokens` to split it; and `_check_step_contract` -- it
    legitimately mirrors the same projection, once, phase-independently,
    to validate it before either target runs.

    The rule is inverted rather than enumerating class names, because
    enumeration goes stale: `DecodeTarget` inherits its forward from
    `_GptOssTarget` (where decode actually executes) and its `step_args`
    delegates to the module-level `_build_step_args`. A gate keyed on the
    `DecodeTarget` class name alone sees neither and only guards the one
    call site that states the literals.

    Read by parsing rather than by importing, like everything else in this
    file: no GPU, no built extensions.
    """
    offenders = []
    for path in _core_modules():
        if not path.is_file():
            continue
        offenders.extend(_phase_field_offenders(path))
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


# The binding layer is only safe because `raw_call` is independent of it. These
# two gates are what keep that true; without them an entry can quietly start
# reading bound state, and CELLS stops covering what the target actually runs.


def _catalog_entries() -> list[tuple[Path, ast.ClassDef]]:
    """Every OpWrapper subclass in the catalog, as (path, class node)."""
    found = []
    for path in sorted((_ROOT / "catalog").glob("*/*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "OpWrapper" for b in node.bases
            ):
                found.append((path, node))
    return found


def test_raw_call_reads_no_bound_state():
    """`raw_call` takes every argument explicitly.

    The moment it reads `self._const`, `self._layered`, or any other attribute,
    the arguments a cell drives it with stop being the whole input -- and
    `CELLS` silently narrows to whatever the last target happened to bind.
    """
    offenders = []
    for path, cls in _catalog_entries():
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef) or node.name != "raw_call":
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self"
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{inner.lineno} self.{inner.attr}")
    assert not offenders, (
        "raw_call must take every argument explicitly; these read bound state: "
        + ", ".join(offenders)
    )


def test_is_valid_accepts_everything_raw_call_accepts():
    """`is_valid` must accept every argument `raw_call` accepts.

    Not parameter-for-parameter mirroring -- an earlier version of this gate
    demanded that, and it was wrong: it demanded a property the tree does not
    have and does not need. `is_valid` may narrow to only the parameters it
    actually inspects, as long as it also carries a `**kwargs` absorber.
    `validating()` forwards the call's full, merged argument set to
    `is_valid` verbatim, so the absorber is what keeps that forward from
    raising `TypeError` on the arguments the guard does not name --
    `thop_attention.is_valid` checks three of `raw_call`'s ~115 parameters
    and absorbs the rest in `**unused_kwargs`; that is not a gap the gate
    missed, it is the mechanism the tree chose so a 115-parameter signature
    does not have to be restated to be validated. An entry with no `is_valid`
    of its own inherits the base class's `(*args, **kwargs)` no-op, which
    already accepts everything, so it is exempt from this check the same way.

    `reference` is not checked here at all, and was wrong to check before.
    It is driven by a cell's own explicit argument list, never by the merged
    forward that reaches `is_valid`, so it never receives an argument it did
    not declare -- there is nothing for it to absorb and nothing to mirror.
    `thop_attention.reference` takes 16 of `raw_call`'s ~115 parameters on
    purpose: a 115-parameter reference implementation would be unmaintainable,
    and the cell driving it only ever supplies those 16. Eight entries narrow
    `reference` this way; all eight are correct.

    The `self`-is-first-parameter check is kept as-is: an entry once shipped
    `raw_call` and `is_valid` both missing `self`, and a mirror check that
    assumed the first parameter was `self` dropped a real argument from both
    and reported a match.
    """
    problems = []
    for path, cls in _catalog_entries():
        funcs = {
            n.name: n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("raw_call", "reference", "is_valid")
        }
        raw_call = funcs.get("raw_call")
        if raw_call is None:
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
        raw_names = {a.arg for a in raw_call.args.args + raw_call.args.kwonlyargs} - {"self"}
        valid_names = {a.arg for a in is_valid.args.args + is_valid.args.kwonlyargs} - {"self"}
        missing = sorted(raw_names - valid_names)
        if missing:
            problems.append(
                f"{path.relative_to(_ROOT)}: {cls.name}.is_valid does not accept "
                f"{missing} that raw_call accepts, and has no ** absorber"
            )
    assert not problems, "; ".join(problems)
