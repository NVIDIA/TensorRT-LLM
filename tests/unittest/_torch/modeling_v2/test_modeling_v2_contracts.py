# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code contracts of the catalog entries and their cells.

An entry computes a pure function of its arguments, so its ``CELLS`` cover its
whole input space. These checks keep that true. They read the source and import
nothing, so they run on any CPU machine.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/modeling_v2"


def _entry_files() -> list[Path]:
    """Every entry module: the shared catalog, each family's ``ops/`` and each target's."""
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

    Once it reads an attribute of `self`, the arguments a cell drives it with are
    no longer its whole input, and `CELLS` covers only what that instance holds.
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
    """`is_valid` accepts every argument `__call__` accepts.

    `validating()` forwards the call's full argument set to `is_valid`, so
    `is_valid` names each of them or absorbs the rest with `**kwargs`. An entry
    without its own `is_valid` inherits the base class's `(*args, **kwargs)`.
    `reference` is not checked: a cell calls it with the cell's own arguments.
    The first parameter of `__call__`, `reference` and `is_valid` is `self`, so
    the comparison does not drop a real argument.
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
            continue
        call_names = {a.arg for a in call.args.args + call.args.kwonlyargs} - {"self"}
        valid_names = {a.arg for a in is_valid.args.args + is_valid.args.kwonlyargs} - {"self"}
        missing = sorted(call_names - valid_names)
        if missing:
            problems.append(
                f"{path.relative_to(_ROOT)}: {cls.name}.is_valid does not accept "
                f"{missing} that __call__ accepts, and has no ** absorber"
            )
    assert not problems, "; ".join(problems)


# `CELLS = certified_cells(op, TARGET_CELLS)`; the test function's own name
# `test_certified_cells(` does not count as driving the cells.
_DRIVES_TARGET_CELLS = re.compile(r"^CELLS\s*=\s*certified_cells\(\w+,\s*TARGET_CELLS\)", re.M)


def test_every_entry_with_cells_has_a_test_that_drives_them():
    """A cell list no test runs reads as coverage that does not exist.

    An entry that declares `Cell(` needs a test under this tree that names both
    the entry and `.CELLS`, or assigns `CELLS = certified_cells(op, TARGET_CELLS)`,
    which drives the entry's cells and its targets'. This reads the source, so it
    shows that some test refers to the cells, not that they ran.
    """
    sources = [t.read_text() for t in Path(__file__).resolve().parent.rglob("*.py")]
    undriven = []
    for entry in _entry_files():
        if "Cell(" not in entry.read_text():
            continue
        if not any(
            entry.stem in src and (".CELLS" in src or _DRIVES_TARGET_CELLS.search(src))
            for src in sources
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
    `catalog/<area>/<entry>.py`. Each key is a target directory under `models/`,
    and that target's code imports the entry: a cell for an entry the target
    never calls certifies nothing it runs.
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
        assert target_dir.is_dir(), f"{test.name}: {target!r} is not a directory under models/"
        used = "".join(
            p.read_text() for p in target_dir.rglob("*.py") if "__pycache__" not in p.parts
        )
        assert re.search(rf"catalog\.{area}\.{entry} import", used), (
            f"{test.name}: declares cells for {target}, which never imports {area}/{entry}"
        )
