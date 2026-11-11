# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conventions scan: what a landed unit added, read by code and reported, never a gate.

At each landing, on CPU, the checkout is read for what a reviewer of the delivery looks
for first:

- each entry the target and its family keep (``ops/<category>/<name>.py``): an
  ``OpWrapper`` with ``__call__`` and ``reference``, ``CELLS`` it certifies and ``ARCHS``
  that name this architecture, and a test under the target's tests that reads its cells;
- Modeling V1 imports in the target;
- the landing name: the module a kernel task was to land is where its row places it.

A finding never holds a task and gets no repair turn. Each is kept in
``conventions.json`` in the workspace with the task that first showed it, for the report
and for the final ``fix`` task. A worker's own statement is not evidence; this scan is.
The scan is shallow: it reads source, imports nothing and follows no call.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ..infra.files import write_text_atomic
from ..operators.coverage import MODEL_OPS_DIRNAME, OP_MATRIX_SUFFIX, model_ops_tests_dir

#: In the workspace: every finding a landing showed, with the task that first showed it.
CONVENTIONS_FILENAME = "conventions.json"
#: Where Modeling V1's models live: what a Modeling V2 target imports from there is said.
V1_MODELS = "tensorrt_llm._torch.models"


def _named(node: ast.AST) -> str:
    return getattr(node, "id", None) or getattr(node, "attr", None) or ""


def _parsed(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None


def entry_findings(path: Path, arch: str = "") -> list[str]:
    """What an entry's source does not state of its contract; ``arch`` must be in its ``ARCHS``."""
    tree = _parsed(path)
    classes = [
        node
        for node in (ast.walk(tree) if tree else ())
        if isinstance(node, ast.ClassDef) and any(_named(b) == "OpWrapper" for b in node.bases)
    ]
    if not classes:
        return [
            "no OpWrapper subclass: an entry is its contract (__call__, reference, CELLS, ARCHS)"
        ]
    found: list[str] = []
    for cls in classes:
        methods = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        stated: dict[str, ast.AST] = {}
        for node in cls.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                for target in getattr(node, "targets", None) or [node.target]:
                    stated[_named(target)] = node.value
        words = [
            str(n.value if isinstance(n, ast.Constant) else _named(n)).lower()
            for n in ast.walk(stated.get("ARCHS", ast.Pass()))
        ]
        archs = sorted({word for word in words if word.startswith("sm_")})
        cells = [
            n
            for n in ast.walk(stated.get("CELLS", ast.Pass()))
            if isinstance(n, ast.Call) and _named(n.func) == "Cell"
        ]
        found += [
            f"{cls.name} defines no {m}" for m in ("__call__", "reference") if m not in methods
        ]
        if not cells:
            found.append(f"{cls.name} declares no CELLS: nothing is certified")
        if not archs:
            found.append(f"{cls.name} declares no ARCHS: its cells were driven on no architecture")
        elif arch and arch not in archs:
            found.append(f"{cls.name}'s ARCHS names {', '.join(archs)}, not {arch}")
    return found


def untested(root: Path, tests_dir: str, category: str, name: str) -> str:
    """Why entry ``name``'s test does not show its cells were driven; ``""`` when it does."""
    folder = root / tests_dir / category
    names = (f"test_modeling_v2_{name}", f"test_modeling_v2_{name}{OP_MATRIX_SUFFIX}")
    tests = [folder / f"{stem}.py" for stem in names if (folder / f"{stem}.py").is_file()]
    if not tests:
        return f"no test at {tests_dir}/{category}/{names[0]}.py"
    # A multi-rank entry's collected file only starts its rank body, which drives the cells.
    read = [*tests, folder / f"_{name}{OP_MATRIX_SUFFIX}.py"]
    texts = [path.read_text(encoding="utf-8", errors="replace") for path in read if path.is_file()]
    if any(".CELLS" in text or "certified_cells(" in text for text in texts):
        return ""
    return f"{tests[0].relative_to(root)} never reads the entry's CELLS: they were not driven"


def v1_imports(root: Path, target: Path) -> list[str]:
    """Each import of Modeling V1's models in the target's sources, by file and line."""
    found: list[str] = []
    for source in sorted(target.rglob("*.py")):
        tree = _parsed(source)
        for node in ast.walk(tree) if tree else ():
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module or ""]
            for module in modules:
                if module == V1_MODELS or module.startswith(f"{V1_MODELS}."):
                    where = f"{source.relative_to(root)}:{node.lineno}"
                    found.append(f"{where} imports Modeling V1 ({module})")
    return found


def scan(
    tree: str | Path, target_rel: str, arch: str = "", *, landing: str = "", named: str = ""
) -> list[str]:
    """The findings of the checkout ``tree`` for the target at ``target_rel``, one line each.

    ``landing`` is the module a kernel task was to land (repo-relative) and ``named`` the
    one its decision names, when it names one.
    """
    root, target = Path(tree), Path(tree) / target_rel
    found: list[str] = []
    if landing and not (root / landing).is_file():
        at = f", it is at `{named}`" if named and (root / named).is_file() else ""
        found.append(f"`{landing}` is not in the tree: the entry is not where its row lands it{at}")
    # The target's own entries and the ones its family shares.
    for owner in (target, target.parent):
        for entry in sorted((owner / MODEL_OPS_DIRNAME).glob("*/*.py")):
            if entry.name.startswith("_"):
                continue
            rel, category = entry.relative_to(root), entry.parent.name
            tests_dir = model_ops_tests_dir(owner.relative_to(root).as_posix())
            said = [*entry_findings(entry, arch), untested(root, tests_dir, category, entry.stem)]
            found += [f"{rel}: {line}" for line in said if line]
    return found + v1_imports(root, target)


def load(workspace: str | Path) -> dict[str, str]:
    """Every finding a landing of this run showed, with the task that first showed it."""
    try:
        held = json.loads((Path(workspace) / CONVENTIONS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in held.items()} if isinstance(held, dict) else {}


def record(workspace: str | Path, task_id: str, findings: list[str]) -> list[str]:
    """Keep ``findings`` under ``task_id``, the first to show them; the ones not seen before."""
    held = load(workspace)
    new = [finding for finding in findings if finding not in held]
    if new:
        held.update({finding: task_id for finding in new})
        Path(workspace).mkdir(parents=True, exist_ok=True)
        write_text_atomic(Path(workspace) / CONVENTIONS_FILENAME, json.dumps(held, indent=1))
    return new


__all__ = [
    "CONVENTIONS_FILENAME",
    "entry_findings",
    "load",
    "record",
    "scan",
    "untested",
    "v1_imports",
]
