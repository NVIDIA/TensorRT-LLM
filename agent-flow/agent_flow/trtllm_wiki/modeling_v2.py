# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelingV2 families, targets and operation wrappers, read off the checkout's source.

Everything here comes from parsing files, never importing them:

* a family is a ``models/<family>/`` directory; the architectures it routes are
  the string arguments of the ``@routes(...)`` decorators in its ``routing.py``;
* a target is a directory under a family that holds ``modeling.py``; its name
  splits into checkpoint, GPU architecture and parallel layout when it follows
  the ``<checkpoint>__<sm>__<parallel>`` convention;
* an operation is a class deriving from ``OpWrapper`` (by name) in the catalog,
  a family or a target; its ``ARCHS``, whether it declares ``CELLS``, the calls
  its ``__call__`` makes into ``torch.ops``, FlashInfer or a family/target
  native library, and the unit tests that name its module or class.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterable

MODELING_V2_REL = "tensorrt_llm/_torch/_experimental/modeling_v2"
MODELING_V2_TESTS_REL = "tests/unittest/_torch/modeling_v2"
MV2_ACCURACY_TESTS_REL = "tests/integration/defs/accuracy/modeling_v2"

_TARGET_NAME = re.compile(r"^(?P<checkpoint>.+?)__(?P<sm>sm_[0-9x]+)__(?P<parallel>[a-z0-9_]+)$")


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None


def _rel(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _decorator_name(decorator: ast.expr) -> str | None:
    func = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def routed_architectures(tree: ast.Module) -> list[tuple[str, int]]:
    """``(architecture, line)`` for every ``@routes("...")`` argument in ``tree``."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if _decorator_name(decorator) != "routes" or not isinstance(decorator, ast.Call):
                continue
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append((arg.value, decorator.lineno))
    return found


def _classes_named(tree: ast.Module | None, prefix: str) -> list[str]:
    if tree is None:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith(prefix)]


def _target(repo: Path, family: str, directory: Path) -> dict[str, Any]:
    match = _TARGET_NAME.match(directory.name)
    tests = repo / MODELING_V2_TESTS_REL / "models" / family / directory.name
    return {
        "name": directory.name,
        "path": _rel(repo, directory),
        "checkpoint": match["checkpoint"] if match else "",
        "sm": match["sm"] if match else "",
        "parallel": match["parallel"] if match else "",
        "classes": _classes_named(_parse(directory / "modeling.py"), "ModelingV2"),
        "has_weights": (directory / "weights.py").is_file(),
        "has_native": (directory / "csrc").is_dir(),
        "unit_tests": _rel(repo, tests) if tests.is_dir() else "",
    }


def _accuracy_tests(repo: Path, family: str) -> dict[str, Any]:
    path = repo / MV2_ACCURACY_TESTS_REL / f"test_{family}.py"
    tree = _parse(path)
    if tree is None:
        return {"path": "", "classes": []}
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                m.name
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and m.name.startswith("test_")
            ]
            model = next(
                (
                    item.value.value
                    for item in node.body
                    if isinstance(item, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "MODEL_NAME" for t in item.targets)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ),
                "",
            )
            classes.append(
                {"name": node.name, "line": node.lineno, "tests": methods, "model": model}
            )
    return {"path": _rel(repo, path), "classes": classes}


def extract_families(repo: Path) -> list[dict[str, Any]]:
    """One record per family directory, sorted by name."""
    models = Path(repo) / MODELING_V2_REL / "models"
    families: list[dict[str, Any]] = []
    for directory in sorted(p for p in models.glob("*") if p.is_dir()):
        if directory.name.startswith(("_", ".")):
            continue
        routing = directory / "routing.py"
        tree = _parse(routing)
        routes = routed_architectures(tree) if tree is not None else []
        targets = [
            _target(Path(repo), directory.name, child)
            for child in sorted(directory.iterdir())
            if child.is_dir() and (child / "modeling.py").is_file()
        ]
        families.append(
            {
                "family": directory.name,
                "architectures": [arch for arch, _ in routes],
                "routing": f"{_rel(Path(repo), routing)}:{routes[0][1]}" if routes else "",
                "targets": targets,
                "has_native": (directory / "csrc").is_dir(),
                "accuracy_tests": _accuracy_tests(Path(repo), directory.name),
            }
        )
    return families


def _dotted(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _wrapped_calls(method: ast.AST) -> list[str]:
    """What ``__call__`` calls into: ``torch.ops.<ns>.<op>``, ``flashinfer.*``, ``native:<op>``."""
    calls: list[str] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name.startswith("torch.ops.") and name.count(".") >= 3:
            calls.append(".".join(name.split(".")[:4]))
        elif name.startswith("flashinfer."):
            calls.append(name)
        elif "native" in name.split(".")[0] and "." in name:
            calls.append(f"native:{name.split('.', 1)[1]}")
    return list(dict.fromkeys(calls))


def _archs(value: ast.expr) -> list[str]:
    names: list[str] = []
    for node in ast.walk(value):
        if isinstance(node, ast.Attribute) and node.attr.upper().startswith("SM_"):
            names.append(node.attr.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.lower().startswith("sm_"):
                names.append(node.value.lower())
    return sorted(dict.fromkeys(names))


def _is_wrapper(node: ast.ClassDef) -> bool:
    return any(_dotted(base).endswith("OpWrapper") for base in node.bases)


def _scope(package: Path, parts: tuple[str, ...]) -> tuple[str, str, str, str]:
    """``(scope, family, target, category)`` of a file at ``parts`` under the package.

    ``catalog/<category>/x.py``; ``models/<family>/ops/[<category>/]x.py``;
    ``models/<family>/<target>/ops/[<category>/]x.py``.
    """
    dirs = parts[:-1]
    if dirs[0] == "catalog":
        return "catalog", "", "", dirs[1] if len(dirs) > 1 else ""
    family = dirs[1] if len(dirs) > 1 else ""
    target = ""
    if len(dirs) > 2 and (package / "models" / family / dirs[2] / "modeling.py").is_file():
        target = dirs[2]
    category = dirs[dirs.index("ops") + 1] if "ops" in dirs[:-1] else ""
    return ("target" if target else "family"), family, target, category


def _op_records(repo: Path, path: Path, package: Path) -> Iterable[dict[str, Any]]:
    tree = _parse(path)
    if tree is None:
        return
    module = _rel(repo, path)[: -len(".py")].replace("/", ".")
    scope, family, target, category = _scope(package, path.relative_to(package).parts)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_wrapper(node):
            continue
        archs: list[str] = []
        has_cells = False
        calls: list[str] = []
        for item in node.body:
            targets = item.targets if isinstance(item, ast.Assign) else []
            if isinstance(item, ast.AnnAssign):
                targets = [item.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if "ARCHS" in names and getattr(item, "value", None) is not None:
                archs = _archs(item.value)
            has_cells = has_cells or "CELLS" in names
            if isinstance(item, ast.FunctionDef) and item.name == "__call__":
                calls = _wrapped_calls(item)
        yield {
            "name": node.name,
            "module": module,
            "path": f"{_rel(repo, path)}:{node.lineno}",
            "scope": scope,
            "family": family,
            "target": target,
            "category": category,
            "archs": archs,
            "has_cells": has_cells,
            "calls": calls,
            "tests": [],
        }


def _imported_modules(repo: Path, path: Path) -> set[str]:
    """The dotted modules (and ``module.name`` of each ``from`` import) the file at ``path`` imports."""
    tree = _parse(path)
    if tree is None:
        return set()
    module = _rel(repo, path)[: -len(".py")].replace("/", ".")
    package = (
        module[: -len(".__init__")] if module.endswith(".__init__") else module.rpartition(".")[0]
    )
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join([*parts, node.module] if node.module else parts)
            else:
                base = node.module or ""
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _target_imports(repo: Path) -> dict[str, set[str]]:
    """Target path -> every module its files import."""
    models = repo / MODELING_V2_REL / "models"
    imports: dict[str, set[str]] = {}
    for modeling in sorted(models.glob("*/*/modeling.py")):
        target = modeling.parent
        imports[_rel(repo, target)] = set().union(
            *(_imported_modules(repo, py) for py in sorted(target.rglob("*.py")))
        )
    return imports


def extract_ops(repo: Path) -> list[dict[str, Any]]:
    """Every ``OpWrapper`` class in the package.

    Each carries the unit tests that name it and the targets whose files import it
    (``used_by``).
    """
    repo = Path(repo)
    package = repo / MODELING_V2_REL
    files = [*package.glob("catalog/*/**/*.py"), *package.glob("models/*/**/*.py")]
    ops: list[dict[str, Any]] = []
    for path in sorted(files):
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        ops.extend(_op_records(repo, path, package))
    tests = sorted((repo / MODELING_V2_TESTS_REL).glob("**/test_*.py"))
    texts = {}
    for test in tests:
        try:
            texts[_rel(repo, test)] = test.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    target_imports = _target_imports(repo)
    for op in ops:
        module = op["module"]
        op["used_by"] = sorted(
            target for target, imported in target_imports.items() if module in imported
        )
        stem = module.rsplit(".", 1)[1]
        op["tests"] = [
            rel
            for rel, text in texts.items()
            if module in text or (op["name"] in text and stem in text)
        ]
    return ops
