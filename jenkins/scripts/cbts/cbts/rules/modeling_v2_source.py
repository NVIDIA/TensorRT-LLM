# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""What modeling_v2 source files import, and which targets a family routes to.

ModelingV2Rule selects tests with these readings, and
`tests/unittest/_torch/modeling_v2/test_modeling_v2_ci_selection.py` checks the
tree with the same ones, so the rule and the check cannot disagree on what a
file imports. The check loads this file by path, so it imports only the
standard library.
"""

from __future__ import annotations

import ast
from typing import Optional

PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"


def resolve(name: str, level: int, package: str) -> Optional[str]:
    """The absolute name of `name` imported with `level` leading dots from `package`."""
    if level == 0:
        return name
    parts = package.split(".")
    if level - 1 >= len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, name] if name else base)


def _is_str(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _loader_aliases(node: ast.ImportFrom) -> list[str]:
    """The names `from importlib import import_module [as x]` binds to `import_module`."""
    if node.level or node.module != "importlib":
        return []
    return [alias.asname or alias.name for alias in node.names if alias.name == "import_module"]


def import_loaders(tree: ast.AST) -> set[str]:
    """The bare names a module can call `importlib.import_module` by."""
    loaders = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            loaders.update(_loader_aliases(node))
    return loaders


def dynamic_import(call: ast.Call, package: str, loaders: set[str]) -> Optional[str]:
    """The module an `import_module(...)` / `__import__(...)` call names; "" if computed.

    None if the call is not an import this reads. `loaders` are the bare names
    bound to `importlib.import_module`; any `<x>.import_module(...)` counts,
    which covers `importlib` imported under another name.
    """
    func = call.func
    if (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
        isinstance(func, ast.Name) and func.id in loaders
    ):
        anchor = call.args[1] if len(call.args) > 1 else None
        anchor = next((k.value for k in call.keywords if k.arg == "package"), anchor)
    elif isinstance(func, ast.Name) and func.id == "__import__":
        # A `level` argument makes the name relative; read it as computed.
        if len(call.args) > 4 or any(k.arg == "level" for k in call.keywords):
            return ""
        anchor = None
    else:
        return None
    if not call.args or not _is_str(call.args[0]):
        return ""
    name = call.args[0].value
    level = len(name) - len(name.lstrip("."))
    if level == 0:
        return name
    if _is_str(anchor):
        anchor_package = anchor.value
    elif isinstance(anchor, ast.Name) and anchor.id == "__package__":
        anchor_package = package
    else:
        return ""
    return resolve(name[level:], level, anchor_package) or ""


def imported_names(tree: ast.AST, package: str) -> tuple[set[str], bool]:
    """Absolute names a module in `package` imports anywhere, and whether it computes one at run time.

    `from X import Y` names both X and X.Y, as Y may be a submodule. String
    literals passed to `import_module` or `__import__` count as imports.
    """
    names: set[str] = set()
    calls: list[ast.Call] = []
    loaders = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve(node.module or "", node.level, package)
            if base:
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            loaders.update(_loader_aliases(node))
        elif isinstance(node, ast.Call):
            calls.append(node)
    computed = False
    for call in calls:
        name = dynamic_import(call, package, loaders)
        if name == "":
            computed = True
        elif name:
            names.add(name)
    return names, computed


def in_package(name: str) -> bool:
    return name == PACKAGE or name.startswith(f"{PACKAGE}.")


def package_names(names: set[str]) -> set[str]:
    """The names under the subtree among `names`, with the packages that hold them.

    Importing a.b.c runs a/__init__.py and a/b/__init__.py too. A name counts
    whether or not its file exists, so the importers of a deleted or moved
    module are still found: CI reads the PR head, where the file is gone.
    """
    depth = PACKAGE.count(".") + 1
    found = set()
    for name in filter(in_package, names):
        parts = name.split(".")
        found.update(".".join(parts[:i]) for i in range(depth, len(parts) + 1))
    return found


def _is_route(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
        isinstance(d, ast.Call) and getattr(d.func, "id", getattr(d.func, "attr", None)) == "routes"
        for d in node.decorator_list
    )


def route_returns(tree: ast.AST) -> list[tuple[ast.Return, Optional[str]]]:
    """Each `return` of a `@routes(...)` function in `routing.py`, with the target it returns.

    The target is the directory `<D>` of the `from .<D>... import <Class>` that
    binds the returned name. It is None when the return is not a name, or the
    name is bound by any other import or by imports from different directories.
    Returns of functions nested in a route function do not count.
    """
    bound: dict[str, set[Optional[str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            target = None
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = node.module.split(".")[0]
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                bound.setdefault(name, set()).add(target)
    returns = []
    for function in filter(_is_route, ast.walk(tree)):
        todo = list(function.body)
        while todo:
            node = todo.pop()
            if isinstance(node, ast.Return):
                returns.append(node)
            elif not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                todo.extend(ast.iter_child_nodes(node))
    found = []
    for node in sorted(returns, key=lambda node: node.lineno):
        targets = bound.get(node.value.id, set()) if isinstance(node.value, ast.Name) else set()
        found.append((node, next(iter(targets)) if len(targets) == 1 else None))
    return found


def routed_targets(tree: ast.AST) -> Optional[set[str]]:
    """The target directories a family's `routing.py` returns classes from.

    None when a return's target is unknown or there is no return, so a caller
    cannot tell targets from family code.
    """
    targets = [target for _, target in route_returns(tree)]
    if not targets or None in targets:
        return None
    return set(targets)
