# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Import-direction contracts between serving layers, read from the source tree.

The import graph is built with ``ast`` from the repository files, not the
installed package, and needs no new dependency. Each package a PR creates adds
its :class:`Contract` to ``tests/unittest/serve/test_layering.py``.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "Contract",
    "ImportEdge",
    "build_import_graph",
    "check_contracts",
    "stale_allowances",
]


@dataclass(frozen=True)
class ImportEdge:
    importer: str  # dotted module of the importing file
    imported: str  # absolute dotted module (relative imports resolved)
    names: Tuple[str, ...]  # names of a from-import; () for "import x"
    lineno: int
    in_function: bool  # inside a def or class body (runs at call time)
    type_checking: bool  # under "if TYPE_CHECKING:"


@dataclass(frozen=True)
class Contract:
    name: str
    importers: Tuple[str, ...]  # module prefixes the rule applies to
    forbidden: Tuple[str, ...]  # imported-module prefixes that are not allowed
    allowed: FrozenSet[Tuple[str, str, str]] = field(default_factory=frozenset)
    include_type_checking: bool = False
    fix: str = ""  # printed on failure: which contract, where the code goes


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, module: str, package: str) -> None:
        self._module = module
        self._package = package
        self._function_depth = 0
        self._type_checking_depth = 0
        self.edges: List[ImportEdge] = []

    def _add(self, imported: Optional[str], names: Tuple[str, ...], lineno: int) -> None:
        if imported is None:
            return
        self.edges.append(
            ImportEdge(
                importer=self._module,
                imported=imported,
                names=names,
                lineno=lineno,
                in_function=self._function_depth > 0,
                type_checking=self._type_checking_depth > 0,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name, (), node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported = _resolve_relative(self._package, node.level, node.module)
        self._add(imported, tuple(alias.name for alias in node.names), node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        # importlib.import_module("<literal>") calls are edges with names=().
        target = node.func
        is_import_module = (
            isinstance(target, ast.Attribute)
            and target.attr == "import_module"
            and isinstance(target.value, ast.Name)
            and target.value.id == "importlib"
        ) or (isinstance(target, ast.Name) and target.id == "import_module")
        if is_import_module and node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                self._add(value, (), node.lineno)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            self._type_checking_depth += 1
            for child in node.body:
                self.visit(child)
            self._type_checking_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def _visit_scope(self, node) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _resolve_relative(package: str, level: int, module: Optional[str]) -> Optional[str]:
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    if module:
        base.extend(module.split("."))
    return ".".join(base) if base else None


def build_import_graph(package_dir: Path, package: str) -> List[ImportEdge]:
    """Every import edge of every ``*.py`` under ``package_dir``.

    Module-level and function-level imports, relative imports resolved, and
    ``importlib.import_module("<literal>")`` calls recorded as edges with
    ``names=()``.
    """
    edges: List[ImportEdge] = []
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)
        parts = [package, *relative.parts]
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        module = ".".join(parts)
        file_package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        collector = _ImportCollector(module, file_package)
        collector.visit(tree)
        edges.extend(collector.edges)
    return edges


def _edge_triples(edge: ImportEdge) -> List[Tuple[str, str, str]]:
    names = edge.names or ("",)
    return [(edge.importer, edge.imported, name) for name in names]


def check_contracts(edges: Sequence[ImportEdge], contracts: Sequence[Contract]) -> List[str]:
    """Violations of each contract, as messages that name the contract and the fix.

    A from-import is matched as ``<module>.<name>``, so ``from pkg import sub``
    and ``from .. import sub`` count as imports of the submodule ``pkg.sub``.
    """
    problems = []
    for contract in contracts:
        for edge in edges:
            if edge.type_checking and not contract.include_type_checking:
                continue
            if not any(_matches_prefix(edge.importer, prefix) for prefix in contract.importers):
                continue
            for triple in _edge_triples(edge):
                importer, imported, name = triple
                target = f"{imported}.{name}" if name else imported
                if not any(_matches_prefix(target, prefix) for prefix in contract.forbidden):
                    continue
                if triple in contract.allowed:
                    continue
                problems.append(
                    f"contract '{contract.name}': {importer} (line {edge.lineno}) imports "
                    f"`{target}`. {contract.fix}"
                )
    return problems


def stale_allowances(edges: Sequence[ImportEdge], contracts: Sequence[Contract]) -> List[str]:
    """Baseline allowances that match no current import; the baseline only shrinks."""
    live = {triple for edge in edges for triple in _edge_triples(edge)}
    problems = []
    for contract in contracts:
        for triple in sorted(contract.allowed):
            if triple not in live:
                importer, imported, name = triple
                problems.append(
                    f"contract '{contract.name}': the allowance ({importer}, {imported}, "
                    f"{name!r}) matches no import anymore; delete it, the baseline only shrinks."
                )
    return problems
