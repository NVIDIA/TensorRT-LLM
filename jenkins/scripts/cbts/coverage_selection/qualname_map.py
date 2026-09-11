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
"""Map changed source lines to co_qualname strings matching the touch DB."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field


@dataclass
class _Scope:
    qualname: str
    sig_start: int
    body_start: int
    body_end: int
    body_attr: str
    sig_attr: str


@dataclass
class PythonChangeAnalysis:
    """Static facts about changed module bindings and local dependencies."""

    changed_bindings: set[str]
    import_consumers: set[str]
    callers: dict[str, set[str]]
    unresolved_reason: str = ""
    caller_escapes: set[str] = field(default_factory=set)


def _substatements(node: ast.stmt):
    """Yield direct sub-statements of a compound statement (no new scope)."""
    for field_name in ("body", "orelse", "finalbody"):
        yield from getattr(node, field_name, None) or []
    for handler in getattr(node, "handlers", None) or []:
        yield from handler.body
    for case in getattr(node, "cases", None) or []:
        yield from case.body


def _collect_scopes(tree: ast.Module) -> list[_Scope]:
    scopes: list[_Scope] = []

    def walk(stmts, prefix: str, enclosing_attr: str) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = prefix + node.name
                recorded = "<locals>" not in qual
                sig_start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                body_start = node.body[0].lineno
                body_attr = qual if recorded else enclosing_attr
                scopes.append(
                    _Scope(qual, sig_start, body_start, node.end_lineno, body_attr, enclosing_attr)
                )
                if isinstance(node, ast.ClassDef):
                    walk(node.body, qual + ".", body_attr)
                else:
                    walk(node.body, qual + ".<locals>.", body_attr)
            else:
                subs = list(_substatements(node))
                if subs:
                    walk(subs, prefix, enclosing_attr)

    walk(tree.body, "", "<module>")
    return scopes


def _innermost(line: int, scopes: list[_Scope]) -> _Scope | None:
    best: _Scope | None = None
    for s in scopes:
        if s.sig_start <= line <= s.body_end and (best is None or s.sig_start > best.sig_start):
            best = s
    return best


def _attribute(line: int, scopes: list[_Scope]) -> str:
    best = _innermost(line, scopes)
    if best is None:
        return "<module>"
    return best.sig_attr if line < best.body_start else best.body_attr


def qualnames_for_lines(source: str, lines: set[int]) -> tuple[set[str], bool]:
    """Return (qualnames, ok); ok=False when the source cannot be parsed."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), False
    scopes = _collect_scopes(tree)
    return {_attribute(ln, scopes) for ln in lines}, True


def import_executed_qualnames(source: str) -> set[str]:
    """Qualnames whose code runs once at import: `<module>` and every class body."""
    out = {"<module>"}

    def walk(stmts, prefix: str) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = prefix + node.name
                if isinstance(node, ast.ClassDef):
                    out.add(qual)
                    walk(node.body, qual + ".")
                else:
                    walk(node.body, qual + ".<locals>.")
            else:
                walk(list(_substatements(node)), prefix)

    try:
        walk(ast.parse(source).body, "")
    except SyntaxError:
        pass
    return out


def closure_attributed_qualnames(source: str, lines: set[int]) -> set[str]:
    """Qualnames a changed line only reaches by walking out of a `<locals>` scope."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    scopes = _collect_scopes(tree)
    out: set[str] = set()
    for line in lines:
        best = _innermost(line, scopes)
        if best is not None and "<locals>" in best.qualname:
            out.add(_attribute(line, scopes))
    return out


def _is_literal_expression(node: ast.expr) -> bool:
    """Return whether evaluating ``node`` cannot call user code."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is None or _is_literal_expression(key) for key in node.keys) and all(
            _is_literal_expression(value) for value in node.values
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_literal_expression(node.operand)
    return False


def _module_binding_names(node: ast.stmt, future_annotations: bool) -> set[str] | None:
    """Names created by a low-risk module statement, or None when unsupported."""
    if isinstance(node, ast.Assign):
        if not _is_literal_expression(node.value):
            return None
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if len(names) != len(node.targets):
            return None
        return names
    if isinstance(node, ast.AnnAssign):
        if (
            not future_annotations
            or not isinstance(node.target, ast.Name)
            or node.value is None
            or not _is_literal_expression(node.value)
        ):
            return None
        return {node.target.id}
    if isinstance(node, ast.Import):
        roots = {alias.name.partition(".")[0] for alias in node.names}
        if not roots or not roots <= set(sys.builtin_module_names):
            return None
        return {alias.asname or alias.name.partition(".")[0] for alias in node.names}
    return None


def _safe_added_function(
    node: ast.stmt, future_annotations: bool
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return a side-effect-free added function declaration, else None."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    arguments = node.args
    annotations = [
        *(argument.annotation for argument in arguments.posonlyargs),
        *(argument.annotation for argument in arguments.args),
        *(argument.annotation for argument in arguments.kwonlyargs),
        arguments.vararg.annotation if arguments.vararg else None,
        arguments.kwarg.annotation if arguments.kwarg else None,
        node.returns,
    ]
    if (
        node.decorator_list
        or arguments.defaults
        or any(default is not None for default in arguments.kw_defaults)
        or (not future_annotations and any(annotation is not None for annotation in annotations))
    ):
        return None
    return node


def _node_for_line(nodes: list[ast.stmt], line: int) -> ast.stmt | None:
    matches = [node for node in nodes if node.lineno <= line <= node.end_lineno]
    return min(matches, key=lambda node: node.end_lineno - node.lineno) if matches else None


def _recorded_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def walk(statements: list[ast.stmt], prefix: str) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[prefix + node.name] = node
            elif isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + ".")

    walk(tree.body, "")
    return functions


class _FunctionNames(ast.NodeVisitor):
    """Names loaded, bound, and directly called in one function body."""

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.loaded: set[str] = set()
        self.nested_loaded: set[str] = set()
        self.bound: set[str] = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)
        self.globals: set[str] = set()
        self.calls: set[str] = set()
        for statement in node.body:
            self.visit(statement)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        self.bound.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.bound.update(alias.asname or alias.name for alias in node.names)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        else:
            self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def _include_nested_loads(self, node: ast.AST) -> None:
        # Nested code objects are not recorded separately. Attribute their
        # binding dependencies conservatively to the recorded outer function,
        # but do not infer caller edges across the nested scope.
        self.nested_loaded.update(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self._include_nested_loads(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bound.add(node.name)
        self._include_nested_loads(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self._include_nested_loads(node)


def analyze_python_changes(
    source: str, changed_lines: set[int], deleted_lines: dict[int, list[str]]
) -> PythonChangeAnalysis:
    """Describe low-risk import-time bindings and their local references.

    Literal assignments (including literal replacements), plain function
    declarations, and newly added builtin-module imports are represented.
    Unsupported syntax is returned as an unresolved fact for policy callers.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PythonChangeAnalysis(set(), set(), {}, "unparsable source")

    scopes = _collect_scopes(tree)
    import_qualnames = import_executed_qualnames(source)
    import_lines = {line for line in changed_lines if _attribute(line, scopes) in import_qualnames}
    module_nodes = list(tree.body)
    future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )

    binding_names: set[str] = set()
    deleted_binding_names: set[str] = set()
    direct_consumers: set[str] = set()
    for line in sorted(import_lines):
        if _attribute(line, scopes) != "<module>":
            return PythonChangeAnalysis(set(), set(), {}, "class/signature import change")
        node = _node_for_line(module_nodes, line)
        added_function = (
            _safe_added_function(node, future_annotations) if node is not None else None
        )
        if added_function is not None and line not in deleted_lines:
            binding_names.add(added_function.name)
            direct_consumers.add(added_function.name)
            continue
        names = _module_binding_names(node, future_annotations) if node is not None else None
        if node is None and line in deleted_lines:
            try:
                old_tree = ast.parse("\n".join(deleted_lines[line]))
            except SyntaxError:
                return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")
            if len(old_tree.body) != 1:
                return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")
            old_node = old_tree.body[0]
            old_names = _module_binding_names(old_node, future_annotations)
            if old_names is None or isinstance(old_node, ast.Import):
                return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")
            deleted_binding_names.update(old_names)
            continue
        if names is None:
            return PythonChangeAnalysis(set(), set(), {}, "effectful module statement")
        if line in deleted_lines:
            try:
                old_tree = ast.parse("\n".join(deleted_lines[line]))
            except SyntaxError:
                return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")
            if len(old_tree.body) != 1:
                return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")
            old_node = old_tree.body[0]
            old_names = _module_binding_names(old_node, future_annotations)
            if (
                old_names != names
                or isinstance(node, ast.Import)
                or isinstance(old_node, ast.Import)
            ):
                return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")
        binding_names.update(names)
    if not deleted_binding_names <= binding_names:
        return PythonChangeAnalysis(set(), set(), {}, "unresolved import replacement")

    functions = _recorded_functions(tree)
    facts = {qualname: _FunctionNames(node) for qualname, node in functions.items()}
    consumers = direct_consumers | {
        qualname
        for qualname, fact in facts.items()
        if (
            fact.nested_loaded
            | {name for name in fact.loaded if name not in fact.bound or name in fact.globals}
        )
        & binding_names
    }
    import_loads = {
        _attribute(node.lineno, scopes)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in binding_names
        and _attribute(node.lineno, scopes) in import_qualnames
    }
    if import_loads:
        return PythonChangeAnalysis(set(), set(), {}, "import-time binding consumer")

    module_functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    callers: dict[str, set[str]] = {}
    for caller, fact in facts.items():
        for callee in fact.calls & module_functions:
            if callee not in fact.bound or callee in fact.globals:
                callers.setdefault(callee, set()).add(caller)

    caller_escapes = {
        name
        for fact in facts.values()
        for name in (
            fact.nested_loaded
            | {
                loaded
                for loaded in fact.loaded
                if loaded not in fact.bound or loaded in fact.globals
            }
        )
        if name in module_functions
    }
    caller_escapes.update(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in module_functions
        and _attribute(node.lineno, scopes) in import_qualnames
    )

    return PythonChangeAnalysis(binding_names, consumers, callers, caller_escapes=caller_escapes)
