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
"""Static Python change facts shared by CBTS selection policies."""

from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import dataclass, field


@dataclass
class _Scope:
    qualname: str
    sig_start: int
    body_start: int
    body_end: int
    body_attr: str
    sig_attr: str


@dataclass(frozen=True, order=True)
class ImportTarget:
    """A statically named binding imported from one module."""

    module: str
    level: int
    name: str


@dataclass
class PythonChangeFacts:
    """Static facts about changed module bindings and local dependencies."""

    changed_bindings: set[str]
    binding_consumers: set[str]
    callers: dict[str, set[str]]
    limitation: str = ""
    callable_escapes: set[str] = field(default_factory=set)
    new_import_targets: set[ImportTarget] = field(default_factory=set)
    old_import_targets: set[ImportTarget] = field(default_factory=set)
    new_import_bindings: set[str] = field(default_factory=set)
    new_declaration_bindings: set[str] = field(default_factory=set)


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


def _import_from_bindings(node: ast.stmt) -> dict[str, ImportTarget] | None:
    """Return local bindings for one statically named ``from`` import."""
    if not isinstance(node, ast.ImportFrom) or node.module is None:
        return None
    if any(alias.name == "*" for alias in node.names):
        return None
    bindings = {
        alias.asname or alias.name: ImportTarget(node.module, node.level, alias.name)
        for alias in node.names
    }
    return bindings if len(bindings) == len(node.names) else None


def _module_import_from_bindings(tree: ast.Module) -> dict[str, ImportTarget] | None:
    """Return unambiguous direct ``from``-import bindings for one module."""
    bindings: dict[str, ImportTarget] = {}
    binding_counts: dict[str, int] = {}
    for node in tree.body:
        for local in _direct_scope_bindings([node]):
            binding_counts[local] = binding_counts.get(local, 0) + 1
        if not isinstance(node, ast.ImportFrom):
            continue
        imported = _import_from_bindings(node)
        if imported is None or imported.keys() & bindings.keys():
            return None
        bindings.update(imported)
    if any(binding_counts[local] != 1 for local in bindings):
        return None
    return bindings


def _module_import_from_delta(
    tree: ast.Module, old_tree: ast.Module
) -> tuple[set[str], set[ImportTarget], set[ImportTarget], set[str]] | None:
    """Describe the complete direct ``from``-import binding delta."""
    current = _module_import_from_bindings(tree)
    previous = _module_import_from_bindings(old_tree)
    if current is None or previous is None:
        return None
    changed_locals = {
        local
        for local in previous.keys() | current.keys()
        if previous.get(local) != current.get(local)
    }
    return (
        changed_locals,
        {previous[local] for local in changed_locals if local in previous},
        {current[local] for local in changed_locals if local in current},
        current.keys() - previous.keys(),
    )


def _statically_bound_names(tree: ast.Module) -> set[str]:
    """Over-approximate names that static syntax could bind in a module."""
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
    return names


def _import_from_replacement(
    node: ast.stmt,
    deleted: list[str],
    old_module_nodes: list[ast.stmt] | None,
) -> tuple[set[str], set[ImportTarget], set[ImportTarget], set[str]] | None:
    """Describe a same-module static ``from``-import replacement."""
    current = _import_from_bindings(node)
    if current is None or not isinstance(node, ast.ImportFrom):
        return None
    if old_module_nodes is not None:
        candidates = [
            old_node
            for old_node in old_module_nodes
            if isinstance(old_node, ast.ImportFrom)
            and old_node.module == node.module
            and old_node.level == node.level
        ]
        if len(candidates) != 1:
            return None
        old_node = candidates[0]
    else:
        try:
            old_tree = ast.parse("\n".join(deleted))
        except SyntaxError:
            return None
        if len(old_tree.body) != 1 or not isinstance(old_tree.body[0], ast.ImportFrom):
            return None
        old_node = old_tree.body[0]
    previous = _import_from_bindings(old_node)
    if previous is None or old_node.module != node.module or old_node.level != node.level:
        return None

    changed_locals = {
        local
        for local in previous.keys() | current.keys()
        if previous.get(local) != current.get(local)
    }
    old_targets = {previous[local] for local in changed_locals if local in previous}
    new_targets = {current[local] for local in changed_locals if local in current}
    added_locals = current.keys() - previous.keys()
    return changed_locals, old_targets, new_targets, added_locals


def _trusted_type_checking_guard(node: ast.If, module_nodes: list[ast.stmt]) -> bool:
    """Return whether ``node`` is a trusted, import-only TYPE_CHECKING block."""
    if (
        node.orelse
        or not node.body
        or not all(isinstance(statement, (ast.Import, ast.ImportFrom)) for statement in node.body)
    ):
        return False

    if isinstance(node.test, ast.Name):
        guard_root = node.test.id

        def establishes_guard(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "typing"
                and any(
                    alias.name == "TYPE_CHECKING" and (alias.asname or alias.name) == guard_root
                    for alias in statement.names
                )
            )

    elif (
        isinstance(node.test, ast.Attribute)
        and node.test.attr == "TYPE_CHECKING"
        and isinstance(node.test.value, ast.Name)
    ):
        guard_root = node.test.value.id

        def establishes_guard(statement: ast.stmt) -> bool:
            return isinstance(statement, ast.Import) and any(
                alias.name == "typing" and (alias.asname or alias.name) == guard_root
                for alias in statement.names
            )

    else:
        return False

    trusted = False
    for statement in module_nodes:
        if statement is node:
            break
        if guard_root in _direct_scope_bindings([statement]):
            trusted = establishes_guard(statement)
    return trusted


def _line_in_import_only_block(node: ast.If, line: int) -> bool:
    """Return whether ``line`` belongs to an import inside ``node``."""
    return any(statement.lineno <= line <= statement.end_lineno for statement in node.body)


_SAFE_ANNOTATION_BUILTINS = {
    "bool",
    "bytes",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "object",
    "set",
    "str",
    "tuple",
    "type",
}
_SAFE_BUILTIN_DECORATORS = {"classmethod", "property", "staticmethod"}
# These module globals permit a plain annotation load, not arbitrary type operators.
_SAFE_ANNOTATION_MODULE_LOADS = {"torch": {"Tensor", "dtype"}}
_SAFE_TYPING_IMPORTS = {
    "Any",
    "Callable",
    "ClassVar",
    "Dict",
    "FrozenSet",
    "List",
    "Optional",
    "Set",
    "Tuple",
    "Type",
    "Union",
}


def _cached_typing_imports(
    tree: ast.Module, old_tree: ast.Module | None, old_bound_names: set[str]
) -> dict[str, ImportTarget]:
    """New known typing globals, when the pre-image already loads that module."""
    if old_tree is None or not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "typing"
        and not node.level
        or isinstance(node, ast.Import)
        and any(alias.name == "typing" for alias in node.names)
        for node in old_tree.body
    ):
        return {}
    imported = _module_import_from_bindings(tree)
    typing_aliases = {
        alias.asname or alias.name
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Import)
        for alias in statement.names
        if alias.name == "typing"
    }
    changed_typing_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and isinstance(node.value, ast.Name)
        and node.value.id in typing_aliases
    }
    return {
        local: target
        for local, target in (imported or {}).items()
        if local not in old_bound_names
        and not target.level
        and target.module == "typing"
        and target.name in _SAFE_TYPING_IMPORTS
        and target.name not in changed_typing_attributes
    }


def _annotation_name_loads(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    load_names: set[str],
    generic_names: set[str],
    modules: set[str],
    *,
    future_annotations: bool,
) -> set[int]:
    """Name nodes in annotations whose evaluation was independently validated."""
    arguments = function.args
    annotations = [
        *(argument.annotation for argument in arguments.posonlyargs),
        *(argument.annotation for argument in arguments.args),
        *(argument.annotation for argument in arguments.kwonlyargs),
        arguments.vararg.annotation if arguments.vararg else None,
        arguments.kwarg.annotation if arguments.kwarg else None,
        function.returns,
    ]
    return {
        id(node)
        for annotation in annotations
        if annotation is not None
        and (future_annotations or _safe_annotation(annotation, load_names, generic_names, modules))
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _trusted_annotation_bindings(
    tree: ast.Module, before_line: int
) -> tuple[set[str], set[str], set[str]]:
    """Return load-safe names, subscript-safe names, and trusted modules."""
    load_names = set(_SAFE_ANNOTATION_BUILTINS)
    generic_names = set(_SAFE_ANNOTATION_BUILTINS)
    modules: set[str] = set()
    for node in tree.body:
        if node.end_lineno >= before_line:
            continue
        changes = _AnnotationScopeChanges()
        changes.visit(node)
        bound_names = changes.names
        changed_attributes = changes.attributes
        generic_names.difference_update(bound_names)
        modules.difference_update(bound_names)
        load_names.difference_update(
            {
                name
                for name in load_names
                if "." in name
                and (
                    "*" in bound_names
                    or name.partition(".")[0] in bound_names
                    or name in changed_attributes
                )
            }
        )
        if isinstance(node, ast.ImportFrom):
            imported_names = {
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            }
            load_names.update(imported_names)
            if not node.level and node.module in {"collections.abc", "typing"}:
                generic_names.update(imported_names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for attribute in _SAFE_ANNOTATION_MODULE_LOADS.get(alias.name, set()):
                    load_names.add(f"{alias.asname or alias.name}.{attribute}")
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"collections.abc", "typing"}
            )
    return load_names, generic_names, modules


def _safe_annotation(
    node: ast.expr | None,
    load_names: set[str],
    generic_names: set[str],
    modules: set[str],
) -> bool:
    """Return whether evaluating a newly added annotation is side-effect-free."""
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in load_names
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id in modules or f"{node.value.id}.{node.attr}" in load_names
    if isinstance(node, ast.Subscript):
        base_is_safe = (
            isinstance(node.value, ast.Name)
            and node.value.id in generic_names
            or isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in modules
        )
        return base_is_safe and _safe_annotation(node.slice, load_names, generic_names, modules)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _safe_annotation_operator_operand(
            node.left, generic_names, modules
        ) and _safe_annotation_operator_operand(node.right, generic_names, modules)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_safe_annotation(item, load_names, generic_names, modules) for item in node.elts)
    return False


def _safe_annotation_operator_operand(
    node: ast.expr, generic_names: set[str], modules: set[str]
) -> bool:
    """Return whether an annotation operand has trusted type operators."""
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.Name):
        return node.id in generic_names
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id in modules
    if isinstance(node, ast.Subscript):
        return _safe_annotation(node, generic_names, generic_names, modules)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _safe_annotation_operator_operand(
            node.left, generic_names, modules
        ) and _safe_annotation_operator_operand(node.right, generic_names, modules)
    return False


def _safe_added_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    future_annotations: bool,
    annotation_load_names: set[str],
    annotation_generic_names: set[str],
    annotation_modules: set[str],
    trusted_decorators: set[str],
) -> bool:
    """Return whether evaluating an added function declaration is low risk."""
    if any(
        not isinstance(decorator, ast.Name) or decorator.id not in trusted_decorators
        for decorator in node.decorator_list
    ):
        return False
    arguments = node.args
    if not all(_is_literal_expression(default) for default in arguments.defaults) or not all(
        default is None or _is_literal_expression(default) for default in arguments.kw_defaults
    ):
        return False
    if future_annotations:
        return True
    annotations = [
        *(argument.annotation for argument in arguments.posonlyargs),
        *(argument.annotation for argument in arguments.args),
        *(argument.annotation for argument in arguments.kwonlyargs),
        arguments.vararg.annotation if arguments.vararg else None,
        arguments.kwarg.annotation if arguments.kwarg else None,
        node.returns,
    ]
    return all(
        _safe_annotation(
            annotation,
            annotation_load_names,
            annotation_generic_names,
            annotation_modules,
        )
        for annotation in annotations
    )


def _same_ast(left: ast.AST | None, right: ast.AST | None) -> bool:
    if left is None or right is None:
        return left is right
    return ast.dump(left) == ast.dump(right)


def _safe_signature_addition(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    old_node: ast.FunctionDef | ast.AsyncFunctionDef,
    annotation_load_names: set[str],
    annotation_generic_names: set[str],
    annotation_modules: set[str],
) -> bool:
    """Return whether a signature adds import-safe parameters or a return annotation."""
    if type(node) is not type(old_node):
        return False
    return_annotation_added = (
        old_node.returns is None
        and node.returns is not None
        and _safe_annotation(
            node.returns, annotation_load_names, annotation_generic_names, annotation_modules
        )
    )
    if (
        not _same_ast(node.returns, old_node.returns)
        and not return_annotation_added
        or node.type_comment != old_node.type_comment
    ):
        return False
    if return_annotation_added and node.decorator_list:
        # A decorator can consume annotation metadata while the module loads.
        return False
    if len(node.decorator_list) != len(old_node.decorator_list) or any(
        not _same_ast(new, old) for new, old in zip(node.decorator_list, old_node.decorator_list)
    ):
        return False

    arguments = node.args
    old_arguments = old_node.args
    added_positional_count = len(arguments.args) - len(old_arguments.args)
    added_keyword_only_count = len(arguments.kwonlyargs) - len(old_arguments.kwonlyargs)
    if (
        (
            added_positional_count <= 0
            and added_keyword_only_count <= 0
            and not return_annotation_added
        )
        or added_positional_count < 0
        or added_keyword_only_count < 0
        or len(arguments.posonlyargs) != len(old_arguments.posonlyargs)
        or not _same_ast(arguments.vararg, old_arguments.vararg)
        or not _same_ast(arguments.kwarg, old_arguments.kwarg)
        or any(
            not _same_ast(new, old)
            for new, old in zip(arguments.posonlyargs, old_arguments.posonlyargs)
        )
    ):
        return False

    positional = [*arguments.posonlyargs, *arguments.args]
    old_positional = [*old_arguments.posonlyargs, *old_arguments.args]
    positional_defaults = [None] * (len(positional) - len(arguments.defaults)) + arguments.defaults
    old_positional_defaults = [None] * (
        len(old_positional) - len(old_arguments.defaults)
    ) + old_arguments.defaults
    for current, previous, defaults, old_defaults in (
        (positional, old_positional, positional_defaults, old_positional_defaults),
        (
            arguments.kwonlyargs,
            old_arguments.kwonlyargs,
            arguments.kw_defaults,
            old_arguments.kw_defaults,
        ),
    ):
        old_index = 0
        old_names = {argument.arg for argument in previous}
        for argument, default in zip(current, defaults):
            if old_index < len(previous) and _same_ast(argument, previous[old_index]):
                if not _same_ast(default, old_defaults[old_index]):
                    return False
                old_index += 1
            elif (
                argument.arg in old_names
                or default is not None
                and not _is_literal_expression(default)
                or not _safe_annotation(
                    argument.annotation,
                    annotation_load_names,
                    annotation_generic_names,
                    annotation_modules,
                )
            ):
                return False
        if old_index != len(previous):
            return False
    return True


class _DirectScopeBindings(ast.NodeVisitor):
    """Collect names bound directly in one module or class scope."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.names.add(node.name)


class _AnnotationScopeChanges(_DirectScopeBindings):
    """Bindings and attribute writes that can run before an eager annotation."""

    def __init__(self) -> None:
        super().__init__()
        self.attributes: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.attributes.add(f"{node.value.id}.{node.attr}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)
        self.visit(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)


def _direct_scope_bindings(statements: list[ast.stmt]) -> set[str]:
    bindings = _DirectScopeBindings()
    for statement in statements:
        bindings.visit(statement)
    return bindings.names


def _direct_scope_bindings_before(statements: list[ast.stmt], line: int) -> set[str]:
    return _direct_scope_bindings(
        [statement for statement in statements if statement.end_lineno < line]
    )


def _definition_start(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])


def _node_for_line(nodes: list[ast.stmt], line: int) -> ast.stmt | None:
    matches = [node for node in nodes if node.lineno <= line <= node.end_lineno]
    return min(matches, key=lambda node: node.end_lineno - node.lineno) if matches else None


def _is_docstring_replacement(
    node: ast.stmt | None,
    statements: list[ast.stmt],
    old_statements: list[ast.stmt] | None,
) -> bool:
    """Recognize literal docstrings in both images of the same scope."""
    return bool(
        statements
        and old_statements
        and node is statements[0]
        and all(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
            for statement in (statements[0], old_statements[0])
        )
    )


def _safe_added_class_literal_bindings(
    node: ast.stmt,
    old_class: ast.ClassDef,
    tree: ast.Module,
    *,
    future_annotations: bool,
) -> set[str] | None:
    """Return import-safe literal bindings newly added to an existing class."""
    if isinstance(node, ast.Assign):
        if not _is_literal_expression(node.value):
            return None
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if len(names) != len(node.targets):
            return None
    elif isinstance(node, ast.AnnAssign):
        if (
            not isinstance(node.target, ast.Name)
            or node.value is None
            or not _is_literal_expression(node.value)
        ):
            return None
        names = {node.target.id}
        if not future_annotations:
            load_names, generic_names, modules = _trusted_annotation_bindings(tree, node.lineno)
            load_names = {name for name in load_names if "." not in name}
            if not _safe_annotation(node.annotation, load_names, generic_names, modules):
                return None
    else:
        return None

    if names & _direct_scope_bindings(old_class.body):
        return None
    return names


def _recorded_definition_groups(
    tree: ast.Module | None,
) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]]:
    """Collect every definition sharing a recorded qualname, including branches."""
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]] = {}

    def walk(statements: list[ast.stmt], prefix: str) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = prefix + node.name
                definitions.setdefault(qualname, []).append(node)
                if isinstance(node, ast.ClassDef):
                    walk(node.body, qualname + ".")
            else:
                walk(list(_substatements(node)), prefix)

    if tree is not None:
        walk(tree.body, "")
    return definitions


def _recorded_definitions(
    tree: ast.Module | None,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Return declarations whose scope and enclosing classes are unambiguous."""
    definitions = _recorded_definition_groups(tree)
    ambiguous = {qualname for qualname, nodes in definitions.items() if len(nodes) > 1}
    return {
        qualname: nodes[0]
        for qualname, nodes in definitions.items()
        if not any(qualname == name or qualname.startswith(name + ".") for name in ambiguous)
    }


def _recorded_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        qualname: node
        for qualname, node in _recorded_definitions(tree).items()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _recorded_classes(tree: ast.Module | None) -> dict[str, ast.ClassDef]:
    return {
        qualname: node
        for qualname, node in _recorded_definitions(tree).items()
        if isinstance(node, ast.ClassDef)
    }


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

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._include_nested_loads(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._include_nested_loads(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._include_nested_loads(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._include_nested_loads(node)

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
    source: str,
    changed_lines: set[int],
    deleted_lines: dict[int, list[str]],
    *,
    pre_source: str | None = None,
) -> PythonChangeFacts:
    """Describe low-risk import-time bindings and their local references.

    Literal assignments (including literal replacements), low-risk added
    function declarations, newly added builtin-module imports, and static
    top-level ``ImportFrom`` binding deltas are represented. Unsupported syntax
    is returned as an unresolved fact for policy callers.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PythonChangeFacts(set(), set(), {}, "unparsable source")
    try:
        old_tree = ast.parse(pre_source) if pre_source is not None else None
    except SyntaxError:
        return PythonChangeFacts(set(), set(), {}, "unparsable pre-image")
    old_module_nodes = list(old_tree.body) if old_tree is not None else None
    old_bound_names = _statically_bound_names(old_tree) if old_tree is not None else set()
    import_from_delta = _module_import_from_delta(tree, old_tree) if old_tree is not None else None

    scopes = _collect_scopes(tree)
    import_qualnames = import_executed_qualnames(source)
    import_lines = {line for line in changed_lines if _attribute(line, scopes) in import_qualnames}
    definition_groups = _recorded_definition_groups(tree)
    old_definition_groups = _recorded_definition_groups(old_tree)
    ambiguous_qualnames = {
        qualname
        for groups in (definition_groups, old_definition_groups)
        for qualname, nodes in groups.items()
        if len(nodes) > 1
    }
    for line in import_lines:
        scope = _innermost(line, scopes)
        if scope is not None and any(
            scope.qualname == qualname or scope.qualname.startswith(qualname + ".")
            for qualname in ambiguous_qualnames
        ):
            return PythonChangeFacts(set(), set(), {}, "ambiguous class/function declaration")
    module_nodes = list(tree.body)
    future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )
    functions = _recorded_functions(tree)
    old_functions = _recorded_functions(old_tree) if old_tree is not None else {}
    classes = _recorded_classes(tree)
    old_classes = _recorded_classes(old_tree)
    module_bindings = _direct_scope_bindings(tree.body)
    old_module_bindings = _direct_scope_bindings(old_tree.body) if old_tree is not None else set()

    binding_names: set[str] = set()
    deleted_binding_names: set[str] = set()
    direct_consumers: set[str] = set()
    new_import_targets: set[ImportTarget] = set()
    old_import_targets: set[ImportTarget] = set()
    new_import_bindings: set[str] = set()
    new_declaration_bindings: set[str] = set()
    cached_typing_imports = _cached_typing_imports(tree, old_tree, old_bound_names)
    safe_annotation_loads: set[int] = set()
    handled_module_nodes: set[int] = set()
    handled_class_nodes: set[int] = set()
    handled_signatures: set[str] = set()
    handled_import_from_delta = False
    changed_class_attributes: set[str] = set()
    for line in sorted(import_lines):
        scope = _innermost(line, scopes)
        signature_qualname = (
            scope.qualname
            if scope is not None and line < scope.body_start and scope.qualname in functions
            else None
        )
        old_function = old_functions.get(signature_qualname or "")
        if signature_qualname is not None and old_function is not None:
            function = functions[signature_qualname]
            definition_line = _definition_start(function)
            annotation_load_names, annotation_generic_names, annotation_modules = (
                _trusted_annotation_bindings(tree, definition_line)
            )
            annotation_load_names.update(_direct_scope_bindings_before(tree.body, definition_line))
            parent_qualname = signature_qualname.rpartition(".")[0]
            if parent_qualname:
                class_bindings = _direct_scope_bindings_before(
                    classes[parent_qualname].body, definition_line
                )
                annotation_generic_names.difference_update(class_bindings)
                annotation_modules.difference_update(class_bindings)
                annotation_load_names.difference_update(
                    {
                        name
                        for name in annotation_load_names
                        if "." in name and name.partition(".")[0] in class_bindings
                    }
                )
                annotation_load_names.update(class_bindings)
            decorators_unchanged = len(function.decorator_list) == len(
                old_function.decorator_list
            ) and all(
                _same_ast(new, old)
                for new, old in zip(function.decorator_list, old_function.decorator_list)
            )
            if decorators_unchanged:
                if signature_qualname in handled_signatures:
                    continue
                handled_signatures.add(signature_qualname)
                if (
                    _same_ast(function.args, old_function.args)
                    and _same_ast(function.returns, old_function.returns)
                    and function.type_comment == old_function.type_comment
                ):
                    deleted = deleted_lines.get(line)
                    if deleted:
                        try:
                            deleted_tree = ast.parse(textwrap.dedent("\n".join(deleted)))
                        except SyntaxError:
                            return PythonChangeFacts(
                                set(), set(), {}, "unresolved import replacement"
                            )
                        old_node = deleted_tree.body[0] if len(deleted_tree.body) == 1 else None
                        old_names = (
                            _module_binding_names(old_node, future_annotations)
                            if old_node is not None and "." not in signature_qualname
                            else None
                        )
                        if (
                            old_names is None
                            or isinstance(old_node, ast.Import)
                            or old_names & module_bindings
                        ):
                            return PythonChangeFacts(
                                set(), set(), {}, "unresolved import replacement"
                            )
                        deleted_binding_names.update(old_names)
                    continue
                if not _safe_signature_addition(
                    function,
                    old_function,
                    annotation_load_names,
                    annotation_generic_names,
                    annotation_modules,
                ):
                    return PythonChangeFacts(set(), set(), {}, "class/signature import change")
                direct_consumers.add(signature_qualname)
                safe_annotation_loads.update(
                    _annotation_name_loads(
                        function,
                        annotation_load_names,
                        annotation_generic_names,
                        annotation_modules,
                        future_annotations=future_annotations,
                    )
                )
                continue
        if signature_qualname is not None and old_function is None:
            if signature_qualname in handled_signatures:
                continue
            if old_tree is None or line in deleted_lines:
                return PythonChangeFacts(set(), set(), {}, "class/signature import change")
            function = functions[signature_qualname]
            definition_line = _definition_start(function)
            annotation_load_names, annotation_generic_names, annotation_modules = (
                _trusted_annotation_bindings(tree, definition_line)
            )
            annotation_load_names.update(_direct_scope_bindings_before(tree.body, definition_line))
            parent_qualname, separator, local_name = signature_qualname.rpartition(".")
            if separator:
                old_parent = old_classes.get(parent_qualname)
                current_parent = classes.get(parent_qualname)
                if (
                    old_parent is None
                    or current_parent is None
                    or local_name in _direct_scope_bindings(old_parent.body)
                ):
                    return PythonChangeFacts(set(), set(), {}, "class/signature import change")
                class_bindings = _direct_scope_bindings_before(current_parent.body, definition_line)
                annotation_generic_names.difference_update(class_bindings)
                annotation_modules.difference_update(class_bindings)
                annotation_load_names.difference_update(
                    {
                        name
                        for name in annotation_load_names
                        if "." in name and name.partition(".")[0] in class_bindings
                    }
                )
                annotation_load_names.update(class_bindings)
                decorator_scope_bindings = module_bindings | _direct_scope_bindings(
                    current_parent.body
                )
            else:
                if local_name in old_module_bindings:
                    return PythonChangeFacts(set(), set(), {}, "class/signature import change")
                decorator_scope_bindings = module_bindings
            trusted_decorators = _SAFE_BUILTIN_DECORATORS - decorator_scope_bindings
            if not _safe_added_function(
                function,
                future_annotations=future_annotations,
                annotation_load_names=annotation_load_names,
                annotation_generic_names=annotation_generic_names,
                annotation_modules=annotation_modules,
                trusted_decorators=trusted_decorators,
            ):
                return PythonChangeFacts(set(), set(), {}, "class/signature import change")
            handled_signatures.add(signature_qualname)
            direct_consumers.add(signature_qualname)
            safe_annotation_loads.update(
                _annotation_name_loads(
                    function,
                    annotation_load_names,
                    annotation_generic_names,
                    annotation_modules,
                    future_annotations=future_annotations,
                )
            )
            if not separator:
                binding_names.add(local_name)
                new_declaration_bindings.add(local_name)
            continue
        attributed_qualname = _attribute(line, scopes)
        if attributed_qualname in classes:
            current_class = classes[attributed_qualname]
            old_class = old_classes.get(attributed_qualname)
            node = _node_for_line(current_class.body, line)
            if node is not None and id(node) in handled_class_nodes:
                continue
            if node is not None:
                handled_class_nodes.add(id(node))
            if old_class is not None and _is_docstring_replacement(
                node, current_class.body, old_class.body
            ):
                changed_class_attributes.add("__doc__")
                direct_consumers.add(attributed_qualname)
                continue
            if old_class is None or node is None or line in deleted_lines:
                return PythonChangeFacts(set(), set(), {}, "class/signature import change")
            names = _safe_added_class_literal_bindings(
                node,
                old_class,
                tree,
                future_annotations=future_annotations,
            )
            if names is None:
                return PythonChangeFacts(set(), set(), {}, "class/signature import change")
            changed_class_attributes.update(names)
            direct_consumers.add(attributed_qualname)
            continue
        if attributed_qualname != "<module>":
            return PythonChangeFacts(set(), set(), {}, "class/signature import change")
        node = _node_for_line(module_nodes, line)
        if node is not None and id(node) in handled_module_nodes:
            continue
        if node is not None:
            handled_module_nodes.add(id(node))
        if _is_docstring_replacement(node, module_nodes, old_module_nodes):
            binding_names.add("__doc__")
            direct_consumers.add("<module>")
            continue
        if (
            isinstance(node, ast.If)
            and _line_in_import_only_block(node, line)
            and _trusted_type_checking_guard(node, module_nodes)
        ):
            continue
        if isinstance(node, ast.ImportFrom) and import_from_delta is not None:
            current_bindings = _import_from_bindings(node)
            changed_locals, old_targets, new_targets, added_locals = import_from_delta
            if current_bindings is None or not current_bindings.keys() & changed_locals:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            if not handled_import_from_delta:
                binding_names.update(changed_locals)
                old_import_targets.update(old_targets)
                new_import_targets.update(new_targets)
                new_import_bindings.update(added_locals - old_bound_names)
                handled_import_from_delta = True
            continue
        if isinstance(node, ast.ImportFrom) and (
            old_module_nodes is not None or line in deleted_lines
        ):
            replacement = _import_from_replacement(
                node, deleted_lines.get(line, []), old_module_nodes
            )
            if replacement is None:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            changed_locals, old_targets, new_targets, added_locals = replacement
            binding_names.update(changed_locals)
            old_import_targets.update(old_targets)
            new_import_targets.update(new_targets)
            if old_module_nodes is not None:
                new_import_bindings.update(added_locals - old_bound_names)
            continue
        names = _module_binding_names(node, future_annotations) if node is not None else None
        if node is None and line in deleted_lines:
            try:
                deleted_tree = ast.parse("\n".join(deleted_lines[line]))
            except SyntaxError:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            if len(deleted_tree.body) != 1:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            old_node = deleted_tree.body[0]
            old_names = _module_binding_names(old_node, future_annotations)
            if old_names is None or isinstance(old_node, ast.Import):
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            deleted_binding_names.update(old_names)
            continue
        if names is None:
            return PythonChangeFacts(set(), set(), {}, "effectful module statement")
        if line in deleted_lines:
            try:
                deleted_tree = ast.parse("\n".join(deleted_lines[line]))
            except SyntaxError:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            if len(deleted_tree.body) != 1:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            old_node = deleted_tree.body[0]
            old_names = _module_binding_names(old_node, future_annotations)
            if (
                old_names is not None
                and not isinstance(old_node, ast.Import)
                and not isinstance(node, ast.Import)
                and not old_names & module_bindings
                and old_module_nodes is not None
                and any(_same_ast(node, statement) for statement in old_module_nodes)
            ):
                # A deletion can be anchored to an unchanged following assignment.
                deleted_binding_names.update(old_names)
                continue
            if (
                old_names != names
                or isinstance(node, ast.Import)
                or isinstance(old_node, ast.Import)
            ):
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
        binding_names.update(names)
    binding_names.update(deleted_binding_names)

    facts = {
        qualname: [
            _FunctionNames(node)
            for node in nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for qualname, nodes in definition_groups.items()
    }
    old_facts = {
        qualname: [
            _FunctionNames(node)
            for node in nodes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for qualname, nodes in old_definition_groups.items()
    }
    consumers = direct_consumers | {
        qualname
        for qualname, variants in facts.items()
        for fact in variants
        if (
            fact.nested_loaded
            | {name for name in fact.calls if name not in fact.bound or name in fact.globals}
            | {name for name in fact.loaded if name not in fact.bound or name in fact.globals}
        )
        & binding_names
    }
    consumers.update(
        qualname
        for qualname, variants in old_facts.items()
        for fact in variants
        if (
            fact.nested_loaded
            | {name for name in fact.calls if name not in fact.bound or name in fact.globals}
            | {name for name in fact.loaded if name not in fact.bound or name in fact.globals}
        )
        & deleted_binding_names
    )
    consumers.update(
        qualname
        for qualname, nodes in definition_groups.items()
        for function in nodes
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        if any(
            isinstance(node, ast.Attribute)
            and node.attr in changed_class_attributes
            or isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in changed_class_attributes
            for node in ast.walk(function)
        )
    )
    if any(
        consumer == qualname or consumer.startswith(qualname + ".")
        for consumer in consumers
        for qualname in ambiguous_qualnames
    ):
        return PythonChangeFacts(set(), set(), {}, "ambiguous class/function declaration")
    import_loads = {
        _attribute(node.lineno, scopes)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in binding_names
        and _attribute(node.lineno, scopes) in import_qualnames
        and not (node.id in cached_typing_imports and id(node) in safe_annotation_loads)
    }
    if old_tree is not None and deleted_binding_names:
        old_scopes = _collect_scopes(old_tree)
        old_import_qualnames = import_executed_qualnames(pre_source or "")
        import_loads.update(
            _attribute(node.lineno, old_scopes)
            for node in ast.walk(old_tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in deleted_binding_names
            and _attribute(node.lineno, old_scopes) in old_import_qualnames
        )
    if import_loads:
        return PythonChangeFacts(set(), set(), {}, "import-time binding consumer")

    module_functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    callers: dict[str, set[str]] = {}
    for caller, variants in facts.items():
        for fact in variants:
            for callee in fact.calls & module_functions:
                if callee not in fact.bound or callee in fact.globals:
                    callers.setdefault(callee, set()).add(caller)

    callable_escapes = {
        name
        for variants in facts.values()
        for fact in variants
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
    callable_escapes.update(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in module_functions
        and _attribute(node.lineno, scopes) in import_qualnames
    )

    return PythonChangeFacts(
        binding_names,
        consumers,
        callers,
        callable_escapes=callable_escapes,
        new_import_targets=new_import_targets - set(cached_typing_imports.values()),
        old_import_targets=old_import_targets,
        new_import_bindings=new_import_bindings,
        new_declaration_bindings=new_declaration_bindings,
    )
