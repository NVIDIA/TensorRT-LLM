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


def _import_from_bindings(node: ast.stmt) -> dict[str, str] | None:
    """Return ``{local name: source name}`` for a static ``from`` import."""
    if not isinstance(node, ast.ImportFrom) or node.module is None:
        return None
    if any(alias.name == "*" for alias in node.names):
        return None
    bindings = {alias.asname or alias.name: alias.name for alias in node.names}
    return bindings if len(bindings) == len(node.names) else None


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
    old_targets = {
        ImportTarget(node.module, node.level, source_name)
        for local in changed_locals
        for source_name in (previous.get(local),)
        if source_name is not None
    }
    new_targets = {
        ImportTarget(node.module, node.level, source_name)
        for local in changed_locals
        for source_name in (current.get(local),)
        if source_name is not None
    }
    added_locals = current.keys() - previous.keys()
    return changed_locals, old_targets, new_targets, added_locals


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


def _trusted_annotation_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return trusted type names and module aliases used by annotations."""
    names = set(_SAFE_ANNOTATION_BUILTINS)
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {"collections.abc", "typing"}:
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"collections.abc", "typing"}
            )
    return names, modules


def _safe_annotation(node: ast.expr | None, names: set[str], modules: set[str]) -> bool:
    """Return whether evaluating a newly added annotation is side-effect-free."""
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id in modules
    if isinstance(node, ast.Subscript):
        return _safe_annotation(node.value, names, modules) and _safe_annotation(
            node.slice, names, modules
        )
    if isinstance(node, ast.Tuple):
        return all(_safe_annotation(item, names, modules) for item in node.elts)
    return False


def _same_ast(left: ast.AST | None, right: ast.AST | None) -> bool:
    if left is None or right is None:
        return left is right
    return ast.dump(left) == ast.dump(right)


def _safe_optional_parameter_addition(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    old_node: ast.FunctionDef | ast.AsyncFunctionDef,
    annotation_names: set[str],
    annotation_modules: set[str],
) -> bool:
    """Return whether a signature only appends literal-default parameters."""
    if type(node) is not type(old_node):
        return False
    if not _same_ast(node.returns, old_node.returns) or node.type_comment != old_node.type_comment:
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
        (added_positional_count <= 0 and added_keyword_only_count <= 0)
        or added_positional_count < 0
        or added_keyword_only_count < 0
        or len(arguments.posonlyargs) != len(old_arguments.posonlyargs)
        or not _same_ast(arguments.vararg, old_arguments.vararg)
        or not _same_ast(arguments.kwarg, old_arguments.kwarg)
        or any(
            not _same_ast(new, old)
            for new, old in zip(arguments.posonlyargs, old_arguments.posonlyargs)
        )
        or any(not _same_ast(new, old) for new, old in zip(arguments.args, old_arguments.args))
        or any(
            not _same_ast(new, old)
            for new, old in zip(arguments.kwonlyargs, old_arguments.kwonlyargs)
        )
        or len(arguments.defaults) != len(old_arguments.defaults) + added_positional_count
        or any(
            not _same_ast(new, old) for new, old in zip(arguments.defaults, old_arguments.defaults)
        )
        or len(arguments.kw_defaults) != len(old_arguments.kw_defaults) + added_keyword_only_count
        or any(
            not _same_ast(new, old)
            for new, old in zip(arguments.kw_defaults, old_arguments.kw_defaults)
        )
    ):
        return False

    added_arguments = arguments.args[len(old_arguments.args) :]
    added_defaults = arguments.defaults[len(old_arguments.defaults) :]
    added_keyword_only = arguments.kwonlyargs[len(old_arguments.kwonlyargs) :]
    added_keyword_defaults = arguments.kw_defaults[len(old_arguments.kw_defaults) :]
    return (
        all(_is_literal_expression(default) for default in added_defaults)
        and all(
            default is not None and _is_literal_expression(default)
            for default in added_keyword_defaults
        )
        and all(
            _safe_annotation(argument.annotation, annotation_names, annotation_modules)
            for argument in [*added_arguments, *added_keyword_only]
        )
    )


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

    Literal assignments (including literal replacements), plain function
    declarations, newly added builtin-module imports, and same-module static
    ``ImportFrom`` replacements are represented. Unsupported syntax is returned
    as an unresolved fact for policy callers.
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
    functions = _recorded_functions(tree)
    old_functions = _recorded_functions(old_tree) if old_tree is not None else {}
    annotation_names, annotation_modules = _trusted_annotation_bindings(tree)

    binding_names: set[str] = set()
    deleted_binding_names: set[str] = set()
    direct_consumers: set[str] = set()
    new_import_targets: set[ImportTarget] = set()
    old_import_targets: set[ImportTarget] = set()
    new_import_bindings: set[str] = set()
    handled_module_nodes: set[int] = set()
    handled_signatures: set[str] = set()
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
                if not _safe_optional_parameter_addition(
                    function,
                    old_function,
                    annotation_names,
                    annotation_modules,
                ):
                    return PythonChangeFacts(set(), set(), {}, "class/signature import change")
                direct_consumers.add(signature_qualname)
                continue
        if _attribute(line, scopes) != "<module>":
            return PythonChangeFacts(set(), set(), {}, "class/signature import change")
        node = _node_for_line(module_nodes, line)
        if node is not None and id(node) in handled_module_nodes:
            continue
        if node is not None:
            handled_module_nodes.add(id(node))
        added_function = (
            _safe_added_function(node, future_annotations) if node is not None else None
        )
        if added_function is not None and line not in deleted_lines:
            binding_names.add(added_function.name)
            direct_consumers.add(added_function.name)
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
                old_tree = ast.parse("\n".join(deleted_lines[line]))
            except SyntaxError:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            if len(old_tree.body) != 1:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            old_node = old_tree.body[0]
            old_names = _module_binding_names(old_node, future_annotations)
            if old_names is None or isinstance(old_node, ast.Import):
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            deleted_binding_names.update(old_names)
            continue
        if names is None:
            return PythonChangeFacts(set(), set(), {}, "effectful module statement")
        if line in deleted_lines:
            try:
                old_tree = ast.parse("\n".join(deleted_lines[line]))
            except SyntaxError:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            if len(old_tree.body) != 1:
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
            old_node = old_tree.body[0]
            old_names = _module_binding_names(old_node, future_annotations)
            if (
                old_names != names
                or isinstance(node, ast.Import)
                or isinstance(old_node, ast.Import)
            ):
                return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")
        binding_names.update(names)
    if not deleted_binding_names <= binding_names:
        return PythonChangeFacts(set(), set(), {}, "unresolved import replacement")

    facts = {qualname: _FunctionNames(node) for qualname, node in functions.items()}
    consumers = direct_consumers | {
        qualname
        for qualname, fact in facts.items()
        if (
            fact.nested_loaded
            | fact.calls
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
        return PythonChangeFacts(set(), set(), {}, "import-time binding consumer")

    module_functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    callers: dict[str, set[str]] = {}
    for caller, fact in facts.items():
        for callee in fact.calls & module_functions:
            if callee not in fact.bound or callee in fact.globals:
                callers.setdefault(callee, set()).add(caller)

    callable_escapes = {
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
        new_import_targets=new_import_targets,
        old_import_targets=old_import_targets,
        new_import_bindings=new_import_bindings,
    )
