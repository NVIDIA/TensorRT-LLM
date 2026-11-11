# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace declaration-time callable flow instead of matching product functions."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from cbts.coverage.selection.python_change_analysis import (
    _AnnotationScopeChanges,
    _direct_scope_bindings,
    _recorded_definition_groups,
    _same_ast,
)
from cbts.repository_reference import _module_name, _resolve_from


@dataclass
class DecoratorFacts:
    signature_safe: set[str] = field(default_factory=set)
    per_call: set[str] = field(default_factory=set)
    wrapper_bounds: dict[str, set[tuple[str, str]]] = field(default_factory=dict)


class _Unresolved(Exception):
    """Callable flow cannot be bounded without executing opaque code."""


_CALLABLE, _OPAQUE, _TEXT = object(), object(), object()


@dataclass
class _Function:
    path: str
    qualname: str
    node: ast.FunctionDef
    closure: dict[str, object]
    image: int


@dataclass(frozen=True)
class _Module:
    name: str


@dataclass(frozen=True)
class _External:
    name: str


def _tainted(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_tainted(item) for item in value)
    if isinstance(value, dict):
        return any(_tainted(item) for item in [*value, *value.values()])
    return (
        value is _CALLABLE
        or isinstance(value, _Function)
        and any(_tainted(item) for item in value.closure.values())
    )


def _unique_binding(tree: ast.Module, name: str) -> ast.stmt | None:
    nodes = [node for node in tree.body if name in _direct_scope_bindings([node])]
    if len(nodes) != 1 or "*" in _direct_scope_bindings(tree.body):
        return None
    if any(isinstance(node, ast.Global) and name in node.names for node in ast.walk(tree)):
        return None
    return nodes[0]


def _unmodified_attributes(tree: ast.Module, names: set[str]) -> bool:
    changes = _AnnotationScopeChanges()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval"}:
                return False
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id in {"globals", "locals", "vars"}:
                    return False
        if isinstance(node, (ast.Attribute, ast.Call)):
            changes.visit(node)
    return not names & changes.unknown_attributes and not any(
        attribute.partition(".")[0] in names for attribute in changes.attributes
    )


class _CallableFlow:
    """Follow imports, factories, aliases and returned closures in both images.

    Callable inspection, invocation and registration during decoration decline.
    Returned Python wrappers bound later calls, including cache hits and opaque
    dispatch. Missing library source gains no exception based on API names.
    """

    def __init__(self, read_images: Callable[[str], tuple[str, str] | None]) -> None:
        self.read_images = read_images
        self.trees: dict[tuple[str, int], ast.Module | None] = {}
        self.symbols: dict[tuple[str, int, str], object] = {}
        self.resolving: set[tuple[str, int, str]] = set()
        self.calls: set[tuple[str, str, int]] = set()
        self.callable_contexts: list[bool] = []

    def tree(self, path: str, image: int) -> ast.Module | None:
        key = path, image
        if key not in self.trees:
            images = self.read_images(path)
            try:
                self.trees[key] = ast.parse(images[image]) if images else None
            except SyntaxError as error:
                raise _Unresolved("unparsable repository source") from error
        return self.trees[key]

    def member(self, module: str, name: str, image: int) -> object:
        root = module.replace(".", "/")
        paths = [
            path
            for path in (root + ".py", root + "/__init__.py")
            if self.tree(path, image) is not None
        ]
        if len(paths) == 1:
            return self.symbol(paths[0], name, image)
        if paths:
            raise _Unresolved("ambiguous module")
        return _External(module + "." + name)

    def symbol(self, path: str, name: str, image: int) -> object:
        key = path, image, name
        if key in self.symbols:
            return self.symbols[key]
        if key in self.resolving:
            raise _Unresolved("cyclic binding")
        tree = self.tree(path, image)
        if tree is None:
            raise _Unresolved("missing source")
        node = _unique_binding(tree, name)
        if node is None or not _unmodified_attributes(tree, {name}):
            raise _Unresolved("unresolved or mutated binding")
        self.resolving.add(key)
        try:
            if isinstance(node, ast.FunctionDef) and not node.decorator_list:
                value = _Function(path, name, node, {}, image)
            elif isinstance(node, ast.ImportFrom):
                module_name, is_package = _module_name(path)
                module = _resolve_from(module_name, is_package, node)
                alias = next(alias for alias in node.names if (alias.asname or alias.name) == name)
                value = self.member(module, alias.name, image)
            elif isinstance(node, ast.Import):
                alias = next(
                    alias
                    for alias in node.names
                    if (alias.asname or alias.name.partition(".")[0]) == name
                )
                value = _Module(alias.name if alias.asname else name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                value = self.expr(node.value, path, {}, image)
            else:
                raise _Unresolved("unsupported binding")
            self.symbols[key] = value
            return value
        finally:
            self.resolving.remove(key)

    def expr(self, node: ast.expr, path: str, env: dict[str, object], image: int) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id] if node.id in env else self.symbol(path, node.id, image)
        if isinstance(node, ast.Attribute):
            value = self.expr(node.value, path, env, image)
            if isinstance(value, _Module):
                return self.member(value.name, node.attr, image)
            if isinstance(value, _External):
                return _External(value.name + "." + node.attr)
            if _tainted(value):
                raise _Unresolved("callable metadata access")
            return _OPAQUE
        if isinstance(node, ast.Call):
            target = self.expr(node.func, path, env, image)
            args = []
            for argument in node.args:
                if isinstance(argument, ast.Starred):
                    value = self.expr(argument.value, path, env, image)
                    if not isinstance(value, tuple):
                        raise _Unresolved("unknown argument expansion")
                    args.extend(value)
                else:
                    args.append(self.expr(argument, path, env, image))
            kwargs = {}
            for keyword in node.keywords:
                value = self.expr(keyword.value, path, env, image)
                addition = value if keyword.arg is None else {keyword.arg: value}
                if not isinstance(addition, dict) or kwargs.keys() & addition.keys():
                    raise _Unresolved("unknown or duplicate keyword expansion")
                kwargs.update(addition)
            return self.apply(target, args, kwargs)
        if isinstance(node, ast.JoinedStr):
            values = [
                self.expr(part.value, path, env, image)
                for part in ast.walk(node)
                if isinstance(part, ast.FormattedValue)
            ]
            if any(_tainted(value) for value in values):
                raise _Unresolved("callable formatted into metadata")
            if any(self.callable_contexts) and not all(
                isinstance(value, (str, int, float, bool)) or value is _TEXT for value in values
            ):
                raise _Unresolved("opaque formatting during callable decoration")
            return _TEXT
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            children = (
                [*node.values, *(key for key in node.keys if key is not None)]
                if isinstance(node, ast.Dict)
                else node.elts
            )
            if any(_tainted(self.expr(child, path, env, image)) for child in children):
                raise _Unresolved("callable stored in a container")
            return _OPAQUE
        if isinstance(node, (ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp)):
            values = [
                self.expr(child, path, env, image)
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.expr)
            ]
            if any(_tainted(value) for value in values):
                raise _Unresolved("callable used in an operator")
            return _OPAQUE
        raise _Unresolved("unsupported expression")

    def apply(self, target: object, args: list[object], kwargs: dict[str, object]) -> object:
        if isinstance(target, list):
            results = [self.apply(item, args, kwargs) for item in target]
            return [
                value
                for result in results
                for value in (result if isinstance(result, list) else [result])
            ]
        if isinstance(target, _External):
            raise _Unresolved("callable implementation is unavailable")
        if not isinstance(target, _Function):
            raise _Unresolved("opaque decorator or declaration-time invocation")
        key = target.path, target.qualname, target.image
        if key in self.calls or len(self.calls) >= 16:
            raise _Unresolved("recursive or excessive decorator expansion")
        parameters = target.node.args
        positional = [*parameters.posonlyargs, *parameters.args]
        if len(args) > len(positional) and parameters.vararg is None:
            raise _Unresolved("excess positional arguments")
        parameter_names = {
            parameter.arg
            for parameter in [
                *positional,
                *parameters.kwonlyargs,
                *([parameters.vararg] if parameters.vararg else []),
                *([parameters.kwarg] if parameters.kwarg else []),
            ]
        }
        env = {name: value for name, value in target.closure.items() if name not in parameter_names}
        env.update(zip((parameter.arg for parameter in positional), args))
        if parameters.vararg:
            env[parameters.vararg.arg] = tuple(args[len(positional) :])
        extra = {}
        for name, value in kwargs.items():
            if name in {parameter.arg for parameter in parameters.posonlyargs}:
                if parameters.kwarg is None:
                    raise _Unresolved("positional-only parameter passed as keyword")
                extra[name] = value
                continue
            if name not in {parameter.arg for parameter in [*positional, *parameters.kwonlyargs]}:
                if parameters.kwarg is None:
                    raise _Unresolved("unexpected keyword argument")
                extra[name] = value
                continue
            if name in env:
                raise _Unresolved("ambiguous arguments")
            env[name] = value
        if parameters.kwarg:
            env[parameters.kwarg.arg] = extra
        defaults = [None] * (len(positional) - len(parameters.defaults)) + parameters.defaults
        for parameter, default in zip(
            [*positional, *parameters.kwonlyargs], [*defaults, *parameters.kw_defaults]
        ):
            if parameter.arg not in env:
                if default is None:
                    raise _Unresolved("missing factory argument")
                env[parameter.arg] = self.expr(default, target.path, target.closure, target.image)
        self.calls.add(key)
        self.callable_contexts.append(any(_tainted(value) for value in env.values()))
        try:
            values, active = self.block(target.node.body, target, [env])
            if active or not values:
                raise _Unresolved("unbounded decorator return")
            unique = []
            for value in values:
                if value not in unique:
                    unique.append(value)
            return unique[0] if len(unique) == 1 else unique
        finally:
            self.callable_contexts.pop()
            self.calls.remove(key)

    def block(
        self, statements: list[ast.stmt], function: _Function, states: list[dict[str, object]]
    ) -> tuple[list[object], list[dict[str, object]]]:
        returned = []
        for statement in statements:
            next_states = []
            for env in states:
                if isinstance(statement, ast.Return) and statement.value is not None:
                    returned.append(self.expr(statement.value, function.path, env, function.image))
                    continue
                if isinstance(statement, ast.FunctionDef):
                    value = _Function(
                        function.path,
                        function.qualname + ".<locals>." + statement.name,
                        statement,
                        dict(env),
                        function.image,
                    )
                    if statement.args.defaults or any(
                        default is not None for default in statement.args.kw_defaults
                    ):
                        raise _Unresolved("executable closure signature")
                    annotations = [statement.returns] + [
                        argument.annotation
                        for argument in [
                            *statement.args.posonlyargs,
                            *statement.args.args,
                            *statement.args.kwonlyargs,
                            *([statement.args.vararg] if statement.args.vararg else []),
                            *([statement.args.kwarg] if statement.args.kwarg else []),
                        ]
                    ]
                    annotations = [
                        annotation for annotation in annotations if annotation is not None
                    ]
                    if annotations:
                        raise _Unresolved("closure annotations")
                    for decorator in reversed(statement.decorator_list):
                        value = self.apply(
                            self.expr(decorator, function.path, env, function.image), [value], {}
                        )
                    env[statement.name] = value
                elif (
                    isinstance(statement, (ast.Assign, ast.AnnAssign))
                    and statement.value is not None
                ):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    value = self.expr(statement.value, function.path, env, function.image)
                    if not all(isinstance(target, ast.Name) for target in targets):
                        raise _Unresolved("declaration-time state write")
                    for target in targets:
                        env[target.id] = value
                elif isinstance(statement, ast.Expr):
                    self.expr(statement.value, function.path, env, function.image)
                elif isinstance(statement, ast.Raise):
                    if statement.exc is not None:
                        self.expr(statement.exc, function.path, env, function.image)
                    if statement.cause is not None:
                        self.expr(statement.cause, function.path, env, function.image)
                    continue
                elif isinstance(statement, ast.If):
                    if _tainted(self.expr(statement.test, function.path, env, function.image)):
                        raise _Unresolved("callable-dependent branch")
                    for branch in (statement.body, statement.orelse):
                        values, active = self.block(branch, function, [dict(env)])
                        returned.extend(values)
                        next_states.extend(active)
                    continue
                elif not isinstance(statement, ast.Pass):
                    raise _Unresolved("unsupported declaration-time control flow")
                next_states.append(env)
            states = next_states
            if len(states) + len(returned) > 32:
                raise _Unresolved("excessive branch expansion")
        return returned, states


def analyze_decorators(
    path: str,
    source: str,
    pre_source: str | None,
    read_images: Callable[[str], tuple[str, str] | None],
) -> DecoratorFacts:
    """Analyze both images, following source instead of product implementation templates."""
    facts = DecoratorFacts()
    if pre_source is None:
        return facts
    try:
        trees = [ast.parse(source), ast.parse(pre_source)]
    except SyntaxError:
        return facts
    flow = _CallableFlow(read_images)
    for image, tree in enumerate(trees):
        flow.trees[path, image] = tree
    groups = [_recorded_definition_groups(tree) for tree in trees]
    for qualname, nodes in groups[0].items():
        previous = groups[1].get(qualname, [])
        if (
            len(nodes) != 1
            or len(previous) != 1
            or not all(isinstance(node, ast.FunctionDef) for node in [nodes[0], previous[0]])
        ):
            continue
        if not nodes[0].decorator_list or not _same_ast(
            ast.Tuple(elts=nodes[0].decorator_list), ast.Tuple(elts=previous[0].decorator_list)
        ):
            continue
        bounds = set()
        try:
            for image, node in enumerate([nodes[0], previous[0]]):
                owners = set().union(
                    *(
                        _direct_scope_bindings(owner.body)
                        for owner_name, owners in groups[image].items()
                        if qualname.startswith(owner_name + ".")
                        for owner in owners
                        if isinstance(owner, ast.ClassDef)
                    )
                )
                names = {
                    child.id
                    for decorator in node.decorator_list
                    for child in ast.walk(decorator)
                    if isinstance(child, ast.Name)
                }
                if names & owners:
                    raise _Unresolved("class namespace shadows decorator")
                value = _CALLABLE
                for decorator in reversed(node.decorator_list):
                    target = flow.expr(decorator, path, {}, image)
                    value = flow.apply(target, [value], {})
                for result in value if isinstance(value, list) else [value]:
                    if result is _CALLABLE:
                        continue
                    if isinstance(result, _Function) and _tainted(result):
                        bounds.add((result.path, result.qualname))
                    else:
                        raise _Unresolved("unresolved wrapped value")
        except _Unresolved:
            continue
        facts.signature_safe.add(qualname)
        if bounds:
            facts.wrapper_bounds[qualname] = bounds
        else:
            facts.per_call.add(qualname)
    return facts
