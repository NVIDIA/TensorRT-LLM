#!/usr/bin/env python3
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
"""Statically compare registered test doubles with current production source.

Compare member declarations, callable signatures, and instance initialization
without importing or executing registered code. Unsupported cases remain incomplete.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import re
import sys
import sysconfig
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

BINDINGS_MODULE = "tensorrt_llm.bindings"
MARKERS = "test_common.interface_checks"
UNKNOWN = "?"
DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
TARGET = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


class Incomplete(ValueError):
    """Available source cannot establish the complete interface."""


def statements(node: ast.AST) -> Iterator[ast.AST]:
    """Visit a lexical scope without borrowing declarations from nested scopes."""
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, DEFINITIONS + (ast.Lambda,)):
            yield from statements(child)


def assignment_leaves(
    target: ast.AST, value: ast.AST | None
) -> Iterator[tuple[ast.AST, ast.AST | None]]:
    """Associate unpacked targets with literal values when their shape is known."""
    if isinstance(target, ast.Starred):
        yield from assignment_leaves(target.value, value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        values: list[ast.AST | None] = [None] * len(target.elts)
        if isinstance(value, (ast.Tuple, ast.List)) and not any(
            isinstance(item, ast.Starred) for item in value.elts
        ):
            stars = [i for i, item in enumerate(target.elts) if isinstance(item, ast.Starred)]
            if not stars and len(target.elts) == len(value.elts):
                values = list(value.elts)
            elif len(stars) == 1 and len(value.elts) >= len(target.elts) - 1:
                index = stars[0]
                tail = len(target.elts) - index - 1
                stop = len(value.elts) - tail
                values = [
                    *value.elts[:index],
                    ast.List(elts=value.elts[index:stop], ctx=ast.Load()),
                    *value.elts[stop:],
                ]
        for child, supplied in zip(target.elts, values):
            yield from assignment_leaves(child, supplied)
    else:
        yield target, value


@dataclass
class Source:
    path: Path
    module: str
    tree: ast.Module
    imports: dict[str, str] = field(default_factory=dict)
    star_imports: list[str] = field(default_factory=list)
    scope_bindings: dict[ast.AST, dict[str, set[ast.AST]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def qualified(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return self.qualified(node.value) + "." + node.attr
        return UNKNOWN

    def bindings(self, scope: ast.AST) -> dict[str, set[ast.AST]]:
        """Collect lexical bindings that can shadow module imports and builtins."""
        if scope not in self.scope_bindings:
            bindings: dict[str, set[ast.AST]] = {}
            for node in statements(scope):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    bindings.setdefault(node.id, set()).add(node)
                elif isinstance(node, ast.arg):
                    bindings.setdefault(node.arg, set()).add(node)
                elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                    if node.name is not None:
                        bindings.setdefault(node.name, set()).add(node)
                elif isinstance(node, ast.MatchMapping) and node.rest is not None:
                    bindings.setdefault(node.rest, set()).add(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if node not in self.tree.body:
                        for alias in node.names:
                            name = alias.asname or alias.name.split(".")[0]
                            bindings.setdefault(name, set()).add(node)
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, DEFINITIONS):
                        bindings.setdefault(child.name, set()).add(child)
            self.scope_bindings[scope] = bindings
        return self.scope_bindings[scope]


@dataclass
class Definition:
    source: Source
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

    def __deepcopy__(self, memo: dict) -> Definition:
        # Source trees are immutable; branch state copies only the tracked values.
        return self


@dataclass
class Member:
    kind: str
    signature_issue: str | None = None
    signatures: list[dict] = field(default_factory=list)
    # Defaults accessible through the class do not require per-instance assignment.
    default: bool = False
    conditional: bool = False
    line: int = 1
    origin: tuple[str, int] | None = None


@dataclass
class Shape:
    members: dict[str, Member] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    skipped_bases: list[str] = field(default_factory=list)
    initializer_fields: dict[str, Member] = field(default_factory=dict)
    initializer_unknown: list[str] = field(default_factory=list)


@dataclass
class Value:
    state: State | None = None
    kind: str = UNKNOWN


@dataclass
class State:
    fields: dict[str, Value] = field(default_factory=dict)
    possible: set[str] = field(default_factory=set)
    unknown: list[str] = field(default_factory=list)
    definition: Definition | None = None


DYNAMIC_MOCK_MEMBERS = "Dynamic mock/module members cannot establish exact equality"


class Sources:
    def __init__(self, roots: list[Path]) -> None:
        self.roots = list(dict.fromkeys(path.resolve() for path in roots))
        self.library_roots = list(
            dict.fromkeys(
                Path(path).resolve()
                for key, path in sysconfig.get_paths().items()
                if key in {"stdlib", "platstdlib", "purelib", "platlib"}
            )
        )
        self.cache: dict[Path, Source] = {}
        self.modules: dict[str, Source] = {}
        self.mros: dict[tuple[Path, int], list[Definition]] = {}
        self.shapes: dict[tuple[Path, int, bool, bool], Shape] = {}
        self.binding_bases: dict[str, Definition] = {}
        # These roots contribute no checked members, but still constrain C3 order.
        tree = ast.parse("class object: pass\nclass ABC: pass\nclass Generic: pass\n")
        source = Source(Path("<mock-interface-roots>"), "", tree)
        self.empty_bases = {
            name: Definition(source, node)
            for name, node in zip(("object", "abc.ABC", "typing.Generic"), tree.body)
            if isinstance(node, ast.ClassDef)
        }
        object_base, abc_base, generic_base = self.empty_bases.values()
        self.mros[(source.path, object_base.node.lineno)] = [object_base]
        self.mros[(source.path, abc_base.node.lineno)] = [abc_base, object_base]
        self.mros[(source.path, generic_base.node.lineno)] = [generic_base, object_base]

    def read(self, path: Path, module: str) -> Source:
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        source = Source(path, module, ast.parse(path.read_text(), filename=str(path)))
        for node in statements(source.tree):
            entries = {}
            if isinstance(node, ast.Import):
                entries = {
                    alias.asname or alias.name.split(".")[0]: alias.name
                    if alias.asname
                    else alias.name.split(".")[0]
                    for alias in node.names
                }
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    parts = module.split(".")
                    if not path.name.startswith("__init__."):
                        parts.pop()
                    prefix = ".".join(
                        parts[: len(parts) - node.level + 1] + ([prefix] if prefix else [])
                    )
                if any(alias.name == "*" for alias in node.names):
                    if node not in source.tree.body:
                        source.star_imports.append(UNKNOWN)
                    else:
                        source.star_imports.append(prefix)
                entries = {
                    alias.asname or alias.name: f"{prefix}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                }
            for name, target in entries.items():
                if name in source.imports and source.imports[name] != target:
                    source.imports[name] = UNKNOWN
                else:
                    source.imports[name] = target
        self.cache[path] = source
        return source

    def module(self, name: str) -> Source:
        if name in self.modules:
            return self.modules[name]
        for root in [*self.roots, *self.library_roots]:
            for suffix in (".pyi", ".py", "/__init__.pyi", "/__init__.py"):
                path = root / (name.replace(".", "/") + suffix)
                if path.is_file():
                    self.modules[name] = self.read(path, name)
                    return self.modules[name]
        raise Incomplete(f"Cannot locate source/stub for {name}")

    def definition(self, target: str, seen: frozenset[str] = frozenset()) -> Definition:
        if target in seen:
            raise Incomplete(f"Cyclic definition: {target}")
        module, _, name = target.partition(":")
        source = self.module(module)
        body = source.tree.body
        found = None
        for component in name.split("."):
            found = next(
                (n for n in body if isinstance(n, DEFINITIONS) and n.name == component), None
            )
            if found is None:
                root, _, tail = name.partition(".")
                if root in source.imports:
                    return self.dotted(
                        source.imports[root] + ("." + tail if tail else ""), seen | {target}
                    )
                candidates = []
                for wildcard in source.star_imports:
                    exported = self.module(wildcard)
                    if root in self.literal_exports(exported):
                        candidates.append(wildcard + ":" + name)
                if len(set(candidates)) == 1:
                    return self.definition(candidates[0], seen | {target})
                if candidates:
                    raise Incomplete(f"Ambiguous wildcard exports for {target}")
                raise Incomplete(
                    f"Cannot resolve {target}; runtime/conditional definitions are not assumed"
                )
            body = found.body
        if found is None:
            raise Incomplete(f"Empty definition: {target}")
        return Definition(source, found)

    @staticmethod
    def literal_exports(source: Source) -> set[str]:
        """Resolve only an unconditional literal __all__; never execute export builders."""
        shadowed = set(source.imports)
        shadowed.update(
            node.id
            for node in statements(source.tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        shadowed.update(node.name for node in source.tree.body if isinstance(node, DEFINITIONS))

        def read_only(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id == "__all__" and isinstance(node.ctx, ast.Load)
            if isinstance(node, ast.Constant):
                return True
            if isinstance(node, (ast.List, ast.Tuple)):
                return all(read_only(element) for element in node.elts)
            if isinstance(node, ast.Compare):
                return (
                    all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
                    and read_only(node.left)
                    and all(read_only(value) for value in node.comparators)
                )
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"sorted", "len"} - shadowed
                and len(node.args) == 1
                and not node.keywords
                and read_only(node.args[0])
            )

        # Validation guards can read __all__ without changing its literal contents.
        guards = {
            id(child)
            for node in statements(source.tree)
            if isinstance(node, (ast.If, ast.Assert)) and read_only(node.test)
            for child in ast.walk(node.test)
        }
        assignments = []
        for node in statements(source.tree):
            if isinstance(node, ast.Name) and node.id == "__all__" and id(node) not in guards:
                assignments.append(node)
        declaration = next(
            (
                node
                for node in source.tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                )
            ),
            None,
        )
        if len(assignments) != 1 or declaration is None:
            raise Incomplete(f"Non-literal wildcard exports in {source.module}")
        try:
            names = ast.literal_eval(declaration.value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise Incomplete(f"Non-literal wildcard exports in {source.module}") from error
        if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names):
            raise Incomplete(f"Non-literal wildcard exports in {source.module}")
        return set(names)

    def dotted(self, name: str, seen: frozenset[str] = frozenset()) -> Definition:
        parts = name.split(".")
        for index in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:index])
            try:
                self.module(module)
            except Incomplete:
                continue
            return self.definition(module + ":" + ".".join(parts[index:]), seen)
        raise Incomplete(f"Cannot resolve definition {name}")

    def reference(self, source: Source, node: ast.AST) -> Definition:
        name = source.qualified(node)
        if "." in name:
            return self.dotted(name)
        return self.definition(source.module + ":" + name)

    def callable(
        self, definition: Definition, *, bound: bool = False, keep_binding: bool = False
    ) -> Member:
        """Read a function or a method as accessed through its class or instance."""
        source, node = definition.source, definition.node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise Incomplete("Expected a function or method")
        owner = next(
            (
                item
                for item in source.tree.body
                if isinstance(item, ast.ClassDef) and node in item.body
            ),
            None,
        )
        if owner is None:
            if bound or node not in source.tree.body:
                raise Incomplete("Bound replacement requires a statically resolved instance method")
            return callable_members(source.tree.body, source, False)[node.name]
        member = callable_members(owner.body, source, True).get(node.name)
        if member is None:
            raise Incomplete(f"Class-body assignment replaces {node.name}")
        if member.kind != "callable":
            raise Incomplete("Property replacements require an owning-class registration")
        binding = member.signatures[0]["binding"]
        if bound and binding != "method":
            raise Incomplete("bound=True requires an instance method")
        if binding == "method" and not bound:
            member = callable_members(owner.body, source, True, keep_receiver=True)[node.name]
        if not keep_binding:
            for sig in member.signatures:
                sig["binding"] = "function"
        return member

    def class_mro(
        self, definition: Definition, seen: frozenset[tuple[Path, int]] = frozenset()
    ) -> list[Definition]:
        """Resolve C3 precedence from source without importing or constructing classes."""
        node, source = definition.node, definition.source
        if not isinstance(node, ast.ClassDef):
            raise Incomplete("Expected a class")
        key = (source.path, node.lineno)
        if key in seen:
            raise Incomplete("Cyclic inheritance")
        if key in self.mros:
            return self.mros[key]
        bases = []
        for base in node.bases:
            origin = base.value if isinstance(base, ast.Subscript) else base
            name = source.qualified(origin).removeprefix("builtins.")
            if name.startswith(BINDINGS_MODULE + "."):
                if name not in self.binding_bases:
                    module, _, class_name = name.rpartition(".")
                    tree = ast.parse(f"class {class_name}: pass")
                    opaque = Definition(Source(Path(f"<{name}>"), module, tree), tree.body[0])
                    self.binding_bases[name] = opaque
                    self.mros[(opaque.source.path, opaque.node.lineno)] = [
                        opaque,
                        self.empty_bases["object"],
                    ]
                bases.append(self.binding_bases[name])
                continue
            resolved = (
                self.empty_bases[name]
                if name in self.empty_bases
                else self.reference(source, origin)
            )
            if isinstance(base, ast.Subscript) and name != "typing.Generic":
                origin_mro = self.class_mro(resolved, seen | {key})
                if self.empty_bases["typing.Generic"] not in origin_mro or any(
                    ancestor.node.keywords
                    or any(
                        isinstance(member, DEFINITIONS) and member.name == "__class_getitem__"
                        for member in ancestor.node.body
                    )
                    for ancestor in origin_mro
                ):
                    raise Incomplete("Parameterized base has a custom or unresolved runtime origin")
            bases.append(resolved)
        if not bases:
            bases = [self.empty_bases["object"]]

        def identity(item: Definition) -> tuple[Path, int]:
            return item.source.path, item.node.lineno

        if len({identity(base) for base in bases}) != len(bases):
            raise Incomplete("Duplicate base class")
        sequences = [list(self.class_mro(base, seen | {key})) for base in bases] + [bases[:]]
        result = [definition]
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (
                    sequence[0]
                    for sequence in sequences
                    if not any(
                        identity(sequence[0]) == identity(tail)
                        for other in sequences
                        for tail in other[1:]
                    )
                ),
                None,
            )
            if candidate is None:
                raise Incomplete("Inconsistent C3 inheritance order")
            result.append(candidate)
            for sequence in sequences:
                if identity(sequence[0]) == identity(candidate):
                    sequence.pop(0)
        self.mros[key] = result
        return result


def default_value(node: ast.AST | None) -> str:
    if node is None:
        return "required"
    try:
        value = ast.literal_eval(node)
        return repr(value)
    except (ValueError, TypeError, SyntaxError):
        # Merely sharing an expression cannot prove equal evaluated default values.
        return UNKNOWN


def signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, binding: str, *, keep_receiver: bool = False
) -> dict:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    has_receiver = binding in {"method", "classmethod", "property-get", "property-set"}
    parameters = []
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if index == 0 and has_receiver and not keep_receiver:
            continue
        kind = "positional-only" if index < len(node.args.posonlyargs) else "positional-or-keyword"
        parameters.append((arg.arg, kind, default_value(default)))
    if node.args.vararg:
        parameters.append((node.args.vararg.arg, "var-positional", "required"))
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        parameters.append((arg.arg, "keyword-only", default_value(default)))
    if node.args.kwarg:
        parameters.append((node.args.kwarg.arg, "var-keyword", "required"))
    result = {
        "binding": binding,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "generator": any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in statements(node)),
        "parameters": parameters,
    }
    if has_receiver and not (positional or node.args.vararg):
        result["missing_receiver"] = True
    return result


def preserves_call_signature(decorator: ast.AST, source: Source) -> bool:
    """Recognize wrappers whose public call contract follows the wrapped function."""
    call = decorator if isinstance(decorator, ast.Call) else None
    name = source.qualified(call.func if call is not None else decorator)
    if call is None:
        return name in {"functools.cache", "functools.lru_cache", "torch.compiler.disable"}
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if None in keywords:
        return False
    if name == "functools.lru_cache":
        if len(call.args) > 1 or keywords.keys() - {"maxsize", "typed"}:
            return False
        if call.args and "maxsize" in keywords:
            return False
        size = call.args[0] if call.args else keywords.get("maxsize", ast.Constant(128))
        typed = keywords.get("typed", ast.Constant(False))
        return (
            isinstance(size, ast.Constant)
            and (size.value is None or type(size.value) is int)
            and isinstance(typed, ast.Constant)
            and type(typed.value) is bool
        )
    if name == "torch.compiler.disable":
        if call.args or keywords.keys() - {"recursive", "reason"}:
            return False
        recursive = keywords.get("recursive", ast.Constant(True))
        reason = keywords.get("reason", ast.Constant(None))
        return (
            isinstance(recursive, ast.Constant)
            and type(recursive.value) is bool
            and isinstance(reason, ast.Constant)
            and (reason.value is None or isinstance(reason.value, str))
        )
    if name in {"torch.no_grad", "torch.inference_mode"}:
        return not call.args and not keywords
    if name == "tensorrt_llm._torch.utils.with_model_extra_attrs":
        return len(call.args) == 1 and not keywords
    if name == "tensorrt_llm._utils.nvtx_range":
        return (
            len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and not keywords
        )
    if name == "torch.library.custom_op":
        # An explicit dispatcher schema can differ from the Python annotations.
        schema = keywords.get("schema", ast.Constant(None))
        return (
            len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and "mutates_args" in keywords
            and not keywords.keys() - {"mutates_args", "device_types", "schema", "tags"}
            and isinstance(schema, ast.Constant)
            and schema.value is None
        )
    return False


def callable_members(
    nodes: list[ast.stmt], source: Source, in_class: bool, *, keep_receiver: bool = False
) -> dict[str, Member]:
    result = {}
    overload_names = set()
    for node in nodes:
        if in_class and isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result.pop(target.id, None)
                    overload_names.discard(target.id)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [source.qualified(d) for d in node.decorator_list]
        binding = "method" if in_class else "function"
        if "staticmethod" in decorators:
            binding = "staticmethod"
        if "classmethod" in decorators:
            binding = "classmethod"
        if "property" in decorators or "functools.cached_property" in decorators:
            binding = "property-get"
        if any(d.endswith(".setter") for d in decorators):
            binding = "property-set"
        overload = any(d in {"typing.overload", "typing_extensions.overload"} for d in decorators)
        if overload:
            overload_names.add(node.name)
        elif node.name in overload_names and binding != "property-set":
            continue
        repeated = node.name in result and not overload and binding != "property-set"
        member = result.setdefault(
            node.name,
            Member(
                "property" if binding.startswith("property-") else "callable",
                line=node.lineno,
                origin=(str(source.path), node.lineno),
            ),
        )
        if repeated:
            member.signature_issue = "Decorator or repeated declaration can change the interface"
        member.signatures.append(
            signature(node, binding, keep_receiver=keep_receiver and binding == "method")
        )
        allowed = {
            "staticmethod",
            "classmethod",
            "property",
            "functools.cached_property",
            "abc.abstractmethod",
            "typing.overload",
            "typing_extensions.overload",
            MARKERS + ".mock_target",
        }
        for dec, name in zip(node.decorator_list, decorators):
            if isinstance(dec, ast.Call) and source.qualified(dec.func) == MARKERS + ".mock_target":
                continue
            if preserves_call_signature(dec, source):
                continue
            if (
                isinstance(dec, ast.Call)
                and source.qualified(dec.func) == "tensorrt_llm.llmapi.utils.set_api_status"
                and len(dec.args) == 1
                and isinstance(dec.args[0], ast.Constant)
                and isinstance(dec.args[0].value, str)
                and not dec.keywords
            ):
                # This metadata decorator returns the original callable, changing
                # only its docstring and the API-status registry.
                continue
            if name not in allowed and not name.endswith(".setter"):
                member.signature_issue = (
                    "Decorator or repeated declaration can change the interface"
                )
    return result


def class_shape(
    definition: Definition,
    sources: Sources,
    seen: frozenset[tuple[Path, int]] = frozenset(),
    *,
    members_only: bool = False,
    include_type_checking: bool = False,
) -> Shape:
    key = definition.source.path, definition.node.lineno
    if key in seen:
        raise Incomplete("Cyclic inheritance")
    cache_key = (*key, members_only, include_type_checking)
    if cache_key in sources.shapes:
        return copy.deepcopy(sources.shapes[cache_key])
    shape = Shape()
    for base in reversed(sources.class_mro(definition, seen)):
        if any(base is empty for empty in sources.empty_bases.values()):
            continue
        if any(base is excluded for excluded in sources.binding_bases.values()):
            shape.skipped_bases.append(base.source.module + "." + base.node.name)
            continue
        _extend_class_shape(
            base, shape, members_only=members_only, include_type_checking=include_type_checking
        )
    for name, member in shape.initializer_fields.items():
        previous = shape.members.get(name)
        if previous is None or (previous.conditional and not member.conditional):
            shape.members[name] = member
    shape.unknown.extend(shape.initializer_unknown)
    sources.shapes[cache_key] = shape
    return copy.deepcopy(shape)


def matches_initializer_call(
    call: ast.Call,
    initializer: Member | None,
    caller: ast.FunctionDef | ast.AsyncFunctionDef,
    source: Source,
) -> bool:
    """Check the arguments of a direct call against a known inherited initializer."""
    if (
        initializer is None
        or initializer.signature_issue
        or len(initializer.signatures) != 1
        or any(isinstance(arg, ast.Starred) for arg in call.args)
    ):
        return False
    signature = initializer.signatures[0]
    if (
        signature["binding"] != "method"
        or signature.get("missing_receiver")
        or signature["async"]
        or signature["generator"]
    ):
        return False
    parameters = [
        inspect.Parameter(
            name,
            getattr(inspect.Parameter, kind.replace("-", "_").upper()),
            default=inspect.Parameter.empty if default == "required" else None,
        )
        for name, kind, default in signature["parameters"]
    ]
    keywords = {kw.arg: None for kw in call.keywords if kw.arg is not None}
    try:
        inspect.Signature(parameters).bind(*([None] * len(call.args)), **keywords)
    except TypeError:
        return False
    unpacked = [kw.value for kw in call.keywords if kw.arg is None]
    if not unpacked:
        return True
    # A direct **kwargs forward cannot supply any of the caller's named parameters.
    # Verify that it can only add keywords the inherited initializer accepts.
    named_parameters = {arg.arg for arg in (*caller.args.args, *caller.args.kwonlyargs)}
    explicitly_bound = keywords.keys() | {
        param.name
        for param in parameters[: len(call.args)]
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    }
    return (
        len(unpacked) == 1
        and isinstance(unpacked[0], ast.Name)
        and caller.args.kwarg is not None
        and unpacked[0].id == caller.args.kwarg.arg
        and source.bindings(caller).get(unpacked[0].id) == {caller.args.kwarg}
        and explicitly_bound <= named_parameters
        and any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters)
    )


def _extend_class_shape(
    definition: Definition,
    shape: Shape,
    *,
    members_only: bool,
    include_type_checking: bool,
) -> None:
    node, source = definition.node, definition.source
    if not isinstance(node, ast.ClassDef):
        raise Incomplete("Expected a class")
    generated_initializer = False
    plain_dataclass = False
    for dec in node.decorator_list:
        decorator = dec.func if isinstance(dec, ast.Call) else dec
        if source.qualified(decorator) == "dataclasses.dataclass":
            plain_dataclass = not isinstance(dec, ast.Call) or (not dec.args and not dec.keywords)
            init = (
                next(
                    (kw.value for kw in dec.keywords if kw.arg == "init"),
                    ast.Constant(value=True),
                )
                if isinstance(dec, ast.Call)
                else ast.Constant(value=True)
            )
            generated_initializer = not (isinstance(init, ast.Constant) and init.value is False)
        if not (
            (isinstance(dec, ast.Call) and source.qualified(dec.func) == MARKERS + ".mock_target")
            or (
                members_only
                and plain_dataclass
                and source.qualified(decorator) == "dataclasses.dataclass"
            )
        ):
            shape.unknown.append("Generated/decorated class interface: " + ast.unparse(dec))
    if node.keywords:
        shape.unknown.append("Metaclass/class keywords can generate members")
    runtime_body = []
    type_checking_properties = {}
    for item in node.body:
        if isinstance(item, ast.If) and source.qualified(item.test) == "typing.TYPE_CHECKING":
            runtime_body.extend(item.orelse)
            if include_type_checking:
                type_checking_properties.update(
                    (name, member)
                    for name, member in callable_members(item.body, source, True).items()
                    if member.kind == "property"
                )
        else:
            runtime_body.append(item)
    declared_callables = callable_members(runtime_body, source, True)
    inherited_initializer = shape.members.get("__init__")
    inherited_fields = shape.initializer_fields
    inherited_unknown = shape.initializer_unknown
    if "__init__" in declared_callables or generated_initializer:
        shape.initializer_fields = {}
        shape.initializer_unknown = []
    shape.members.update(declared_callables)
    if generated_initializer and "__init__" not in declared_callables:
        shape.members["__init__"] = Member(
            "callable", "Generated initializer cannot be verified", line=node.lineno
        )
    initializer = shape.members.get("__init__")
    if initializer is not None and initializer.signature_issue and "__init__" in declared_callables:
        shape.initializer_unknown.append("Initializer decorator can change instance fields")
    if "__init__" not in shape.members:
        shape.members["__init__"] = Member(
            "callable",
            signatures=[
                {
                    "binding": "method",
                    "async": False,
                    "generator": False,
                    "parameters": [],
                }
            ],
        )
    for item in runtime_body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id in declared_callables:
                        # The last binding for this name is a callable.
                        continue
                    if target.id == "__init__" and item.value is not None:
                        shape.initializer_fields = {}
                        shape.initializer_unknown = [
                            "Initializer binding requires source resolution"
                        ]
                    declared_member = shape.members.get(target.id)
                    if (
                        isinstance(item, ast.AnnAssign)
                        and item.value is None
                        and declared_member is not None
                        and declared_member.kind in {"callable", "property"}
                    ):
                        # An annotation without a value does not replace the descriptor.
                        continue
                    annotation = item.annotation if isinstance(item, ast.AnnAssign) else None
                    classvar = isinstance(annotation, ast.Subscript) and source.qualified(
                        annotation.value
                    ) in {"typing.ClassVar", "typing_extensions.ClassVar"}
                    has_default = item.value is not None
                    if (
                        members_only
                        and plain_dataclass
                        and not classvar
                        and isinstance(item.value, ast.Call)
                        and source.qualified(item.value.func) == "dataclasses.field"
                    ):
                        # A factory runs in __init__; object.__new__ does not run it.
                        has_default = any(
                            k.arg == "default"
                            and source.qualified(k.value) != "dataclasses.MISSING"
                            for k in item.value.keywords
                        )
                        if item.value.args or any(k.arg is None for k in item.value.keywords):
                            shape.unknown.append("Dynamic dataclass field options")
                    shape.members[target.id] = Member(
                        "class-attribute" if classvar else "attribute",
                        default=has_default,
                        line=item.lineno,
                    )
                    if target.id == "__slots__":
                        shape.unknown.append("Slot descriptors require separate resolution")
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            member = declared_callables.get(item.name)
            if member is None or item.lineno < member.line:
                continue
            args = [*item.args.posonlyargs, *item.args.args]
            if not args or any(
                source.qualified(d) in {"staticmethod", "classmethod"} for d in item.decorator_list
            ):
                continue
            receiver = args[0].arg
            is_initializer = item.name == "__init__"
            fields = shape.initializer_fields if is_initializer else shape.members
            unknown = shape.initializer_unknown if is_initializer else shape.unknown
            direct_calls = {
                stmt.value
                for stmt in item.body
                if is_initializer
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
            }
            for assignment in statements(item):
                if isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        assignment.targets
                        if isinstance(assignment, ast.Assign)
                        else [assignment.target]
                    )
                    assigned = [
                        pair
                        for target in targets
                        for pair in assignment_leaves(target, assignment.value)
                    ]
                    for target, _ in assigned:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == receiver
                        ):
                            previous = fields.get(target.attr)
                            conditional = not is_initializer or assignment not in item.body
                            if previous is None or (previous.conditional and not conditional):
                                fields[target.attr] = Member(
                                    "attribute", conditional=conditional, line=assignment.lineno
                                )
                if (
                    isinstance(assignment, ast.Call)
                    and source.qualified(assignment.func) == "setattr"
                ):
                    unknown.append("Dynamic setattr in target class")
                if is_initializer and isinstance(assignment, ast.Call):
                    # Passing self or invoking one of its methods may initialize
                    # fields; merely reading self.field as an argument does not.
                    def passes_receiver(value: ast.AST) -> bool:
                        if isinstance(value, ast.Attribute):
                            return False
                        return (isinstance(value, ast.Name) and value.id == receiver) or any(
                            passes_receiver(child) for child in ast.iter_child_nodes(value)
                        )

                    uses_receiver = any(
                        passes_receiver(value)
                        for value in [*assignment.args, *(k.value for k in assignment.keywords)]
                    ) or (
                        isinstance(assignment.func, ast.Attribute)
                        and isinstance(assignment.func.value, ast.Name)
                        and assignment.func.value.id == receiver
                    )
                    super_call = (
                        assignment.func.value
                        if isinstance(assignment.func, ast.Attribute)
                        and isinstance(assignment.func.value, ast.Call)
                        and source.qualified(assignment.func.value.func) == "super"
                        else None
                    )
                    if (
                        isinstance(assignment.func, ast.Attribute)
                        and super_call is not None
                        and assignment.func.attr == "__init__"
                        and not super_call.args
                        and not super_call.keywords
                        and assignment in direct_calls
                        and not shape.skipped_bases
                        and "super" not in source.imports
                        and "super" not in source.bindings(source.tree)
                        and "super" not in source.bindings(item)
                        and matches_initializer_call(
                            assignment, inherited_initializer, item, source
                        )
                    ):
                        for name, member in inherited_fields.items():
                            fields.setdefault(name, member)
                        unknown.extend(inherited_unknown)
                    elif uses_receiver or super_call is not None:
                        unknown.append("Initializer helper effects require source resolution")
            if is_initializer and any(
                isinstance(n, (ast.Return, ast.Delete)) for n in statements(item)
            ):
                unknown.append("Initializer has early exit or deleted fields")
        elif not (isinstance(item, (ast.Pass, ast.Expr))):
            shape.unknown.append("Conditional/dynamic class body")
    # A guarded property may describe an inherited native descriptor.
    for name, member in type_checking_properties.items():
        shape.members.setdefault(name, member)
    if {
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__new__",
    } & shape.members.keys():
        shape.unknown.append("Dynamic attributes or custom __new__")


def issue(code: str, message: str, line: int, uncertain: bool = False) -> dict:
    return {
        "code": code,
        "message": message,
        "line": line,
        "unverified": uncertain,
        "engine": "ast",
    }


def compare_member(name: str, expected: Member, actual: Member, line: int) -> list[dict]:
    if expected.kind != actual.kind:
        return [issue("member-kind", f"{name}: expected {expected.kind}, got {actual.kind}", line)]
    if expected.kind in {"attribute", "class-attribute"}:
        errors = []
        if expected.default != actual.default:
            errors.append(
                issue("member-storage", f"{name}: class default availability differs", line)
            )
        if expected.conditional or actual.conditional:
            errors.append(
                issue(
                    "unverified-member",
                    f"{name}: conditional or lazy member declaration",
                    line,
                    True,
                )
            )
        return errors
    for label, member in (("target", expected), ("replacement", actual)):
        if any(sig.get("missing_receiver") for sig in member.signatures):
            return [issue("signature", f"{name}: {label} cannot accept a bound receiver", line)]
    if reason := expected.signature_issue or actual.signature_issue:
        return [issue("unverified-signature", f"{name}: {reason}", line, True)]
    if expected.origin is not None and expected.origin == actual.origin:
        if expected.signatures == actual.signatures:
            return []

    def structure(signatures: list[dict]) -> list[dict]:
        return [
            {
                **sig,
                "parameters": [
                    (n, kind, default == "required") for n, kind, default in sig["parameters"]
                ],
            }
            for sig in signatures
        ]

    different = structure(expected.signatures) != structure(actual.signatures)
    if not different:
        different = any(
            ep[2] != ap[2] and UNKNOWN not in (ep[2], ap[2])
            for es, ac in zip(expected.signatures, actual.signatures)
            for ep, ap in zip(es["parameters"], ac["parameters"])
        )
    if different:
        return [
            issue(
                "signature",
                f"{name}: expected {expected.signatures}, got {actual.signatures}",
                line,
            )
        ]
    if any(
        default == UNKNOWN
        for sig in expected.signatures + actual.signatures
        for _, _, default in sig["parameters"]
    ):
        return [issue("unverified-signature", f"{name}: unresolved defaults", line, True)]
    return []


def compare_shapes(expected: Shape, actual: Shape, line: int) -> list[dict]:
    errors = [
        issue("unverified-interface", message, line, True)
        for message in dict.fromkeys(expected.unknown + actual.unknown)
    ]
    for name in sorted(expected.members.keys() - actual.members.keys()):
        if name in expected.initializer_fields and actual.initializer_unknown:
            continue
        errors.append(issue("missing-member", name, line))
    for name in sorted(actual.members.keys() - expected.members.keys()):
        errors.append(issue("extra-member", name, actual.members[name].line or line))
    for name in sorted(expected.members.keys() & actual.members.keys()):
        errors.extend(
            compare_member(
                name,
                expected.members[name],
                actual.members[name],
                actual.members[name].line or line,
            )
        )
    return errors


def skip_binding_check(result: dict, member: str, line: int) -> None:
    result.setdefault("skipped_checks", []).append(
        {
            "member": member,
            "line": line,
            "reason": "Compiled tensorrt_llm.bindings interfaces are excluded from test mock checks",
        }
    )


class Analyzer:
    def __init__(self, source: Source, sources: Sources) -> None:
        self.source = source
        self.sources = sources
        self.results: list[dict] = []
        self.handled: set[int] = set()
        self.context: list[str] = []
        self.skipped_bases: set[str] = set()
        self.parents = {
            child: node for node in ast.walk(source.tree) for child in ast.iter_child_nodes(node)
        }

    def shape(
        self,
        definition: Definition,
        *,
        members_only: bool = False,
        include_type_checking: bool = False,
    ) -> Shape:
        shape = class_shape(
            definition,
            self.sources,
            members_only=members_only,
            include_type_checking=include_type_checking,
        )
        self.skipped_bases.update(shape.skipped_bases)
        return shape

    def result(self, target: str, line: int, kind: str, diagnostics: list[dict]) -> None:
        if kind == "instance":
            diagnostics = diagnostics + [
                issue("unverified-control-flow", message, line, True) for message in self.context
            ]
        status = (
            "mismatched"
            if any(not e["unverified"] for e in diagnostics)
            else "unverified"
            if diagnostics
            else "matched"
        )
        self.results.append(
            {
                "path": str(self.source.path),
                "line": line,
                "target": target,
                "kind": kind,
                "status": status,
                "complete": not any(e["unverified"] for e in diagnostics),
                "diagnostics": diagnostics,
            }
        )
        for base in sorted(self.skipped_bases):
            skip_binding_check(self.results[-1], base, line)
        if self.skipped_bases and status == "matched":
            self.results[-1]["status"] = "skipped"
        self.skipped_bases.clear()
        module = target.partition(":")[0]
        if module == BINDINGS_MODULE or module.startswith(BINDINGS_MODULE + "."):
            result = self.results[-1]
            result.update(status="skipped", complete=True, diagnostics=[])
            skip_binding_check(result, target, line)

    def target(self, call: ast.Call, index: int) -> str:
        self.handled.add(id(call))
        allowed_keywords = (
            {"target", "allow_extra", "members_only", "members"}
            if index == 1
            else {"target", "bound"}
        )
        if (
            len(call.args) > index + 1
            or any(k.arg not in allowed_keywords for k in call.keywords)
            or (len(call.args) > index and any(k.arg == "target" for k in call.keywords))
        ):
            raise Incomplete("Invalid marker arguments")
        if index == 0:
            self.bound_replacement(call)
        value = (
            call.args[index]
            if len(call.args) > index
            else next((k.value for k in call.keywords if k.arg == "target"), None)
        )
        if (
            not isinstance(value, ast.Constant)
            or not isinstance(value.value, str)
            or not TARGET.fullmatch(value.value)
        ):
            raise Incomplete("Target must be a literal 'module:ClassOrFunction' string")
        return value.value

    @staticmethod
    def bound_replacement(marker: ast.Call) -> bool:
        value = next(
            (keyword.value for keyword in marker.keywords if keyword.arg == "bound"),
            ast.Constant(value=False),
        )
        if not isinstance(value, ast.Constant) or not isinstance(value.value, bool):
            raise Incomplete("bound must be a literal boolean")
        return value.value

    def expression(self, node: ast.AST | None, env: dict[str, Value]) -> Value:
        if isinstance(node, ast.Name):
            return env.get(node.id, Value())
        if isinstance(node, ast.Attribute):
            owner = self.expression(node.value, env).state
            return owner.fields.get(node.attr, Value()) if owner is not None else Value()
        if isinstance(node, ast.Call):
            name = self.source.qualified(node.func)
            if name == MARKERS + ".verify_instance" and node.args:
                if id(node) not in self.handled:
                    self.instance_registration(node, env)
                return self.expression(node.args[0], env)
            if name == "types.SimpleNamespace":
                if self.uses_rebound_names(node, [node.func]):
                    return Value(State(unknown=["Namespace constructor has an unresolved binding"]))
                state = State()
                if node.args:
                    state.unknown.append("Namespace positional arguments are not supported")
                for keyword in node.keywords:
                    if keyword.arg is None:
                        state.unknown.append("Dynamic namespace unpacking")
                    else:
                        state.fields[keyword.arg] = self.expression(keyword.value, env)
                return Value(state, name)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "__new__" and node.args:
                try:
                    owner = self.source.qualified(node.func.value)
                    if (
                        owner not in {"object", self.source.qualified(node.args[0])}
                        or len(node.args) != 1
                        or node.keywords
                    ):
                        raise Incomplete(
                            "Only object.__new__(Real) or Real.__new__(Real) is supported"
                        )
                    definition = self.sources.reference(self.source, node.args[0])
                    if self.uses_rebound_names(
                        node, [node.func.value, node.args[0]], allowed=definition.node
                    ):
                        raise Incomplete("Allocator has an unresolved binding")
                    if owner != "object":
                        shape = class_shape(definition, self.sources, members_only=True)
                        allocator = shape.members.get("__new__")
                        if allocator is not None and (
                            allocator.kind in {"callable", "property"} or allocator.default
                        ):
                            raise Incomplete(f"Cannot establish the return type of {owner}.__new__")
                    return Value(State(definition=definition), self.source.qualified(node.args[0]))
                except Incomplete as error:
                    return Value(State(unknown=[str(error)]))
            if name.startswith("unittest.mock.") or name == "types.ModuleType":
                if self.uses_rebound_names(node, [node.func]):
                    return Value(State(unknown=["Mock constructor has an unresolved binding"]))
                return Value(
                    State(unknown=[DYNAMIC_MOCK_MEMBERS]),
                    name,
                )
        return Value()

    def instance_registration(self, call: ast.Call, env: dict[str, Value]) -> None:
        """Check an instance marker at its source evaluation point."""
        target = "<unresolved>"
        try:
            target = self.target(call, 1)
            if not call.args:
                raise Incomplete(
                    "verify_instance requires the instance as its first positional argument"
                )
            allow_extra = next(
                (k.value for k in call.keywords if k.arg == "allow_extra"),
                ast.Constant(value=False),
            )
            if not isinstance(allow_extra, ast.Constant) or not isinstance(allow_extra.value, bool):
                raise Incomplete("allow_extra must be a literal boolean")
            members_only = next(
                (k.value for k in call.keywords if k.arg == "members_only"),
                ast.Constant(value=False),
            )
            if not isinstance(members_only, ast.Constant) or not isinstance(
                members_only.value, bool
            ):
                raise Incomplete("members_only must be a literal boolean")
            members = next((k.value for k in call.keywords if k.arg == "members"), None)
            selected = None
            if members is not None and not (
                isinstance(members, ast.Constant) and members.value is None
            ):
                if (
                    not isinstance(members, ast.Tuple)
                    or not members.elts
                    or not all(
                        isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                        and item.value.isidentifier()
                        and item.value != "__init__"
                        for item in members.elts
                    )
                ):
                    raise Incomplete("members must be a non-empty tuple of literal member names")
                selected = tuple(item.value for item in members.elts)
                if len(selected) != len(set(selected)):
                    raise Incomplete("members must name each selected member once")
            errors = self.instance(
                self.expression(call.args[0], env),
                target,
                call.lineno,
                allow_extra=allow_extra.value,
                members_only=members_only.value,
                selected=selected,
            )
            self.result(target, call.lineno, "instance", errors)
            self.results[-1]["scope"] = "direct-members" if members_only.value else "structure"
            if selected is not None:
                self.results[-1].update(scope="selected-members", members=selected)
        except (Incomplete, OSError, SyntaxError, UnicodeError) as error:
            self.result(
                target,
                call.lineno,
                "instance",
                [issue("unverified-state", str(error), call.lineno, True)],
            )

    def uses_rebound_names(
        self, call: ast.Call, references: list[ast.expr], *, allowed: ast.AST | None = None
    ) -> bool:
        names = set()
        for ref in references:
            while isinstance(ref, ast.Attribute):
                ref = ref.value
            if not isinstance(ref, ast.Name):
                return True
            names.add(ref.id)

        scopes = [self.source.tree]
        parent = self.parents.get(call)
        while parent is not None:
            if isinstance(parent, DEFINITIONS + (ast.Lambda,)):
                scopes.append(parent)
            parent = self.parents.get(parent)
        for scope in scopes:
            bindings = self.source.bindings(scope)
            if any(bindings.get(name, set()) - {allowed} for name in names):
                return True
        return False

    def autospecced_patch(self, patch_call: ast.Call, target: str, *, bound: bool) -> list[dict]:
        """Verify a callable patch whose signature is copied by unittest.mock."""
        keywords = {kw.arg: kw.value for kw in patch_call.keywords}
        autospec = keywords.get("autospec")
        if (
            bound
            or not isinstance(autospec, ast.Constant)
            or autospec.value is not True
            or keywords.keys() - {"autospec", "return_value", "side_effect"}
        ):
            raise Incomplete(
                "patch creates a dynamic replacement; use a concrete callable or autospec=True "
                "on the registered callable"
            )
        args = patch_call.args
        patch_name = self.source.qualified(patch_call.func)
        referenced = [patch_call.func]
        if patch_name == "unittest.mock.patch.object" and len(args) == 2:
            referenced.append(args[0])
        if self.uses_rebound_names(patch_call, referenced):
            raise Incomplete("Cannot verify an autospec patch that uses a rebound name")
        if patch_name == "unittest.mock.patch" and len(args) == 1:
            if not isinstance(args[0], ast.Constant) or not isinstance(args[0].value, str):
                raise Incomplete("Autospec requires a literal patch target")
            patched = self.sources.dotted(args[0].value)
        elif patch_name == "unittest.mock.patch.object" and len(args) == 2:
            owner = self.sources.reference(self.source, args[0])
            if (
                not isinstance(owner.node, ast.ClassDef)
                or not isinstance(args[1], ast.Constant)
                or not isinstance(args[1].value, str)
                or not args[1].value.isidentifier()
            ):
                raise Incomplete("Autospec requires a resolved class and a literal method name")
            patched = self.sources.definition(
                f"{owner.source.module}:{owner.node.name}.{args[1].value}"
            )
        else:
            raise Incomplete("Autospec patch uses unsupported target or replacement arguments")
        expected = self.sources.definition(target)
        if patched != expected:
            return [
                issue("patch-target", f"Autospec patch does not patch {target}", patch_call.lineno)
            ]
        member = self.sources.callable(expected)
        if (
            member.signature_issue
            or len(member.signatures) != 1
            or member.signatures[0]["generator"]
            or member.signatures[0].get("missing_receiver")
        ):
            raise Incomplete("Cannot establish the autospecced callable's signature from source")
        return []

    def inline_registration(self, call: ast.Call) -> bool:
        """Discover identity markers embedded in patch arguments without executing them."""
        if not isinstance(call.func, ast.Call):
            return False
        marker = call.func
        if self.source.qualified(marker.func) != MARKERS + ".mock_target":
            return False
        target = "<unresolved>"
        try:
            target = self.target(marker, 0)
            if len(call.args) != 1 or call.keywords:
                raise Incomplete("Apply mock_target to exactly one replacement")
            replacement = call.args[0]
            if isinstance(replacement, ast.Call) and self.source.qualified(replacement.func) in {
                "unittest.mock.patch",
                "unittest.mock.patch.object",
            }:
                errors = self.autospecced_patch(
                    replacement, target, bound=self.bound_replacement(marker)
                )
                self.result(target, call.lineno, "replacement", errors)
                return True
            real = self.sources.definition(target)
            binding = "function"
            if isinstance(replacement, ast.Call):
                descriptor = self.source.qualified(replacement.func).removeprefix("builtins.")
                if (
                    descriptor in {"staticmethod", "classmethod"}
                    and len(replacement.args) == 1
                    and not replacement.keywords
                ):
                    if self.uses_rebound_names(replacement, [replacement.func]):
                        raise Incomplete("Descriptor constructor has an unresolved binding")
                    binding = descriptor
                    replacement = replacement.args[0]
            if not isinstance(replacement, ast.Lambda):
                raise Incomplete(
                    "Dynamic replacement needs a separately registered named definition; "
                    "its interface is not inferred from Mock/spec or its target"
                )
            if isinstance(real.node, ast.ClassDef):
                errors = [issue("target-kind", "Lambda mock targets a class", call.lineno)]
            else:
                bound = self.bound_replacement(marker)
                expected = self.sources.callable(real, bound=bound, keep_binding=True)
                bindings = {sig["binding"] for sig in expected.signatures}
                if binding != "function" and (bound or bindings != {binding}):
                    expected_binding = "function" if bound else ", ".join(sorted(bindings))
                    self.result(
                        target,
                        call.lineno,
                        "replacement",
                        [
                            issue(
                                "signature",
                                f"Expected {expected_binding} binding, got {binding}",
                                call.lineno,
                            )
                        ],
                    )
                    return True
                function = ast.FunctionDef(
                    name="_mock_lambda",
                    args=replacement.args,
                    body=[ast.Return(value=replacement.body)],
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                )
                ast.copy_location(function, replacement)
                ast.fix_missing_locations(function)
                sig = signature(function, binding)
                sig["binding"] = "function"
                actual = Member("callable", signatures=[sig])
                for expected_signature in expected.signatures:
                    expected_signature["binding"] = "function"
                errors = compare_member(function.name, expected, actual, call.lineno)
            self.result(target, call.lineno, "replacement", errors)
        except (Incomplete, OSError, SyntaxError, UnicodeError) as error:
            self.result(
                target,
                call.lineno,
                "replacement",
                [issue("unverified-interface", str(error), call.lineno, True)],
            )
        return True

    def instance(
        self,
        value: Value,
        target: str,
        line: int,
        *,
        allow_extra: bool = False,
        members_only: bool = False,
        selected: tuple[str, ...] | None = None,
    ) -> list[dict]:
        expected_definition = self.sources.definition(target)
        if selected is not None and value.state is not None:
            return self.selected_members(value, expected_definition, selected, line)
        expected = self.shape(
            expected_definition, members_only=members_only, include_type_checking=True
        )
        if value.state is None:
            return [
                issue(
                    "unverified-state",
                    "No statically tracked namespace or bypassed-constructor instance",
                    line,
                    True,
                )
            ]
        state = value.state
        errors = [
            issue("unverified-state", msg, line, True)
            for msg in dict.fromkeys(state.unknown + expected.unknown)
        ]
        available = {}
        if state.definition:
            actual_shape = self.shape(state.definition, members_only=members_only)
            if not members_only and (
                state.definition.source.path,
                state.definition.node.lineno,
            ) != (
                expected_definition.source.path,
                expected_definition.node.lineno,
            ):
                errors.extend(compare_shapes(expected, actual_shape, line))
            available = {
                name: member
                for name, member in actual_shape.members.items()
                if member.default or member.kind in {"callable", "property"}
            }
            if members_only:
                errors.extend(
                    issue("unverified-state", msg, line, True)
                    for msg in actual_shape.unknown
                    if msg not in expected.unknown
                )
                if not allow_extra:
                    errors.extend(
                        issue("extra-member", name, line)
                        for name in available.keys() - expected.members.keys() - state.fields.keys()
                    )
        for name, member in expected.members.items():
            if name == "__init__" or name in available:
                continue
            if name not in state.fields:
                code = (
                    "possibly-uninitialized" if name in state.possible else "missing-initialization"
                )
                if state.unknown:
                    code = "unverified-initialization"
                errors.append(issue(code, name, line, member.conditional or bool(state.unknown)))
        for name in state.fields:
            if name not in expected.members:
                if not allow_extra:
                    errors.append(issue("extra-member", name, line))
                continue
            if members_only:
                continue
            member = expected.members[name]
            if member.kind not in {"attribute", "class-attribute"}:
                errors.append(
                    issue(
                        "unverified-callable",
                        f"{name}: supplied callable/descriptor needs a named definition",
                        line,
                        True,
                    )
                )
                continue
        return errors

    def selected_members(
        self, value: Value, definition: Definition, selected: tuple[str, ...], line: int
    ) -> list[dict]:
        expected = class_shape(
            definition, self.sources, members_only=True, include_type_checking=True
        )
        state = value.state
        assert state is not None
        unknown = set(state.unknown)
        if value.kind in {"unittest.mock.Mock", "unittest.mock.MagicMock"}:
            unknown.discard(DYNAMIC_MOCK_MEMBERS)
        available = state.fields.keys()
        if state.definition:
            actual = class_shape(state.definition, self.sources, members_only=True)
            available = available | {
                name
                for name, member in actual.members.items()
                if member.default or member.kind in {"callable", "property"}
            }
            if "__getattribute__" in actual.members or (
                "__getattr__" in actual.members and any(name not in available for name in selected)
            ):
                unknown.add("Custom attribute lookup on the supplied instance")
        errors = [issue("unverified-state", message, line, True) for message in sorted(unknown)]
        for name in selected:
            if name not in expected.members:
                if expected.skipped_bases:
                    self.skipped_bases.update(expected.skipped_bases)
                elif expected.initializer_unknown:
                    errors.append(issue("unverified-target-member", name, line, True))
                else:
                    errors.append(issue("unknown-target-member", name, line))
            elif name not in available:
                code = (
                    "possibly-uninitialized" if name in state.possible else "missing-initialization"
                )
                if unknown:
                    code = "unverified-initialization"
                errors.append(issue(code, name, line, bool(unknown)))
        return errors

    def assignment(self, target: ast.AST, value: Value, env: dict[str, Value]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, ast.Attribute):
            owner = self.expression(target.value, env).state
            if owner is not None:
                owner.fields[target.attr] = value

    def effects(self, expression: ast.AST, env: dict[str, Value]) -> None:
        """Invalidate tracked instances passed to calls whose effects are not modeled."""
        if isinstance(expression, ast.Lambda):
            return
        nodes = list(statements(expression))
        for node in nodes:
            if isinstance(node, ast.NamedExpr):
                self.assignment(node.target, Value(), env)
        for call in nodes:
            if (
                isinstance(call, ast.Call)
                and self.source.qualified(call.func) == MARKERS + ".verify_instance"
                and id(call) not in self.handled
            ):
                self.instance_registration(call, env)
        for call in nodes:
            if not isinstance(call, ast.Call):
                continue
            name = self.source.qualified(call.func)
            if name in {
                MARKERS + ".verify_instance",
                MARKERS + ".mock_target",
            } or (
                name == "types.SimpleNamespace" and not self.uses_rebound_names(call, [call.func])
            ):
                continue
            arguments = [*call.args, *(kw.value for kw in call.keywords)]
            if isinstance(call.func, ast.Attribute):
                arguments.append(call.func.value)
            for argument in arguments:
                for node in ast.walk(argument):
                    if isinstance(node, ast.Name):
                        state = env.get(node.id, Value()).state
                        if state is not None:
                            state.unknown.append(f"Unresolved call effects at line {call.lineno}")

    def block(self, body: list[ast.stmt], env: dict[str, Value]) -> bool:
        """Track statements and report whether execution can reach the end of the block."""
        for statement in body:
            if isinstance(statement, DEFINITIONS):
                for decorator in statement.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and self.source.qualified(decorator.func) == MARKERS + ".mock_target"
                    ):
                        target = "<unresolved>"
                        try:
                            target = self.target(decorator, 0)
                            real = self.sources.definition(target)
                            if isinstance(statement, ast.ClassDef):
                                if self.bound_replacement(decorator):
                                    raise Incomplete("bound=True requires an instance method")
                                errors = compare_shapes(
                                    self.shape(real, include_type_checking=True),
                                    self.shape(Definition(self.source, statement)),
                                    statement.lineno,
                                )
                            elif isinstance(real.node, ast.ClassDef):
                                errors = [
                                    issue(
                                        "target-kind",
                                        "Function mock targets a class",
                                        statement.lineno,
                                    )
                                ]
                            else:
                                expected = self.sources.callable(
                                    real, bound=self.bound_replacement(decorator)
                                )
                                actual = callable_members(body, self.source, False).get(
                                    statement.name
                                )
                                if expected is None or actual is None:
                                    raise Incomplete(
                                        "Register the owning class for method comparison"
                                    )
                                # Compare descriptors through class access, just like
                                # the target method. Classmethod receivers are implicit.
                                for sig in actual.signatures:
                                    if sig["binding"] in {"classmethod", "staticmethod"}:
                                        sig["binding"] = "function"
                                errors = compare_member(
                                    statement.name,
                                    expected,
                                    actual,
                                    statement.lineno,
                                )
                            self.result(target, decorator.lineno, "definition", errors)
                        except (Incomplete, OSError, SyntaxError, UnicodeError) as error:
                            self.result(
                                target,
                                decorator.lineno,
                                "definition",
                                [issue("unverified-interface", str(error), decorator.lineno, True)],
                            )
                # Function bodies run later, when captured objects may have changed.
                nested = copy.deepcopy(env) if isinstance(statement, ast.ClassDef) else {}
                self.block(statement.body, nested)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                if isinstance(statement, ast.AnnAssign) and statement.value is None:
                    continue
                if statement.value is not None:
                    self.effects(statement.value, env)
                value = self.expression(statement.value, env)
                targets = (
                    statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                )
                values = {id(statement.value): value}
                assignments = []
                for target in targets:
                    for leaf, supplied in assignment_leaves(target, statement.value):
                        if id(supplied) not in values:
                            values[id(supplied)] = self.expression(supplied, env)
                        assignments.append((leaf, values[id(supplied)]))
                # All RHS values are evaluated before any target is assigned,
                # including swaps and chained assignments to shared objects.
                for target, supplied in assignments:
                    self.assignment(target, supplied, env)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                name = self.source.qualified(call.func)
                if name == MARKERS + ".verify_instance":
                    self.instance_registration(call, env)
                elif (
                    name == "setattr"
                    and len(call.args) == 3
                    and isinstance(call.args[1], ast.Constant)
                    and isinstance(call.args[1].value, str)
                ):
                    owner = self.expression(call.args[0], env).state
                    if owner:
                        owner.fields[call.args[1].value] = self.expression(call.args[2], env)
                else:
                    self.effects(call, env)
            elif isinstance(statement, ast.If):
                self.effects(statement.test, env)
                left, right = copy.deepcopy(env), copy.deepcopy(env)
                left_continues = self.block(statement.body, left)
                right_continues = self.block(statement.orelse, right)
                if not (left_continues or right_continues):
                    return False
                if left_continues != right_continues:
                    env.clear()
                    env.update(left if left_continues else right)
                    continue
                merged = {}
                for name in env.keys() | left.keys() | right.keys():
                    a, b = left.get(name, Value()), right.get(name, Value())
                    if a.state is not None and b.state is not None:
                        key = (id(a.state), id(b.state))
                        if key in merged:
                            env[name] = merged[key]
                            continue
                        state = State(definition=a.state.definition)
                        state.possible = (
                            set(a.state.fields)
                            | set(b.state.fields)
                            | a.state.possible
                            | b.state.possible
                        )
                        state.unknown = a.state.unknown + b.state.unknown
                        if a.kind != b.kind or a.state.definition != b.state.definition:
                            state.unknown.append("Branches construct different instance classes")
                        for field_name in a.state.fields.keys() & b.state.fields.keys():
                            av, bv = a.state.fields[field_name], b.state.fields[field_name]
                            state.fields[field_name] = av if av is bv else Value()
                        env[name] = Value(state, a.kind if a.kind == b.kind else UNKNOWN)
                        merged[key] = env[name]
                    else:
                        env[name] = Value()
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    if isinstance(target, ast.Attribute):
                        state = self.expression(target.value, env).state
                        if state:
                            state.fields.pop(target.attr, None)
                    elif isinstance(target, ast.Name):
                        env.pop(target.id, None)
            elif isinstance(
                statement,
                (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match),
            ):
                for value in env.values():
                    if value.state:
                        value.state.unknown.append(
                            f"Unsupported control flow at line {statement.lineno}"
                        )
                # Discovery still visits markers inside unsupported regions.
                self.context.append(f"Unsupported control flow at line {statement.lineno}")
                for field_name in ("body", "orelse", "finalbody"):
                    self.block(getattr(statement, field_name, []), copy.deepcopy(env))
                for handler in getattr(statement, "handlers", []):
                    self.block(handler.body, copy.deepcopy(env))
                self.context.pop()
            elif isinstance(statement, ast.Assert):
                self.effects(statement, env)
            elif isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                self.effects(statement, env)
                # Remaining markers are discovered as unsupported, never as successful checks.
                return False
            elif not isinstance(statement, (ast.Import, ast.ImportFrom, ast.Pass)):
                self.effects(statement, env)
                for value in env.values():
                    if value.state:
                        value.state.unknown.append(
                            f"Unsupported statement at line {statement.lineno}"
                        )
        return True

    def analyze(self) -> list[dict]:
        for node in ast.walk(self.source.tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == MARKERS
                and node not in self.source.tree.body
            ):
                self.result(
                    "<unresolved>",
                    node.lineno,
                    "registration",
                    [
                        issue(
                            "unverified-registration",
                            "Import markers at module scope",
                            node.lineno,
                            True,
                        )
                    ],
                )
        self.block(self.source.tree.body, {})
        for node in ast.walk(self.source.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Call):
                if id(node.func) not in self.handled:
                    self.inline_registration(node)
        for node in ast.walk(self.source.tree):
            if (
                isinstance(node, ast.Call)
                and self.source.qualified(node.func)
                in {MARKERS + ".mock_target", MARKERS + ".verify_instance"}
                and id(node) not in self.handled
            ):
                target = "<unresolved>"
                message = "Registration occurs outside supported eager initialization flow"
                try:
                    target = self.target(
                        node,
                        1
                        if self.source.qualified(node.func) == MARKERS + ".verify_instance"
                        else 0,
                    )
                except Incomplete as error:
                    message = str(error)
                self.result(
                    target,
                    node.lineno,
                    "registration",
                    [
                        issue(
                            "unverified-registration",
                            message,
                            node.lineno,
                            True,
                        )
                    ],
                )
        return self.results


def check(
    root: Path,
    paths: list[Path],
    search_paths: list[Path] | None = None,
) -> dict:
    root = root.resolve()
    sources = Sources([root, root / "tests", *(search_paths or [])])
    results = []
    failures = []
    for path in sorted(set(p.resolve() for p in paths)):
        try:
            text = path.read_text()
            if "interface_checks" not in text:
                continue
            parts = path.relative_to(root).with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            results.extend(Analyzer(sources.read(path, module), sources).analyze())
        except (OSError, UnicodeError, SyntaxError, ValueError) as error:
            failures.append({"path": str(path), "message": str(error)})
    for result in results:
        result["path"] = Path(result["path"]).relative_to(root).as_posix()
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in ("matched", "mismatched", "unverified", "skipped")
    }
    return {
        "tool": "static-mock-interfaces",
        "verification": "ast-only",
        "registrations": len(results),
        "counts": counts,
        "results": results,
        "analysis_errors": failures,
    }


def check_status(report: dict) -> int:
    """Distinguish mismatches from incomplete verification for either entry point."""
    if (
        report["analysis_errors"]
        or report["counts"]["unverified"]
        or any(not r["complete"] for r in report["results"])
    ):
        return 2
    return int(bool(report["counts"]["mismatched"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Test files/directories; default: all tests (also on production-only changes)",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--search-path", action="append", type=Path, default=[])
    parser.add_argument("--report", type=Path, default=Path("mock_interface_report.json"))
    parser.add_argument("--ast-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--advisory", action="store_true", help="Report diagnostics without failing the hook"
    )
    args = parser.parse_args()
    paths = []
    for entry in args.paths or [Path("tests")]:
        entry = entry if entry.is_absolute() else args.root / entry
        paths.extend(entry.rglob("*.py") if entry.is_dir() else [entry])
    report = check(args.root, paths, args.search_path)
    check_exit_code = check_status(report)
    report["check_exit_code"] = check_exit_code
    report["advisory"] = args.advisory
    report["exit_code"] = 0 if args.advisory else check_exit_code
    print(f"Verification: {report['verification']}")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for result in report["results"]:
        print(f"{result['path']}:{result['line']}: {result['status']} {result['target']}")
        for diagnostic in result["diagnostics"]:
            print(f"  [{diagnostic['code']}] {diagnostic['message']}")
        for skipped in result.get("skipped_checks", []):
            print(f"  [skipped] {skipped['member']}: {skipped['reason']}")
    for error in report["analysis_errors"]:
        print(f"Analysis error: {error['path']}: {error['message']}", file=sys.stderr)
    print(
        f"Checked {report['registrations']} registrations: {report['counts']}. Report: {args.report.resolve()}"
    )
    if not report["registrations"]:
        print("No registered interfaces were verified. Add markers in test code to opt in.")
    if args.advisory:
        print(f"Mock interface check exit status: {check_exit_code}; advisory exit status: 0")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
