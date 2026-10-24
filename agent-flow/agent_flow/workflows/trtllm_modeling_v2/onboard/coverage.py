# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether a certification's cells are already an entry's, read from the entry's source.

A certification names the cells it needs: a verify queue row's ``cells``, or
a fenced block in a certification task's goal::

    ```cells
    - {q_len: 4, captured: true}
    - {q_len: 8, captured: true}
    ```

Each required cell is a mapping of the entry's own spec keys. It is covered
by a cell of the entry's ``CELLS`` whose spec holds every one of those keys
at an equal value, on an architecture the entry's ``ARCHS`` names; a key the
requirement does not name is free. The entry's test drives every cell it
lists, so a cell on the tree is a cell CI holds the entry to.

The entry is read without importing it (:func:`read_entry_cells`): the
orchestrator has no torch. ``CELLS`` is evaluated from its source over a
closed set of forms -- literals, ``dict``/``tuple``/``list`` calls, ``Cell``,
module constants, ``range``, comprehensions, ``+`` and ``*``, f-strings. A
name or call outside them stays a symbol (``torch.bfloat16``) or is
unreadable; an unreadable value matches nothing, so what cannot be read is
never taken as covered. Values compare as written: numbers by value, a
symbol by its dotted name or, when the requirement names it without a dot,
by its last part (``bfloat16`` is ``torch.bfloat16``), sequences and
mappings element by element.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import yaml

#: The queue row key, and the fenced block's info string, that name the required cells.
CELLS_KEY = "cells"
_FENCE = re.compile(r"```[ \t]*cells[ \t]*\n(.*?)```", re.DOTALL)
#: How many elements a comprehension or ``range`` may produce before it is unreadable.
_MAX_ITEMS = 4096


@dataclass(frozen=True)
class Symbol:
    """A name the source does not define: an import, an enum member, a dtype."""

    name: str


@dataclass(frozen=True)
class Named:
    """A module constant: its name and the value it is assigned."""

    name: str
    value: Any


class _Unreadable:
    """A value outside the forms read here; equal to nothing."""

    def __repr__(self) -> str:
        return "<unreadable>"


UNREADABLE = _Unreadable()


class _Unknown(Exception):
    """An expression outside the forms read here."""


@dataclass(frozen=True)
class CellRead:
    """One element of ``CELLS`` as read: its ``why`` and ``spec``, or why it could not be read."""

    index: int
    why: str = ""
    spec: Mapping[str, Any] | None = None
    problem: str = ""


@dataclass(frozen=True)
class EntryCells:
    """An entry's ``ARCHS`` and ``CELLS`` as its source states them."""

    archs: tuple[str, ...] = ()
    cells: tuple[CellRead, ...] = ()
    #: Why the cells could not be read at all; empty when they were.
    problem: str = ""


@dataclass(frozen=True)
class Coverage:
    """Whether every required cell is covered, by which entry cell, and what is missing."""

    covered: bool
    matches: tuple[tuple[Mapping[str, Any], CellRead], ...] = ()
    missing: tuple[Mapping[str, Any], ...] = ()
    reason: str = ""


@dataclass
class _Cell:
    why: Any = ""
    spec: Any = field(default_factory=dict)


# ------------------------------------------------------------- requirements


def _cell_list(value: Any) -> list[dict[str, Any]] | None:
    """``value`` as a list of required cells; ``None`` when any element is not a mapping."""
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list) or not value:
        return None
    cells = []
    for item in value:
        if not isinstance(item, Mapping) or not item:
            return None
        cells.append({str(k): v for k, v in item.items()})
    return cells


def required_cells(row: Mapping[str, Any] | None, text: str = "") -> tuple[list[dict], str]:
    """The cells a certification requires, and where they are stated; ``([], why)`` when none.

    A queue row's ``cells`` first, then a fenced ```` ```cells ```` block in
    ``text`` (the task's goal). A statement that does not parse as a list of
    non-empty mappings states nothing, and the certification is dispatched.
    """
    if isinstance(row, Mapping) and row.get(CELLS_KEY) is not None:
        cells = _cell_list(row.get(CELLS_KEY))
        if cells is None:
            return [], "the queue row's `cells` is not a list of mappings"
        return cells, "the queue row's `cells`"
    match = _FENCE.search(text or "")
    if match is None:
        return [], "no required cells are stated"
    try:
        cells = _cell_list(yaml.safe_load(match.group(1)))
    except yaml.YAMLError as exc:
        return [], f"the goal's ```cells block is not YAML: {exc}"
    if cells is None:
        return [], "the goal's ```cells block is not a list of mappings"
    return cells, "the goal's ```cells block"


# ------------------------------------------------------------- reading


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _plain(value: Any) -> Any:
    """``value`` with module constants replaced by what they are assigned."""
    return value.value if isinstance(value, Named) else value


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.Div: operator.truediv,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class _Reader:
    """Evaluates the forms the module docstring lists, over a module's and a class's names."""

    def __init__(self, module: Mapping[str, ast.expr], klass: Mapping[str, ast.expr]) -> None:
        self.module = dict(module)
        self.klass = dict(klass)
        self.done: dict[str, Any] = {}
        self.busy: set[str] = set()

    def name(self, name: str, scope: Mapping[str, Any]) -> Any:
        if name in scope:
            return scope[name]
        for table, label in ((self.klass, "class"), (self.module, "module")):
            if name in table:
                key = f"{label}:{name}"
                if key not in self.done:
                    if key in self.busy:
                        raise _Unknown(f"{name} refers to itself")
                    self.busy.add(key)
                    try:
                        self.done[key] = self.eval(table[name], {})
                    finally:
                        self.busy.discard(key)
                value = self.done[key]
                return value if label == "class" else Named(name, _plain(value))
        if name in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[name]
        return Symbol(name)

    def value(self, node: ast.expr, scope: Mapping[str, Any]) -> Any:
        """A mapping's value: :data:`UNREADABLE` when it is outside the forms read here.

        So one key no reader can evaluate leaves the rest of a cell's spec,
        and matches nothing.
        """
        try:
            return self.eval(node, scope)
        except _Unknown:
            return UNREADABLE

    def items(self, node: ast.expr, scope: Mapping[str, Any]) -> list[Any]:
        value = _plain(self.eval(node, scope))
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict):
            return list(value)
        raise _Unknown("not a sequence")

    def sequence(self, elts: Sequence[ast.expr], scope: Mapping[str, Any]) -> list[Any]:
        out: list[Any] = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                out.extend(self.items(elt.value, scope))
            else:
                out.append(self.eval(elt, scope))
        return out

    def comprehension(self, node: Any, scope: Mapping[str, Any]) -> list[Any]:
        results: list[Any] = []

        def bind(target: ast.expr, value: Any, into: dict[str, Any]) -> None:
            if isinstance(target, ast.Name):
                into[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)):
                value = _plain(value)
                if not isinstance(value, (list, tuple)) or len(value) != len(target.elts):
                    raise _Unknown("unpacking")
                for sub, item in zip(target.elts, value):
                    bind(sub, item, into)
            else:
                raise _Unknown("comprehension target")

        def walk(index: int, local: dict[str, Any]) -> None:
            if index == len(node.generators):
                results.append(self.eval(node.elt, local))
                if len(results) > _MAX_ITEMS:
                    raise _Unknown("too many items")
                return
            gen = node.generators[index]
            if gen.is_async:
                raise _Unknown("async comprehension")
            for item in self.items(gen.iter, local):
                inner = dict(local)
                bind(gen.target, item, inner)
                if all(_plain(self.eval(test, inner)) is True for test in gen.ifs):
                    walk(index + 1, inner)

        walk(0, dict(scope))
        return results

    def call(self, node: ast.Call, scope: Mapping[str, Any]) -> Any:
        func = _dotted(node.func) or ""
        last = func.rsplit(".", 1)[-1]
        args = self.sequence(node.args, scope)
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                spread = _plain(self.eval(keyword.value, scope))
                if not isinstance(spread, dict):
                    raise _Unknown("** of a non-mapping")
                kwargs.update(spread)
            elif func == "dict":
                kwargs[keyword.arg] = self.value(keyword.value, scope)
            else:
                kwargs[keyword.arg] = self.eval(keyword.value, scope)
        if last == "Cell":
            why = kwargs.get("why", args[0] if args else "")
            spec = kwargs.get("spec", args[1] if len(args) > 1 else {})
            return _Cell(why=why, spec=spec)
        if func == "dict":
            base = dict(_plain(args[0])) if args else {}
            base.update(kwargs)
            return base
        if (
            func in ("tuple", "list", "frozenset", "set", "sorted")
            and len(args) <= 1
            and not kwargs
        ):
            values = list(_plain(args[0])) if args else []
            return sorted(values) if func == "sorted" else values
        if func == "range" and not kwargs and 1 <= len(args) <= 3:
            bounds = [_plain(a) for a in args]
            if not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds):
                raise _Unknown("range of non-integers")
            values = list(range(*bounds))
            if len(values) > _MAX_ITEMS:
                raise _Unknown("range too long")
            return values
        raise _Unknown(f"call to {func or 'an expression'}")

    def eval(self, node: ast.expr, scope: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.name(node.id, scope)
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted is None:
                raise _Unknown("attribute of an expression")
            return Symbol(dotted)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return self.sequence(node.elts, scope)
        if isinstance(node, ast.Dict):
            out: dict[Any, Any] = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    spread = _plain(self.eval(value, scope))
                    if not isinstance(spread, dict):
                        raise _Unknown("** of a non-mapping")
                    out.update(spread)
                    continue
                name = _plain(self.eval(key, scope))
                if not isinstance(name, (str, int, float, bool)) and name is not None:
                    raise _Unknown("a key that is not a constant")
                out[name] = self.value(value, scope)
            return out
        if isinstance(node, ast.Call):
            return self.call(node, scope)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            return self.comprehension(node, scope)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left = _plain(self.eval(node.left, scope))
            right = _plain(self.eval(node.right, scope))
            if isinstance(left, (Symbol, _Unreadable)) or isinstance(right, (Symbol, _Unreadable)):
                raise _Unknown("arithmetic on a symbol")
            if isinstance(left, list) and isinstance(right, list) and isinstance(node.op, ast.Add):
                return left + right
            try:
                return _BINARY[type(node.op)](left, right)
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise _Unknown(str(exc)) from exc
        if isinstance(node, ast.UnaryOp):
            value = _plain(self.eval(node.operand, scope))
            if isinstance(node.op, ast.Not):
                if isinstance(value, (Symbol, _Unreadable)):
                    raise _Unknown("not of a symbol")
                return not value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value if isinstance(node.op, ast.USub) else +value
            raise _Unknown("unary operator")
        if isinstance(node, ast.Compare):
            left = _plain(self.eval(node.left, scope))
            for op, right_node in zip(node.ops, node.comparators):
                right = _plain(self.eval(right_node, scope))
                if (
                    type(op) not in _COMPARE
                    or isinstance(left, Symbol)
                    or isinstance(right, Symbol)
                ):
                    raise _Unknown("comparison")
                try:
                    if not _COMPARE[type(op)](left, right):
                        return False
                except TypeError as exc:
                    raise _Unknown(str(exc)) from exc
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            values = [_plain(self.eval(v, scope)) for v in node.values]
            if not all(isinstance(v, bool) for v in values):
                raise _Unknown("boolean of non-booleans")
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.IfExp):
            test = _plain(self.eval(node.test, scope))
            if not isinstance(test, bool):
                raise _Unknown("condition that is not a boolean")
            return self.eval(node.body if test else node.orelse, scope)
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    try:
                        parts.append(str(_plain(self.eval(value.value, scope))))
                    except _Unknown:
                        parts.append("{?}")
                else:
                    parts.append(str(getattr(value, "value", "")))
            return "".join(parts)
        if isinstance(node, ast.Subscript):
            container = _plain(self.eval(node.value, scope))
            index = _plain(self.eval(node.slice, scope))
            try:
                return container[index]
            except (TypeError, KeyError, IndexError) as exc:
                raise _Unknown(str(exc)) from exc
        raise _Unknown(type(node).__name__)


def _assignments(body: Iterable[ast.stmt]) -> dict[str, ast.expr]:
    found: dict[str, ast.expr] = {}
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value:
            found[stmt.target.id] = stmt.value
    return found


def _is_wrapper(node: ast.ClassDef) -> bool:
    return any(
        (isinstance(b, ast.Name) and b.id == "OpWrapper")
        or (isinstance(b, ast.Attribute) and b.attr == "OpWrapper")
        for b in node.bases
    )


def _arch_names(value: Any) -> tuple[str, ...]:
    found: set[str] = set()

    def walk(item: Any) -> None:
        item = item.value if isinstance(item, Named) else item
        if isinstance(item, Symbol) and item.name.rsplit(".", 1)[-1].startswith("SM_"):
            found.add(item.name.rsplit(".", 1)[-1].lower())
        elif isinstance(item, str) and item.startswith("sm_"):
            found.add(item)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)

    walk(value)
    return tuple(sorted(found))


def _spec_values(spec: Any) -> Mapping[str, Any] | None:
    spec = _plain(spec)
    if not isinstance(spec, dict):
        return None
    return {str(k): v for k, v in spec.items()}


def read_entry_cells(source: str) -> EntryCells:
    """An entry's ``ARCHS`` and ``CELLS``, read from its source; see the module docstring."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return EntryCells(problem=f"the entry does not parse: {exc}")
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and _is_wrapper(n)]
    if len(classes) != 1:
        return EntryCells(problem=f"the entry holds {len(classes)} OpWrapper classes, not one")
    module = _assignments(tree.body)
    klass = _assignments(classes[0].body)
    reader = _Reader(module, klass)
    if "ARCHS" not in klass:
        return EntryCells(problem="the entry declares no ARCHS")
    try:
        archs = _arch_names(reader.name("ARCHS", {}))
    except _Unknown as exc:
        return EntryCells(problem=f"ARCHS cannot be read: {exc}")
    if "CELLS" not in klass:
        return EntryCells(archs=archs, problem="the entry declares no CELLS")
    node = klass["CELLS"]
    elements: list[tuple[int, ast.expr | None]] = []
    if isinstance(node, (ast.Tuple, ast.List)) and not any(
        isinstance(e, ast.Starred) for e in node.elts
    ):
        elements = list(enumerate(node.elts))
    cells: list[CellRead] = []
    if elements:
        # One element at a time, so one unreadable cell leaves the others.
        for index, elt in elements:
            try:
                value = reader.eval(elt, {})
            except _Unknown as exc:
                cells.append(CellRead(index=index, problem=str(exc)))
                continue
            cells.append(_cell_read(index, value))
    else:
        try:
            values = reader.items(node, {})
        except _Unknown as exc:
            return EntryCells(archs=archs, problem=f"CELLS cannot be read: {exc}")
        cells = [_cell_read(index, value) for index, value in enumerate(values)]
    return EntryCells(archs=archs, cells=tuple(cells))


def _cell_read(index: int, value: Any) -> CellRead:
    value = _plain(value)
    if not isinstance(value, _Cell):
        return CellRead(index=index, problem="not a Cell(...)")
    spec = _spec_values(value.spec)
    why = _plain(value.why)
    if spec is None:
        return CellRead(index=index, why=str(why), problem="its spec is not a mapping")
    return CellRead(index=index, why=str(why), spec=spec)


# ------------------------------------------------------------- matching


def _name_matches(want: str, name: str) -> bool:
    return want == name or ("." not in want and name.rsplit(".", 1)[-1] == want)


def same(want: Any, have: Any) -> bool:
    """Whether a required value ``want`` equals an entry's ``have``; see the module docstring."""
    if isinstance(have, _Unreadable) or isinstance(want, _Unreadable):
        return False
    if isinstance(have, Named):
        return same(want, have.value) or (isinstance(want, str) and _name_matches(want, have.name))
    if isinstance(have, Symbol):
        return isinstance(want, str) and _name_matches(want, have.name)
    if isinstance(want, bool) or isinstance(have, bool):
        return isinstance(want, bool) and isinstance(have, bool) and want == have
    if isinstance(want, (int, float)) and isinstance(have, (int, float)):
        return float(want) == float(have)
    if want is None or have is None:
        return want is None and have is None
    if isinstance(want, str) and isinstance(have, str):
        return want == have
    if isinstance(want, (list, tuple)) and isinstance(have, (list, tuple)):
        return len(want) == len(have) and all(same(w, h) for w, h in zip(want, have))
    if isinstance(want, Mapping) and isinstance(have, Mapping):
        return set(map(str, want)) == set(map(str, have)) and all(
            same(want[k], have.get(k, have.get(str(k)))) for k in want
        )
    return False


def covers(required: Mapping[str, Any], cell: CellRead) -> bool:
    """Whether ``cell`` holds every key of ``required`` at an equal value."""
    if cell.spec is None:
        return False
    return all(key in cell.spec and same(value, cell.spec[key]) for key, value in required.items())


def coverage(required: Sequence[Mapping[str, Any]], entry: EntryCells, arch: str) -> Coverage:
    """Whether ``entry`` covers every required cell on ``arch``; see the module docstring."""
    if not required:
        return Coverage(False, reason="no required cells are stated")
    if entry.problem:
        return Coverage(False, missing=tuple(required), reason=entry.problem)
    if arch not in entry.archs:
        return Coverage(
            False,
            missing=tuple(required),
            reason=f"ARCHS names {', '.join(entry.archs) or 'no architecture'}, not {arch}",
        )
    matches, missing = [], []
    for want in required:
        cell = next((c for c in entry.cells if covers(want, c)), None)
        if cell is None:
            missing.append(want)
        else:
            matches.append((want, cell))
    if missing:
        unread = sum(1 for c in entry.cells if c.spec is None)
        reason = f"{len(missing)} of {len(required)} required cell(s) are in no cell of CELLS"
        if unread:
            reason += f" ({unread} cell(s) could not be read from the source)"
        return Coverage(False, tuple(matches), tuple(missing), reason)
    return Coverage(True, tuple(matches), (), f"every required cell is in CELLS on {arch}")


def describe_cell(cell: Mapping[str, Any]) -> str:
    """A required cell on one line."""
    return "{" + ", ".join(f"{k}: {v!r}" for k, v in cell.items()) + "}"


__all__ = [
    "CELLS_KEY",
    "CellRead",
    "Coverage",
    "EntryCells",
    "Named",
    "Symbol",
    "UNREADABLE",
    "coverage",
    "covers",
    "describe_cell",
    "read_entry_cells",
    "required_cells",
    "same",
]
