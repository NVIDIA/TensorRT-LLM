# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every fixed size in a target must answer to an engine limit.

A catalog entry is certified over the serving envelope, kernel by kernel. A
target can still set a ceiling none of those receipts sees: a module-level
``_TOPK_MAX_ROWS = 8192`` that sizes an offsets buffer at construction makes
every engine with ``max_num_tokens`` above 8192 die in warm-up, while every
kernel it feeds is certified to 16384. No gate launched above 8192, so no gate
saw it either.

This module reads a target statically -- no import, no torch -- and lists:

- **capacity constants**: module- and class-level integer constants whose
  name says they bound something (``MAX``, ``ROWS``, ``TOKENS``, ``BATCH``,
  ``CAPACITY``, ...), or that size an allocation, or that a comparison or a
  ``min``/``max`` uses as a ceiling;
- **fixed-size buffers**: allocation calls (``torch.empty``/``zeros``/
  ``arange``/...) sized by such a constant or by an integer literal of 1024 or
  more.

Each one must be **tied** to the engine: sized from an engine argument
(``model_config.max_num_tokens``, ``max_batch_size``, ``max_seq_len``, ...),
which the audit recognizes and passes; or **bound** by a declaration, which
the audit then checks against the envelope. Two spellings of the declaration,
both read without importing the target::

    _TOPK_MAX_ROWS = 16384  # capacity: max_num_tokens
    _WINDOW = 128  # capacity: fixed: the checkpoint's sliding window

    CAPACITY_BOUNDS = {
        "_GATHER_ROWS": "max_num_tokens * world_size",
        "Indexer.SLOTS": "max_batch_size",
    }

A bound constant passes when its value is at least the envelope's limit for
that axis (times the factor, where ``world_size`` is the widest attention-DP
gather the envelope claims). ``fixed: <reason>`` records that the number is
not a capacity at all -- a tile width, a window the checkpoint fixes -- and the
reason is what the reviewer checks.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .envelope import ENVELOPE_AXES, ServingEnvelope

#: Engine arguments a size can be derived from. A size expression that reads
#: one of these -- as a name or as an attribute of anything -- follows the
#: engine's configuration and is tied by construction.
ENGINE_ARGS: tuple[str, ...] = (
    *ENVELOPE_AXES,
    "max_draft_len",
    "max_beam_width",
    "max_num_requests",
    "max_num_sequences",
    "tokens_per_block",
    "max_total_draft_tokens",
)

#: What a binding may name: an envelope axis, or the draft length.
BINDABLE_AXES: tuple[str, ...] = (*ENVELOPE_AXES, "max_draft_len")

#: The module-level mapping a target may declare its bindings in.
BOUNDS_NAME = "CAPACITY_BOUNDS"

#: ``# capacity: <binding>`` on the constant's or the allocation's line.
_INLINE = re.compile(r"#\s*capacity:\s*(?P<binding>.+?)\s*$")

#: Name segments that say a constant bounds something.
CAPACITY_WORDS: tuple[str, ...] = (
    "MAX",
    "ROWS",
    "TOKENS",
    "BATCH",
    "CAPACITY",
    "CAP",
    "SEQ",
    "SEQLEN",
    "LEN",
    "LIMIT",
    "SLOTS",
    "POOL",
    "PAGES",
    "BLOCKS",
    "ENTRIES",
    "REQUESTS",
    "BUF",
    "BUFFER",
)

#: Calls that allocate a buffer whose size is their argument.
ALLOC_FUNCS: frozenset[str] = frozenset(
    {
        "empty",
        "zeros",
        "ones",
        "full",
        "arange",
        "empty_strided",
        "randn",
        "rand",
        "randint",
        "new_empty",
        "new_zeros",
        "new_ones",
        "new_full",
    }
)

#: An integer literal at least this large, sizing a buffer, is a capacity.
LITERAL_THRESHOLD = 1024

_CONSTANT_NAME = re.compile(r"^_?[A-Z][A-Z0-9_]*$")

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.LShift: operator.lshift,
    ast.Pow: operator.pow,
}

STATUS_OK = "ok"
STATUS_FAIL = "fail"


@dataclass(frozen=True)
class Binding:
    """A parsed declaration: an axis times a factor, or ``fixed`` with a reason."""

    raw: str
    axis: str | None = None
    factor: int = 1
    world: bool = False
    reason: str = ""
    error: str = ""

    @property
    def fixed(self) -> bool:
        return self.axis is None and not self.error

    def required(self, envelope: ServingEnvelope) -> int | None:
        """The smallest value that covers the envelope, or None if unknowable."""
        if self.axis is None:
            return None
        if self.axis == "max_draft_len":
            base = envelope.max_draft_len or None
        else:
            base = envelope.limit(self.axis)
        if base is None:
            return None
        world = envelope.max_gather_world_size if self.world else 1
        return base * self.factor * world


def parse_binding(text: str) -> Binding:
    """``max_num_tokens``, ``max_num_tokens * 4``, ``... * world_size``, ``fixed: why``."""
    raw = text.strip()
    if raw.lower().startswith("fixed"):
        reason = raw.partition(":")[2].strip()
        if not reason:
            return Binding(
                raw=raw, error="`fixed` needs a reason: `fixed: <why this is not a capacity>`"
            )
        return Binding(raw=raw, reason=reason)
    parts = [part.strip() for part in raw.split("*")]
    axis = parts[0]
    if axis not in BINDABLE_AXES:
        return Binding(
            raw=raw,
            error=f"binds to {axis!r}, which is not one of {list(BINDABLE_AXES)} or `fixed: <why>`",
        )
    factor = 1
    world = False
    for part in parts[1:]:
        if part == "world_size":
            world = True
        elif part.isdigit() and int(part) >= 1:
            factor *= int(part)
        else:
            return Binding(raw=raw, error=f"factor {part!r} is not an integer or `world_size`")
    return Binding(raw=raw, axis=axis, factor=factor, world=world)


@dataclass(frozen=True)
class Constant:
    """One integer constant at module or class scope."""

    name: str
    value: int
    file: str
    line: int
    inline: str = ""


@dataclass(frozen=True)
class Allocation:
    """One allocation call sized by a constant or a large literal."""

    call: str
    file: str
    line: int
    scope: str
    constants: tuple[str, ...] = ()
    literal: int | None = None
    tied: bool = False
    inline: str = ""


@dataclass(frozen=True)
class CapacityFinding:
    """One audited capacity, and its verdict."""

    kind: str
    name: str
    file: str
    line: int
    value: int | None
    status: str
    detail: str
    binding: str = ""

    @property
    def passed(self) -> bool:
        return self.status == STATUS_OK


@dataclass(frozen=True)
class CapacityAudit:
    """The audit of one target against one envelope."""

    findings: tuple[CapacityFinding, ...] = ()
    files: tuple[str, ...] = ()
    tied: tuple[Allocation, ...] = ()
    envelope_declared: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> tuple[CapacityFinding, ...]:
        return tuple(f for f in self.findings if not f.passed)

    @property
    def passed(self) -> bool:
        return not self.failures and not self.errors

    def checks(self) -> list[Any]:
        """Delivery checks: one per failing capacity, or one summary when clean."""
        from .delivery import Check

        checks = [
            Check(name=f"capacity:{f.name}", passed=False, detail=f"{f.file}:{f.line}: {f.detail}")
            for f in self.failures
        ]
        checks.extend(Check(name="capacity:parse", passed=False, detail=e) for e in self.errors)
        if not checks:
            checks.append(
                Check(
                    name="capacity:audit",
                    passed=True,
                    detail=(
                        f"{len(self.findings)} capacity constant(s)/buffer(s) bound or fixed, "
                        f"{len(self.tied)} buffer(s) sized from engine arguments, over "
                        f"{len(self.files)} file(s)"
                    ),
                )
            )
        return checks

    def render(self) -> str:
        """The reviewer's table: every capacity, its binding and its verdict."""
        basis = (
            "the declared serving envelope"
            if self.envelope_declared
            else "a DERIVED envelope (the gate tuples' hull) -- the spec declares none, so a "
            "constant that passes here is proven only up to what the gates ran"
        )
        lines = [
            "## Capacity constants in the target",
            "",
            f"Audited statically against {basis}. Each row is a fixed size: a FAIL is "
            "unbound or below the envelope; an ok row is a CLAIM -- check that the "
            "binding is true of the code (the constant really caps that axis, the "
            "`fixed:` reason really holds).",
            "",
        ]
        if not self.findings and not self.errors:
            lines.append(
                f"No capacity constant or fixed-size buffer found in {len(self.files)} file(s); "
                f"{len(self.tied)} allocation(s) are sized from engine arguments."
            )
            return "\n".join(lines) + "\n"
        lines += [
            "| verdict | kind | name | value | binding | where | detail |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in self.findings:
            value = "" if f.value is None else str(f.value)
            lines.append(
                f"| {'ok' if f.passed else 'FAIL'} | {f.kind} | `{f.name}` | {value} | "
                f"{f.binding or '(none)'} | {f.file}:{f.line} | {f.detail} |"
            )
        for error in self.errors:
            lines.append(f"| FAIL | parse | | | | | {error} |")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ parsing


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _references(node: ast.AST) -> tuple[set[str], bool]:
    """Constant-like names an expression reads, and whether it reads an engine arg."""
    names: set[str] = set()
    engine = False
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if child.id in ENGINE_ARGS:
                engine = True
            elif _CONSTANT_NAME.match(child.id):
                names.add(child.id)
        elif isinstance(child, ast.Attribute):
            if child.attr in ENGINE_ARGS:
                engine = True
            elif _CONSTANT_NAME.match(child.attr):
                # `Class.CONST` or `self.CONST`: keep both spellings.
                names.add(child.attr)
                if isinstance(child.value, ast.Name):
                    names.add(f"{child.value.id}.{child.attr}")
    return names, engine


def _evaluate(node: ast.AST, known: Mapping[str, int]) -> int | None:
    """An integer constant expression's value, or None."""
    if isinstance(node, ast.Constant):
        value = node.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(node, ast.Name):
        return known.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _evaluate(node.operand, known)
        return -inner if inner is not None else None
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left, known)
        right = _evaluate(node.right, known)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow) and (right < 0 or right > 64):
            return None
        try:
            return int(_BINARY[type(node.op)](left, right))
        except (ArithmeticError, ValueError):
            return None
    return None


def _inline(lines: list[str], node: ast.AST) -> str:
    for number in {getattr(node, "lineno", 0), getattr(node, "end_lineno", 0)}:
        if 1 <= number <= len(lines):
            match = _INLINE.search(lines[number - 1])
            if match:
                return match.group("binding")
    return ""


def _size_arguments(call: ast.Call) -> list[ast.expr]:
    """The arguments of an allocation call that give its size."""
    name = _call_name(call.func)
    sizes = [kw.value for kw in call.keywords if kw.arg in ("size", "end", "high")]
    if name in ("full", "new_full"):
        sizes.extend(call.args[:1])
    elif name == "randint":
        sizes.extend(call.args[1:3])
    elif name == "arange":
        sizes.extend(call.args[:2])
    else:
        sizes.extend(call.args)
    return sizes


class _Collector(ast.NodeVisitor):
    def __init__(self, file: str, lines: list[str]) -> None:
        self.file = file
        self.lines = lines
        self.constants: dict[str, Constant] = {}
        self.values: dict[str, int] = {}
        self.allocations: list[Allocation] = []
        self.ceilings: set[str] = set()
        self.bounds: dict[str, str] = {}
        self.bounds_line = 0
        self.errors: list[str] = []
        self._scope: list[str] = []

    # Constants live at module scope and directly in class bodies.
    def _record_constant(self, target: ast.expr, value: ast.AST, node: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        if target.id == BOUNDS_NAME and not self._scope:
            self._record_bounds(value, node)
            return
        if not _CONSTANT_NAME.match(target.id):
            return
        number = _evaluate(value, self.values)
        if number is None:
            return
        owner = ".".join(s for s in self._scope if not s.startswith("def "))
        qualified = f"{owner}.{target.id}" if owner else target.id
        self.values[target.id] = number
        self.constants[qualified] = Constant(
            name=qualified,
            value=number,
            file=self.file,
            line=getattr(node, "lineno", 0),
            inline=_inline(self.lines, node),
        )

    def _record_bounds(self, value: ast.AST, node: ast.AST) -> None:
        self.bounds_line = getattr(node, "lineno", 0)
        try:
            parsed = ast.literal_eval(value)
        except ValueError:
            self.errors.append(
                f"{self.file}:{self.bounds_line}: {BOUNDS_NAME} must be a literal mapping of "
                f"constant name to binding"
            )
            return
        if not isinstance(parsed, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
        ):
            self.errors.append(
                f"{self.file}:{self.bounds_line}: {BOUNDS_NAME} must map constant names to "
                f"binding strings"
            )
            return
        self.bounds.update(parsed)

    def _in_function(self) -> bool:
        return any(s.startswith("def ") for s in self._scope)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._in_function() and len(node.targets) == 1:
            self._record_constant(node.targets[0], node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._in_function() and node.value is not None:
            self._record_constant(node.target, node.value, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope.append(f"def {node.name}")
        self.generic_visit(node)
        self._scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Compare(self, node: ast.Compare) -> None:
        for operand in (node.left, *node.comparators):
            names, _ = _references(operand)
            self.ceilings.update(names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in ("min", "max", "clamp", "clamp_max", "clip"):
            for arg in node.args:
                names, _ = _references(arg)
                self.ceilings.update(names)
        if name in ALLOC_FUNCS:
            self._record_allocation(node, name)
        self.generic_visit(node)

    def _record_allocation(self, node: ast.Call, name: str) -> None:
        names: set[str] = set()
        tied = False
        literal: int | None = None
        for size in _size_arguments(node):
            found, engine = _references(size)
            names.update(found)
            tied = tied or engine
            # A literal expression (`1 << 20`) as a whole, then every literal
            # inside a shape tuple (`(8192, 4)`).
            for child in ast.walk(size):
                value = _evaluate(child, {})
                if value is not None and value >= LITERAL_THRESHOLD:
                    literal = max(literal or 0, value)
        if not names and literal is None and not tied:
            return
        functions = [s[4:] for s in self._scope if s.startswith("def ")]
        scope = "import" if not functions else functions[-1]
        self.allocations.append(
            Allocation(
                call=name,
                file=self.file,
                line=node.lineno,
                scope=scope,
                constants=tuple(sorted(names)),
                literal=literal,
                tied=tied,
                inline=_inline(self.lines, node),
            )
        )


def _is_capacity_name(name: str) -> bool:
    words = name.split(".")[-1].strip("_").split("_")
    return any(word in CAPACITY_WORDS for word in words)


def target_files(target_dir: Path) -> list[Path]:
    """Every Python file of a target, tests and caches excluded."""
    return sorted(
        path
        for path in Path(target_dir).rglob("*.py")
        if "__pycache__" not in path.parts and not path.name.startswith("test_")
    )


def audit_target(target_dir: Path | str, envelope: ServingEnvelope) -> CapacityAudit:
    """Audit every capacity constant and fixed-size buffer under ``target_dir``."""
    root = Path(target_dir)
    findings: list[CapacityFinding] = []
    tied: list[Allocation] = []
    errors: list[str] = []
    files = target_files(root) if root.is_dir() else []
    for path in files:
        rel = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"{rel}: cannot parse: {exc}")
            continue
        collector = _Collector(rel, source.splitlines())
        collector.visit(tree)
        errors.extend(collector.errors)
        findings.extend(_judge_file(collector, envelope))
        tied.extend(
            a for a in collector.allocations if a.tied and not a.constants and not a.literal
        )
    return CapacityAudit(
        findings=tuple(findings),
        files=tuple(str(p.relative_to(root)) for p in files),
        tied=tuple(tied),
        envelope_declared=envelope.declared,
        errors=tuple(errors),
    )


def _lookup(collector: _Collector, name: str) -> Constant | None:
    if name in collector.constants:
        return collector.constants[name]
    matches = [c for key, c in collector.constants.items() if key.split(".")[-1] == name]
    return matches[0] if len(matches) == 1 else None


def _judge_file(collector: _Collector, envelope: ServingEnvelope) -> list[CapacityFinding]:
    sized = {name for alloc in collector.allocations for name in alloc.constants}
    relevant: dict[str, Constant] = {}
    for qualified, constant in collector.constants.items():
        short = qualified.split(".")[-1]
        # A small constant that sizes a buffer is a dimension (a head width, a
        # tile), not a capacity; it still counts when its name or its use as
        # a ceiling says otherwise.
        sizes_large = constant.value >= LITERAL_THRESHOLD and (short in sized or qualified in sized)
        if (
            _is_capacity_name(qualified)
            or sizes_large
            or short in collector.ceilings
            or qualified in collector.ceilings
            or qualified in collector.bounds
            or short in collector.bounds
            or constant.inline
        ):
            relevant[qualified] = constant

    findings: list[CapacityFinding] = []
    for name in sorted(collector.bounds):
        if _lookup(collector, name) is None:
            findings.append(
                CapacityFinding(
                    kind="binding",
                    name=name,
                    file=collector.file,
                    line=collector.bounds_line,
                    value=None,
                    status=STATUS_FAIL,
                    detail=f"{BOUNDS_NAME} binds {name!r}, which this file does not define",
                    binding=collector.bounds[name],
                )
            )

    for qualified, constant in sorted(relevant.items()):
        short = qualified.split(".")[-1]
        declared = constant.inline or collector.bounds.get(qualified) or collector.bounds.get(short)
        uses = []
        if short in sized or qualified in sized:
            uses.append("sizes a buffer")
        if short in collector.ceilings or qualified in collector.ceilings:
            uses.append("used as a ceiling")
        findings.append(
            _judge(
                kind="constant",
                name=qualified,
                file=collector.file,
                line=constant.line,
                value=constant.value,
                declared=declared,
                uses=uses,
                envelope=envelope,
            )
        )

    for alloc in collector.allocations:
        if alloc.literal is None or alloc.tied:
            # No large literal: any constant in the size carries the verdict.
            # Sized from an engine argument: tied by construction.
            continue
        findings.append(
            _judge(
                kind="buffer",
                name=f"{alloc.call}@{alloc.file}:{alloc.line}",
                file=alloc.file,
                line=alloc.line,
                value=alloc.literal,
                declared=alloc.inline,
                uses=[f"literal size at {alloc.scope}"],
                envelope=envelope,
            )
        )
    return findings


def _judge(
    *,
    kind: str,
    name: str,
    file: str,
    line: int,
    value: int | None,
    declared: str,
    uses: Iterable[str],
    envelope: ServingEnvelope,
) -> CapacityFinding:
    context = f" ({', '.join(uses)})" if uses else ""
    if not declared:
        return CapacityFinding(
            kind=kind,
            name=name,
            file=file,
            line=line,
            value=value,
            status=STATUS_FAIL,
            detail=(
                f"unbound capacity{context}: derive it from an engine argument at "
                f"construction, or declare `# capacity: <axis>` / `# capacity: fixed: <why>` "
                f"so it can be checked against the envelope"
            ),
        )
    binding = parse_binding(declared)
    if binding.error:
        return CapacityFinding(
            kind=kind,
            name=name,
            file=file,
            line=line,
            value=value,
            status=STATUS_FAIL,
            detail=binding.error,
            binding=binding.raw,
        )
    if binding.fixed:
        return CapacityFinding(
            kind=kind,
            name=name,
            file=file,
            line=line,
            value=value,
            status=STATUS_OK,
            detail=f"not a capacity: {binding.reason}{context}",
            binding=binding.raw,
        )
    required = binding.required(envelope)
    if required is None:
        return CapacityFinding(
            kind=kind,
            name=name,
            file=file,
            line=line,
            value=value,
            status=STATUS_FAIL,
            detail=(
                f"bound to {binding.raw}, but the envelope states no {binding.axis}, so the "
                f"bound cannot be proven"
            ),
            binding=binding.raw,
        )
    if value is None or value < required:
        return CapacityFinding(
            kind=kind,
            name=name,
            file=file,
            line=line,
            value=value,
            status=STATUS_FAIL,
            detail=(
                f"{value} < {binding.raw} = {required} at the envelope{context}: every engine "
                f"configured above {value} exceeds this"
            ),
            binding=binding.raw,
        )
    return CapacityFinding(
        kind=kind,
        name=name,
        file=file,
        line=line,
        value=value,
        status=STATUS_OK,
        detail=f"{value} >= {binding.raw} = {required}{context}",
        binding=binding.raw,
    )


__all__ = [
    "ALLOC_FUNCS",
    "BINDABLE_AXES",
    "BOUNDS_NAME",
    "CAPACITY_WORDS",
    "ENGINE_ARGS",
    "LITERAL_THRESHOLD",
    "Allocation",
    "Binding",
    "CapacityAudit",
    "CapacityFinding",
    "Constant",
    "audit_target",
    "parse_binding",
    "target_files",
]
