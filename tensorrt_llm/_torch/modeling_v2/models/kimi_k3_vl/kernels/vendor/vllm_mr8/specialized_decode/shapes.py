# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode shapes, row-keyed policies and the specialized decode's coverage plan.

A deployment verifies ``q`` tokens per request (1 + draft length; 1 without
speculation) and captures one FULL decode graph per token-row count M (``Shape``).
Each kernel family decides per shape whether it runs a tuned kernel, a generic one,
or a stock seam (``Choice``); the coverage plan records those decisions for every
captured shape, and a shape that some family cannot run stays on the stock forward.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .cutedsl.common.policy import batch_table

# Rows covered by the tuned tables and the 64-row mailboxes (B8 x q8 verify steps).
TUNED_ROWS = 64


@dataclass(frozen=True)
class Shape:
    """One captured decode graph: ``rows`` token rows of ``rows // q`` requests."""

    rows: int
    q: int = 1

    @property
    def requests(self) -> int:
        return self.rows // self.q

    def __str__(self) -> str:
        return f"M{self.rows}(q{self.q})"


class Path(Enum):
    TUNED = "tuned"  # a measured table row
    GENERIC = "generic"  # a specialized kernel, correct for the shape, untuned
    STOCK = "stock"  # the family's stock seam inside the specialized graph


@dataclass(frozen=True)
class Choice:
    """A family's decision for one shape; ``tactic`` is family-specific (hashable),
    ``note`` names its provenance in the coverage log."""

    path: Path
    tactic: Any = None
    note: str = ""

    def __str__(self) -> str:
        return f"{self.path.value}({self.note})" if self.note else self.path.value


@dataclass(frozen=True)
class Context:
    """Deployment facts every family's ``choose`` may read (built once per process)."""

    arch: str  # "rubin" (compute capability 10.7) or "gb300"
    tp: int
    rank: int
    dcp: int
    sms: int
    q: int  # the deployment's verify length
    max_rows: int  # the largest captured decode graph: capacity above TUNED_ROWS
    max_context: int
    bf16_state: bool


_MISSING = object()


def rows_policy(entry, shape: Shape, default: Any = _MISSING, name: str = "policy"):
    """The value of a rows-keyed policy ``entry`` at ``shape``.

    ``entry`` is a ``batch_table`` spec ("1-8:1,64:3.16.4.2"), the same for every q, or
    a dict {q | "*": spec} read at the exact q first, then at "*". A miss returns
    ``default``; without one it raises: the table must cover the shape but does not."""
    specs = entry if isinstance(entry, dict) else {"*": entry}
    for key in (shape.q, "*"):
        if key in specs:
            value = batch_table(specs[key]).get(shape.rows, _MISSING)
            if value is not _MISSING:
                return value
    if default is _MISSING:
        raise KeyError(f"{name}: no entry for {shape}")
    return default


def captured_rows(capture_sizes: Iterable[int], q: int, limit: int) -> tuple[int, ...]:
    """Token rows of the FULL decode graphs a deployment captures: the runner rounds
    each capture size up to whole q-token requests and drops those above ``limit``
    (max_num_seqs * q and the largest capture size)."""
    rows = {-(-size // q) * q for size in capture_sizes}
    return tuple(sorted(m for m in rows if m <= limit))


class Coverage:
    """Per captured shape, each family's ``Choice`` (``None`` for the whole shape: the
    step runs the stock forward, with the reason)."""

    def __init__(self, q: int):
        self.q = q
        self.plan: dict[int, dict[str, Choice] | None] = {}
        self.reasons: dict[int, str] = {}

    def specialize(self, rows: int, choices: dict[str, Choice]) -> None:
        self.plan[rows] = dict(choices)

    def stock(self, rows: int, reason: str) -> None:
        self.plan[rows], self.reasons[rows] = None, reason

    def lines(self) -> list[str]:
        out = []
        for rows in sorted(self.plan):
            choices = self.plan[rows]
            shape = Shape(rows, self.q)
            if choices is None:
                out.append(f"{shape}: stock ({self.reasons[rows]})")
            else:
                body = " ".join(f"{name}={choice}" for name, choice in choices.items())
                out.append(f"{shape}: {body}")
        return out
