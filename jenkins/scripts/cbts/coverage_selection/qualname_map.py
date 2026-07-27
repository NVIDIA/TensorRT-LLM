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
from dataclasses import dataclass


@dataclass
class _Scope:
    qualname: str
    sig_start: int
    body_start: int
    body_end: int
    body_attr: str
    sig_attr: str


def _substatements(node: ast.stmt):
    """Yield direct sub-statements of a compound statement (no new scope)."""
    for field in ("body", "orelse", "finalbody"):
        yield from getattr(node, field, None) or []
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


def _attribute(line: int, scopes: list[_Scope]) -> str:
    best: _Scope | None = None
    for s in scopes:
        if s.sig_start <= line <= s.body_end and (best is None or s.sig_start > best.sig_start):
            best = s
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
