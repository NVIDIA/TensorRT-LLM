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
"""Separate literal Python docstring edits from executable diff lines."""

from __future__ import annotations

import ast
import re

from cbts.rules._helpers import iter_diff_changes, reconstruct_diff_pre_image, strip_noop_diff_lines

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_SCOPE_TYPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring(node: ast.AST) -> ast.Expr | None:
    if not isinstance(node, _SCOPE_TYPES) or not node.body:
        return None
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def _scope_docstrings(tree: ast.Module) -> dict[tuple[str, ...], ast.Expr | None] | None:
    scopes: dict[tuple[str, ...], ast.Expr | None] = {}

    def visit(node: ast.AST, owner: tuple[str, ...]) -> bool:
        if isinstance(node, _SCOPE_TYPES):
            if not isinstance(node, ast.Module):
                owner += (node.name,)
            if owner in scopes:
                return False
            scopes[owner] = _docstring(node)
        return all(visit(child, owner) for child in ast.iter_child_nodes(node))

    return scopes if visit(tree, ()) else None


def _docstring_lines(node: ast.Expr | None, source: str) -> set[int]:
    if node is None:
        return set()
    lines = source.splitlines()
    # AST columns count UTF-8 bytes. A statement sharing a line with the
    # docstring must remain in the executable diff.
    prefix = lines[node.lineno - 1].encode("utf-8")[: node.col_offset].strip()
    suffix = lines[node.end_lineno - 1].encode("utf-8")[node.end_col_offset :].strip()
    if prefix or (suffix and not suffix.startswith(b"#")):
        return set()
    return set(range(node.lineno, node.end_lineno + 1))


def _without_docstrings(tree: ast.Module) -> str:
    for node in ast.walk(tree):
        if _docstring(node) is not None:
            del node.body[0]
    return ast.dump(tree, include_attributes=False)


def split_docstring_diff(post_source: str, diff: str) -> tuple[str, bool]:
    """Return the executable diff and whether an existing scope's docs changed.

    Both source images must parse and the raw patch must match the post-image.
    Missing, ambiguous, or newly created scopes keep their normal code impact.
    The returned diff is for line selection, not patch application; raw hunk
    counts remain unchanged, as in ``strip_noop_diff_lines``.
    """
    pre_source = reconstruct_diff_pre_image(post_source, diff)
    if pre_source is None:
        return diff, False
    try:
        before = ast.parse(pre_source, type_comments=True)
        after = ast.parse(post_source, type_comments=True)
    except (SyntaxError, ValueError):
        return diff, False
    if not before.body:
        return diff, False
    old_scopes = _scope_docstrings(before)
    new_scopes = _scope_docstrings(after)
    if old_scopes is None or new_scopes is None:
        return diff, False
    old_docs: set[int] = set()
    new_docs: set[int] = set()
    for owner in old_scopes.keys() & new_scopes.keys():
        old_docs.update(_docstring_lines(old_scopes[owner], pre_source))
        new_docs.update(_docstring_lines(new_scopes[owner], post_source))

    out: list[str] = []
    pending_deleted: list[str] = []
    old_line = new_line = 0
    in_hunk = False
    changed = False
    for raw in diff.splitlines(keepends=True):
        match = _HUNK_HEADER_RE.match(raw)
        if match is not None:
            out.extend(pending_deleted)
            pending_deleted.clear()
            old_line, new_line = int(match.group(1)), int(match.group(2))
            in_hunk = True
            out.append(raw)
            continue
        if raw.startswith("\\ No newline"):
            continue
        if not raw or (not in_hunk and raw.startswith(("+++", "---"))):
            out.append(raw)
            continue
        sign = raw[0]
        if sign == "-":
            if old_line in old_docs:
                changed = True
            else:
                pending_deleted.append(raw)
            old_line += 1
        elif sign == "+":
            if new_line in new_docs:
                changed = True
                # Deleted code anchors after inserted documentation, at the
                # next executable addition or surviving context line.
                out.append(" " + raw[1:])
            else:
                out.extend(pending_deleted)
                pending_deleted.clear()
                out.append(raw)
            new_line += 1
        else:
            out.extend(pending_deleted)
            pending_deleted.clear()
            out.append(raw)
            if sign == " ":
                old_line += 1
                new_line += 1
    out.extend(pending_deleted)
    executable = "".join(out)
    if changed and not any(iter_diff_changes(strip_noop_diff_lines(executable))):
        if _without_docstrings(before) != _without_docstrings(after):
            return diff, False
    return executable, changed
