# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AST table helpers for specialized-target source checks."""

import ast


def defined(node: ast.stmt) -> tuple[str, ...]:
    """The names a top-level statement defines (``A, B = ...`` defines both)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, ast.Assign):
        return tuple(
            n.id
            for t in node.targets
            for n in (t.elts if isinstance(t, ast.Tuple) else (t,))
            if isinstance(n, ast.Name)
        )
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return (node.target.id,)
    return ()


def block_rows(tree: ast.Module) -> dict[tuple, dict[str, str]]:
    """The default target's ``_BLOCKS`` table: key -> {field: the name it binds, as source}; {} for
    a module without one (a specialized target)."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and "_BLOCKS" in defined(node):
            table = node.value
            if isinstance(table, ast.Call) and table.args:  # MappingProxyType({...})
                table = table.args[0]
            assert isinstance(table, ast.Dict), "_BLOCKS is not a dict literal"
            rows = {}
            for key, value in zip(table.keys, table.values):
                assert isinstance(value, ast.Call), ast.unparse(value)[:80]
                rows[ast.literal_eval(key)] = {
                    kw.arg: ast.unparse(kw.value) for kw in value.keywords
                }
            return rows
    return {}
