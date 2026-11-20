# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helpers the fused-V-transpose kernel takes from the 2x kernel module fit both kernels.

``fp4_mla_cutedsl_mufu16_fused_v_transpose`` imports helpers that ``fp4_mla_cutedsl_mufu16``
defines. Such a helper runs with the globals of the 2x module, while the rest of the fused-V
kernel reads its own module's. Both modules define most of their constants themselves, and a
few differ. A shared helper is right for both kernels only while every module-level name it
reaches, through the functions it calls and the constants those are computed from, is defined
the same way in both modules. Parsing only: no GPU, no import of the kernel modules.
"""

import ast
import importlib.util
import symtable
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.cpu_only

BACKEND = (
    Path(importlib.util.find_spec("tensorrt_llm").origin).parent
    / "_torch/attention/backends/fp4_mla"
)
TWO_X = "fp4_mla_cutedsl_mufu16"
FUSED_V = "fp4_mla_cutedsl_mufu16_fused_v_transpose"
# Called once at import; rebinds constants of its module through ``globals().update(...)``.
REFRESH = "_refresh_sf_vec_config"


class _Definition(NamedTuple):
    """How a module binds a name: the statements that bind it, and the module-level names they read."""

    statements: tuple
    reads: frozenset


def _loads(*nodes: ast.AST) -> set[str]:
    return {
        name.id
        for node in nodes
        for name in ast.walk(node)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
    }


def _targets(statement: ast.stmt) -> list[str]:
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    return [
        name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)
    ]


def _scope_reads(scope: symtable.SymbolTable) -> set[str]:
    """Globals a function or class body reads, nested scopes included."""
    names = {s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()}
    for child in scope.get_children():
        names |= _scope_reads(child)
    return names


def _only_raises(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.If) and all(
        isinstance(inner, ast.Raise) for inner in statement.body + statement.orelse
    )


def _definitions(source: str) -> dict[str, _Definition]:
    """The module-level names of ``source``, with what binds each of them."""
    tree = ast.parse(source)
    scopes = {
        scope.get_name(): scope for scope in symtable.symtable(source, "m", "exec").get_children()
    }
    found: dict[str, list[tuple[object, set[str]]]] = {}

    def bind(name: str, statement: object, reads: set[str]) -> None:
        found.setdefault(name, []).append((statement, reads))

    refresh, refresh_argument_reads = None, set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # the body reads its globals when it runs; decorators, defaults and bases when it is defined
            header = node.decorator_list + (
                [node.args] if isinstance(node, ast.FunctionDef) else node.bases
            )
            bind(node.name, ast.dump(node), _scope_reads(scopes[node.name]) | _loads(*header))
            if node.name == REFRESH:
                refresh = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bind(
                    alias.asname or alias.name, ("from", node.level, node.module, alias.name), set()
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bind((alias.asname or alias.name).split(".")[0], ("import", alias.name), set())
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _targets(node):
                bind(name, ast.dump(node), _loads(node))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert ast.unparse(node.value.func) == REFRESH, ast.unparse(node)
            refresh_argument_reads = _loads(*node.value.args)
        else:
            # anything else that could bind a name would escape this check
            assert _only_raises(node) or isinstance(node, ast.Expr), ast.unparse(node)[:200]

    if refresh is not None:
        # straight-line local assignments, checks that raise, and one globals().update(NAME=...)
        parameters = {argument.arg for argument in refresh.args.args}
        local_statements: dict[str, list[ast.stmt]] = {}
        rebinding = []
        for statement in refresh.body:
            if isinstance(statement, ast.Assign):
                for name in _targets(statement):
                    local_statements.setdefault(name, []).append(statement)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                assert ast.unparse(statement.value.func) == "globals().update", ast.unparse(
                    statement
                )
                rebinding += statement.value.keywords
            else:
                assert _only_raises(statement), ast.unparse(statement)[:200]
        for keyword in rebinding:
            statements, reads, pending, seen = (
                [ast.dump(keyword.value)],
                set(),
                _loads(keyword.value),
                set(),
            )
            while pending:
                name = pending.pop()
                if name in seen:
                    continue
                seen.add(name)
                if name in local_statements:
                    statements += [ast.dump(statement) for statement in local_statements[name]]
                    pending |= _loads(*local_statements[name])
                elif name in parameters:
                    reads |= refresh_argument_reads
                else:
                    reads.add(name)
            bind(keyword.arg, tuple(sorted(statements)), reads)

    return {
        name: _Definition(
            tuple(statement for statement, _ in bindings),
            frozenset(read for _, reads in bindings for read in reads),
        )
        for name, bindings in found.items()
    }


def _shared_names(fused_v_source: str) -> list[str]:
    """Names the fused-V module imports from the 2x module."""
    return [
        alias.name
        for node in ast.parse(fused_v_source).body
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == TWO_X
        for alias in node.names
    ]


def _path_to_a_difference(
    name: str,
    two_x: dict[str, _Definition],
    fused_v: dict[str, _Definition],
    shared: set[str],
    seen: set[str],
) -> list[str] | None:
    """Names read from ``name`` down to one the two modules define differently; None without one."""
    if name not in two_x or name in seen:
        return None
    seen.add(name)
    if (
        name in fused_v
        and name not in shared
        and fused_v[name].statements != two_x[name].statements
    ):
        return [name]
    for read in sorted(two_x[name].reads):
        path = _path_to_a_difference(read, two_x, fused_v, shared, seen)
        if path is not None:
            return [name, *path]
    return None


def _unfit_helpers(two_x_source: str, fused_v_source: str) -> dict[str, list[str]]:
    two_x, fused_v = _definitions(two_x_source), _definitions(fused_v_source)
    shared = _shared_names(fused_v_source)
    assert shared and set(shared) <= set(two_x), shared
    paths = {
        name: _path_to_a_difference(name, two_x, fused_v, set(shared), set()) for name in shared
    }
    return {name: path for name, path in paths.items() if path is not None}


def test_helpers_shared_with_the_fused_v_kernel_read_what_both_modules_define_alike() -> None:
    unfit = _unfit_helpers(
        (BACKEND / f"{TWO_X}.py").read_text(), (BACKEND / f"{FUSED_V}.py").read_text()
    )
    assert not unfit, (
        "the fused-V kernel takes these helpers from the 2x kernel module, where they reach a "
        f"name the two modules define differently (helper: names read down to it): {unfit}"
    )


def test_the_check_follows_a_constant_through_a_function_and_the_rebinding_at_import() -> None:
    two_x = """
STAGES = 3
WIDTH = 16
def _refresh_sf_vec_config(width):
    ring = STAGES * width
    globals().update(RING=ring)
_refresh_sf_vec_config(WIDTH)
def _slot(index):
    return index % RING
def helper(index):
    return _slot(index)
def other(index):
    return index * WIDTH
"""
    fused_v = f"""
from .{TWO_X} import helper, other
STAGES = 2
WIDTH = 16
def _refresh_sf_vec_config(width):
    ring = STAGES * width
    globals().update(RING=ring)
_refresh_sf_vec_config(WIDTH)
"""
    # RING is rebound alike in both modules, from a constant they define differently
    assert _unfit_helpers(two_x, fused_v) == {"helper": ["helper", "_slot", "RING", "STAGES"]}
    assert _unfit_helpers(two_x, fused_v.replace("STAGES = 2", "STAGES = 3")) == {}
