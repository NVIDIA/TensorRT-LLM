# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MSA page table is written on the host side only, before the graph replay.

The index scorer's early issue (``ops/m3_index_score.py``, ``early_issue``) reads ``block_table[b, split]`` BEFORE
its PDL grid-dependency wait and loads that page's K tile right after it. That is exact only while no kernel of the
step writes ``MsaMetadata.msa_block_table``: today its one writer is the host-to-device copy in
``_build_msa_fields`` (called from ``prepare()``, before the replay), besides the allocation. This test scans
``tensorrt_llm/_torch`` and fails on any other write site, so a new in-graph writer has to revisit the early issue.

(``msa_seq_lens_cuda`` IS patched in-graph by ``on_update_kv_lens``; the scorer reads it before the wait only as a hint
and bounds every load and store by the value it reads after the wait.)

CPU only: it parses the sources.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

pytestmark = pytest.mark.cpu_only

_ROOT = Path(tensorrt_llm.__file__).resolve().parent / "_torch"
_NAME = "msa_block_table"
# In-place tensor methods that write their receiver.
_MUTATORS = {
    "copy_",
    "fill_",
    "zero_",
    "index_put_",
    "index_copy_",
    "scatter_",
    "masked_fill_",
    "add_",
    "sub_",
}
# Helpers whose first argument is the destination they write.
_COPY_HELPERS = {"copy_h2d_if_changed", "copy_h2d_pool_shared"}
# (file relative to tensorrt_llm/_torch, enclosing function) -> what it may do
_ALLOWED = {
    (
        "attention/backends/sparse/minimax_m3/msa_backend.py",
        "_create_msa_buffers",
    ),  # the allocation
    (
        "attention/backends/sparse/minimax_m3/msa_backend.py",
        "_build_msa_fields",
    ),  # the H2D copy in prepare()
}


def _mentions(node) -> bool:
    """Whether the expression's receiver chain (attributes / subscripts / slices) is ``*.msa_block_table``."""
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Call)):
        if isinstance(node, ast.Attribute) and node.attr == _NAME:
            return True
        node = node.func if isinstance(node, ast.Call) else node.value
    return isinstance(node, ast.Name) and node.id == _NAME


def _write_sites(tree: ast.AST):
    sites = []

    def visit(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_mentions(t) for t in targets) and not isinstance(node, ast.AnnAssign):
                sites.append((func, node.lineno))
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in _MUTATORS and _mentions(f.value):
                sites.append((func, node.lineno))
            name = (
                f.id
                if isinstance(f, ast.Name)
                else (f.attr if isinstance(f, ast.Attribute) else "")
            )
            if name in _COPY_HELPERS and node.args and _mentions(node.args[0]):
                sites.append((func, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(tree, None)
    return sites


def test_only_prepare_writes_the_msa_page_table() -> None:
    found = set()
    for path in sorted(_ROOT.rglob("*.py")):
        src = path.read_text()
        if _NAME not in src:
            continue
        rel = path.relative_to(_ROOT).as_posix()
        for func, line in _write_sites(ast.parse(src)):
            assert (rel, func) in _ALLOWED, (
                f"{rel}:{line} ({func}) writes msa_block_table; the index scorer's early issue reads it before its "
                "PDL grid wait (ops/m3_index_score.py), so an in-graph writer must be ordered before it or the "
                "early page read moved after the wait"
            )
            found.add((rel, func))
    assert found == _ALLOWED, f"expected write sites {_ALLOWED}, found {found}"
