# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-token valid-block count is written only before the step's first layer.

The long-row select (``kernels/m3_select_long_dsl.py``, ``SELECT_N_VALID_PREWAIT``) reads its row's entry of
``MsaMetadata.msa_n_valid_blocks`` BEFORE its PDL grid-dependency wait. That is exact only while every write to the
buffer happens before the step's first layer:

- the allocation (``_alloc_msa_proxy_scratch``);
- the host-to-device copy in ``prepare()`` (``_build_step_plans``);
- ``on_update_kv_lens``, which the runner calls before the forward. It writes through the plain
  ``<<<>>>`` launch of ``msa_kv_update_kernel`` or through torch copies, so any PDL grid of the step starts only
  after it completes.

This test scans ``tensorrt_llm/_torch`` and fails on any other write site (including writes through a local alias of
the buffer), so a new writer has to revisit the pre-wait read. A buffer handed to an opaque kernel launch is not
tracked; such a launch inside a layer must be reviewed by hand.

CPU only: it parses the sources.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

pytestmark = pytest.mark.cpu_only

_ROOT = Path(tensorrt_llm.__file__).resolve().parent / "_torch"
_NAME = "msa_n_valid_blocks"
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
# Calls whose first argument is the destination they write: ``step_inputs.copy_(dst, src)`` and the H2D helpers.
_COPY_HELPERS = {"copy_", "copy_h2d_if_changed", "copy_h2d_pool_shared"}
_BACKEND = "attention/backends/sparse/minimax_m3/msa_backend.py"
# (file relative to tensorrt_llm/_torch, enclosing function) -> writes allowed there
_ALLOWED = {
    (_BACKEND, "_alloc_msa_proxy_scratch"),  # the allocation
    (_BACKEND, "_build_step_plans"),  # the H2D copy in prepare()
    (_BACKEND, "on_update_kv_lens"),  # the device-side correction before the forward
}


def _base(node):
    """The innermost receiver of an attribute / subscript / call chain."""
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Call)):
        if isinstance(node, ast.Attribute) and node.attr == _NAME:
            return node
        node = node.func if isinstance(node, ast.Call) else node.value
    return node


def _mentions(node, aliases) -> bool:
    """Whether the expression's receiver chain is ``*.msa_n_valid_blocks`` or a local alias of it."""
    b = _base(node)
    if isinstance(b, ast.Attribute):
        return b.attr == _NAME
    return isinstance(b, ast.Name) and (b.id == _NAME or b.id in aliases)


def _refers(node, aliases) -> bool:
    """Whether any sub-expression (e.g. a branch of a conditional) names the buffer or an alias."""
    return any(
        (isinstance(n, ast.Attribute) and n.attr == _NAME)
        or (isinstance(n, ast.Name) and (n.id == _NAME or n.id in aliases))
        for n in ast.walk(node)
    )


def _write_sites(tree: ast.AST):
    sites = []

    def visit(node, func, aliases):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func, aliases = node.name, set()
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for t in targets:
                if isinstance(t, ast.Name) and value is not None and _refers(value, aliases):
                    aliases.add(t.id)  # a local view of the buffer
                elif (
                    not isinstance(t, ast.Name)
                    and _mentions(t, aliases)
                    and not isinstance(node, ast.AnnAssign)
                ):
                    sites.append((func, node.lineno))
                elif (
                    isinstance(t, ast.Attribute)
                    and t.attr == _NAME
                    and isinstance(node, ast.AnnAssign)
                ):
                    pass  # a dataclass field declaration
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in _MUTATORS and _mentions(f.value, aliases):
                sites.append((func, node.lineno))
            name = (
                f.id
                if isinstance(f, ast.Name)
                else (f.attr if isinstance(f, ast.Attribute) else "")
            )
            if name in _COPY_HELPERS and node.args and _mentions(node.args[0], aliases):
                sites.append((func, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, func, aliases)

    visit(tree, None, set())
    return sites


def test_only_the_pre_forward_paths_write_the_valid_block_count() -> None:
    found = set()
    for path in sorted(_ROOT.rglob("*.py")):
        src = path.read_text()
        if _NAME not in src:
            continue
        rel = path.relative_to(_ROOT).as_posix()
        for func, line in _write_sites(ast.parse(src)):
            assert (rel, func) in _ALLOWED, (
                f"{rel}:{line} ({func}) writes msa_n_valid_blocks; the long-row select reads it before its PDL grid "
                "wait (kernels/m3_select_long_dsl.py, SELECT_N_VALID_PREWAIT), so a writer inside the "
                "step must be ordered before the select's grid or the read moved after the wait"
            )
            found.add((rel, func))
    assert found == _ALLOWED, f"expected write sites {_ALLOWED}, found {found}"
