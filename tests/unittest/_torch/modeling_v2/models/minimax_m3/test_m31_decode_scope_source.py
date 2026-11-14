# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the MiniMax-M3.1-Flash sm_107 tp2 / tp8 decode-dispatch predicates, read from source (no CUDA import).

A pure generation step runs the decode target's kernels only for flat token ids without input embeddings and at most
``DECODE_STEP_MAX`` rows; every other step runs the generic layers.
"""

import ast
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

FAMILY = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
)
TARGETS = ("minimax_m3_1_flash_nvfp4__sm_107__tp2", "minimax_m3_1_flash_nvfp4__sm_107__tp8")


def dispatch(target, rows, step_max, ndim=1, embeddings=False):
    """Evaluates the decode target's ``kernels`` predicate with ``is_decode_step`` at a step bound of ``step_max``."""
    path = FAMILY / target / "modeling.py"
    tree = ast.parse(path.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_decode_step"
    )
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DecodeTarget"
    )
    forward = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    predicate = next(
        node.value
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "kernels" for target in node.targets)
    )
    env = {
        "DECODE_STEP_MAX": step_max,
        "input_ids": types.SimpleNamespace(ndim=ndim, numel=lambda: rows),
        "inputs_embeds": object() if embeddings else None,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), env)
    return eval(compile(ast.Expression(predicate), str(path), "eval"), env)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("step_max", [8, 32, 48])
def test_the_row_bound(target, step_max):
    for rows in (1, 7, step_max):
        assert dispatch(target, rows, step_max), rows
    for rows in (step_max + 1, 2 * step_max, 4096):
        assert not dispatch(target, rows, step_max), rows


@pytest.mark.parametrize("target", TARGETS)
def test_embeddings_and_nonflat_ids_use_the_generic_path(target):
    assert dispatch(target, 8, 32)
    assert not dispatch(target, 8, 32, ndim=2)
    assert not dispatch(target, 8, 32, embeddings=True)


_BOUND_NAMES = (
    "TP2_ROWS",
    "R1VR8",
    "DECODE_MAX_TOKENS",
    "VR_BANDS",
    "TP2WIDE_DECODE",
    "DECODE_STEP_MAX",
)


def step_max(target, disabled):
    """DECODE_STEP_MAX as the target computes it at import, with ``disabled(item_id)`` as its switch registry."""
    path = FAMILY / target / "modeling.py"
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in _BOUND_NAMES for t in node.targets)
    ]
    env = {"m31_disabled": disabled}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env["DECODE_STEP_MAX"]


@pytest.mark.parametrize(("target", "default"), zip(TARGETS, (448, 64)))
def test_the_step_bound_and_its_kill_switch(target, default):
    """Every switch on: the widest decode step. DECODE_LAYER_WIDE_ROWS off: the 8-row bound, whatever the rest."""
    assert step_max(target, lambda item: False) == default
    assert step_max(target, lambda item: item == "DECODE_LAYER_WIDE_ROWS") == 8
    assert step_max(target, lambda item: True) == 8
