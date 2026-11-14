# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W6 FSEL_WIDE GPU oracle: the TP2 decode select at the served K7 c1..c4 shape (two KV heads per rank, the 1M
engine's 8193-block scores in the scorer's block-major layout, 1..4 requests of 8 query rows at 128K..200K tokens)
through the target adapter at every row count 1..33 (33 = the bound + 1, where the DSL select keeps the step), at the
trigger phase the target passes for that row count, eagerly and inside a CUDA graph: the table equals the select
contract's reference and the TP2 SelectLongDsl's, as a set and in order (bitwise)."""

import ast
from pathlib import Path

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_decode_fsel
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import m3_fsel_topk as fsel
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.m3_fsel_topk import (
    m3_fsel_topk as op,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import (  # noqa: F401
        minimax_m3_1_flash_nvfp4__sm_107__tp2,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 TP2 target runs on SM107 only", allow_module_level=True)


@pytest.fixture(autouse=True)
def _op_built():
    assert fsel.fsel_op() is not None, (
        "this build lacks mv2_minimax_m3::minimax_m3_sm107tp4_fsel_topk"
    )


_MODELING = (
    Path(tensorrt_llm.__file__).resolve().parent
    / "_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/modeling.py"
)
NUM_BLOCKS = 8193
HEADS = 2
ROWS_PER_REQUEST = 8


def _constants():
    """The target's FSEL bound and trigger phases as written (default: no kill switch set)."""
    tree = ast.parse(_MODELING.read_text())
    values = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            values[n.targets[0].id] = n.value
    lit = {
        k: ast.literal_eval(values[k])
        for k in (
            "FSEL_NARROW_MAX_ROWS",
            "FSEL_TRIGGER_PHASE",
            "FSEL_WIDE_TRIGGER_PHASE",
        )
    }
    lit["SELECT_LONG_TRIGGER_MAX_ROWS"] = ast.literal_eval(values["SELECT_LONG_TRIGGER_MAX_ROWS"])
    bound = eval(
        compile(ast.Expression(values["FSEL_MAX_ROWS"]), "bound", "eval"),
        {"FSEL_WIDE": True, **lit},
    )
    return bound, lit


def _dsl():
    # Imported unguarded: a broken DSL module must fail the test, not drop the DSL comparison.
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.attention.m3_select_long_dsl import (  # noqa: E501
        m3_select_long_dsl,
    )

    return m3_select_long_dsl


def _case(total_q: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # The scorer's layout: [heads, blocks, total_q] with block stride total_q.
    scores = torch.randn(
        HEADS, NUM_BLOCKS, total_q, generator=g, device="cuda", dtype=torch.float32
    )
    contexts = (131072, 150000, 176000, 200000)
    vals = []
    for t in range(total_q):
        req, pos = divmod(t, ROWS_PER_REQUEST)
        kv = contexts[req % len(contexts)] - ROWS_PER_REQUEST + pos + 1
        vals.append((kv + 127) // 128)
    return scores, torch.tensor(vals, dtype=torch.int32, device="cuda")


def _check(scores, nvb, table):
    ref = op.reference(scores, nvb, table, 0, 1)
    assert torch.equal(table, ref), "FSEL table differs from the select contract's reference"
    out = torch.empty_like(table)
    _dsl()(scores, nvb, out, 0, 1, False)
    assert torch.equal(table, out), "FSEL table differs from the TP2 SelectLongDsl"


def _phase(rows, lit):
    if rows <= lit["SELECT_LONG_TRIGGER_MAX_ROWS"]:
        return lit["FSEL_TRIGGER_PHASE"]
    return lit["FSEL_WIDE_TRIGGER_PHASE"]


@pytest.mark.parametrize("total_q", list(range(1, 34)))
def test_served_rows_and_the_bound_plus_one(total_q) -> None:
    bound, lit = _constants()
    assert bound == 32
    fsel.prepare(torch.device("cuda"))
    for seed in (5, 6):
        scores, nvb = _case(total_q, seed)
        table = m31_decode_fsel.select(
            scores,
            n_valid_blocks=nvb,
            num_tokens=total_q,
            num_kv_heads=HEADS,
            init_blocks=0,
            local_blocks=1,
            enabled=True,
            max_rows=bound,
            use_pdl=True,
            trigger_phase=_phase(total_q, lit),
        )
        if total_q > bound:
            assert table is None
            continue
        assert table is not None and tuple(table.shape) == (total_q, HEADS, 16)
        _check(scores, nvb, table)


@pytest.mark.parametrize("total_q", [16, 24, 32])
def test_graph_replay_at_wide_rows(total_q) -> None:
    bound, lit = _constants()
    fsel.prepare(torch.device("cuda"))
    scores, nvb = _case(total_q, 17)
    kwargs = dict(
        n_valid_blocks=nvb,
        num_tokens=total_q,
        num_kv_heads=HEADS,
        init_blocks=0,
        local_blocks=1,
        enabled=True,
        max_rows=bound,
        use_pdl=True,
        trigger_phase=_phase(total_q, lit),
    )
    eager = m31_decode_fsel.select(scores, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        m31_decode_fsel.select(scores, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = m31_decode_fsel.select(scores, **kwargs)
    for _ in range(3):
        captured.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)
    _check(scores, nvb, captured)
