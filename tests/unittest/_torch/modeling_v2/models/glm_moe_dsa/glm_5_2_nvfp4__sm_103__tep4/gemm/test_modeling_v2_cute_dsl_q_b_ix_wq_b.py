# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_q_b_ix_wq_b entry (MTP3 / MTP5 wide decode)."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_q_b_ix_wq_b runs an sm_103 CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection as projection_op_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_q_b_ix_wq_b import (  # noqa: E402, E501
    MTP_ROWS,
    K,
    N,
    at_rows,
    build_all,
    stack_weight,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_q_b_ix_wq_b import (  # noqa: E402, E501
    cute_dsl_q_b_ix_wq_b as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

_CACHE: dict = {}


def _cells():
    """The fused cells and the served q_b / ix_qb MTP cells, built once."""
    if not _CACHE:
        dev = torch.cuda.current_device()
        w = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
        _CACHE["fold"] = build_all(dev)
        _CACHE["q_b"] = projection_op_module.build_all("q_b", w, MTP_ROWS)
        _CACHE["ix_qb"] = projection_op_module.build_all("ix_qb", w, MTP_ROWS)
    return _CACHE


def _operands(rows: int, seed: int):
    """`q_a` O(1), the two weights scaled by K**-0.5 (each output O(1)), their stack."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_a = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    w_q = (torch.randn(N, K, generator=g, device="cuda") * K**-0.5).to(torch.bfloat16)
    w_ix = (torch.randn(N, K, generator=g, device="cuda") * K**-0.5).to(torch.bfloat16)
    return q_a, w_q, w_ix, stack_weight(w_q, w_ix)


def _runner(rows: int, cell: int):
    runner = _cells()["fold"][cell]
    return runner if rows == cell else at_rows(runner, rows)


def _served(rows: int, q_a, w_q, w_ix):
    """The served wide decode path's q and indexer q: the q_b / ix_qb MTP cells at `rows` (on
    the next cell up, re-pinned to the exact rows, as `_wide_cell` runs them)."""
    cell = min(m for m in MTP_ROWS if m >= rows)
    out = []
    for name, w in (("q_b", w_q), ("ix_qb", w_ix)):
        runner = _cells()[name][cell]
        if rows != cell:
            runner = projection_op_module.at_rows(runner, rows)
        o = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
        projection_op_module.cute_dsl_projection(q_a, w, o, runner)
        out.append(o)
    return out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Both outputs are the served cells' bit for bit, and within the fp64 band (4 bf16 ulp
    element-wise, 2 ulp RMS of the row scale: one rounding of nearly equal fp32 sums)."""
    rows = cell.spec["rows"]
    q_a, w_q, w_ix, w2 = _operands(rows, zlib.crc32(cell.why.encode()) % 2**31)
    out2 = torch.full((2, rows, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        q, ix_q = op(q_a, w2, out2, _runner(rows, cell.spec["cell"]))
    torch.cuda.synchronize()
    assert q.data_ptr() == out2[0].data_ptr() and ix_q.data_ptr() == out2[1].data_ptr()
    ref_q, ref_ix = op.reference(q_a, w2, out2, None)
    assert float(ref_q.float().abs().amax()) > 0.5 and float(ref_ix.float().std()) > 0.1
    assert_within_ulp(q, ref_q, element_ulp=4.0, rms_ulp=2.0)
    assert_within_ulp(ix_q, ref_ix, element_ulp=4.0, rms_ulp=2.0)
    s_q, s_ix = _served(rows, q_a, w_q, w_ix)
    assert torch.equal(q, s_q), f"{rows} rows: q differs from the served q_b cell"
    assert torch.equal(ix_q, s_ix), f"{rows} rows: indexer q differs from the served wq_b cell"


@pytest.mark.parametrize("rows", MTP_ROWS)
def test_replay_is_eager_bitwise_on_new_operands(rows) -> None:
    """In a CUDA graph: replay writes the eager launch's bytes; after new operands are copied
    into the captured buffers, the eager launch's product on them; eager twice is bitwise."""
    runner = _runner(rows, rows)
    q_a, _, _, w2 = _operands(rows, seed=rows)
    out2 = torch.empty(2, rows, N, dtype=torch.bfloat16, device="cuda")
    op(q_a, w2, out2, runner)
    eager = out2.clone()
    op(q_a, w2, out2, runner)
    assert torch.equal(out2, eager), "eager twice differs"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(q_a, w2, out2, runner)
    out2.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2, eager)
    q2, _, _, w2b = _operands(rows, seed=rows + 1)
    q_a.copy_(q2)
    w2.copy_(w2b)
    fresh = torch.empty_like(out2)
    op(q_a, w2, fresh, runner)
    out2.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out2, fresh) and not torch.equal(out2, eager)


def test_partial_rows_are_the_cells_rows_bitwise() -> None:
    """A row count between 64 and 192 runs on the next cell up: each row's outputs are
    bitwise the cell's for that row with the input padded by other rows."""
    cells = _cells()["fold"]
    for rows in (65, 68, 96, 97, 127, 129, 144, 150, 191):
        cell = min(m for m in MTP_ROWS if m >= rows)
        q_a, _, _, w2 = _operands(rows, seed=rows)
        out2 = torch.empty(2, rows, N, dtype=torch.bfloat16, device="cuda")
        op(q_a, w2, out2, at_rows(cells[cell], rows))
        qp, _, _, _ = _operands(cell, seed=rows + 1000)
        qp[:rows] = q_a
        outp = torch.empty(2, cell, N, dtype=torch.bfloat16, device="cuda")
        op(qp, w2, outp, cells[cell])
        torch.cuda.synchronize()
        assert torch.equal(out2, outp[:, :rows]), f"{rows} rows: not the {cell} cell's rows"


def test_guards() -> None:
    cell = _cells()["fold"][MTP_ROWS[0]]
    assert at_rows(cell, MTP_ROWS[0] + 1) is None and at_rows(cell, 0) is None
    w = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="stack_weight takes"):
        stack_weight(w, torch.zeros(N, K - 8, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="q_b \\| wq_b fold"):
        op(torch.zeros(4, K, dtype=torch.bfloat16, device="cuda"), stack_weight(w, w),
           torch.empty(2, 5, N, dtype=torch.bfloat16, device="cuda"), cell)  # fmt: skip
