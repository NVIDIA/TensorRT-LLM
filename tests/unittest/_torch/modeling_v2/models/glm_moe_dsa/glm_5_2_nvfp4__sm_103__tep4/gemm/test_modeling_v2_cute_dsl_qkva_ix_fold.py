# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_qkva_ix_fold entry (MTP3 / MTP5 wide decode)."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_qkva_ix_fold runs an sm_103 CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_indexer_pre_proj_split as split_op_module,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection as projection_op_module,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_qkva_ix_fold import (  # noqa: E402, E501
    IX_HEADS,
    IX_KEY,
    MTP_ROWS,
    QKVA_N,
    K,
    at_rows,
    build_all,
    stack_weight,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_qkva_ix_fold import (  # noqa: E402, E501
    cute_dsl_qkva_ix_fold as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

#: One unit of fp32's mantissa.
FP32_ULP = 2.0**-23

_CACHE: dict = {}


def _cells():
    """The fold cells, the served qkva MTP cells and the indexer split handles, built once."""
    if not _CACHE:
        dev = torch.cuda.current_device()
        g = torch.Generator(device="cuda").manual_seed(1)
        qkva_w = (torch.randn(QKVA_N, K, generator=g, device="cuda") * K**-0.5).to(torch.bfloat16)
        _CACHE["fold"] = build_all(dev)
        _CACHE["qkva"] = projection_op_module.build_all("qkva", qkva_w, MTP_ROWS)
        _CACHE["split"] = {m: split_op_module.compile(dev, m) for m in MTP_ROWS}
    return _CACHE


def _operands(rows: int, seed: int):
    """`x`, the qkva weight, the indexer's bf16 key / head-weight rows, their stack. Each
    output is O(1): weights scaled by K**-0.5."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    qkva = (torch.randn(QKVA_N, K, generator=g, device="cuda") * K**-0.5).to(torch.bfloat16)
    ix = (torch.randn(IX_KEY + IX_HEADS, K, generator=g, device="cuda") * K**-0.5).to(
        torch.bfloat16
    )
    return x, qkva, ix, stack_weight(qkva, ix)


def _served(rows: int, x, qkva, ix):
    """What the served wide decode path writes for these operands: the qkva MTP cell at
    `rows` (on the next cell up, as `_wide_cell` runs it) and the split-output indexer
    projection's 128 / 192-row handle on the rows padded with zeros."""
    cells = _cells()
    cell = min(m for m in MTP_ROWS if m >= rows)
    runner = cells["qkva"][cell]
    if rows != cell:
        runner = projection_op_module.at_rows(runner, rows)
    out = torch.empty(rows, QKVA_N, dtype=torch.bfloat16, device="cuda")
    projection_op_module.cute_dsl_projection(x, qkva, out, runner)
    xp = torch.zeros(cell, K, dtype=torch.bfloat16, device="cuda")
    xp[:rows] = x
    key = torch.empty(cell, IX_KEY, dtype=torch.bfloat16, device="cuda")
    heads = torch.empty(cell, IX_HEADS, dtype=torch.float32, device="cuda")
    split_op_module.cute_dsl_indexer_pre_proj_split(xp, ix, key, heads, cells["split"][cell])
    return out, key[:rows], heads[:rows]


def _runner(rows: int, cell: int):
    runner = _cells()["fold"][cell]
    return runner if rows == cell else at_rows(runner, rows)


def _check_indexer(key, heads, ref_key, ref_heads) -> None:
    """The fold's indexer columns: fp32 sums of K = 6144 exact bf16 products (8 chains,
    summed in chain order), against fp64 rounded once. Keys rounded to bf16 on both sides:
    4 ulp element-wise, 2 ulp RMS (every bf16 projection in this target). Head weights fp32:
    the split-output projection's band, 512 ulp (2**-14) element-wise and 128 ulp RMS of the
    row scale -- a reordering of K fp32 additions; a wrong column is O(1)."""
    assert_within_ulp(key, ref_key, element_ulp=4.0, rms_ulp=2.0)
    assert_within_ulp(heads, ref_heads, 512.0, 128.0, ulp=FP32_ULP)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """All three outputs are the served path's bit for bit: the qkva output the qkva MTP
    cell's, the keys and head weights the split-output indexer projection's (the fold's
    indexer tiles accumulate in its chains and reduction order); and within the fp64 band."""
    rows = cell.spec["rows"]
    x, qkva, ix, w_ext = _operands(rows, zlib.crc32(cell.why.encode()) % 2**31)
    out = torch.full((rows, QKVA_N), float("nan"), dtype=torch.bfloat16, device="cuda")
    runner = _runner(rows, cell.spec["cell"])
    runner.kb.fill_(float("nan"))
    runner.wq.fill_(float("nan"))
    with validating(op):
        got_out, key, heads = op(x, w_ext, out, runner)
    torch.cuda.synchronize()
    assert got_out is out
    ref_out, ref_key, ref_heads = op.reference(x, w_ext, out, runner)
    assert_within_ulp(out, ref_out, element_ulp=4.0, rms_ulp=2.0)
    _check_indexer(key, heads, ref_key, ref_heads)
    s_out, s_key, s_heads = _served(rows, x, qkva, ix)
    assert torch.equal(out, s_out), f"{rows} rows: qkva differs from the served qkva cell"
    assert torch.equal(key, s_key), f"{rows} rows: keys differ from the served split"
    assert torch.equal(heads.view(torch.int32), s_heads.view(torch.int32)), (
        f"{rows} rows: head weights differ from the served split"
    )


@pytest.mark.parametrize("rows", MTP_ROWS)
def test_replay_is_eager_bitwise_on_new_operands(rows) -> None:
    """Inside the decode CUDA graph: a replay writes the eager launch's bytes (all three
    outputs), and after new operands are copied into the captured buffers, the eager
    launch's product on them; twice-eager is bitwise too."""
    runner = _runner(rows, rows)
    x, _, _, w_ext = _operands(rows, seed=rows)
    out = torch.empty(rows, QKVA_N, dtype=torch.bfloat16, device="cuda")
    op(x, w_ext, out, runner)
    eager = [t.clone() for t in (out, runner.kb[:rows], runner.wq[:rows])]
    op(x, w_ext, out, runner)
    again = (out, runner.kb[:rows], runner.wq[:rows])
    assert all(torch.equal(a, b) for a, b in zip(eager, again)), "eager twice differs"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, w_ext, out, runner)
    for t in (out, runner.kb, runner.wq):
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, (out, runner.kb[:rows], runner.wq[:rows])))
    x2, _, _, w2 = _operands(rows, seed=rows + 1)
    x.copy_(x2)
    w_ext.copy_(w2)
    out2 = torch.empty_like(out)
    op(x, w_ext, out2, runner)
    eager2 = [t.clone() for t in (out2, runner.kb[:rows], runner.wq[:rows])]
    for t in (out, runner.kb, runner.wq):
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager2, (out, runner.kb[:rows], runner.wq[:rows])))
    assert not torch.equal(out, eager[0]), "the new operands did not change the product"


def test_partial_rows_are_the_cells_rows_bitwise() -> None:
    """A row count between 64 and 192 runs on the next cell up: each row's three outputs are
    bitwise what the cell writes for that row with the input padded by other rows."""
    cells = _cells()["fold"]
    for rows in (65, 68, 96, 97, 127, 129, 144, 150, 191):
        cell = min(m for m in MTP_ROWS if m >= rows)
        x, _, _, w_ext = _operands(rows, seed=rows)
        out = torch.empty(rows, QKVA_N, dtype=torch.bfloat16, device="cuda")
        _, key, heads = op(x, w_ext, out, at_rows(cells[cell], rows))
        got = (out.clone(), key.clone(), heads.clone())
        xp, _, _, _ = _operands(cell, seed=rows + 1000)
        xp[:rows] = x
        outp = torch.empty(cell, QKVA_N, dtype=torch.bfloat16, device="cuda")
        _, keyp, headsp = op(xp, w_ext, outp, cells[cell])
        torch.cuda.synchronize()
        assert torch.equal(got[0], outp[:rows]), f"{rows} rows: qkva not the {cell} cell's"
        assert torch.equal(got[1], keyp[:rows]) and torch.equal(got[2], headsp[:rows]), (
            f"{rows} rows: indexer columns not the {cell} cell's"
        )


def test_at_rows_refuses_rows_past_the_cell() -> None:
    cell = _cells()["fold"][MTP_ROWS[0]]
    assert at_rows(cell, MTP_ROWS[0] + 1) is None
    assert at_rows(cell, 0) is None


def test_stack_weight_refuses_wrong_shapes() -> None:
    qkva = torch.zeros(QKVA_N, K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="stack_weight takes"):
        stack_weight(qkva, torch.zeros(IX_KEY, K, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="bf16"):
        stack_weight(qkva, torch.zeros(IX_KEY + IX_HEADS, K, dtype=torch.float32, device="cuda"))
