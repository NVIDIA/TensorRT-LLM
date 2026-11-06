# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_pre_proj_split entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "cute_dsl_indexer_pre_proj_split runs an sm_103 CuTe DSL kernel", allow_module_level=True
    )

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_indexer_pre_proj_split import (  # noqa: E402, E501
    MTP_ROWS,
    PREFILL_MAX_ROWS,
    PrefillSplitHandle,
    build_prefill,
    compile_ragged,
    project_unsplit,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_indexer_pre_proj_split import (  # noqa: E402, E501
    compile as compile_handle,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_indexer_pre_proj_split import (  # noqa: E402, E501
    cute_dsl_indexer_pre_proj_split as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, KEY, HEADS = 6144, 128, 32
#: One unit of fp32's mantissa.
FP32_ULP = 2.0**-23


def _compare(key, head_weights, ref_key, ref_head_weights) -> None:
    """Products of bf16 operands are exact in fp32; kernel and reference
    differ only in how K = 6144 of them are summed (fp32 over an 8-way
    split-K cluster, or the prefill GEMM's single fp32 accumulator, against
    fp64). The prefill path stores fp32 and rounds the keys to bf16 in a copy:
    the same one rounding.

    key, rounded once to bf16 on both sides: 4 ulp element-wise and 2 ulp
    RMS, as for every bf16 projection in this target.

    head_weights, fp32: a random-walk estimate of the reordering is a few
    fp32 ulps of the row scale. The band is 512 ulp (2**-14) element-wise and
    128 ulp (2**-16) RMS, wide enough for an accumulator that truncates
    instead of rounding; a wrong column or row is O(1).
    """
    assert_within_ulp(key, ref_key, element_ulp=4.0, rms_ulp=2.0)
    assert_within_ulp(head_weights, ref_head_weights, 512.0, 128.0, ulp=FP32_ULP)


def _build(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    weight = torch.randn(KEY + HEADS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5
    # NaN, so an element the kernel never writes fails the comparison.
    key = torch.full((rows, KEY), float("nan"), dtype=torch.bfloat16, device="cuda")
    head_weights = torch.full((rows, HEADS), float("nan"), dtype=torch.float32, device="cuda")
    return x, weight.to(torch.bfloat16), key, head_weights


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    rows = cell.spec["rows"]
    x, weight, key, head_weights = _build(rows, zlib.crc32(cell.why.encode()) % 2**31)
    if cell.spec.get("prefill"):
        # Built on another weight: the target builds once and runs every layer through it.
        compiled = build_prefill(torch.zeros_like(weight))
    else:
        compiled = compile_handle(torch.cuda.current_device(), rows)
    with validating(op):
        got = op(x, weight, key, head_weights, compiled)
    assert got[0] is key and got[1] is head_weights, "the op writes both buffers and returns them"
    _compare(key, head_weights, *op.reference(x, weight, key, head_weights, compiled))


@pytest.mark.parametrize("rows", [65, 129, 1099, 4097, 8192])
def test_unsplit_columns_are_the_split_outputs_bitwise(rows) -> None:
    """`project_unsplit` (the served context path, `TRTLLM_GLM52_CTX_IX_NOSPLIT=1`): its key
    columns rounded to bf16 are the split's keys and its head-weight columns the split's head
    weights, bit for bit. With the K write's and q side's own pitch-160 tests this pins the
    unsplit path to the split one end to end."""
    x, weight, key, head_weights = _build(rows, zlib.crc32(f"unsplit {rows}".encode()) % 2**31)
    handle = build_prefill(weight)
    op(x, weight, key, head_weights, handle)
    kb, wq = project_unsplit(x, weight, handle)
    torch.cuda.synchronize()
    assert kb.dtype == torch.float32 and kb.stride() == (KEY + HEADS, 1)
    assert wq.stride() == (KEY + HEADS, 1) and wq.data_ptr() == kb.data_ptr() + KEY * 4
    assert bool(torch.isfinite(key.float()).all()), "a key was not written"
    assert torch.equal(kb.to(torch.bfloat16).view(torch.int16), key.view(torch.int16))
    assert torch.equal(wq.contiguous().view(torch.int32), head_weights.view(torch.int32))


def test_a_handle_refuses_another_row_count() -> None:
    """Each handle is compiled for one row count; the target picks it by the
    step's rows, so a mismatch must raise rather than compute a partial tile."""
    compiled = compile_handle(torch.cuda.current_device(), 2)
    with pytest.raises(ValueError, match="compiled for M=2"):
        op(*_build(4, seed=4), compiled)


@pytest.mark.parametrize("rows", MTP_ROWS)
def test_mtp_rows_replay_bitwise_in_a_cuda_graph(rows) -> None:
    """The MTP3 / MTP5 handles run inside the decode CUDA graph: a replay writes the eager
    launch's bytes, and a replay after new operands are copied into the captured buffers
    writes the eager launch's product on them."""
    compiled = compile_handle(torch.cuda.current_device(), rows)
    x, weight, key, head_weights = _build(rows, seed=rows)
    op(x, weight, key, head_weights, compiled)
    eager = (key.clone(), head_weights.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, key, head_weights, compiled)
    key.fill_(float("nan"))
    head_weights.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(key, eager[0]) and torch.equal(head_weights, eager[1])
    x2, weight2, key2, head_weights2 = _build(rows, seed=rows + 1)
    x.copy_(x2)
    weight.copy_(weight2)
    op(x, weight, key2, head_weights2, compiled)
    key.fill_(float("nan"))
    head_weights.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(key, key2) and torch.equal(head_weights, head_weights2)
    assert not torch.equal(key, eager[0]), "the new operands did not change the product"
    _compare(key, head_weights, *op.reference(x, weight, key, head_weights, compiled))


#: MTP3 / MTP5 decode row counts without a handle: each bucket's edges, the 24-request
#: batches (96, 144) and partial batches at widths 4 / 6.
WIDE_ROWS = (65, 66, 68, 84, 96, 100, 126, 127, 129, 144, 150, 190, 191)


def test_runtime_m_handle_is_the_compiled_handles_bitwise() -> None:
    """`compile_ragged`: the kernel compiled once on a symbolic row count, for the MTP3 /
    MTP5 decode row counts without a handle of their own. At 128 and 192 rows it writes the
    compiled handles' bytes; at every other wide row count each row is bitwise what the
    next compiled handle up writes for it with the input padded by other (random) rows;
    within the fp64 reference band; a captured launch replays bitwise on new operands; and
    it refuses more rows than it was built for."""
    dev = torch.cuda.current_device()
    rag = compile_ragged(dev)
    cells = {m: compile_handle(dev, m) for m in MTP_ROWS}
    for rows in (*MTP_ROWS, *WIDE_ROWS):
        x, weight, key, head_weights = _build(rows, seed=rows)
        with validating(op):
            op(x, weight, key, head_weights, rag)
        cell = min(m for m in cells if m >= rows)
        xp, _, kp, wp = _build(cell, seed=rows + 1000)
        xp[:rows] = x
        op(xp, weight, kp, wp, cells[cell])
        torch.cuda.synchronize()
        assert torch.equal(key, kp[:rows]) and torch.equal(head_weights, wp[:rows]), (
            f"{rows} rows: not the {cell}-row handle's"
        )
        _compare(key, head_weights, *op.reference(x, weight, key, head_weights, rag))
    x, weight, key, head_weights = _build(96, seed=96)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, key, head_weights, rag)
    x2, _, key2, head_weights2 = _build(96, seed=97)
    x.copy_(x2)
    op(x, weight, key2, head_weights2, rag)
    key.fill_(float("nan"))
    head_weights.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(key, key2) and torch.equal(head_weights, head_weights2)
    with pytest.raises(ValueError, match="M=1..192"):
        op(*_build(193, seed=5), rag)


#: MTP5 at 1-3 requests (6 / 12 / 18 rows; MTP3 at 3 is 12): captured decode row counts without
#: a handle, which the target runs on the runtime-M handle (`_cell_up`) instead of `x.float()`
#: and the fp32 cuBLAS projection. 16, a compiled handle, is the control.
DECODE_ROWS = (6, 12, 18, 16)


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_runtime_m_handle_at_small_decode_rows(rows) -> None:
    """At the small-batch MTP decode counts the runtime-M handle writes, row for row, what
    the handle compiled for that count writes and what the next decode handle up (16 / 32)
    writes for it with the input padded by other (random) rows; within the fp64 reference
    band; and a captured launch replays bitwise on new operands."""
    dev = torch.cuda.current_device()
    rag = compile_ragged(dev)
    up = min(m for m in (16, 32, 48, 64) if m >= rows)
    own, cell = compile_handle(dev, rows), compile_handle(dev, up)
    x, weight, key, head_weights = _build(rows, seed=rows)
    with validating(op):
        op(x, weight, key, head_weights, rag)
    _, _, ko, wo = _build(rows, seed=rows + 500)
    op(x, weight, ko, wo, own)
    xp, _, kp, wp = _build(up, seed=rows + 1000)
    xp[:rows] = x
    op(xp, weight, kp, wp, cell)
    torch.cuda.synchronize()
    assert torch.equal(key, ko) and torch.equal(head_weights, wo), f"{rows}: not its own handle's"
    assert torch.equal(key, kp[:rows]) and torch.equal(head_weights, wp[:rows]), (
        f"{rows} rows: not the {up}-row handle's"
    )
    _compare(key, head_weights, *op.reference(x, weight, key, head_weights, rag))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, key, head_weights, rag)
    x2, _, key2, head_weights2 = _build(rows, seed=rows + 7)
    x.copy_(x2)
    op(x, weight, key2, head_weights2, rag)
    key.fill_(float("nan"))
    head_weights.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(key, key2) and torch.equal(head_weights, head_weights2)


def test_guard_refuses_bf16_head_weights_on_the_prefill_handle() -> None:
    """The prefill path copies the fp32 product into the buffers, and a copy casts:
    bf16 head weights would round silently where the scorers expect fp32."""
    x, weight, key, head_weights = _build(80, seed=5)
    handle = PrefillSplitHandle(PREFILL_MAX_ROWS, None)
    with pytest.raises(AssertionError, match="head_weights must be contiguous fp32"):
        with validating(op):
            op(x, weight, key, head_weights.to(torch.bfloat16), handle)
