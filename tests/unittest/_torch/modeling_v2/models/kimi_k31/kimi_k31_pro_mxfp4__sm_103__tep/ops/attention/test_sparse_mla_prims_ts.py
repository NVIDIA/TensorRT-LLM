# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_sparse_mla_prims_ts entry (PrimTS D576 token-sparse MLA, 96 heads).

The reference is an FP32 softmax attention over the selected E4M3 rows as the kernel reads them, so the only
differences left are the kernel's own arithmetic: the FP8 probabilities of its P.V MMA and the BF16 output. Band:
measured on GB300 over every cell (worst 12.4 BF16 ulp of a row's largest magnitude on one element, 2.4 ulp
relative RMS), then doubled: 25 and 5 ulp.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.attention.mla_query_e4m3 import (  # noqa: E501
    QUERY_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.attention.sparse_mla_prims_ts import (  # noqa: E501
    full_lengths,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.attention.sparse_mla_prims_ts import (  # noqa: E501
    k31_sparse_mla_prims_ts as op,
)

__extra_import_path__ = ["../../../../..", "../../../ops/attention"]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

HEADS = 96
TOP_K = 2048
#: The band in BF16 ulp (see the module docstring).
ELEMENT_ULP = 25
RMS_ULP = 5
_POOLS: dict = {}


def _pool(kv_format: str, pool_rows: int) -> torch.Tensor:
    """One pool per format and size, shared by the cells: E4M3 rows of N(0, 1) latents."""
    key = (kv_format, pool_rows)
    if key not in _POOLS:
        gen = torch.Generator(device="cuda").manual_seed(1234)
        x = torch.randn(pool_rows, QUERY_DIM, generator=gen, device="cuda")
        _POOLS[key] = x.to(torch.float8_e4m3fn)
    return _POOLS[key]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, kv_format, pool_rows = spec["rows"], spec["kv_format"], spec["pool_rows"]
    query = (torch.randn(rows, HEADS, QUERY_DIM, generator=gen, device="cuda") * 2.0).to(
        torch.float8_e4m3fn
    )
    keys = torch.rand(rows, pool_rows, generator=gen, device="cuda").topk(TOP_K, dim=1).indices
    indices = keys.sort(dim=1).values.int()
    if spec["holes"]:
        indices[:, TOP_K - spec["holes"] :] = -1
    softmax_scale = 1.0 / math.sqrt(192)
    output_scale = 1.0
    return dict(
        query=query,
        kv_pool=_pool(kv_format, pool_rows),
        indices=indices,
        lengths=full_lengths(query.device, rows, TOP_K),
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=torch.empty(rows, HEADS, 512, dtype=torch.bfloat16, device="cuda"),
        kv_format=kv_format,
        plan_rows=spec["plan_rows"] or rows,
    )


def _reference(args) -> torch.Tensor:
    ref = torch.empty_like(args["out"])
    with true_fp32_matmul():
        op.reference(**dict(args, out=ref))
    return ref


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    assert out.data_ptr() == args["out"].data_ptr()
    ref = _reference(args)
    torch.cuda.synchronize()
    assert not bool(out.isnan().any())
    assert_within_ulp(out, ref, ELEMENT_ULP, RMS_ULP)


def test_cuda_graph_replay_reads_new_inputs():
    """A graph captured after the eager (planning) call replays on whatever the captured buffers hold."""
    spec = dict(rows=56, kv_format="native", pool_rows=262_144, holes=0, plan_rows=None)
    args = _build(spec, seed=21)
    op(**args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        op(**args)
        with torch.cuda.graph(graph, stream=stream):
            op(**args)
    torch.cuda.current_stream().wait_stream(stream)
    fresh = _build(spec, seed=22)
    args["query"].copy_(fresh["query"])
    args["indices"].copy_(fresh["indices"])
    args["out"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert_within_ulp(args["out"], _reference(args), ELEMENT_ULP, RMS_ULP)


def test_one_plan_serves_fewer_rows_bitwise():
    """Rows run on a larger plan give the same bits as the rows of a full-size call on that plan."""
    spec = dict(rows=64, kv_format="native", pool_rows=262_144, holes=0, plan_rows=None)
    full = _build(spec, seed=31)
    op(**full)
    part = dict(full, query=full["query"][:40], indices=full["indices"][:40])
    part["lengths"] = full["lengths"][:40]
    part["out"] = torch.empty(40, HEADS, 512, dtype=torch.bfloat16, device="cuda")
    op(**part)
    torch.cuda.synchronize()
    assert torch.equal(part["out"], full["out"][:40])


def test_guard_refuses_an_index_past_the_pool():
    args = _build(op.CELLS[0].spec, seed=3)
    indices = args["indices"].clone()
    indices[0, 0] = args["kv_pool"].shape[0]
    with pytest.raises(AssertionError, match="past the pool"):
        with validating(op):
            op(**dict(args, indices=indices))


def test_guard_refuses_more_rows_than_planned():
    args = _build(op.CELLS[1].spec, seed=4)
    with pytest.raises(AssertionError, match="more rows than planned"):
        with validating(op):
            op(**dict(args, plan_rows=args["query"].shape[0] - 1))
