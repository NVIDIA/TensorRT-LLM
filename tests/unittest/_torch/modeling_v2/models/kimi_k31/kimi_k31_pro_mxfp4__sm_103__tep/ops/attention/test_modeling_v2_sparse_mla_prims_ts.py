# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_sparse_mla_prims_ts entry (PrimTS D576 token-sparse MLA, 96 heads) and the
k31_mla_query_e4m3 entry that quantizes its query (absorbed query -> per-tensor E4M3 rows, compared byte for byte),
each on a sample of its cells (``sample_cells``).

The reference is an FP32 softmax attention over the selected E4M3 rows as the kernel reads them, so the only
differences left are the kernel's own arithmetic: the FP8 probabilities of its P.V MMA and the BF16 output. Band:
measured on GB300 over every cell (worst 12.4 BF16 ulp of a row's largest magnitude on one element, 2.4 ulp
relative RMS), then doubled: 25 and 5 ulp.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.attention.mla_query_e4m3 import (  # noqa: E501
    QUERY_DIM,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.attention.mla_query_e4m3 import (  # noqa: E501
    k31_mla_query_e4m3 as query_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.attention.sparse_mla_prims_ts import (  # noqa: E501
    full_lengths,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.attention.sparse_mla_prims_ts import (  # noqa: E501
    k31_sparse_mla_prims_ts as op,
)

__extra_import_path__ = ["../../../../..", "../../..", "../../../ops/attention"]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import on_sm, sample_cells  # noqa: E402
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = on_sm(103)

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


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]
QUERY_CELLS = sample_cells(query_op.CELLS, size_key="tokens")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    assert out.data_ptr() == args["out"].data_ptr()
    ref = _reference(args)
    torch.cuda.synchronize()
    assert not bool(out.isnan().any())
    assert_within_ulp(out, ref, ELEMENT_ULP, RMS_ULP)


def _query_build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    n, heads = spec["tokens"], spec["heads"]
    spread = 2000.0 if spec.get("data") == "wide" else 30.0
    fused = (torch.randn(n, heads, 576, generator=gen) * spread).bfloat16().cuda()
    if spec["fused"]:
        q_latent, q_pe = fused[..., :512], fused[..., 512:]
    else:
        q_latent, q_pe = fused[..., :512].contiguous(), fused[..., 512:].contiguous()
    scale = torch.tensor([0.875], dtype=torch.float32, device="cuda")
    out = torch.empty(n, heads, 576, dtype=torch.float8_e4m3fn, device="cuda")
    return dict(q_latent=q_latent, q_pe=q_pe, scale=scale, out=out)


@pytest.mark.parametrize("cell", QUERY_CELLS, ids=[c.why[:48] for c in QUERY_CELLS])
def test_query_e4m3_certified_cells(cell, certifying):  # noqa: F811
    certifying(query_op)
    args = _query_build(cell.spec, inputs.seed_of(cell))
    ref_out = torch.empty_like(args["out"])
    query_op(**args)
    query_op.reference(**dict(args, out=ref_out))
    torch.cuda.synchronize()
    assert torch.equal(args["out"].view(torch.uint8), ref_out.view(torch.uint8))
