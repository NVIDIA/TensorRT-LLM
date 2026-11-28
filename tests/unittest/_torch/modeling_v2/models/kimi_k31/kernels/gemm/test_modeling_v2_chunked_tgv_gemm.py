# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 chunked TGV GEMM kernel (K summed in given chunks: the bits of a split-K GEMM).

A chunk map a caller holds as plan data is checked bit for bit against the GEMM it replaces (cuBLAS through the op
the target runs there) on the bit battery (``_bit_battery``), which includes exact-cancellation inputs whose outputs
are the summation order's own rounding: one map per output dtype, the router's FP32 three-chunk map and the latent
projection's BF16 two-chunk map.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.chunked_tgv_gemm import (
    kind_tiles,
    run_chunked_tgv,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import k31_fused_a_gemm
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
    dsv3_router_gemm_op,
)

__extra_import_path__ = [".", "../.."]
from _bit_battery import bit_battery  # noqa: E402 — needs the path declared above
from _k31_utils import on_sm

assert torch.cuda.is_available(), "the chunked TGV GEMM requires a CUDA device"
pytestmark = on_sm(103)

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def chunked_tgv(x, kinds, cta_n=16, stages=12, pdl=True):
    """For each kind ``(weight [N, K] BF16, bounds, out_dtype)``: the kernel's ``[M, N]`` output, a column view of an
    output padded to whole 128-row weight tiles."""
    outs = [
        torch.empty(x.shape[0], kind_tiles(w.shape[0]) * 64, dtype=dtype, device=x.device)
        for w, _, dtype in kinds
    ]
    run_chunked_tgv(
        x,
        [(w, out, tuple(bounds)) for (w, bounds, _), out in zip(kinds, outs)],
        cta_n,
        stages,
        pdl,
    )
    return [out[:, : w.shape[0]] for (w, _, _), out in zip(kinds, outs)]


# Chunk maps checked against the GEMM they replace: (name, N, K, rows, bounds, out dtype, cta_n, stages, reference).
# The replicated Kimi K3.1 MoE head (router: dsv3_router_gemm_op, cuBLAS split-K with FP32 output; latent down:
# k31_fused_a_gemm, cuBLAS): cuBLAS's split on GB300, at the last row of each map's range.
def _router(x, w):
    return dsv3_router_gemm_op(x, w.t())


def _latent(x, w):
    return k31_fused_a_gemm(x, w.t())


CERTIFIED = (
    ("router", 896, 7168, 64, (19, 38), "float32", 16, 12, _router),
    ("latent", 3584, 7168, 48, (28,), "bfloat16", 16, 12, _latent),
)
BATTERY = 20


@pytest.mark.parametrize(
    "name,n,k,m,bounds,dtype,cta_n,stages,ref_fn",
    CERTIFIED,
    ids=[f"{c[0]}-m{c[3]}-{len(c[4]) + 1}chunks" for c in CERTIFIED],
)
def test_certified_maps_are_bitwise(name, n, k, m, bounds, dtype, cta_n, stages, ref_fn) -> None:
    """The map's output equals the replaced GEMM's bit for bit on the whole battery, for the full weight and for
    TEP16 rank slices of it (the sharded MoE head's views)."""
    dt = DTYPES[dtype]
    bits = torch.int32 if dt == torch.float32 else torch.int16
    for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 131 + n, count=BATTERY)):
        ref = ref_fn(x, w)
        views = [(0, n)] + [(r * n // 16, (r + 1) * n // 16) for r in (0, 7, 15)]
        for lo, hi in views:
            (out,) = chunked_tgv(x, [(w[lo:hi], bounds, dt)], cta_n, stages)
            diff = int((out.view(bits) != ref[:, lo:hi].view(bits)).sum())
            assert diff == 0, (
                f"{name} m={m} rows {lo}-{hi} battery input {i}: {diff} outputs differ"
            )
