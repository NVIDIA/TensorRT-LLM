# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_router_logits entry: every cell against the true-FP32 reference and against
dsv3_router_gemm_op's bits on SM107; the reference on the CPU."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    router_logits_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_router_logits import (  # noqa: E501
    rubin_router_logits,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 -- path above

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

# FP32 ulps of each row's largest logit. The kernel has dsv3_router_gemm_op's M <= 16 bits, which
# that entry measured within 5 ulp per element (4 RMS) of an FP64 product on GB300, the true-FP32
# reference within 36 / 7.4; its gate of 256 / 128 is kept, which a dropped 128-wide K slice
# (~10^5 ulp) still fails by far.
ELEMENT_ULP, RMS_ULP = 256.0, 128.0


def _operands(rows, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
    weight = (torch.randn(NUM_EXPERTS, HIDDEN, generator=g) * HIDDEN**-0.5).to(torch.bfloat16)
    return hidden.to(device), weight.to(device)


@gpu
@pytest.mark.parametrize(
    "cell", rubin_router_logits.CELLS, ids=[c.why[:40] for c in rubin_router_logits.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers dsv3_router_gemm_op

    certifying(rubin_router_logits)
    hidden, weight = _operands(cell.spec["rows"], zlib.crc32(cell.why.encode()), "cuda")
    out = rubin_router_logits(hidden, weight)
    assert out.shape == (hidden.shape[0], NUM_EXPERTS) and out.dtype == torch.float32
    assert torch.equal(out, rubin_router_logits(hidden, weight)), "two calls disagree"
    assert_within_ulp(
        out, rubin_router_logits.reference(hidden, weight), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP
    )
    stock = torch.ops.trtllm.dsv3_router_gemm_op(
        hidden, weight.t(), bias=None, out_dtype=torch.float32
    )
    assert torch.equal(out, stock), "not dsv3_router_gemm_op's bits"


@gpu
def test_cuda_graph_replay_matches_eager():
    hidden, weight = _operands(8, 3, "cuda")
    eager = rubin_router_logits(hidden, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rubin_router_logits(hidden, weight)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


@gpu
def test_gate_refuses_a_dropped_k_slice():
    hidden, weight = _operands(8, 1, "cuda")
    out = rubin_router_logits(hidden, weight)
    short = hidden.clone()
    short[:, :128] = 0
    with pytest.raises(AssertionError):
        assert_within_ulp(
            out, router_logits_reference(short, weight), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP
        )


@pytest.mark.cpu_only
def test_reference_is_the_fp32_rounding_of_the_exact_product():
    hidden, weight = _operands(3, 7, "cpu")
    ref = router_logits_reference(hidden, weight)
    exact = hidden.double() @ weight.double().t()
    assert ref.dtype == torch.float32 and ref.shape == (3, NUM_EXPERTS)
    # 7168 FP32 additions in some order: within a few hundred ulp of each row's largest value.
    scale = exact.abs().amax(dim=1, keepdim=True)
    assert ((ref.double() - exact).abs() / scale).max() < 512 * 2**-23
