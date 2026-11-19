# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_router_logits entry: a sample of its cells against the true-FP32 reference and against
dsv3_router_gemm_op's bits on SM107."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_router_logits import (  # noqa: E501
    rubin_router_logits,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 -- path above

pytestmark = cpu_or_sm(107)
gpu = on_sm(107)

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


CELLS = sample_cells(rubin_router_logits.CELLS, size_key="rows")[:2]


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
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
