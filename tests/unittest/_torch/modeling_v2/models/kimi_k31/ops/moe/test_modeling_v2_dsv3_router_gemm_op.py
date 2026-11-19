# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 dsv3_router_gemm_op entry: router logits [M, 896] in fp32."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
    dsv3_router_gemm_op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

pytestmark = on_sm(103, 107)

HIDDEN, EXPERTS = 7168, 896
# Gate in fp32 ulps of each row's largest logit. Both kernels and the reference sum 7168 exact bf16
# products in fp32 in different orders. Measured on GB300 against an fp64 product: the M <= 16
# tensor-core kernel at most 5 ulp per element and 4 ulp RMS; the cuBLAS path (M > 16) at most 108 ulp
# per element and 67 ulp RMS, flat in M from 256 rows up; the true-fp32 reference itself at most 36 / 7.4.
# Dropping one 128-wide K slice moves a logit by ~10^5 ulp.
ELEMENT_ULP, RMS_ULP = 256.0, 128.0


def _build(spec: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["rows"]
    width = spec.get("row_width", HIDDEN)
    hidden = torch.randn(rows, width, generator=gen, device="cuda").to(torch.bfloat16)[:, :HIDDEN]
    gate = (torch.randn(EXPERTS, HIDDEN, generator=gen, device="cuda") * HIDDEN**-0.5).to(
        torch.bfloat16
    )
    return hidden, gate


CELLS = sample_cells(dsv3_router_gemm_op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(dsv3_router_gemm_op)
    hidden, gate = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    if "row_width" in cell.spec:
        assert not hidden.is_contiguous() and hidden.stride(0) == cell.spec["row_width"]
    else:
        hidden = hidden.contiguous()
    # The target's call: the [896, 7168] weight's transpose, fp32 logits.
    out = dsv3_router_gemm_op(hidden, gate.t())
    again = dsv3_router_gemm_op(hidden, gate.t())
    assert out.shape == (hidden.shape[0], EXPERTS) and out.dtype == torch.float32
    assert torch.equal(out, again), "two identical calls disagree"
    ref = dsv3_router_gemm_op.reference(hidden, gate.t())
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
