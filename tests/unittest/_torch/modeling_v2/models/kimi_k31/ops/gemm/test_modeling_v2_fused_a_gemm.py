# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_a_gemm entry (MoE latent projections)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (  # noqa: E501
    k31_fused_a_gemm as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "fused_a_gemm requires a CUDA device"

CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(spec["tokens"], spec["k"], generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(spec["n"], spec["k"], generator=g, device="cuda") / spec["k"] ** 0.5).to(
        torch.bfloat16
    )
    return x, w.t()


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    ref = op.reference(*args)
    # One BF16 rounding of an FP32 sum taken in another order. ULP is 2^-8 of the row's largest
    # magnitude, while BF16's spacing at that magnitude is up to 2^-7 of it: one rounding step is
    # up to two of these units per element, a fraction of one in RMS.
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)
