# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_fp32_linear entry of the MiniMax-M3 tp8 target (the router beyond the decode routers' rows)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_fp32_linear import (  # noqa: E501
    m3_fp32_linear as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_fp32_linear requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """torch's default fp32 matmul precision on this hardware may be TF32 (~2^-11 relative per product): the band is
    stated in fp32 ulps of each row's largest logit and covers it (K = 6144 TF32 products: ~2^13 ulps worst element);
    measured on GB300 with the engine's defaults: worst element ~1700 ulps, RMS ~300."""
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(spec["m"], spec["k"], generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    w = torch.randn(spec["e"], spec["k"], generator=g, device="cuda") * 0.02
    with validating(op):
        out = op(x, w)
    assert out.dtype == torch.float32 and out.shape == (spec["m"], spec["e"])
    assert_within_ulp(out, op.reference(x, w), element_ulp=8192.0, rms_ulp=2048.0)
