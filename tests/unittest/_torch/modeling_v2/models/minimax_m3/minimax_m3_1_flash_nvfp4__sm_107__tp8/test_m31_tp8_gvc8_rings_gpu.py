# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_GEMV_RING_RETUNE (default on): the his-layout drafter QKV and o GEMVs at ring depth 2 return the default
rings' (6 / 12) bits at the drafter row counts; TRTLLM_M31_DISABLE_DRAFT_GEMV_RING_RETUNE restores the default rings."""

import importlib

import pytest
import torch

import tensorrt_llm  # noqa: F401 -- register native ops before evaluating skips

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires SM107 and the TP8 GEMV provider",
)
SERVED = {(1280, 6144): (8, 6), (6144, 1024): (2, 12), (3072, 6144): (4, 12)}
RING_RETUNE_CFGS = {(1280, 6144): (8, 2), (6144, 1024): (2, 2), (3072, 6144): (4, 12)}


def test_default_is_the_retune() -> None:
    mxd = importlib.import_module(f"{PKG}.drafter_mxfp8")
    sw = importlib.import_module(f"{PKG}.m31_switches")
    assert mxd.HIS_CFGS == (
        SERVED if sw.m31_disabled("DRAFT_GEMV_RING_RETUNE") else RING_RETUNE_CFGS
    )


@pytest.mark.parametrize("rows", [7, 8])
@pytest.mark.parametrize("proj", ["qkv_proj", "o_proj"])
def test_ring_retune_rings_are_bitwise_the_served(proj: str, rows: int) -> None:
    mxd = importlib.import_module(f"{PKG}.drafter_mxfp8")
    gemv = importlib.import_module(f"{PKG}.kernels.mx_gemv.op")
    n, k = mxd.HIS_SHAPES[proj]
    g = torch.Generator(device="cuda").manual_seed(83 + rows)
    weight = (
        (torch.randn(n, k, device="cuda", generator=g) * 16)
        .to(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    rec = mxd.mxfp8_from_bf16(weight)
    assert rec is not None, "an exact MXFP8 dequantization"
    w8, sf = rec[0], mxd.swizzle_sf(rec[1])
    x = torch.randn(rows, k, device="cuda", generator=g).to(torch.bfloat16)
    ref = gemv.m3_mx_gemv_bf16(x, w8, sf, True, SERVED[(n, k)])
    assert ref is not None
    ref = ref.clone()
    for rep in range(3):
        got = gemv.m3_mx_gemv_bf16(x, w8, sf, True, RING_RETUNE_CFGS[(n, k)])
        torch.cuda.synchronize()
        assert got is not None
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (proj, rows, rep)
