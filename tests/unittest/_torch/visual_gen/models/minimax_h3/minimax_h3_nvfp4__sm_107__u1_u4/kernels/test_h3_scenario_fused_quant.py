# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kernels/fused_quant.py``: producers that emit the static-NVFP4 Linears' inputs directly.

Reference for each producer: the family's unfused kernel (``fused_adaln.py`` or
``trtllm::silu_and_mul``) writing BF16, then ``trtllm::fp4_quantize(x, input_scale, 16, False)``,
which is what the block runs without the fusion. Codes and swizzled block scales are compared byte
for byte, and the gated residual the second producer also returns is compared bit for bit.
AdaLN byte identity uses the family kernel's four-warp reduction split.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  -- loads libth_common.so (registers the trtllm ops)
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_adaln
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.config import (
    ScenarioConfig,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
    fused_quant,
)

pytestmark = skip_pre_blackwell

HIDDEN = 5120
INTER = 4 * HIDDEN
ROWS = (1017, 9455)
EPS = 1e-5
N_MOD = 6


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    """A calibrated-style NVFP4 input scale, 0-dim fp32 like ``Linear.input_scale``."""
    amax = x.float().abs().max().clamp(min=1e-3)
    return (448.0 * 6.0 / amax).to(torch.float32)


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.dtype == torch.uint8 and expected.dtype == torch.uint8, what
    assert actual.numel() == expected.numel(), (
        f"{what}: {actual.numel()} vs {expected.numel()} bytes"
    )
    assert torch.equal(actual.reshape(-1), expected.reshape(-1)), what


def _adaln_inputs(rows: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)

    x = randn(1, rows, HIDDEN)
    x[0, 0, :16] *= 64.0  # an outlier block in the first row
    x[0, rows // 2] = 0.0  # an all-zero row: the norm runs on eps alone
    a = randn(1, rows, HIDDEN)
    w = 1.0 + 0.1 * randn(HIDDEN)
    mod = 0.5 * randn(N_MOD, 6 * HIDDEN)
    idx = torch.randint(0, N_MOD, (rows,), generator=gen, device="cuda")
    bf16 = torch.bfloat16
    return x.to(bf16), a.to(bf16), w.to(bf16), mod.to(bf16), idx


@pytest.mark.parametrize("rows", ROWS)
def test_norm_mod_fp4_equals_family_norm_then_fp4_quantize(rows):
    x, _, w, mod, idx = _adaln_inputs(rows, seed=rows)
    ref = fused_adaln.h3_norm_mod(x, w, mod, idx, EPS, 1, 0).reshape(rows, HIDDEN)
    gs = _global_scale(ref)
    ref_codes, ref_scales = torch.ops.trtllm.fp4_quantize(ref, gs, 16, False)
    _, codes, scales = fused_quant._launch_adaln(x, None, w, mod, idx, gs, EPS, 0, 1, 0)
    _assert_same_bytes(codes, ref_codes, "packed e2m1 codes")
    _assert_same_bytes(scales, ref_scales, "swizzled e4m3 block scales")


@pytest.mark.parametrize("rows", ROWS)
def test_gate_res_norm_mod_fp4_equals_family_then_fp4_quantize(rows):
    x, a, w, mod, idx = _adaln_inputs(rows, seed=rows + 1)
    ref_x1, ref = fused_adaln.h3_gate_res_norm_mod(x, a, w, mod, idx, EPS, 2, 4, 3)
    ref = ref.reshape(rows, HIDDEN)
    gs = _global_scale(ref)
    ref_codes, ref_scales = torch.ops.trtllm.fp4_quantize(ref, gs, 16, False)
    x1, codes, scales = fused_quant._launch_adaln(x, a, w, mod, idx, gs, EPS, 2, 4, 3)
    assert x1.dtype == torch.bfloat16
    assert torch.equal(x1.reshape(-1).view(torch.int16), ref_x1.reshape(-1).view(torch.int16)), (
        "gated residual (BF16 bits)"
    )
    _assert_same_bytes(codes, ref_codes, "packed e2m1 codes")
    _assert_same_bytes(scales, ref_scales, "swizzled e4m3 block scales")


def test_adaln_fp4_warps_match_family_kernel():
    # Byte identity with the family AdaLN kernel needs the same reduction split.
    assert ScenarioConfig().fused_quant_warps == 4


# The tiled SwiGLU kernel at the scenario's default tile, and the per-row kernel (no tile).
@pytest.mark.parametrize("tile", ((8, 512, 4), None), ids=("tile8x512x4", "row"))
@pytest.mark.parametrize("rows", ROWS)
def test_swiglu_fp4_equals_silu_and_mul_then_fp4_quantize(rows, tile, monkeypatch):
    gen = torch.Generator(device="cuda").manual_seed(rows + 2)
    h = torch.randn((rows, 2 * INTER), generator=gen, device="cuda", dtype=torch.float32)
    h[0, :16] *= 16.0  # gate outliers
    h[0, INTER : INTER + 16] *= 16.0  # up outliers on the same block
    h[rows // 2] = 0.0
    h = h.to(torch.bfloat16)
    ref = torch.ops.trtllm.silu_and_mul(h)
    gs = _global_scale(ref)
    ref_codes, ref_scales = torch.ops.trtllm.fp4_quantize(ref, gs, 16, False)
    # The launcher takes its tile from the scenario config; the test pins it without touching the config.
    monkeypatch.setattr(fused_quant, "_swiglu_tile", lambda: tile)
    codes, scales = fused_quant._launch_swiglu(h, gs)
    assert codes.shape == (rows, INTER // 2)
    _assert_same_bytes(codes, ref_codes, "packed e2m1 codes")
    _assert_same_bytes(scales, ref_scales, "swizzled e4m3 block scales")
