# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the load-time NVFP4 recipes (`ops/quantization/_nvfp4_recipes`).

Each recipe against an independent construction of what it must produce, bitwise:
the FP8 -> NVFP4 decode values against one `fp4_quantize` call with the global
scale computed separately (linear scale layout requested directly, so the recipe's
swizzle-and-read-back path is checked too); the Marlin pack by unpacking it again
with the reference reader (`_nvfp4_reference.dequantize_marlin`'s helpers), which is
also the proof that the Marlin entry's reference reads the layout the GEMM reads;
the W4A4 copy against `fp4_quantize(BF16(w8 * s), 1/g)`; the gate_up permutation
against its defining map; the lm_head copies against the checkpoint tensors.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_recipes as recipes,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_reference as nvref,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 recipes are certified on sm_120"
)

DEV = "cuda"


def _fp8_weight(n: int, k: int, seed: int):
    """An FP8 per-tensor weight like the checkpoint's projections, plus the corners the
    decode recipe has to survive: 64 rows 1e-4 smaller (subnormal block scales) and
    all-zero blocks (zero scales)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=DEV) * 1.5 + 0.25)
    w[:64] *= 1e-4
    w[64:72, :1024] = 0
    s = (w.abs().amax() / 448.0).float()
    return (w / s).to(torch.float8_e4m3fn), s.reshape(())


def _checkpoint_nvfp4(n: int, k: int, seed: int):
    """Random NVFP4 checkpoint tensors: codes, normal E4M3 scales [n, k/16], fp32 scales."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), generator=g, device=DEV, dtype=torch.int32).to(
        torch.uint8
    )
    sf = torch.randint(0x28, 0x48, (n, k // 16), generator=g, device=DEV, dtype=torch.int32)
    sf = sf.to(torch.uint8).view(torch.float8_e4m3fn)
    weight_scale_2 = torch.tensor(3.7e-4, device=DEV)
    input_scale = torch.tensor(0.0123, device=DEV)
    return packed, sf, weight_scale_2, input_scale


@pytest.mark.parametrize("shape", [(16384, 5120), (5120, 6144)], ids=["qkvz", "out_proj"])
def test_fp8_decode_values_match_an_independent_quantize(shape) -> None:
    n, k = shape
    w8, s = _fp8_weight(n, k, seed=n + k)
    values = recipes.fp8_to_nvfp4_values(w8, s)

    amax = float(w8.to(torch.bfloat16).abs().amax().float().item()) * float(s.item())
    g = 2.0 ** math.ceil(math.log2(amax / 2688.0))
    gs = torch.tensor([float(s.item()) / g], dtype=torch.float32, device=DEV)
    packed, sf = torch.ops.trtllm.fp4_quantize(w8.to(torch.bfloat16), gs, 16, False, False)

    assert float(values.global_scale.item()) == g
    assert values.global_scale.dim() == 0 and values.global_scale.dtype == torch.float32
    assert torch.equal(values.packed, packed.view(torch.uint8))
    assert torch.equal(values.sf, sf.view(torch.uint8).reshape(n, k // 16))
    subnormal = (values.sf > 0) & (values.sf < 0x08)
    assert bool(subnormal.any()), "the weight was built to exercise subnormal block scales"


def test_marlin_pack_round_trips_through_the_reference_reader() -> None:
    n, k = 16384, 5120
    values = recipes.fp8_to_nvfp4_values(*_fp8_weight(n, k, seed=7))
    m = recipes.marlin_pack(values)
    assert (m.size_n, m.size_k) == (n, k)
    assert m.weight.dtype == torch.int32 and tuple(m.weight.shape) == (k // 16, 2 * n)
    assert m.weight_global_scale.dtype == torch.bfloat16

    for n0, n1 in ((0, 64), (8192, 8320), (n - 128, n)):
        codes = nvref.unpack_marlin_codes(m.weight, k, n, n0, n1)
        assert torch.equal(codes, nvref.unpack_codes(values.packed[n0:n1])), (n0, n1)
    scales = nvref.unpack_marlin_scales(m.weight_scale, k, n)
    assert torch.equal(scales, nvref.e4m3_to_float(values.sf))
    assert nvref.marlin_global_scale(m.weight_global_scale) == float(values.global_scale.item())


def test_gemv_layout_is_the_same_values() -> None:
    values = recipes.fp8_to_nvfp4_values(*_fp8_weight(5120, 6144, seed=8))
    gemv = recipes.gemv_layout(values)
    assert torch.equal(gemv.weight, values.packed) and gemv.weight.is_contiguous()
    assert torch.equal(gemv.weight_sf, values.sf) and gemv.weight_sf.shape == (5120, 6144 // 16)
    assert isinstance(gemv.global_scale, float)
    assert gemv.global_scale == float(values.global_scale.item())


def test_w4a4_copy_rounds_w_times_s_to_bf16_first() -> None:
    n, k = 14336, 5120
    w8, s = _fp8_weight(n, k, seed=9)
    a = torch.tensor(0.0371, device=DEV)
    copy = recipes.fp8_to_nvfp4_w4a4(w8, s, a)

    w_bf16 = (w8.float() * s.float()).to(torch.bfloat16)
    g = 2.0 ** math.ceil(math.log2(float(w_bf16.abs().amax().float().item()) / 2688.0))
    packed, sf = torch.ops.trtllm.fp4_quantize(
        w_bf16, torch.tensor([1.0 / g], device=DEV), 16, False, True
    )
    assert torch.equal(copy.weight, packed.view(torch.uint8))
    assert torch.equal(copy.weight_sf.view(torch.uint8), sf.view(torch.uint8))
    a_val = float(a.item())
    assert copy.act_global.dtype == torch.float32 and copy.act_global.shape == (1,)
    assert copy.act_global.item() == torch.tensor(6.0 / a_val, dtype=torch.float32).item()
    assert copy.alpha.item() == torch.tensor((a_val / 6.0) * g, dtype=torch.float32).item()


def test_gate_up_permutation_is_the_epilogue_map() -> None:
    n, k = 34816, 5120
    d = n // 2
    perm = recipes.gate_up_swiglu_row_permutation(n)
    assert torch.equal(torch.sort(perm).values, torch.arange(n))
    # Row c of tile t holds gate (p = 0) or up (p = 1) of output
    # j = 64t + 32nw + 16(a // 2) + 4lq + 2(a % 2) + b for c = 32a + 16nw + 8b + 2lq + p.
    for r in (0, 1, 2, 9, 31, 32, 77, 127, 128, 4095, n - 1):
        t, c = divmod(r, 128)
        a, nw, b, lq, p = c // 32, (c % 32) // 16, (c % 16) // 8, (c % 8) // 2, c % 2
        j = 64 * t + 32 * nw + 16 * (a // 2) + 4 * lq + 2 * (a % 2) + b
        assert perm[r].item() == (j if p == 0 else d + j), r

    g = torch.Generator(device=DEV).manual_seed(10)
    w = (torch.randn(n, k, generator=g, device=DEV) * 0.05).to(torch.bfloat16)
    packed, sf = torch.ops.trtllm.fp4_quantize(w, torch.tensor([64.0], device=DEV), 16, False, True)
    new_w, new_sf = recipes.permute_gate_up_for_swiglu_epilogue(packed, sf, n, k)
    perm = perm.to(DEV)
    assert torch.equal(new_w, packed[perm])
    assert new_sf.shape == sf.shape
    assert torch.equal(
        nvref.unswizzle_sf(new_sf, n, k // 16), nvref.unswizzle_sf(sf, n, k // 16)[perm]
    )


def test_lm_head_copies_are_the_checkpoint_values() -> None:
    v, k = 2048, 5120
    packed, sf, ws2, a = _checkpoint_nvfp4(v, k, seed=11)
    m = recipes.lm_head_marlin(packed, sf, ws2)
    assert (m.size_n, m.size_k) == (v, k)
    assert torch.equal(nvref.unpack_marlin_codes(m.weight, k, v), nvref.unpack_codes(packed))
    assert torch.equal(nvref.unpack_marlin_scales(m.weight_scale, k, v), nvref.e4m3_to_float(sf))
    # Marlin carries the global scale in BF16.
    assert nvref.marlin_global_scale(m.weight_global_scale) == ws2.to(torch.bfloat16).float().item()

    w4a4 = recipes.lm_head_w4a4(packed, sf, ws2, a)
    assert torch.equal(w4a4.weight, packed)
    assert torch.equal(
        nvref.unswizzle_sf(w4a4.weight_sf, v, k // 16), sf.view(torch.uint8).reshape(v, k // 16)
    )
    assert w4a4.act_global.shape == (1,) and w4a4.alpha.shape == (1,)
    assert w4a4.act_global.item() == (1.0 / a.float()).item()
    assert w4a4.alpha.item() == (a.float() * ws2.float()).item()
