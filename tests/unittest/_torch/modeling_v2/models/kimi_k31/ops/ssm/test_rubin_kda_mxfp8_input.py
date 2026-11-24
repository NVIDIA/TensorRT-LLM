# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 output mode of rubin_kda_mtp_verify (sm_107).

With `quantize_output` the verify writes its output as an MXFP8 GEMM operand: E4M3 codes and one
UE8M0 scale per (row, head) in the R128c4 layout. It must be bitwise the engine's 1x128 UE8M0
quantization of the BF16 output the same call makes without it, and the entry's reference must
quantize the same way. With `mxfp8_exact` as well it must be bitwise the catalog `mxfp8_quantize`
of that BF16 output (one scale per 32 values: the operand `rubin_mxfp8_linear` gives o_proj), and
the calls must commit the same state and caches; also where the output norm weight puts groups of
that output in the subnormal-scale band (`amax * rcp(448)` in (2^-127, 2^-126)), gives subnormal
values, infinities or NaNs, NaNs by parity included (every odd value a NaN: a NaN maximum; every
even one: the odd maximum; one number among NaNs at an even and at an odd index).
"""

import pytest
import torch
from _rubin_kda_verify_inputs import MUTATED, bits, clone_mutated, scale_bytes, verify_args

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers trtllm::fp8_quantize_1x128_packed_ue8m0
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (
    mxfp8_scales_from_r128c4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    _catalog_mxfp8_rows,
    _mxfp8_rows,
    rubin_kda_mtp_verify,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)


def _engine_mxfp8(bf16_rows: torch.Tensor, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    codes, scales = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        bf16_rows.reshape(-1, heads * 128)
    )
    return codes, scales


@pytest.mark.parametrize("gate", ["g", "fa"])
@pytest.mark.parametrize("batch,num_spec", [(16, 5), (24, 5), (11, 2)])
@torch.inference_mode()
def test_mxfp8_output_is_the_quantized_bf16_output(batch, num_spec, gate) -> None:
    heads = 12
    spec = dict(N=batch, H=heads, num_spec=num_spec, pool=2 * batch, gate=gate)
    bf16_args = verify_args(dict(spec, out="bf16"), seed=batch + num_spec)
    mx_args = dict(clone_mutated(bf16_args), quantize_output=True)
    out = rubin_kda_mtp_verify(**bf16_args)
    codes, scales = rubin_kda_mtp_verify(**mx_args)
    rows = batch * (num_spec + 1)
    want_codes, want_scales = _engine_mxfp8(out.view(rows, heads, 128), heads)
    ref_codes, ref_scales = _mxfp8_rows(out.view(rows, heads, 128))
    torch.cuda.synchronize()
    assert torch.equal(bits(codes), bits(want_codes.view(codes.shape)))
    assert torch.equal(scale_bytes(scales, rows, heads), scale_bytes(want_scales, rows, heads))
    # The reference's quantizer is the engine's: same codes, same scale bytes.
    assert torch.equal(bits(ref_codes), bits(want_codes.view(ref_codes.shape)))
    assert torch.equal(scale_bytes(ref_scales, rows, heads), scale_bytes(want_scales, rows, heads))


def _catalog_mxfp8(bf16_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
        mxfp8_quantize,
    )

    return mxfp8_quantize(bf16_rows, True, 32)


@pytest.mark.parametrize(
    "batch,heads,num_spec,stacked", [(12, 96, 7, True), (16, 12, 5, False), (11, 12, 2, False)]
)
@torch.inference_mode()
def test_mxfp8_exact_output_is_the_catalog_quantized_bf16_output(
    batch, heads, num_spec, stacked
) -> None:
    spec = dict(N=batch, H=heads, num_spec=num_spec, pool=2 * batch, stacked=stacked)
    bf16_args = verify_args(dict(spec, out="bf16"), seed=batch + heads + num_spec)
    mx_args = dict(clone_mutated(bf16_args), quantize_output=True, mxfp8_exact=True)
    out = rubin_kda_mtp_verify(**bf16_args)
    codes, scales = rubin_kda_mtp_verify(**mx_args)
    rows, k = batch * (num_spec + 1), heads * 128
    want_codes, want_scales = _catalog_mxfp8(out.reshape(rows, k))
    ref_codes, ref_scales = _catalog_mxfp8_rows(out.view(rows, heads, 128))
    torch.cuda.synchronize()
    assert torch.equal(bits(codes), bits(want_codes.view(codes.shape)))
    want_bytes = mxfp8_scales_from_r128c4(want_scales.view(torch.uint8), rows, k)
    assert torch.equal(mxfp8_scales_from_r128c4(scales, rows, k), want_bytes)
    # The reference's quantizer is the catalog one: same codes, same scale bytes.
    assert torch.equal(bits(ref_codes), bits(want_codes.view(ref_codes.shape)))
    assert torch.equal(mxfp8_scales_from_r128c4(ref_scales, rows, k), want_bytes)
    # Only the epilogue differs: both calls commit the same state, windows and replay caches.
    for name in MUTATED:
        assert torch.equal(bits(mx_args[name]), bits(bf16_args[name])), name


@pytest.mark.parametrize("planted", ["tiny_inf", "nan", "nan_parity"])
@torch.inference_mode()
def test_mxfp8_exact_edge_groups_are_the_catalog_quantized_bf16_output(planted) -> None:
    batch, heads, num_spec = 16, 12, 5
    spec = dict(N=batch, H=heads, num_spec=num_spec, pool=2 * batch, stacked=False)
    bf16_args = verify_args(dict(spec, out="bf16"), seed=1112)
    w = bf16_args["onorm_weight"].clone()  # one weight per value column of every head
    if planted == "tiny_inf":
        w[0:32] = 1.6e-36  # group 0: the subnormal-scale band
        w[32:64] = 1.0e-34  # group 1: maxima ~1e-34 ...
        w[40:44] = 1.0e-40  # ... with subnormal values
        w[64:96] = 2.2e-36  # group 2: around the band
        w[100] = float("inf")  # group 3: one infinity per row and head
    elif planted == "nan":
        w[0:32] = float("nan")  # group 0: a group of NaN
        w[40] = float("nan")  # group 1: one NaN
    else:  # the quantizer's NaN rule by parity: vecMax = even > odd ? even : odd
        lane = torch.arange(32, device=w.device)
        w[1:32:2] = float("nan")  # group 0: every odd value a NaN: a NaN maximum, byte 255
        w[32:64:2] = float("nan")  # group 1: every even value a NaN: the odd maximum
        w[64:96] = torch.where(lane == 6, w[64:96], float("nan"))  # group 2: one even number
        w[96:128] = torch.where(lane == 5, w[96:128], float("nan"))  # group 3: one odd number
    bf16_args["onorm_weight"] = w
    mx_args = dict(clone_mutated(bf16_args), quantize_output=True, mxfp8_exact=True)
    out = rubin_kda_mtp_verify(**bf16_args)
    codes, scales = rubin_kda_mtp_verify(**mx_args)
    rows, k = batch * (num_spec + 1), heads * 128
    flat = out.reshape(rows, k)
    if planted == "tiny_inf":
        sf = flat.float().abs().view(rows, -1, 32).amax(-1) * (1.0 / 448.0)
        assert int(((sf > 2.0**-127) & (sf < 2.0**-126)).sum()) > 0, "no group in the band"
        assert bool(flat.isinf().any())
    else:
        assert bool(flat.isnan().any())
    if planted == "nan_parity":
        g = flat.view(rows, -1, 32).isnan()
        odd_nan = g[..., 1::2].all(-1) & ~g[..., 0::2].all(-1)  # vecMax NaN, finite evens
        even_nan = g[..., 0::2].all(-1) & ~g[..., 1::2].all(-1)  # vecMax = the odd maximum
        assert bool(odd_nan.any()) and bool(even_nan.any()), "no group of the NaN rule by parity"
    want_codes, want_scales = _catalog_mxfp8(flat)
    torch.cuda.synchronize()
    assert torch.equal(bits(codes), bits(want_codes.view(codes.shape)))
    want_bytes = mxfp8_scales_from_r128c4(want_scales.view(torch.uint8), rows, k)
    assert torch.equal(mxfp8_scales_from_r128c4(scales, rows, k), want_bytes)
    for name in MUTATED:
        assert torch.equal(bits(mx_args[name]), bits(bf16_args[name])), name
