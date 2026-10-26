# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_GU_SWIGLU_EPI (F5): the drafter MLP's gate_up GEMV with the swigluoai in its epilogue is bit for bit today's
two launches, m3_mx_gemv_bf16(x, gate_up) followed by the Triton silu_and_mul (limit 7, alpha 1.702, beta 1)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (
    mxdraft,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.mx_gemv import (  # noqa: E501
    op as gemv,
)
from tensorrt_llm._torch.modules.swiglu import swiglu

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 kernel (certified on sm_107 only)",
)

F, K = 1536, 6144  # the drafter MLP's per-rank gate_up at TP8: [2 F, K] = [3072, 6144]
LIMIT, ALPHA, BETA = 7.0, 1.702, 1.0


def _mxfp8_exact_bf16(n: int, k: int, scale_lo: int, scale_hi: int, seed: int) -> torch.Tensor:
    """A bf16 weight that is exactly an MXFP8 dequantization (E4M3 codes x 2^e per 32 columns), as the checkpoint's."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = (
        (torch.randn(n, k, device="cuda", generator=g) * 64)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    exp = torch.randint(scale_lo, scale_hi + 1, (n, k // 32), device="cuda", generator=g)
    return (codes.float() * torch.exp2(exp.float()).repeat_interleave(32, dim=1)).to(torch.bfloat16)


def _forms(w_bf16: torch.Tensor):
    plain = mxdraft.mxfp8_from_bf16(w_bf16)
    inter = mxdraft.mxfp8_from_bf16(gemv.gu_interleave(w_bf16))
    assert plain is not None and inter is not None, "weight is not an exact MXFP8 dequantization"
    return (
        (plain[0], mxdraft.swizzle_sf(plain[1])),
        (inter[0], mxdraft.swizzle_sf(inter[1])),
    )


def _two_launch(x, w8, sf):
    y = gemv.m3_mx_gemv_bf16(x, w8, sf)
    assert y is not None
    return swiglu(y, swiglu_limit=LIMIT, swiglu_alpha=ALPHA, swiglu_beta=BETA)


def test_interleave_is_a_row_permutation():
    # float32: row ids up to 3071 are exact (bf16 holds integers exactly only to 256); gu_interleave is dtype-agnostic.
    w = (
        torch.arange(2 * F, device="cuda", dtype=torch.float32)
        .view(-1, 1)
        .expand(-1, 32)
        .contiguous()
    )
    wi = gemv.gu_interleave(w)
    rows = wi[:, 0].float().long()
    tile = torch.arange(2 * F, device="cuda") // 128
    pos = torch.arange(2 * F, device="cuda") % 128
    expect = torch.where(pos < 64, tile * 64 + pos, F + tile * 64 + pos - 64)
    assert torch.equal(rows, expect)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("case", ["natural", "clamps"])
def test_epilogue_bitwise_vs_two_launches(m, case):
    # natural: small outputs (no clamp). clamps: scales large enough that many gate > 7 and |up| > 7 (both clamps).
    lo, hi = (-14, -10) if case == "natural" else (-6, -2)
    w = _mxfp8_exact_bf16(2 * F, K, lo, hi, seed=1000 + m)
    (w8, sf), (w8i, sfi) = _forms(w)
    g = torch.Generator(device="cuda").manual_seed(m)
    x = (torch.randn(m, K, device="cuda", generator=g) * (1.0 if case == "natural" else 4.0)).to(
        torch.bfloat16
    )
    ref = _two_launch(x, w8, sf)
    out = gemv.m3_mx_gemv_gu_swiglu(x, w8i, sfi)
    assert out is not None and out.shape == (m, F) and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
        f"m={m} {case}: {int((out.view(torch.int16) != ref.view(torch.int16)).sum())} differing elements"
    )
    if case == "clamps":
        y = gemv.m3_mx_gemv_bf16(x, w8, sf).float()
        assert int((y[:, :F] > LIMIT).sum()) > 0 and int((y[:, F:].abs() > LIMIT).sum()) > 0, (
            "clamps not exercised"
        )


def test_cuda_graph_replay_matches_eager():
    m = 8
    w = _mxfp8_exact_bf16(2 * F, K, -12, -8, seed=7)
    (_, _), (w8i, sfi) = _forms(w)
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    eager = gemv.m3_mx_gemv_gu_swiglu(x, w8i, sfi).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gemv.m3_mx_gemv_gu_swiglu(x, w8i, sfi)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


def test_declines():
    w = _mxfp8_exact_bf16(2 * F, K, -12, -8, seed=3)
    (_, _), (w8i, sfi) = _forms(w)
    # more rows than the epilogue's 128 owner threads take at split 4 (16 pairs x 8 tokens)
    x = torch.randn(9, K, device="cuda").to(torch.bfloat16)
    assert gemv.m3_mx_gemv_gu_swiglu(x, w8i, sfi) is None
    # a non-bf16 activation
    assert gemv.m3_mx_gemv_gu_swiglu(x[:8].float(), w8i, sfi) is None


def test_kill_switch_source():
    import inspect

    src = inspect.getsource(mxdraft.route_gu_swiglu)
    assert "TRTLLM_M31_DISABLE_DRAFT_GU_SWIGLU_EPI" in src
    assert mxdraft.DRAFT_GU_SWIGLU_EPI_PATH == "DRAFT_GU_SWIGLU_EPI"
