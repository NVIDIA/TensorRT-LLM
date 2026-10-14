# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the router entries of the MiniMax-M3 tp8 target: m3_router_gemv, m3_router_gemv_fp4 and
m3_router_tc (one kernel source, one test module)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.moe.m3_router_gemv import (  # noqa: E501
    m3_router_gemv as gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.moe.m3_router_gemv import (  # noqa: E501
    near_tie,
    nvfp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.moe.m3_router_gemv_fp4 import (  # noqa: E501
    m3_router_gemv_fp4 as gemv_fp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.moe.m3_router_tc import (  # noqa: E501
    m3_router_tc as tc,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "the router entries require a CUDA device"

#: The experts' NVFP4 input scale the tc cells use: 448 * 6 / 8, block scales well inside E4M3.
SF_SCALE = 336.0


def _build(m: int, e: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    w = torch.randn(e, k, generator=g, device="cuda") * 0.02
    return x, w


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _compare_logits(out: torch.Tensor, ref: torch.Tensor) -> None:
    """fp32 logits that differ from a true fp32 product only in summation order.

    Over K = 6144 products a reordered fp32 sum moves by a few ulps of the sum of magnitudes, which near
    cancellation is ~sqrt(K) times the row's largest logit's ulp at most. Gated in fp32 ulps of each row's largest
    logit: 64 element-wise, 8 in relative RMS. Measured on GB300: GEMV 1.6 / 1.4 ulp, tensor-core router 3.8 / 2.0
    ulp at 2..48 tokens; the tensor-core logits against a true fp32 product (the TF32 rounding the band must see)
    sit at 2200+ / 1700 ulp.
    """
    assert out.dtype == ref.dtype == torch.float32 and out.shape == ref.shape
    assert_within_ulp(out, ref, element_ulp=64.0, rms_ulp=8.0)


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """[M, K / 2] packed e2m1 bytes -> [M, K] 4-bit codes, low nibble first."""
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def _compare_fp4(
    q: torch.Tensor, sf: torch.Tensor, x: torch.Tensor, sf_scale: torch.Tensor
) -> None:
    """Scales bit-exact; codes bit-exact except within ``TIE_WINDOW`` of an e2m1 midpoint, where the kernel's
    approximate reciprocals choose the side (and then only by one code)."""
    ref_q, ref_sf, scaled = nvfp4_quantize(x, sf_scale)
    assert q.dtype == torch.uint8 and q.shape == ref_q.shape
    assert torch.equal(sf, ref_sf), f"{(sf != ref_sf).sum().item()} block scales differ"
    got, want = _codes(q), _codes(ref_q)
    differ = got != want
    assert not bool((differ & ~near_tie(scaled)).any()), (
        f"{(differ & ~near_tie(scaled)).sum().item()} e2m1 codes differ away from a rounding tie"
    )
    magnitude = ((got & 7).int() - (want & 7).int()).abs()
    assert bool((magnitude[differ] <= 1).all()), "a tie resolved more than one code away"


@pytest.mark.parametrize("cell", gemv.CELLS, ids=[c.why[:48] for c in gemv.CELLS])
def test_gemv_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], spec["e"], spec["k"], _seed(cell))
    with validating(gemv):
        out = gemv(x, w, 1, False)
    _compare_logits(out, gemv.reference(x, w, 1, False))


@pytest.mark.parametrize("cell", gemv_fp4.CELLS, ids=[c.why[:48] for c in gemv_fp4.CELLS])
def test_gemv_fp4_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], spec["e"], spec["k"], _seed(cell))
    sf_scale = torch.tensor([spec["sf_scale"]], dtype=torch.float32, device="cuda")
    with validating(gemv_fp4):
        logits, q, sf = gemv_fp4(x, w, sf_scale, False)
    ref_logits, _, _ = gemv_fp4.reference(x, w, sf_scale, False)
    _compare_logits(logits, ref_logits)
    _compare_fp4(q, sf, x, sf_scale)


@pytest.mark.parametrize("cell", tc.CELLS, ids=[c.why[:48] for c in tc.CELLS])
def test_tc_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], 128, spec["k"], _seed(cell))
    sf_scale = torch.tensor([SF_SCALE], dtype=torch.float32, device="cuda") if spec["fp4"] else None
    with validating(tc):
        outs = tc(x, w, sf_scale, False)
    ref = tc.reference(x, w, sf_scale, False)
    assert len(outs) == len(ref) == (3 if spec["fp4"] else 1)
    _compare_logits(outs[0], ref[0])
    if spec["fp4"]:
        _compare_fp4(outs[1], outs[2], x, sf_scale)


def test_tc_reference_is_tf32_and_the_gate_sees_it() -> None:
    """The tensor-core router rounds the weight to TF32: against a true fp32 product its logits sit far outside
    the band, so the TF32 rounding in ``reference`` is load-bearing rather than decoration."""
    x, w = _build(16, 128, 6144, 5)
    out = tc(x, w, None, False)[0]
    with pytest.raises(AssertionError):
        _compare_logits(out, gemv.reference(x, w, 1, False))


def test_the_gemv_is_a_true_fp32_product() -> None:
    """And the GEMV, by contrast, is no TF32 product: the TF32 reference rejects it."""
    x, w = _build(1, 128, 6144, 6)
    out = gemv(x, w, 1, False)
    with pytest.raises(AssertionError):
        _compare_logits(out, tc.reference(x, w, None, False)[0])


def test_guard_refuses_a_misaligned_activation() -> None:
    """A row view starting 8 bytes into the buffer keeps a legal row stride but not a legal bulk-copy source."""
    x, w = _build(2, 128, 6144, 3)
    flat = torch.empty(2 * 6144 + 4, dtype=torch.bfloat16, device="cuda")
    shifted = flat[4:].view(2, 6144)
    shifted.copy_(x)
    with pytest.raises(AssertionError, match="16-byte aligned"):
        with validating(tc):
            tc(shifted, w, None, False)


def test_guard_refuses_a_zero_global_scale() -> None:
    x, w = _build(1, 128, 6144, 4)
    with pytest.raises(AssertionError, match="positive and finite"):
        with validating(gemv_fp4):
            gemv_fp4(x, w, torch.zeros(1, dtype=torch.float32, device="cuda"), False)
