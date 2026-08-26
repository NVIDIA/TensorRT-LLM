# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tests for the unified DiT norm-site producer kernel.
#
# Oracles are the MODEL dtype laws (never re-derived from the kernel):
#   - fp32 mode (WAN): fp32 composition throughout, one rounding per stored
#     tensor; the norm reads the ROUNDED residual_out.
#   - bf16 mode (LTX/Qwen/FLUX): bf16-narrow-first modulator combine and
#     bf16 rounding at every eager elementwise op boundary; fp32 only
#     inside the norm.
# Store gates (each form ships its own gate; none is relaxed):
#   - bf16: dynamic error bound vs the fp32 oracle - the kernel must not be
#     worse than 2x the eager-bf16 baseline's own error (plus the in-tree
#     2e-2 assert_close floor).
#   - nvfp4-static: payload + swizzled SF BITWISE vs
#     torch.ops.trtllm.fp4_quantize(y_bf16, s, 16, False) on the op's own
#     bf16 y, including pad rows and M/scale edges.
#   - nvfp4-deferred: raw == fp32(a/6) EXACT; payload BITWISE vs a
#     div.full.f32 reference (Triton `6.0 / a` lowering + an exact e2m1 RNE
#     emulation); K2-finalized SF BITWISE vs fp4_quantize at the K2 s,
#     including 128x4 pad rows.

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")
pytest.importorskip("cuda.bindings.driver")

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.norm_producer import (  # noqa: E402
    fused_norm_producer,
    sfc_finalize,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.norm_producer.norm_producer import (  # noqa: E402
    WARP_SIZE,
    NormProducer,
)

_EPS = 1e-6


def _require_sm100() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The fused norm producer kernel requires an SM100 GPU.")
    return torch.device("cuda", torch.cuda.current_device())


def _make(shape, dtype=torch.bfloat16, seed_tensor=None):
    device = _require_sm100()
    return torch.randn(*shape, dtype=dtype, device=device)


def _row3(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """[B, D] per-batch row -> [B, 1, D] broadcast view for eager math."""
    if t is None:
        return None
    return t.unsqueeze(1) if t.ndim == 2 else t


def _oracle_fp32(
    x,
    residual=None,
    gate=None,
    gate_table=None,
    weight=None,
    bias=None,
    shift=None,
    scale=None,
    shift_table=None,
    scale_table=None,
    norm_type="layer",
    eps=_EPS,
):
    """WAN dtype law in fp64-free eager torch: fp32 composition, one
    rounding per stored tensor; the norm reads the ROUNDED residual."""
    D = x.shape[-1]

    def compose(row, table):
        if row is None and table is None:
            return None
        v = 0.0
        if table is not None:
            v = table
        if row is not None:
            v = v + _row3(row).float()
        return v

    residual_out = None
    if residual is not None:
        v = x.float()
        g = compose(gate, gate_table)
        if g is not None:
            v = v * g
        residual_out = (residual.float() + v).to(x.dtype)
        ln_in = residual_out.float()
    else:
        ln_in = x.float()
    if norm_type == "layer":
        n = F.layer_norm(ln_in, (D,), weight, bias, eps)
    else:
        n = F.rms_norm(ln_in, (D,), None, eps)
    sc = compose(scale, scale_table)
    sh = compose(shift, shift_table)
    if sc is not None:
        n = n * (1 + sc)
    if sh is not None:
        n = n + sh
    return n, residual_out  # n is fp32 (unrounded); caller narrows


def _oracle_bf16(
    x,
    residual=None,
    gate=None,
    gate_table=None,
    shift=None,
    scale=None,
    shift_table=None,
    scale_table=None,
    norm_type="rms",
    eps=_EPS,
):
    """LTX/Qwen/FLUX narrow-first law: (table.to(bf16) + row) combine and
    plain bf16 eager elementwise ops; fp32 only inside the norm."""
    D = x.shape[-1]

    def compose(row, table):
        if row is None and table is None:
            return None
        if table is None:
            return _row3(row)
        v = table.to(x.dtype)
        if row is not None:
            v = v + _row3(row)
        return v

    residual_out = None
    if residual is not None:
        v = x
        g = compose(gate, gate_table)
        if g is not None:
            v = v * g
        residual_out = residual + v
        ln_in = residual_out
    else:
        ln_in = x
    if norm_type == "layer":
        n = F.layer_norm(ln_in.float(), (D,), None, None, eps).to(x.dtype)
    else:
        n = F.rms_norm(ln_in.float(), (D,), None, eps).to(x.dtype)
    sc = compose(scale, scale_table)
    sh = compose(shift, shift_table)
    if sc is not None:
        n = n * (1 + sc)
    if sh is not None:
        n = n + sh
    return n, residual_out


def _assert_error_bounded(actual: torch.Tensor, baseline: torch.Tensor, oracle32: torch.Tensor):
    """The kernel's error against the fp32 oracle must not exceed 2x the
    eager baseline's own bf16 error (entry-11 precedent operationalized),
    with the in-tree assert_close(2e-2) as the absolute floor."""
    torch.testing.assert_close(actual, oracle32.to(actual.dtype), atol=2e-2, rtol=2e-2)
    cand_err = (actual.float() - oracle32).abs().max().item()
    base_err = (baseline.float() - oracle32).abs().max().item()
    assert cand_err <= 2 * base_err + 1e-6, (
        f"kernel error {cand_err:.3e} exceeds 2x eager baseline {base_err:.3e}"
    )


# ---------------------------------------------------------------------------
# Slice 1: per-batch LN-noaffine + modulate, bf16 store (Qwen/FLUX class)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_size", [512, 3072])
@pytest.mark.parametrize("row_dtype", [torch.bfloat16, torch.float32])
def test_per_batch_ln_noaffine_modulate(hidden_size: int, row_dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    x = _make((2, 33, hidden_size))
    shift = torch.randn(2, hidden_size, dtype=row_dtype, device=x.device)
    scale = torch.randn(2, hidden_size, dtype=row_dtype, device=x.device)

    (y,) = fused_norm_producer(x, shift=shift, scale=scale)
    oracle32, _ = _oracle_fp32(x, shift=shift, scale=scale)
    baseline = F.layer_norm(x.float(), (hidden_size,), eps=_EPS).to(x.dtype) * (
        1 + _row3(scale).to(x.dtype)
    ) + _row3(shift).to(x.dtype)
    _assert_error_bounded(y, baseline, oracle32)


def test_per_batch_broadcast_and_b1d_rows() -> None:
    torch.manual_seed(0)
    x = _make((3, 17, 512))
    shift = torch.randn(3, 1, 512, dtype=torch.bfloat16, device=x.device)  # [B,1,D]
    scale = torch.randn(1, 512, dtype=torch.bfloat16, device=x.device)  # [1,D] broadcast

    (y,) = fused_norm_producer(x, shift=shift, scale=scale)
    oracle32, _ = _oracle_fp32(x, shift=shift.squeeze(1), scale=scale.expand(3, 512))
    torch.testing.assert_close(y, oracle32.to(x.dtype), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Slice 2: per-token inline composition (WAN per-token law, #17695 seam)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_size", [768, 1536, 3072])
def test_per_token_chunks_with_tables(hidden_size: int) -> None:
    torch.manual_seed(42)
    x = _make((2, 3, hidden_size))
    temb = _make((2, 3, 6, hidden_size))
    table = torch.randn(6, hidden_size, dtype=torch.float32, device=x.device)

    (y,) = fused_norm_producer(
        x,
        shift=temb[:, :, 0],
        scale=temb[:, :, 1],
        shift_table=table[0],
        scale_table=table[1],
    )
    oracle32, _ = _oracle_fp32(
        x, shift=temb[:, :, 0], scale=temb[:, :, 1], shift_table=table[0], scale_table=table[1]
    )
    torch.testing.assert_close(y, oracle32.to(x.dtype), atol=2e-2, rtol=2e-2)


def test_per_token_residual_norm3_form() -> None:
    hidden_size = 768
    torch.manual_seed(1)
    residual = _make((2, 3, hidden_size))
    x = _make((2, 3, hidden_size))
    temb = _make((2, 3, 6, hidden_size))
    table = torch.randn(6, hidden_size, dtype=torch.float32, device=x.device)

    y, residual_out = fused_norm_producer(
        x,
        residual=residual,
        shift=temb[:, :, 3],
        scale=temb[:, :, 4],
        shift_table=table[3],
        scale_table=table[4],
    )
    oracle32, ref_residual = _oracle_fp32(
        x,
        residual=residual,
        shift=temb[:, :, 3],
        scale=temb[:, :, 4],
        shift_table=table[3],
        scale_table=table[4],
    )
    torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(y, oracle32.to(x.dtype), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Slice 3: gate+resid prologue; LTX RMS bf16-narrow-first; dual output
# ---------------------------------------------------------------------------
def test_gate_resid_affine_ln_norm2_form() -> None:
    hidden_size = 5120 // 4  # 1280: keeps the CI shape small but D%256==0
    torch.manual_seed(2)
    x = _make((2, 5, hidden_size))
    residual = _make((2, 5, hidden_size))
    gate = torch.randn(2, 1, hidden_size, dtype=torch.float32, device=x.device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=x.device)
    bias = torch.randn(hidden_size, dtype=torch.float32, device=x.device)

    y, residual_out = fused_norm_producer(x, residual=residual, gate=gate, weight=weight, bias=bias)
    oracle32, ref_residual = _oracle_fp32(x, residual=residual, gate=gate, weight=weight, bias=bias)
    torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(y, oracle32.to(x.dtype), atol=2e-2, rtol=2e-2)


def test_gate_resid_pertoken_gate_table() -> None:
    hidden_size = 768
    torch.manual_seed(3)
    x = _make((2, 3, hidden_size))
    residual = _make((2, 3, hidden_size))
    temb = _make((2, 3, 6, hidden_size))
    table = torch.randn(6, hidden_size, dtype=torch.float32, device=x.device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=x.device)
    bias = torch.randn(hidden_size, dtype=torch.float32, device=x.device)

    y, residual_out = fused_norm_producer(
        x,
        residual=residual,
        gate=temb[:, :, 2],
        gate_table=table[2],
        weight=weight,
        bias=bias,
    )
    oracle32, ref_residual = _oracle_fp32(
        x, residual=residual, gate=temb[:, :, 2], gate_table=table[2], weight=weight, bias=bias
    )
    torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(y, oracle32.to(x.dtype), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("hidden_size", [2048, 4096])
def test_ltx_rms_bf16_mode_gate_resid(hidden_size: int) -> None:
    """LTX law: bf16-narrow-first combine, all gate/resid/modulate math
    bf16, fp32 only inside the weightless RMS norm."""
    torch.manual_seed(4)
    x = _make((1, 37, hidden_size))
    residual = _make((1, 37, hidden_size))
    gate = _make((1, hidden_size))
    gate_table = torch.randn(hidden_size, dtype=torch.float32, device=x.device)
    shift = _make((1, hidden_size))
    scale = _make((1, hidden_size))
    shift_table = torch.randn(hidden_size, dtype=torch.float32, device=x.device)
    scale_table = torch.randn(hidden_size, dtype=torch.float32, device=x.device)

    y, residual_out = fused_norm_producer(
        x,
        residual=residual,
        gate=gate,
        gate_table=gate_table,
        shift=shift,
        scale=scale,
        shift_table=shift_table,
        scale_table=scale_table,
        norm_type="rms",
        math_mode="bf16",
    )
    ref_y, ref_residual = _oracle_bf16(
        x,
        residual=residual,
        gate=gate,
        gate_table=gate_table,
        shift=shift,
        scale=scale,
        shift_table=shift_table,
        scale_table=scale_table,
        norm_type="rms",
    )
    torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(y, ref_y, atol=2e-2, rtol=2e-2)


def test_ltx_dual_output() -> None:
    """LTX dual site: x_new = x + attn (no gate), one norm, two
    shift/scale outputs."""
    hidden_size = 2048
    torch.manual_seed(5)
    x = _make((1, 29, hidden_size))
    residual = _make((1, 29, hidden_size))
    rows = [_make((1, hidden_size)) for _ in range(4)]
    tables = [torch.randn(hidden_size, dtype=torch.float32, device=x.device) for _ in range(4)]

    y, y2, residual_out = fused_norm_producer(
        x,
        residual=residual,
        shift=rows[0],
        scale=rows[1],
        shift_table=tables[0],
        scale_table=tables[1],
        shift2=rows[2],
        scale2=rows[3],
        shift2_table=tables[2],
        scale2_table=tables[3],
        norm_type="rms",
        math_mode="bf16",
    )
    ref_y, ref_residual = _oracle_bf16(
        x,
        residual=residual,
        shift=rows[0],
        scale=rows[1],
        shift_table=tables[0],
        scale_table=tables[1],
        norm_type="rms",
    )
    ref_y2, _ = _oracle_bf16(
        x,
        residual=residual,
        shift=rows[2],
        scale=rows[3],
        shift_table=tables[2],
        scale_table=tables[3],
        norm_type="rms",
    )
    torch.testing.assert_close(residual_out, ref_residual, atol=0, rtol=0)
    torch.testing.assert_close(y, ref_y, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(y2, ref_y2, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Slice 5: NVFP4 static store - BITWISE vs fp4_quantize on the op's own y
# ---------------------------------------------------------------------------
def _fp4_static_case(x, gs, **kwargs):
    outs_bf16 = fused_norm_producer(x, **kwargs)
    outs_fp4 = fused_norm_producer(x, store="nvfp4_static", global_scale=gs, **kwargs)
    y_bf16 = outs_bf16[0]
    y_fp4, y_sf = outs_fp4[0], outs_fp4[1]
    B, S, D = x.shape
    ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(y_bf16.view(B * S, D), gs, 16, False)
    assert torch.equal(y_fp4.view(-1), ref_fp4.view(torch.uint8).view(-1)), (
        "fp4 payload not bitwise"
    )
    assert torch.equal(y_sf, ref_sf.view(torch.uint8).view(-1)), "fp4 swizzled SF not bitwise"
    # The non-payload outputs (residual_out, y2) must be bit-identical
    # between the bf16-store and fp4-store runs.
    for a, b in zip(outs_bf16[1:], outs_fp4[2:]):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "batch_seq",
    [(1, 128), (2, 133), (1, 512), (3, 100)],  # M multiple/non-multiple of 128
)
@pytest.mark.parametrize("gs_val", [1.0, 0.0037, 913.0])  # scale extremes
def test_fp4_static_bitwise_vs_fp4_quantize(batch_seq, gs_val: float) -> None:
    B, S = batch_seq
    hidden_size = 512
    torch.manual_seed(6)
    x = _make((B, S, hidden_size))
    gs = torch.tensor([gs_val], dtype=torch.float32, device=x.device)
    shift = _make((B, hidden_size))
    scale = _make((B, hidden_size))
    _fp4_static_case(x, gs, shift=shift, scale=scale)


@pytest.mark.parametrize("vec", [16, 32])
def test_fp4_stores_bitwise_across_vec(vec: int) -> None:
    """The 16-element group amax is exactly associative, so payload + SF
    (and deferred raw) must be bitwise-identical at every vec."""
    hidden_size = 1024  # vec=32 requires D % 1024 == 0
    B, S = 2, 133
    torch.manual_seed(12)
    x = _make((B, S, hidden_size))
    gs = torch.tensor([0.42], dtype=torch.float32, device=x.device)
    rows = _make((B, hidden_size))
    ref4, refsf = fused_norm_producer(
        x, shift=rows, scale=rows, store="nvfp4_static", global_scale=gs, vec=8
    )
    y4, sf = fused_norm_producer(
        x, shift=rows, scale=rows, store="nvfp4_static", global_scale=gs, vec=vec
    )
    assert torch.equal(y4, ref4) and torch.equal(sf, refsf)
    ref4d, refraw = fused_norm_producer(x, shift=rows, scale=rows, store="nvfp4_deferred", vec=8)
    y4d, raw = fused_norm_producer(x, shift=rows, scale=rows, store="nvfp4_deferred", vec=vec)
    assert torch.equal(y4d, ref4d) and torch.equal(raw, refraw)


def test_fp4_static_carriers_and_edges() -> None:
    hidden_size = 768
    torch.manual_seed(7)
    device = _require_sm100()
    gs = torch.tensor([0.42], dtype=torch.float32, device=device)
    x = _make((2, 65, hidden_size))
    residual = _make((2, 65, hidden_size))
    gate = torch.randn(2, 1, hidden_size, dtype=torch.float32, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    bias = torch.randn(hidden_size, dtype=torch.float32, device=device)
    rows = _make((2, hidden_size))

    # gate+resid + affine LN carrier (WAN norm2 srnss form)
    _fp4_static_case(x, gs, residual=residual, gate=gate, weight=weight, bias=bias)
    # resid + modulate carrier (WAN norm3 srnss form)
    _fp4_static_case(x, gs, residual=residual, shift=rows, scale=rows)
    # RMS carrier
    _fp4_static_case(x, gs, shift=rows, scale=rows, norm_type="rms")
    # zero rows exercise the vecMax == 0 branch
    x0 = x.clone()
    x0[:, ::7] = 0
    _fp4_static_case(x0, gs, shift=rows, scale=rows)


# ---------------------------------------------------------------------------
# Slice 6: NVFP4 deferred store + K2 finalize
# ---------------------------------------------------------------------------
_E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _e2m1_rne_codes(v: torch.Tensor) -> torch.Tensor:
    """Exact emulation of cvt.rn.satfinite.e2m1x2.f32 on finite fp32:
    round-to-nearest-even over the e2m1 magnitude set with saturation."""
    assert torch.isfinite(v).all()
    mag = v.abs()
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    code = torch.searchsorted(mids, mag.contiguous().view(-1), right=True).view(mag.shape)
    # Exact ties (mag == mids[k], representable in fp32) round to the even
    # code of {k, k+1}: with right=True the tie lands on k+1; move even ties
    # down to k.
    for k, m in enumerate(_E2M1_MIDS):
        tie = mag == m
        if k % 2 == 0:  # even lower code
            code = torch.where(tie, torch.full_like(code, k), code)
    code = code.clamp(max=7)
    sign = torch.signbit(v)
    return (code + sign * 8).to(torch.uint8)


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """[.., D] codes -> [.., D//2] packed bytes (element 2k low nibble)."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _div_full_ref(a: torch.Tensor) -> torch.Tensor:
    """6.0 / a with Triton's fp32 division lowering (div.full.f32) - the
    pinned instruction of the deferred payload recipe."""
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")

    @triton.jit
    def _k(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        av = tl.load(a_ptr + i, mask=m)
        tl.store(o_ptr + i, 6.0 / av, mask=m)

    out = torch.empty_like(a)
    n = a.numel()
    _k[(triton.cdiv(n, 1024),)](a.view(-1), out.view(-1), n, BLOCK=1024)
    return out


@pytest.mark.parametrize(
    "batch_seq", [(1, 1), (1, 130), (2, 133), (1, 512)]
)  # incl. M=1 and non-128 multiples (no pad constraint for deferred)
@pytest.mark.parametrize("amp", [1.0, 1e-3, 1e3])  # scale extremes
def test_fp4_deferred_raw_exact_and_payload_bitwise(batch_seq, amp: float) -> None:
    B, S = batch_seq
    hidden_size = 512
    torch.manual_seed(8)
    x = (_make((B, S, hidden_size)) * amp).to(torch.bfloat16)
    shift = _make((B, hidden_size))
    scale = _make((B, hidden_size))

    (y_bf16,) = fused_norm_producer(x, shift=shift, scale=scale)
    y_fp4, raw = fused_norm_producer(x, shift=shift, scale=scale, store="nvfp4_deferred")

    M, D = B * S, hidden_size
    # Gate (b): raw == fp32(a/6) EXACT (single fp32 multiply, no tolerance).
    a = y_bf16.view(M, D // 16, 16).float().abs().amax(-1)
    ref_raw = a * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    assert torch.equal(raw, ref_raw), "deferred raw scales not exact a/6"

    # Payload BITWISE vs the pinned recipe: oscale = div.full.f32(6, a)
    # (Triton's fp32 `/` lowering), payload = e2m1_rne(y * oscale) with
    # zero-amax groups producing oscale = 0.
    oscale = _div_full_ref(a)
    oscale = torch.where(a > 0, oscale, torch.zeros_like(oscale))
    scaled = y_bf16.view(M, D // 16, 16).float() * oscale.unsqueeze(-1)
    ref_payload = _pack_e2m1(_e2m1_rne_codes(scaled.view(M, D)))
    assert torch.equal(y_fp4.view(M, D // 2), ref_payload), "deferred payload not bitwise"

    # Gate (d): K2-finalized swizzled SF bitwise vs fp4_quantize at the K2
    # s, including pad rows.
    sf, s, max_raw = sfc_finalize(raw)
    assert torch.equal(max_raw, raw.max()) and torch.equal(s, 448.0 / max_raw)
    ref_fp4_s, ref_sf_s = torch.ops.trtllm.fp4_quantize(y_bf16.view(M, D), s.reshape(1), 16, False)
    assert torch.equal(sf, ref_sf_s.view(torch.uint8).view(-1)), "K2 SF not bitwise at K2 s"


def test_k2_pad_columns_zero_filled() -> None:
    """K/16 not a multiple of 4 exercises K2's column padding."""
    device = _require_sm100()
    torch.manual_seed(9)
    M, KB = 130, 6  # 6 raw cols -> 2 k-tiles, 2 pad cols; 130 rows -> 126 pad rows
    raw = torch.rand(M, KB, dtype=torch.float32, device=device) + 0.01
    sf, _s, _max_raw = sfc_finalize(raw)
    # Every byte belonging to a pad row or pad col must be zero, so the
    # nonzero count equals the M x KB real bytes (raw*s >= 4 here, so every
    # real byte encodes nonzero).
    total_nonzero = int((sf != 0).sum().item())
    # Count expected nonzero bytes: M rows x KB cols (raw > 0 so e4m3 > 0).
    assert total_nonzero == M * KB, f"pad bytes not zero-filled ({total_nonzero} != {M * KB})"


# ---------------------------------------------------------------------------
# torch.compile parity
# ---------------------------------------------------------------------------
def test_torch_compile_fullgraph_parity() -> None:
    hidden_size = 512
    torch.manual_seed(10)
    x = _make((2, 65, hidden_size))
    residual = _make((2, 65, hidden_size))
    rows = _make((2, hidden_size))
    gs = torch.tensor([0.9], dtype=torch.float32, device=x.device)

    def run(x, residual, rows, gs):
        (y,) = fused_norm_producer(x, shift=rows, scale=rows)
        y2, ro = fused_norm_producer(x, residual=residual, shift=rows, scale=rows)
        f4, sf = fused_norm_producer(
            x, shift=rows, scale=rows, store="nvfp4_static", global_scale=gs
        )
        f4d, raw = fused_norm_producer(x, shift=rows, scale=rows, store="nvfp4_deferred")
        return y, y2, ro, f4, sf, f4d, raw

    eager = run(x, residual, rows, gs)
    compiled = torch.compile(run, fullgraph=True)(x, residual, rows, gs)
    for actual, expected in zip(compiled, eager):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Contract rejection matrix + guard rails
# ---------------------------------------------------------------------------
def test_rejects_invalid_contracts() -> None:
    hidden_size = 512
    torch.manual_seed(11)
    x = _make((1, 130, hidden_size))
    rows = _make((1, hidden_size))
    gs = torch.tensor([1.0], dtype=torch.float32, device=x.device)

    with pytest.raises(ValueError, match="unsupported dtype"):
        fused_norm_producer(x.half(), shift=rows.half(), scale=rows.half())
    with pytest.raises(ValueError, match="must be passed together"):
        fused_norm_producer(x, shift=rows)
    with pytest.raises(ValueError, match="gate requires residual"):
        fused_norm_producer(x, gate=rows)
    with pytest.raises(ValueError, match="rms norm is weightless"):
        fused_norm_producer(
            x,
            weight=torch.randn(hidden_size, dtype=torch.float32, device=x.device),
            norm_type="rms",
        )
    with pytest.raises(ValueError, match="bias requires weight"):
        fused_norm_producer(x, bias=torch.randn(hidden_size, dtype=torch.float32, device=x.device))
    with pytest.raises(ValueError, match="bf16 math mode requires bf16"):
        fused_norm_producer(
            x, shift=rows.float(), scale=rows.float(), math_mode="bf16", norm_type="rms"
        )
    with pytest.raises(ValueError, match="requires global_scale"):
        fused_norm_producer(x, shift=rows, scale=rows, store="nvfp4_static")
    with pytest.raises(ValueError, match="global_scale must be None"):
        fused_norm_producer(x, shift=rows, scale=rows, store="nvfp4_deferred", global_scale=gs)
    with pytest.raises(ValueError, match="only accepted with"):
        fused_norm_producer(x, shift=rows, scale=rows, global_scale=gs)
    with pytest.raises(ValueError, match="not supported"):
        fused_norm_producer(_make((1, 4, 500)), vec=8)  # D % 256 != 0
    with pytest.raises(ValueError, match="vec=4 not supported"):
        fused_norm_producer(x, vec=4)
    noncontig = _make((1, hidden_size, 130)).transpose(1, 2)
    with pytest.raises(ValueError, match="last dim must be contiguous"):
        fused_norm_producer(noncontig)
    misaligned = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:].view_as(x)
    with pytest.raises(ValueError, match="32-byte aligned"):
        fused_norm_producer(misaligned)
    small = _make((1, 60, hidden_size))
    with pytest.raises(ValueError, match="too small for in-kernel SF pad zeroing"):
        fused_norm_producer(small, shift=rows, scale=rows, store="nvfp4_static", global_scale=gs)


def test_rejects_num_warps_above_warp_size() -> None:
    with pytest.raises(AssertionError, match="cta_reduce_sum"):
        NormProducer(D=16384, vec=8)  # 64 warps
    assert NormProducer(D=8192, vec=8).num_warps == WARP_SIZE


# ---------------------------------------------------------------------------
# Staged (bulk-copy smem) path, wide-vec geometry, bf16 tables
# ---------------------------------------------------------------------------
def test_staged_bitwise_vs_unstaged() -> None:
    """stage=True moves x/residual through a bulk-async smem prefetch but
    performs identical arithmetic in identical order: every output must be
    BITWISE identical to the unstaged path, for every store form."""
    torch.manual_seed(13)
    device = _require_sm100()
    for hidden_size in (5120, 2048):
        x = _make((2, 130, hidden_size))
        res = _make((2, 130, hidden_size))
        rows = _make((2, hidden_size))
        roww = torch.randn(2, hidden_size, dtype=torch.float32, device=device)
        w = torch.randn(hidden_size, dtype=torch.float32, device=device)
        b = torch.randn(hidden_size, dtype=torch.float32, device=device)
        gs = torch.tensor([0.8], dtype=torch.float32, device=device)
        cases = [
            dict(shift=rows, scale=rows),
            dict(weight=w, bias=b),
            dict(residual=res, gate=roww, weight=w, bias=b),
            dict(
                residual=res, gate=rows, shift=rows, scale=rows, norm_type="rms", math_mode="bf16"
            ),
            dict(shift=rows, scale=rows, store="nvfp4_static", global_scale=gs),
            dict(shift=rows, scale=rows, store="nvfp4_deferred"),
        ]
        for kw in cases:
            ref = fused_norm_producer(x, stage=False, **kw)
            got = fused_norm_producer(x, stage=True, **kw)
            for a, c in zip(ref, got):
                assert torch.equal(a, c), f"staged output differs: D={hidden_size} {sorted(kw)}"


def test_vec40_geometry() -> None:
    """vec=40 (128-thread CTAs at D=5120, the fusedAdaptiveLayerNorm
    geometry) - numerics within the standard gate, staged form bitwise
    vs unstaged."""
    torch.manual_seed(14)
    x = _make((2, 65, 5120))
    rows = _make((2, 5120))
    (y40,) = fused_norm_producer(x, shift=rows, scale=rows, vec=40)
    oracle32, _ = _oracle_fp32(x, shift=rows, scale=rows)
    torch.testing.assert_close(y40, oracle32.to(x.dtype), atol=2e-2, rtol=2e-2)
    (y40s,) = fused_norm_producer(x, shift=rows, scale=rows, vec=40, stage=True)
    assert torch.equal(y40s, y40)


def test_bf16_affine_tables() -> None:
    """bf16 [D] weight/bias (the fused_adaptive_layernorm numerics class,
    pre-narrowed weights) vs the matching bf16-weight oracle."""
    torch.manual_seed(15)
    x = _make((2, 65, 1024))
    w16 = _make((1024,))
    b16 = _make((1024,))
    ref = F.layer_norm(x.float(), (1024,), w16.float(), b16.float(), _EPS).to(x.dtype)
    (y,) = fused_norm_producer(x, weight=w16, bias=b16)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)
