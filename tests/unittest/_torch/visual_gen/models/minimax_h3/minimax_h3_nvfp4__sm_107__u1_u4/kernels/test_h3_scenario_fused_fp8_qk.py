# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kernels/fused_fp8_qk.py``: per-head RMSNorm + partial split-half RoPE + FP8 E4M3 per-tensor cast.

The module docstring states two contracts, and each is checked against a reference that does not
share the kernel's code:

* the cast: ``scale = bf16(max(amax / 448, 1 / (448 * 512)))``, codes ``rn.satfinite(x * (1 / scale))``,
  descale ``float(scale)``, the scale arithmetic of TensorRT-LLM's per-tensor quantizer. The reference
  is ``torch.ops.tensorrt_llm.quantize_e4m3_per_tensor`` on the BF16 normalised / rotated tensor, and
  the same arithmetic written in torch where the C++ op has no equivalent (padded rows, the static Q /
  K descale);
* the norm and RoPE: the documented formula (fp32 sum of squares and ``rsqrt``, one BF16 rounding
  after ``x * rsqrt(mean(x^2) + eps)``, the weight and the rotation in fp32 with BF16-rounded
  ``cos`` / ``sin``, one final BF16 rounding), evaluated independently in torch. The only freedom
  between two fp32 evaluations is the rounding of the normalised value, which the sum-of-squares
  order and ``rsqrt`` can move by one BF16 step; the bound below is built from those terms.

The kernel variants the scenario can select (split vs grouped Q / K cast, V-amax fold, scale mode,
plain vs packed Ulysses send buffer) are pinned per case with ``monkeypatch`` on the module globals;
the process scenario configuration is never re-selected.
"""

import pytest
import torch
from utils.util import skip_pre_ada

import tensorrt_llm  # noqa: F401  -- loads libth_common.so (registers the tensorrt_llm ops)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
    fused_fp8_qk as fq,
)

pytestmark = skip_pre_ada

H, D, ROT, EPS = 56, 128, 96, 1e-6
U = 4
E4M3 = torch.float8_e4m3fn
MIN_SCALE = 1.0 / (448.0 * 512.0)
ROWS = (1017, 9455)
# (``_V2`` grouped cast, ``_VAMAX_FOLD``): the split kernels, the grouped Q / K cast, and the
# grouped cast with V's amax folded into the Q launch (inert unless the scale mode is qkstatic).
KERNELS = {"split": (False, False), "grouped": (True, False), "grouped_fold": (True, True)}


def _inputs(rows: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn((1, rows, 3 * H * D), generator=gen, device="cuda") * 2.0
    # V outliers in the first rows: the dynamic V scale is set there, and the padded case keeps them.
    qkv[0, : max(rows // 97, 1), 2 * H * D :: 977] *= 40.0
    qkv = qkv.to(torch.bfloat16)
    wq = (1.0 + 0.1 * torch.randn(D, generator=gen, device="cuda")).to(torch.bfloat16)
    wk = (1.0 + 0.1 * torch.randn(D, generator=gen, device="cuda")).to(torch.bfloat16)
    pos = torch.arange(rows, dtype=torch.float32, device="cuda")[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT))
    ang = torch.cat([pos * inv, pos * inv], dim=-1)
    return qkv, wq, wk, ang.cos(), ang.sin()


def _planes(qkv: torch.Tensor):
    """Q, K, V of the packed projection output as contiguous ``[1, S, H, D]`` BF16 tensors."""
    s = qkv.shape[1]
    return [
        qkv[..., i * H * D : (i + 1) * H * D].reshape(1, s, H, D).contiguous() for i in range(3)
    ]


def _norm_rope_bf16(x, w, cos, sin):
    out = fq.norm_rope_tensor(x, w, cos, sin, EPS)
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    return out


def _descale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """fp32 ``float(bf16(max(amax / 448, 1 / (448 * 512))))``, the documented scale arithmetic."""
    return (
        torch.maximum(amax.float() / 448.0, torch.tensor(MIN_SCALE, device=amax.device))
        .to(torch.bfloat16)
        .float()
    )


def _cast(y: torch.Tensor, descale: torch.Tensor) -> torch.Tensor:
    """``rn.satfinite.e4m3(y * (1 / descale))`` in fp32; the clamp makes torch's cast saturating."""
    return (y.float() * (1.0 / descale.float())).clamp(-448.0, 448.0).to(E4M3)


def _reference(qkv, wq, wk, cos, sin, mode, valid_len=None):
    """Codes ``[3][1, S, H * D]`` and the ``[3]`` fp32 descale table for one scale mode."""
    s = qkv.shape[1]
    q, k, v = _planes(qkv)
    tensors = (_norm_rope_bf16(q, wq, cos, sin), _norm_rope_bf16(k, wk, cos, sin), v)
    codes, descales = [], []
    for i, y in enumerate(tensors):
        if mode == "qkstatic" and i < 2:
            d = torch.tensor(fq.qk_static_descale_value(D), dtype=torch.float32, device=y.device)
        else:
            rows = y if valid_len is None else y[:, : int(valid_len.item())]
            d = _descale_from_amax(rows.float().abs().amax())
            if valid_len is None:
                # The C++ per-tensor quantizer must agree with the documented arithmetic.
                ref_codes, ref_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(y)
                assert torch.equal(ref_scale.float().reshape(()), d), (i, ref_scale, d)
                assert torch.equal(ref_codes.view(torch.uint8), _cast(y, d).view(torch.uint8)), i
        codes.append(_cast(y, d).reshape(1, s, H * D))
        descales.append(d.reshape(()))
    return codes, torch.stack(descales)


def _run(qkv, wq, wk, cos, sin, mode, ulysses, valid_len=None, rope_halves_equal=False):
    static = fq.qk_static_descale(D, qkv.device) if mode == "qkstatic" else None
    outs, descale = fq._qk_norm_rope_fp8_impl(
        qkv,
        wq,
        wk,
        cos,
        sin,
        static,
        EPS,
        H,
        D,
        0,
        valid_len,
        ulysses,
        rope_halves_equal=rope_halves_equal,
    )
    if ulysses > 1:
        # Send buffer [U, S, 3, H / U, D] -> the three plain [1, S, H * D] tensors.
        send = outs[0]
        s = qkv.shape[1]
        plain = [send[:, :, i].permute(1, 0, 2, 3).reshape(1, s, H * D) for i in range(3)]
        return plain, descale
    return list(outs), descale


def _pin(monkeypatch, mode: str, kernel: str) -> None:
    v2, fold = KERNELS[kernel]
    monkeypatch.setattr(fq, "_MODE", mode)
    monkeypatch.setattr(fq, "_V2", v2)
    monkeypatch.setattr(fq, "_VAMAX_FOLD", fold)
    monkeypatch.setattr(fq, "_split_launch", lambda: True)
    monkeypatch.setattr(fq, "_REDUCE_GROUP", None)


def _assert_codes(actual, expected, what):
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.dtype == E4M3 and a.shape == e.shape, (what, i, a.dtype, a.shape, e.shape)
        assert torch.equal(a.view(torch.uint8), e.view(torch.uint8)), (what, "QKV"[i])


@pytest.mark.parametrize("ulysses", (0, U), ids=("plain", "packed_u4"))
@pytest.mark.parametrize("kernel", tuple(KERNELS))
@pytest.mark.parametrize("mode", ("dynamic", "qkstatic"))
@pytest.mark.parametrize("rows", ROWS)
def test_cast_matches_reference(monkeypatch, rows, mode, kernel, ulysses):
    _pin(monkeypatch, mode, kernel)
    qkv, wq, wk, cos, sin = _inputs(rows, seed=rows)
    ref_codes, ref_descale = _reference(qkv, wq, wk, cos, sin, mode)
    codes, descale = _run(qkv, wq, wk, cos, sin, mode, ulysses)
    _assert_codes(codes, ref_codes, f"{mode}/{kernel}/{ulysses}")
    assert torch.equal(descale.view(torch.int32), ref_descale.view(torch.int32)), (
        descale,
        ref_descale,
    )


@pytest.mark.parametrize("kernel", tuple(KERNELS))
@pytest.mark.parametrize("mode", ("dynamic", "qkstatic"))
@pytest.mark.parametrize("rows", ROWS)
def test_equal_rope_halves_option_keeps_the_codes(monkeypatch, rows, mode, kernel):
    """``rope_halves_equal`` skips loading the table's second half; ``_inputs`` repeats the angles."""
    _pin(monkeypatch, mode, kernel)
    qkv, wq, wk, cos, sin = _inputs(rows, seed=rows)
    half = cos.shape[-1] // 2
    assert torch.equal(cos[:, :half], cos[:, half:]) and torch.equal(sin[:, :half], sin[:, half:])
    ref_codes, ref_descale = _reference(qkv, wq, wk, cos, sin, mode)
    codes, descale = _run(qkv, wq, wk, cos, sin, mode, 0, rope_halves_equal=True)
    _assert_codes(codes, ref_codes, f"halves-equal/{mode}/{kernel}")
    assert torch.equal(descale.view(torch.int32), ref_descale.view(torch.int32))


@pytest.mark.parametrize(
    "mode,kernel,ulysses",
    (("dynamic", "split", 0), ("qkstatic", "grouped_fold", U)),
    ids=("dynamic-split-plain", "qkstatic-fold-packed"),
)
def test_padded_rows_leave_the_scale(monkeypatch, mode, kernel, ulysses):
    """Rows past ``valid_len`` are cast but do not enter the amax."""
    _pin(monkeypatch, mode, kernel)
    rows = 1017
    qkv, wq, wk, cos, sin = _inputs(rows, seed=7)
    valid = rows - rows // 10
    qkv[0, valid:] *= 1000.0  # padded rows far above the valid ones: they would set the scale
    valid_len = torch.tensor([valid], dtype=torch.int32, device="cuda")
    ref_codes, ref_descale = _reference(qkv, wq, wk, cos, sin, mode, valid_len)
    codes, descale = _run(qkv, wq, wk, cos, sin, mode, ulysses, valid_len)
    _assert_codes(codes, ref_codes, f"padded {mode}/{kernel}")
    assert torch.equal(descale.view(torch.int32), ref_descale.view(torch.int32)), (
        descale,
        ref_descale,
    )


def test_zero_tensor_takes_the_scale_floor(monkeypatch):
    _pin(monkeypatch, "dynamic", "grouped")
    qkv, wq, wk, cos, sin = _inputs(1017, seed=11)
    qkv[..., H * D : 2 * H * D] = 0.0  # K all zero: descale = bf16(1 / (448 * 512)), codes 0
    codes, descale = _run(qkv, wq, wk, cos, sin, "dynamic", 0)
    floor = torch.tensor(MIN_SCALE, dtype=torch.float32).to(torch.bfloat16).float().item()
    assert descale[1].item() == floor
    # Zero elements may carry the sign of the rotated sum (FP8 -0.0, byte 0x80): compare values.
    assert torch.equal(codes[1].float(), torch.zeros_like(codes[1].float()))


def _bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    """One BF16 step at |v| (8 significand bits), 2**-126 at zero."""
    _, e = torch.frexp(v.float().abs())
    return torch.where(
        v == 0,
        torch.full_like(v.float(), 2.0**-126),
        torch.ldexp(torch.ones_like(v.float()), e - 8),
    )


def _documented_norm_rope(x, w, cos, sin):
    """The formula of the module docstring in torch fp32, with the terms the bound needs."""
    xf = x.float()
    inv = torch.rsqrt((xf * xf).sum(-1, keepdim=True) * (1.0 / D) + EPS)
    yn = (xf * inv).to(torch.bfloat16).float()  # the one rounding after the normalise
    half = ROT // 2
    partner = torch.cat([yn[..., half:ROT], yn[..., :half], yn[..., ROT:]], dim=-1)
    w_d = w.float()
    w_p = torch.cat([w_d[half:ROT], w_d[:half], w_d[ROT:]])
    sign = torch.cat([-torch.ones(half), torch.ones(D - half)]).to(xf.device)
    c = torch.ones((x.shape[1], D), device=xf.device)
    sn = torch.zeros((x.shape[1], D), device=xf.device)
    c[:, :ROT] = cos.to(torch.bfloat16).float()
    sn[:, :ROT] = sin.to(torch.bfloat16).float()
    c, sn = c[None, :, None, :], sn[None, :, None, :]
    pre = yn * w_d * c + (sign * sn) * (partner * w_p)
    out = pre.to(torch.bfloat16)
    # Another correct fp32 evaluation may round yn or its partner one BF16 step the other way; that
    # moves the pre-rounding value by at most `moved`, and the final rounding of two values `moved`
    # apart differs by at most `moved` plus one step at the larger of them.
    moved = _bf16_ulp(yn) * (w_d * c).abs() + _bf16_ulp(partner) * (sn * w_p).abs()
    bound = moved + _bf16_ulp(pre.abs() + moved)
    return out, bound


@pytest.mark.parametrize("rows", ROWS)
def test_norm_rope_bf16_matches_documented_formula(rows):
    """The BF16 norm + RoPE value against an independent evaluation of the documented formula."""
    qkv, wq, wk, cos, sin = _inputs(rows, seed=rows + 1)
    q, k, _ = _planes(qkv)
    for x, w, name in ((q, wq, "Q"), (k, wk, "K")):
        expected, bound = _documented_norm_rope(x, w, cos, sin)
        actual = _norm_rope_bf16(x, w, cos, sin)
        diff = (actual.float() - expected.float()).abs()
        differ = int((diff != 0).sum())
        assert differ <= 1e-5 * diff.numel(), (name, differ, diff.numel())
        worst = int((diff - bound).argmax())
        assert bool((diff <= bound).all()), (
            name,
            float(diff.view(-1)[worst]),
            float(bound.view(-1)[worst]),
        )


@pytest.mark.parametrize("mode", ("dynamic", "qkstatic"))
def test_tensor_major_send_is_the_row_major_send_permuted(monkeypatch, mode):
    """The tensor-major producer layout ``[3, U, S, H / U, D]`` holds the row-major codes."""
    _pin(monkeypatch, mode, "grouped")
    rows, hl = 1017, H // U
    qkv, wq, wk, cos, sin = _inputs(rows, seed=13)
    static = fq.qk_static_descale(D, qkv.device) if mode == "qkstatic" else None
    block_h, num_warps = fq._launch_config()
    send_rm = torch.empty((U, rows, 3, hl, D), dtype=E4M3, device="cuda")
    send_tm = torch.empty((3, U, rows, hl, D), dtype=E4M3, device="cuda")
    for tensor_major, outs in (
        (False, (send_rm,) * 3),
        (True, (send_tm[0], send_tm[1], send_tm[2])),
    ):
        fq._qk_norm_rope_fp8_split(
            qkv,
            wq,
            wk,
            cos,
            sin,
            static,
            EPS,
            H,
            D,
            rows,
            rows,
            outs,
            block_h,
            num_warps,
            None,
            packed_hl=hl,
            qk_static=static is not None,
            tensor_major=tensor_major,
        )
    expected = send_rm.permute(2, 0, 1, 3, 4).contiguous()
    assert torch.equal(send_tm.view(torch.uint8), expected.view(torch.uint8))
