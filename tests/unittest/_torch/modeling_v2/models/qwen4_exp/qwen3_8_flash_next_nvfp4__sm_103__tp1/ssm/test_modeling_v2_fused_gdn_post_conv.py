# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_gdn_post_conv entry.

Geometry is a Qwen3.8-Flash-Next linear-attention layer: 16 x 128 key heads,
48 x 128 value heads, qkv_dim 10240, bf16 activations.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_post_conv import (  # noqa: E501
    A_LOG_RANGE,
    DT_BIAS_RANGE,
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    QKV_DIM,
    SOFTPLUS_THRESHOLD,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_post_conv import (  # noqa: E501
    fused_gdn_post_conv as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_gdn_post_conv requires a CUDA device"


#: Relative error of the SASS approximate transcendentals (`ex2.approx.f32`,
#: `lg2.approx.f32`) Triton lowers `tl.exp` / `tl.log` to.
_APPROX_ULP = 2.0**-22


def _compare(out: tuple, ref: tuple) -> None:
    """Per output, each gate derived rather than tuned.

    q, k -- torch's bf16 band: both sides round one fp32 value, so they
    differ by at most one bf16 ulp. v -- bitwise: the kernel moves bits.
    g -- a product of two SASS approximate transcendentals (relative error
    ~2^-22 each) with the softplus magnitude bounded by the threshold and
    every term scaled by `exp(A_log)`:
    `atol = 4 * 2^-22 * softplus_threshold * exp(max A_log)`, `rtol = 8 * 2^-22`.
    At this checkpoint's A_log ceiling that is 6.7e-04 against a measured
    worst case of 7.1e-05; five wrong gate formulas land at 1.2e+02 and up.
    beta -- fp32: torch's fp32 band (one well-conditioned transcendental);
    narrower: bitwise, the fp32 sigmoid rounded exactly once.
    """
    assert len(out) == len(ref) == 5
    for got, want in zip(out, ref):
        assert got.dtype == want.dtype and got.shape == want.shape, (
            (got.dtype, got.shape),
            (want.dtype, want.shape),
        )
        assert got.is_contiguous()
    q, k, v, g, beta = out
    q_ref, k_ref, v_ref, g_ref, beta_ref = ref
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    assert torch.equal(v, v_ref), "v must be a bit-exact copy"
    g_atol = 4 * _APPROX_ULP * SOFTPLUS_THRESHOLD * math.exp(A_LOG_RANGE[1])
    torch.testing.assert_close(g, g_ref, rtol=8 * _APPROX_ULP, atol=g_atol)
    if beta.dtype == torch.float32:
        torch.testing.assert_close(beta, beta_ref)
    else:
        assert torch.equal(beta, beta_ref), "beta must be the fp32 sigmoid rounded once"


DEV = "cuda"
DTYPE = torch.bfloat16
HEADS = (NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM)
L2_NORM_EPS = 1e-6
#: The stock mixer's `z` half of `in_proj_qkvz`, and its `[b|a]` projection width.
Z_WIDTH = NUM_V_HEADS * HEAD_V_DIM
BA_WIDTH = 2 * NUM_V_HEADS


def _build(spec, seed):
    """`__call__`'s positional arguments, then its keyword `beta_dtype`."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_p, n_d = spec["num_prefill"], spec["num_decode"]
    n = n_p + n_d
    layout = spec.get("layout", "contiguous")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(DTYPE)

    if layout == "production":
        mixed = randn(n, QKV_DIM + Z_WIDTH)[:, :QKV_DIM]
        prefill, decode = mixed[:n_p].t().contiguous(), mixed[n_p:]
        ba = randn(n, BA_WIDTH)
        b, a = ba[:, :NUM_V_HEADS], ba[:, NUM_V_HEADS:]
        assert not decode.is_contiguous() and not a.is_contiguous()
    elif layout == "target_mixed":
        # `_gdn_block`: mixed_qkv = x @ w_qkv is [T, 10240]; the decode half is
        # its row split, the prefill half the transposing extraction of the rest.
        mixed = randn(n, QKV_DIM)
        prefill, decode = mixed[:n_p].t().contiguous(), mixed[n_p:]
        assert decode.stride() == (QKV_DIM, 1) and decode.storage_offset() == n_p * QKV_DIM
        a, b = randn(n, NUM_V_HEADS), randn(n, NUM_V_HEADS)
    elif layout == "prefill_view":
        prefill = randn(n_p, QKV_DIM + Z_WIDTH)[:, :QKV_DIM].t()
        assert prefill.stride() == (1, QKV_DIM + Z_WIDTH)
        decode = None
        a, b = randn(n, NUM_V_HEADS), randn(n, NUM_V_HEADS)
    else:
        prefill = randn(QKV_DIM, n_p)
        decode = randn(n_d, QKV_DIM) if n_d else None
        a, b = randn(n, NUM_V_HEADS), randn(n, NUM_V_HEADS)

    gate_dtype = spec.get("gate_dtype", torch.float32)
    lo, hi = A_LOG_RANGE
    A_log = (torch.rand(NUM_V_HEADS, generator=g, device=DEV) * (hi - lo) + lo).to(gate_dtype)
    lo, hi = DT_BIAS_RANGE
    dt_bias = (torch.rand(NUM_V_HEADS, generator=g, device=DEV) * (hi - lo) + lo).to(gate_dtype)
    if "dt_bias_fill" in spec:
        dt_bias = torch.full((NUM_V_HEADS,), spec["dt_bias_fill"], device=DEV).to(gate_dtype)
    args = [prefill, decode, a, b, A_log, dt_bias, *HEADS, L2_NORM_EPS, SOFTPLUS_THRESHOLD]
    return args, spec.get("beta_dtype", torch.float32)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, beta_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    if "dt_bias_fill" in cell.spec:
        # Every head strictly on the side of the threshold the cell names.
        x = args[2].float() + args[5].float()
        above = cell.spec["dt_bias_fill"] > SOFTPLUS_THRESHOLD
        assert bool((x > SOFTPLUS_THRESHOLD).all() if above else (x <= SOFTPLUS_THRESHOLD).all())
    before = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]
    out = op(*args, beta_dtype=beta_dtype)
    for arg, old in zip(args, before):
        if isinstance(arg, torch.Tensor):
            assert torch.equal(arg, old), "inputs are read-only"
    _compare(out, op.reference(*args, beta_dtype=beta_dtype))


def test_g_gate_discriminates_wrong_variants() -> None:
    """`g`'s floor is derived, so prove it still separates. Each variant is a
    plausible way to get the gate wrong, and must land >1e4x outside it."""
    args, _ = _build(dict(num_prefill=96, num_decode=0), seed=106)
    prefill, decode, a, b, A_log, dt_bias = args[:6]
    out = op(*args)
    ref = op.reference(*args)
    _compare(out, ref)
    g_ref = ref[3].double()
    atol = 4 * 2.0**-22 * SOFTPLUS_THRESHOLD * math.exp(A_LOG_RANGE[1])
    observed = (out[3].double() - g_ref).abs().max().item()
    assert observed < atol, f"correct result {observed} outside its own gate {atol}"

    x = a.double() + dt_bias.double()
    softplus = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)

    def g_of(*overrides):
        return op.reference(*[*overrides, *args[len(overrides) :]])[3].double()

    variants = {
        "sign flipped": -g_ref,
        "dt_bias omitted": g_of(prefill, decode, a, b, A_log, torch.zeros_like(dt_bias)),
        "A_log not exponentiated": (-A_log.double() * softplus).unsqueeze(0),
        "softplus replaced by identity": (-torch.exp(A_log.double()) * x).unsqueeze(0),
        "a and b swapped": g_of(prefill, decode, b, a),
    }
    for name, variant in variants.items():
        distance = (variant - g_ref).abs().max().item()
        assert distance > 1e4 * atol, f"{name} lands {distance / atol:.1f}x the gate"
        with pytest.raises(AssertionError):
            _compare((*out[:3], variant.float(), out[4]), ref)


def test_qk_normalization_is_not_a_pass_through() -> None:
    """The default bf16 band on q/k separates a missing norm and a q/k swap."""
    args, _ = _build(dict(num_prefill=96, num_decode=0), seed=107)
    out = op(*args)
    ref = op.reference(*args)
    _compare(out, ref)
    q_raw = args[0].t()[:, : NUM_K_HEADS * HEAD_K_DIM].reshape(1, -1, NUM_K_HEADS, HEAD_K_DIM)
    for wrong in ((q_raw.contiguous(), *out[1:]), (out[1], *out[1:])):
        with pytest.raises(AssertionError):
            _compare(wrong, ref)


def test_guard_refuses_strided_gate_parameters() -> None:
    """Unguarded, a stride-2 view is read as `ptr + head` -- the interleaved
    zeros -- and returns a finite wrong g."""
    args, _ = _build(dict(num_prefill=32, num_decode=0), seed=108)
    for position in (4, 5):
        strided = torch.stack([args[position], torch.zeros_like(args[position])], dim=1)[:, 0]
        assert strided.stride() == (2,)
        bad = list(args)
        bad[position] = strided
        with pytest.raises(AssertionError):
            _compare(op(*bad), op.reference(*bad))
        with pytest.raises(AssertionError, match="must be contiguous 1-D"):
            with validating(op):
                op(*bad)


def test_guard_refuses_short_gate_parameters_and_a_b() -> None:
    """Each of these is an unmasked out-of-bounds read, so the hazard is not
    replayed here; the contract's measurements are in the entry's note."""
    args, _ = _build(dict(num_prefill=32, num_decode=8), seed=109)
    for position, match in ((4, "at least num_v_heads"), (5, "at least num_v_heads")):
        bad = list(args)
        bad[position] = args[position][: NUM_V_HEADS - 8].clone()
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(*bad)
    for position in (2, 3):
        bad = list(args)
        # 32 rows cover the prefill block but not prefill + decode = 40.
        bad[position] = args[position][:32].clone()
        with pytest.raises(AssertionError, match="at least num_tokens=40"):
            with validating(op):
                op(*bad)
