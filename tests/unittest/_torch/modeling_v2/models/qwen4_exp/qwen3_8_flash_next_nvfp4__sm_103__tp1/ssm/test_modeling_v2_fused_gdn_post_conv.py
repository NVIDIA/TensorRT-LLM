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
#: The `z` columns of the input projection.
Z_WIDTH = NUM_V_HEADS * HEAD_V_DIM
#: The target's one GDN input-projection output: [q|k|v | z | a | b].
IN_PROJ_WIDTH = QKV_DIM + Z_WIDTH + 2 * NUM_V_HEADS  # 16480


def _build(spec, seed):
    """`__call__`'s positional arguments (its keyword `beta_dtype` left at fp32)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_p, n_d = spec["num_prefill"], spec["num_decode"]
    n = n_p + n_d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(DTYPE)

    # `_gdn_block`: one [T, 16480] GEMM output; q|k|v, a and b are its column
    # views, the decode half the row split of q|k|v, the prefill half the
    # transposing extraction of the rest.
    proj = randn(n, IN_PROJ_WIDTH)
    mixed = proj[:, :QKV_DIM]
    prefill = mixed[:n_p].t().contiguous()
    decode = mixed[n_p:] if n_d else None
    a = proj[:, QKV_DIM + Z_WIDTH : QKV_DIM + Z_WIDTH + NUM_V_HEADS]
    b = proj[:, QKV_DIM + Z_WIDTH + NUM_V_HEADS :]
    assert a.stride(0) == b.stride(0) == IN_PROJ_WIDTH
    assert decode is None or decode.stride() == (IN_PROJ_WIDTH, 1)

    lo, hi = A_LOG_RANGE
    A_log = torch.rand(NUM_V_HEADS, generator=g, device=DEV) * (hi - lo) + lo
    lo, hi = DT_BIAS_RANGE
    dt_bias = torch.rand(NUM_V_HEADS, generator=g, device=DEV) * (hi - lo) + lo
    if "dt_bias_fill" in spec:
        dt_bias = torch.full((NUM_V_HEADS,), spec["dt_bias_fill"], device=DEV)
    return [prefill, decode, a, b, A_log, dt_bias, *HEADS, L2_NORM_EPS, SOFTPLUS_THRESHOLD]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    if "dt_bias_fill" in cell.spec:
        # Every head strictly on the side of the threshold the cell names.
        x = args[2].float() + args[5].float()
        above = cell.spec["dt_bias_fill"] > SOFTPLUS_THRESHOLD
        assert bool((x > SOFTPLUS_THRESHOLD).all() if above else (x <= SOFTPLUS_THRESHOLD).all())
    before = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]
    out = op(*args)
    for arg, old in zip(args, before):
        if isinstance(arg, torch.Tensor):
            assert torch.equal(arg, old), "inputs are read-only"
    _compare(out, op.reference(*args))


def test_guard_refuses_strided_gate_parameters() -> None:
    """Unguarded, a stride-2 `A_log` view is read as `ptr + head` -- the
    interleaved zeros -- and returns a finite wrong g."""
    args = _build(dict(num_prefill=32, num_decode=0), seed=108)
    strided = torch.stack([args[4], torch.zeros_like(args[4])], dim=1)[:, 0]
    assert strided.stride() == (2,)
    bad = list(args)
    bad[4] = strided
    with pytest.raises(AssertionError):
        _compare(op(*bad), op.reference(*bad))
    with pytest.raises(AssertionError, match="must be contiguous 1-D"):
        with validating(op):
            op(*bad)
