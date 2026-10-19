# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_prefill_conv_post entry.

Every cell is checked twice: against the entry's fp32/fp64 reference, and
against the three-call composition it replaces in the target
(`extract_transpose_prefill_slice` -> `causal_conv1d_fwd` ->
`fused_gdn_post_conv`), with the conv pool compared bitwise both times.
Geometry is a Qwen3.8-Flash-Next linear-attention layer: 16 x 128 key heads,
48 x 128 value heads, qkv_dim 10240, conv width 4, bf16 activations.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.causal_conv1d_fwd import (  # noqa: E501
    causal_conv1d_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.extract_transpose_prefill_slice import (  # noqa: E501
    extract_transpose_prefill_slice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.fused_gdn_post_conv import (  # noqa: E501
    fused_gdn_post_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_prefill_conv_post import (  # noqa: E501
    A_LOG_RANGE,
    CONV_WIDTH,
    DT_BIAS_RANGE,
    HEAD_K_DIM,
    HEAD_V_DIM,
    IN_PROJ_WIDTH,
    NUM_K_HEADS,
    NUM_V_HEADS,
    QKV_DIM,
    SOFTPLUS_THRESHOLD,
    _engine_slots,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_prefill_conv_post import (  # noqa: E501
    gdn_prefill_conv_post as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_prefill_conv_post requires a CUDA device"

DEV = "cuda"
DTYPE = torch.bfloat16
HEADS = (NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM)
PAD = -1
L2_NORM_EPS = 1e-6
#: `g`'s band, derived as in the fused_gdn_post_conv entry test: two SASS
#: approximate transcendentals (~2^-22 relative each), the softplus bounded by
#: the threshold, every term scaled by exp(A_log).
_APPROX_ULP = 2.0**-22
G_ATOL = 4 * _APPROX_ULP * SOFTPLUS_THRESHOLD * math.exp(A_LOG_RANGE[1])
A_COL = QKV_DIM + NUM_V_HEADS * HEAD_V_DIM


def _build(spec, seed):
    """`__call__`'s positional arguments for one cell."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lengths = spec["lengths"]
    n_tok = sum(lengths)
    layout = spec.get("layout", "in_proj")

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(DTYPE)

    if layout == "contiguous":
        x, a, b = randn(n_tok, QKV_DIM), randn(n_tok, NUM_V_HEADS), randn(n_tok, NUM_V_HEADS)
    else:
        proj = randn(n_tok + spec.get("extra_rows", 0), IN_PROJ_WIDTH)
        x = proj[:n_tok, :QKV_DIM]
        a = proj[:n_tok, A_COL : A_COL + NUM_V_HEADS]
        b = proj[:n_tok, A_COL + NUM_V_HEADS :]
        assert x.stride(0) == a.stride(0) == IN_PROJ_WIDTH
    weight = (torch.randn(QKV_DIM, CONV_WIDTH, generator=g, device=DEV) * 0.5).to(DTYPE)
    pool_slots = spec["pool_slots"]
    # The pool is a per-layer view of a two-layer buffer, as a cache manager
    # hands it out: slot stride twice the slot's own size.
    conv_states = randn(pool_slots, 2, QKV_DIM, CONV_WIDTH - 1)[:, 1]
    slots = spec.get("slots") or _engine_slots(len(lengths))
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    qsl = torch.tensor(bounds, dtype=torch.int32, device=DEV)
    si = torch.tensor(slots, dtype=torch.int32, device=DEV)
    hi = torch.tensor(spec["initial"], dtype=torch.bool, device=DEV)
    lo, hi_ = A_LOG_RANGE
    A_log = torch.rand(NUM_V_HEADS, generator=g, device=DEV) * (hi_ - lo) + lo
    lo, hi_ = DT_BIAS_RANGE
    dt_bias = torch.rand(NUM_V_HEADS, generator=g, device=DEV) * (hi_ - lo) + lo
    if "dt_bias_fill" in spec:
        dt_bias = torch.full((NUM_V_HEADS,), spec["dt_bias_fill"], device=DEV)
    return [
        x,
        weight,
        conv_states,
        qsl,
        si,
        hi,
        a,
        b,
        A_log,
        dt_bias,
        *HEADS,
        PAD,
        L2_NORM_EPS,
        SOFTPLUS_THRESHOLD,
    ]


def _unfused(args):
    """The target's previous three calls on a copy of the pool: `(q, k, v, g,
    beta, pool)`."""
    x, weight, conv_states, qsl, si, hi, a, b, A_log, dt_bias = args[:10]
    pool = conv_states.clone()
    xt = extract_transpose_prefill_slice(x, x.shape[0], 0, QKV_DIM)
    causal_conv1d_fwd(xt, weight, None, pool, qsl, si, hi, True, PAD, None)
    out = fused_gdn_post_conv(
        xt, None, a, b, A_log, dt_bias, *HEADS, L2_NORM_EPS, SOFTPLUS_THRESHOLD
    )
    return (*out, pool)


def _live_rows(args):
    """Token rows of the contexts whose slot is not the pad sentinel."""
    bounds, slots = args[3].tolist(), args[4].tolist()
    keep = torch.zeros(args[0].shape[0], dtype=torch.bool, device=DEV)
    for s, slot in enumerate(slots):
        if slot != PAD:
            keep[bounds[s] : bounds[s + 1]] = True
    return keep


def _compare(out, pool, want, rows=None, what=""):
    q, k, v, g, beta = out
    q_r, k_r, v_r, g_r, beta_r, pool_r = want
    for got, ref in zip(out, want[:5]):
        assert got.dtype == ref.dtype and got.shape == ref.shape and got.is_contiguous()
    sel = (lambda t: t[0, rows]) if rows is not None else (lambda t: t)
    for name, got, ref in (("q", q, q_r), ("k", k, k_r), ("v", v, v_r)):
        d = (sel(got).float() - sel(ref).float()).abs()
        flips = (d > 0).float().mean().item() if d.numel() else 0.0
        print(
            f"{what} {name}: max abs {d.max().item() if d.numel() else 0:.3e}, "
            f"differing fraction {flips:.2e}"
        )
        torch.testing.assert_close(sel(got), sel(ref))
    torch.testing.assert_close(g, g_r, rtol=8 * _APPROX_ULP, atol=G_ATOL)
    torch.testing.assert_close(beta, beta_r)
    assert torch.equal(pool, pool_r), "conv pool differs"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    pool0 = args[2].clone()
    want = op.reference(*args)
    old = _unfused([*args[:2], pool0.clone(), *args[3:]])
    inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]
    out = op(*args)
    for i, (arg, before) in enumerate(zip(args, inputs)):
        if isinstance(arg, torch.Tensor) and i != 2:
            assert torch.equal(arg, before), f"input {i} must be read-only"
    _compare(out, args[2], want, what=f"{cell.why[:30]} vs reference")
    _compare(out, args[2], old, rows=_live_rows(args), what=f"{cell.why[:30]} vs unfused")
    if PAD in args[4].tolist():
        pad_free = [s for s in range(pool0.shape[0]) if s not in args[4].tolist()]
        assert torch.equal(args[2][pad_free], pool0[pad_free]), "unnamed slots written"


def test_carry_and_silu_are_load_bearing() -> None:
    """The band separates a dropped carry and a dropped SiLU."""
    spec = dict(lengths=(5, 70), pool_slots=4, slots=(1, 3), initial=(True, True))
    args = _build(spec, seed=301)
    want = op.reference(*args)
    no_carry = list(args)
    no_carry[5] = torch.zeros_like(args[5])
    out = op(*[t.clone() if isinstance(t, torch.Tensor) else t for t in no_carry])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out[2], want[2])
    x, weight = args[0], args[1]
    # No SiLU: the pre-activation of the second context's last token, v channels.
    pre = sum(x[74 - 3 + j, :QKV_DIM].float() * weight[:, j].float() for j in range(4))
    pre = pre[2 * NUM_K_HEADS * HEAD_K_DIM :]
    got = op(*_build(spec, seed=301))[2][0, 74].reshape(-1).float()
    assert (got - torch.nn.functional.silu(pre)).abs().max() < 0.05
    assert (got - pre).abs().max() > 0.5


def test_guard_refuses_channel_strided_x() -> None:
    args = _build(dict(lengths=(16,), pool_slots=2, slots=(0,), initial=(True,)), seed=302)
    bad = list(args)
    bad[0] = torch.empty(QKV_DIM, 16, dtype=DTYPE, device=DEV).t()
    with pytest.raises(AssertionError, match="unit channel stride"):
        with validating(op):
            op(*bad)


def test_guard_refuses_short_a_b_and_strided_gates() -> None:
    args = _build(dict(lengths=(16, 8), pool_slots=2, slots=(0, 1), initial=(True, True)), seed=303)
    for position in (6, 7):
        bad = list(args)
        bad[position] = args[position][:16]
        with pytest.raises(AssertionError, match="at least T=24 rows"):
            with validating(op):
                op(*bad)
    bad = list(args)
    bad[8] = torch.stack([args[8], args[8]], dim=1)[:, 0]
    with pytest.raises(AssertionError, match="contiguous 1-D"):
        with validating(op):
            op(*bad)
