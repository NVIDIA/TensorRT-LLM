# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gdn_prefill_conv_post entry.

Every cell is checked against the entry's fp32/fp64 reference, the conv pool
bitwise. Geometry is a Qwen3.8-Flash-Next linear-attention layer, conv width 4, bf16
activations: the whole layer (16 x 128 key heads, 48 x 128 value heads, qkv_dim 10240,
a 16480-wide projection) at tp1 and on a dep4 rank, a quarter of it (4 / 12 heads,
qkv_dim 2560, a 4120-wide projection) on a tp4 or tep4 rank. The pool is a view of a
bigger buffer, which the test also checks the call leaves alone outside the view: a
two-layer buffer for the tp1 cells, the stock manager's page (36 GDN layers a slot,
the view at the last) at an engine's measured slot count for a rank's cells.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.gdn_prefill_conv_post__r27 import (  # noqa: E501
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
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ssm.gdn_prefill_conv_post__r27 import (  # noqa: E501
    gdn_prefill_conv_post as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gdn_prefill_conv_post requires a CUDA device"

DEV = "cuda"
DTYPE = torch.bfloat16
PAD = -1
L2_NORM_EPS = 1e-6
#: `g`'s band, derived as in the fused_gdn_post_conv entry test: two SASS
#: approximate transcendentals (~2^-22 relative each), the softplus bounded by
#: the threshold, every term scaled by exp(A_log).
_APPROX_ULP = 2.0**-22
G_ATOL = 4 * _APPROX_ULP * SOFTPLUS_THRESHOLD * math.exp(A_LOG_RANGE[1])


def _geometry(spec):
    """`(num_k_heads, num_v_heads, head_dim, qkv_dim, projection width)` of a cell: its
    own, or the tp1 layer's."""
    hk = spec.get("num_k_heads", NUM_K_HEADS)
    hv = spec.get("num_v_heads", NUM_V_HEADS)
    d = spec.get("head_dim", HEAD_K_DIM)
    assert d == HEAD_K_DIM == HEAD_V_DIM
    qkv = 2 * hk * d + hv * d
    width = spec.get("x_row_stride", IN_PROJ_WIDTH)
    assert width == qkv + hv * d + 2 * hv, (
        f"x_row_stride {width} is not the [q|k|v|z|a|b] projection of {hk}/{hv} heads"
    )
    return hk, hv, d, qkv, width


def _build(spec, seed):
    """`__call__`'s positional arguments for one cell."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lengths = spec["lengths"]
    n_tok = sum(lengths)
    hk, hv, d, qkv, width = _geometry(spec)
    a_col = qkv + hv * d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(DTYPE)

    # "in_proj" and "rows" both carve x, a and b out of one projection; "rows"
    # leaves `extra_rows` trailing rows the call must not read.
    proj = randn(n_tok + spec.get("extra_rows", 0), width)
    x = proj[:n_tok, :qkv]
    a = proj[:n_tok, a_col : a_col + hv]
    b = proj[:n_tok, a_col + hv : a_col + 2 * hv]
    assert x.stride(0) == a.stride(0) == width
    weight = (torch.randn(qkv, CONV_WIDTH, generator=g, device=DEV) * 0.5).to(DTYPE)
    pool_slots = spec["pool_slots"]
    # The pool is a per-layer view of a multi-layer buffer, as a cache manager
    # hands it out: two layers for the tp1 cells (slot stride twice the slot's own
    # size); for a rank's cells the stock page, `page_layers` (36) GDN layers'
    # states a slot, viewed at the last -- the strides and offset the manager's
    # `get_conv_states` returns, measured.
    layers = spec.get("page_layers", 2)
    conv_states = randn(pool_slots, layers, qkv, CONV_WIDTH - 1)[:, layers - 1]
    slots = spec.get("slots") or _engine_slots(len(lengths))
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    qsl = torch.tensor(bounds, dtype=torch.int32, device=DEV)
    si = torch.tensor(slots, dtype=torch.int32, device=DEV)
    hi = torch.tensor(spec["initial"], dtype=torch.bool, device=DEV)
    lo, hi_ = A_LOG_RANGE
    A_log = torch.rand(hv, generator=g, device=DEV) * (hi_ - lo) + lo
    lo, hi_ = DT_BIAS_RANGE
    dt_bias = torch.rand(hv, generator=g, device=DEV) * (hi_ - lo) + lo
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
        hk,
        d,
        hv,
        d,
        PAD,
        L2_NORM_EPS,
        SOFTPLUS_THRESHOLD,
        spec.get("linear_gate", False),
    ]


def _compare(out, pool, want, what=""):
    q, k, v, g, beta = out
    q_r, k_r, v_r, g_r, beta_r, pool_r = want
    for got, ref in zip(out, want[:5]):
        assert got.dtype == ref.dtype and got.shape == ref.shape and got.is_contiguous()
    for name, got, ref in (("q", q, q_r), ("k", k, k_r), ("v", v, v_r)):
        d = (got.float() - ref.float()).abs()
        flips = (d > 0).float().mean().item() if d.numel() else 0.0
        print(
            f"{what} {name}: max abs {d.max().item() if d.numel() else 0:.3e}, "
            f"differing fraction {flips:.2e}"
        )
        torch.testing.assert_close(got, ref)
    torch.testing.assert_close(g, g_r, rtol=8 * _APPROX_ULP, atol=G_ATOL)
    torch.testing.assert_close(beta, beta_r)
    assert torch.equal(pool, pool_r), "conv pool differs"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    want = op.reference(*args)
    inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in args]
    pool = args[2]
    page = pool._base
    page_before = page.clone()
    out = op(*args)
    for i, (arg, before) in enumerate(zip(args, inputs)):
        if isinstance(arg, torch.Tensor) and i != 2:
            assert torch.equal(arg, before), f"input {i} must be read-only"
    _compare(out, pool, want, what=f"{cell.why[:30]} vs reference")
    # The rest of the buffer -- the other layers' sub-pages of every slot -- is untouched.
    layer = (pool.storage_offset() - page.storage_offset()) // page.stride(1)
    page_before[:, layer] = want[5]
    assert torch.equal(page, page_before), "the call wrote outside its layer's view of the pool"


def test_guard_refuses_channel_strided_x() -> None:
    args = _build(dict(lengths=(16,), pool_slots=2, slots=(0,), initial=(True,)), seed=302)
    bad = list(args)
    bad[0] = torch.empty(QKV_DIM, 16, dtype=DTYPE, device=DEV).t()
    with pytest.raises(AssertionError, match="unit channel stride"):
        with validating(op):
            op(*bad)


def test_linear_gate_is_torch_exp_of_the_log_gate() -> None:
    """`linear_gate=True` stores `torch.exp` of the log-space decay bitwise, and leaves
    every other output and the conv pool as without it."""
    spec = next(c for c in op.CELLS if c.spec.get("linear_gate")).spec
    args = _build(spec, seed=77)
    log_args = list(args)
    log_args[-1] = False
    log_args[2] = args[2].clone()
    pool_lin = args[2]
    log_out = op(*log_args)
    lin_out = op(*args)
    assert torch.equal(lin_out[3], torch.exp(log_out[3]))
    for got, want in zip(lin_out[:3] + lin_out[4:], log_out[:3] + log_out[4:]):
        assert torch.equal(got, want)
    assert torch.equal(pool_lin, log_args[2])
