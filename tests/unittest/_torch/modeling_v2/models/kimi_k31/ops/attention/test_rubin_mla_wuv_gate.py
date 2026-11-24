# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_mla_wuv_gate, W_UV with the MLA output gate in its epilogue.

CPU: the reference (the BF16 value heads, zero past the attention's rows, times ``gate.sigmoid()`` in BF16) on
hand-checked values, the guards. GPU (sm_107): every cell bit for bit against the two-pass form the dep16 target
ran before (``bmm_out`` into the head columns, the padding rows zeroed, ``k31_sigmoid_gate_mul``), and ``o_proj``'s
output on both bit for bit; against the reference within one BF16 step of the product (it sums in another order).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_mla_wuv_gate import (  # noqa: E501
    rubin_mla_wuv_gate as op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

# --- CPU -------------------------------------------------------------------------------------


def test_reference_gates_the_bf16_heads():
    heads, lora, v = 2, 128, 16
    lat = torch.zeros(3, 4, lora, dtype=torch.bfloat16)
    lat[:, :, 0] = 1.0
    w = torch.zeros(heads, lora, v, dtype=torch.bfloat16)
    w[:, 0] = torch.arange(heads * v, dtype=torch.float32).view(heads, v).to(torch.bfloat16)
    gate = torch.full((4, heads * v), 0.0, dtype=torch.bfloat16)  # sigmoid(0) = 0.5
    out = torch.full((4, heads * v), float("nan"), dtype=torch.bfloat16)
    ref = op.reference(out, lat, w, gate)
    want = torch.arange(heads * v, dtype=torch.float32).to(torch.bfloat16) * 0.5
    assert torch.equal(ref[:3], want.expand(3, -1))
    assert (ref[3] == 0).all() and not torch.signbit(ref[3]).any(), "padding rows are gated +0"
    assert out.isnan().all(), "the reference leaves out untouched"


def test_guards():
    out = torch.zeros(4, 2 * 128, dtype=torch.bfloat16)
    lat = torch.zeros(3, 4, 128, dtype=torch.bfloat16)
    w = torch.zeros(2, 128, 128, dtype=torch.bfloat16)
    g = torch.zeros(4, 256, dtype=torch.bfloat16)
    op.is_valid(out, lat, w, g)
    op.is_valid(out, lat, w.transpose(1, 2).contiguous().transpose(1, 2), g)
    with pytest.raises(AssertionError):
        op.is_valid(out.float(), lat, w, g)
    with pytest.raises(AssertionError):
        op.is_valid(out, torch.zeros(5, 4, 128, dtype=torch.bfloat16), w, g)
    with pytest.raises(AssertionError):
        op.is_valid(out, lat, w, torch.zeros(4, 240, dtype=torch.bfloat16))
    with pytest.raises(AssertionError, match="unit column"):
        op.is_valid(out, lat, w, torch.zeros(4, 512, dtype=torch.bfloat16)[:, ::2])
    with pytest.raises(AssertionError, match="64-wide"):
        op.is_valid(
            out,
            torch.zeros(3, 4, 32, dtype=torch.bfloat16),
            torch.zeros(2, 32, 128, dtype=torch.bfloat16),
            g,
        )
    with pytest.raises(AssertionError, match="128-wide"):
        op.is_valid(
            torch.zeros(4, 2 * 16, dtype=torch.bfloat16),
            lat,
            torch.zeros(2, 128, 16, dtype=torch.bfloat16),
            torch.zeros(4, 32, dtype=torch.bfloat16),
        )


# --- GPU -------------------------------------------------------------------------------------


def _inputs(spec, gen, g_scale):
    """The FMHA's latent output ``[rows_lat, pad * lora]`` (the real heads a view), W_UV as the target holds it
    (the transposed view of kv_b's v rows), the gate ``[rows, heads * v]`` and o_proj's weight."""
    rows, rows_lat, pad, heads = spec["rows"], spec["rows_lat"], spec["pad"], spec["heads"]
    lora, v = spec["lora"], spec["v"]
    lat = torch.randn(rows_lat, pad * lora, generator=gen, device="cuda").to(torch.bfloat16)
    lat96 = lat.view(rows_lat, pad, lora)[:, :heads]
    v_b = (torch.randn(heads, v, lora, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    gate = (torch.randn(rows, heads * v, generator=gen, device="cuda") * g_scale).to(torch.bfloat16)
    w_o = (torch.randn(1024, heads * v, generator=gen, device="cuda") * 0.01).to(torch.bfloat16)
    return lat96, v_b.transpose(1, 2), gate, w_o


def _two_pass(lat96, w_uv, gate):
    """The dep16 target's context rows before this entry: W_UV into the head columns, the padding rows zeroed,
    then the gate."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.sigmoid_gate_mul import (  # noqa: E501
        k31_sigmoid_gate_mul,
    )

    rows, rows_lat = gate.shape[0], lat96.shape[0]
    heads, _, v = w_uv.shape
    out = gate.new_empty(rows, heads * v)
    bmm_out(lat96.transpose(0, 1), w_uv, out[:rows_lat].view(rows_lat, heads, v).transpose(0, 1))
    if rows > rows_lat:
        out[rows_lat:].zero_()
    return k31_sigmoid_gate_mul(out, gate)


def _o_proj(x, w):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import (
        rubin_linear,
    )

    return rubin_linear(x, w, False)


@on_sm107
@pytest.mark.parametrize("g_scale", (3.0, 60.0), ids=("typical", "wide_gate"))
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, g_scale, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    lat96, w_uv, gate, w_o = _inputs(
        s, torch.Generator(device="cuda").manual_seed(s["rows"]), g_scale
    )
    ref = op.reference(torch.empty_like(gate), lat96, w_uv, gate)
    old = _two_pass(lat96, w_uv, gate)
    out = torch.full_like(gate, float("nan"))
    res = op(out, lat96, w_uv, gate)
    assert res.data_ptr() == out.data_ptr()
    assert torch.equal(out.view(torch.int16), old.view(torch.int16)), "not the two-pass bytes"
    assert torch.equal(_o_proj(out, w_o).view(torch.int16), _o_proj(old, w_o).view(torch.int16))
    # The reference sums in FP32 in another order, and the tensor cores add 16 products at a time: the head
    # value may move by a few FP32 ulps of the terms' magnitude (a near-cancelling sum by more of itself), then
    # by one BF16 step (2^-8 to 2^-7 relative) when rounded, which the gate scales by at most 1 before the
    # product's own rounding (half a step).
    a, b = out.float(), ref.float()
    heads, _, v = w_uv.shape
    mag = torch.zeros_like(a)
    mag[: lat96.shape[0]] = (
        torch.bmm(lat96.transpose(0, 1).abs().float(), w_uv.abs().float())
        .transpose(0, 1)
        .reshape(-1, heads * v)
    )
    step = torch.maximum(a.abs(), b.abs()).mul(2.0**-6) + mag.mul(2.0**-12) + 2.0**-126
    assert ((a - b).abs() <= step).all()


@on_sm107
def test_strided_gate_and_guard():
    """The gate as a column slice of a wider row-major buffer (a packed projection's columns)."""
    s = op.CELLS[2].spec
    gen = torch.Generator(device="cuda").manual_seed(11)
    lat96, w_uv, gate, _ = _inputs(s, gen, 3.0)
    wide = torch.randn(s["rows"], gate.shape[1] + 256, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    wide[:, : gate.shape[1]] = gate
    out = torch.empty_like(gate)
    with validating(op):
        op(out, lat96, w_uv, wide[:, : gate.shape[1]])
    assert torch.equal(out.view(torch.int16), _two_pass(lat96, w_uv, gate).view(torch.int16))
