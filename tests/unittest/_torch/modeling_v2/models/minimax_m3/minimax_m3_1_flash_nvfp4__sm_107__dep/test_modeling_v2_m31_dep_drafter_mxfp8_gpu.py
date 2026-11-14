# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_DRAFTER_MXFP8 on an SM107 GPU, at the DSpark drafter's shapes (hidden 6144, 64 + 2 x 4 heads of 128, FFN 12288).

A routed projection is the reference MXFP8 product (``trtllm::mxfp8_quantize`` of the rows, dequantized, times the
exactly recovered MXFP8 weight, in fp32) up to the GEMM's accumulation order and one BF16 rounding, and differs from
the BF16 GEMM by the MXFP8 activation alone (rel-L2 inside a band); the routed MLP is the same reference with the
SwiGLU (``silu_and_mul`` constants) quantized; graph replay is the eager output bit for bit; rows outside the band and
TRTLLM_M31_DISABLE_DEP_DRAFTER_MXFP8=1 are the BF16 path bit for bit.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    drafter_mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import mxfp8_recover


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="needs an SM107 GPU")

HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM, FFN = 6144, 64, 4, 128, 12288
QKV = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
ROWS = (33, 42, 77, 112, 128)
#: The routed output vs the fp32 MXFP8 reference: accumulation order + one BF16 rounding.
REF_TOL = 4e-3
#: The routed output vs the BF16 GEMM: the MXFP8 activation's error (MB: 2.66e-2..2.68e-2).
BF16_BAND = (5e-3, 6e-2)


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def _mxfp8_weight(n, k, gen):
    """A checkpoint-like BF16 weight that is exactly an MXFP8 dequantization: N(0, 1 / K) rounded to E4M3 codes at
    each 32-column block's power-of-two scale (so x @ W.T keeps the drafter's activation scale and the SwiGLU-OAI clamp
    at 7 is not saturated)."""
    w = torch.randn((n, k), generator=gen, device="cuda") * k**-0.5
    blocks = w.view(n, k // 32, 32)
    exp = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(2.0**-126) / 448.0))
    scale = torch.exp2(exp).unsqueeze(-1)
    codes = (blocks / scale).to(torch.float8_e4m3fn).float()
    return (codes * scale).view(n, k).to(torch.bfloat16)


def _dequant_act(x):
    """fp32 rows of trtllm::mxfp8_quantize(x, True)."""
    q, sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    rows, k = x.shape
    s = mxfp8_recover.unswizzle_sf(sf, rows, k // 32).float()
    return (
        q.view(torch.float8_e4m3fn).float().view(rows, k // 32, 32)
        * torch.exp2(s - 127.0).unsqueeze(-1)
    ).view(rows, k)


def _ref(x, w):
    return (_dequant_act(x) @ w.float().t()).to(torch.bfloat16)


class _Linear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.tp_size = 1

    def apply_linear(self, input, bias, lora_params=None, layer_idx=None):
        return F.linear(input, self.weight)

    def forward(self, input, all_reduce_params=None, layer_idx=None):
        return self.apply_linear(input, self.bias, None, layer_idx)


def _swiglu_oai(h1):
    return torch.ops.trtllm.silu_and_mul(h1, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)


class _MLP(torch.nn.Module):
    def __init__(self, gate_up, down):
        super().__init__()
        self.gate_up_proj, self.down_proj = gate_up, down
        self.split_gate_up = False
        self.activation = F.silu
        self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit = 1.702, 1.0, 7.0
        self.layer_idx = 0

    def _apply_activation(self, h1):
        return _swiglu_oai(h1)

    def forward(
        self, x, all_rank_num_tokens=None, final_all_reduce_params=None, lora_params=None, **kwargs
    ):
        return self.down_proj(self._apply_activation(self.gate_up_proj(x)))


def _drafter(gen):
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    attn = types.SimpleNamespace(
        qkv_proj=_Linear(_mxfp8_weight(QKV, HIDDEN, gen)),
        o_proj=_Linear(_mxfp8_weight(HIDDEN, Q_HEADS * HEAD_DIM, gen)),
    )
    mlp = _MLP(
        _Linear(_mxfp8_weight(2 * FFN, HIDDEN, gen)), _Linear(_mxfp8_weight(HIDDEN, FFN, gen))
    )
    return types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn, mlp=mlp)])
    )


@pytest.fixture(scope="module")
def drafter():
    gen = torch.Generator(device="cuda").manual_seed(20261101)
    draft = _drafter(gen)
    stock = _drafter(torch.Generator(device="cuda").manual_seed(20261101))
    return draft, stock, gen


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(drafter_mxfp8, "DEP_MXDRAFT", True)


def test_install_routes_the_drafter_shapes(drafter, on):
    draft, _, _ = drafter
    assert drafter_mxfp8.install(draft) == (4, 1)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("kind", ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
def test_routed_projection_is_the_mxfp8_reference(drafter, on, kind, rows):
    draft, stock, gen = drafter
    drafter_mxfp8.install(draft)
    layer, ref_layer = draft.model.layers[0], stock.model.layers[0]
    owner = layer.self_attn if kind in ("qkv_proj", "o_proj") else layer.mlp
    ref_owner = ref_layer.self_attn if kind in ("qkv_proj", "o_proj") else ref_layer.mlp
    lin, ref_lin = getattr(owner, kind), getattr(ref_owner, kind)
    assert torch.equal(lin.weight, ref_lin.weight)
    x = (torch.randn(rows, lin.weight.shape[1], device="cuda", generator=gen) * 0.5).to(
        torch.bfloat16
    )
    out = lin(x)
    got = _rel(out, _ref(x, lin.weight))
    assert got < REF_TOL, (kind, rows, got)
    low, high = BF16_BAND
    got = _rel(out, ref_lin(x))
    assert low < got < high, (kind, rows, got)


@pytest.mark.parametrize("rows", ROWS)
def test_routed_mlp_is_the_mxfp8_reference(drafter, on, rows):
    draft, stock, gen = drafter
    drafter_mxfp8.install(draft)
    mlp, ref_mlp = draft.model.layers[0].mlp, stock.model.layers[0].mlp
    x = (torch.randn(rows, HIDDEN, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    out = mlp(x)
    h1 = mlp.gate_up_proj(x)  # the routed gate_up (deterministic): isolates the down chain
    ref = _ref(_swiglu_oai(h1), mlp.down_proj.weight)
    got = _rel(out, ref)
    assert got < REF_TOL, (rows, got)
    low, high = BF16_BAND
    got = _rel(out, ref_mlp(x))
    assert low < got < high, (rows, got)


@pytest.mark.parametrize("rows", [32, 129, 256])
def test_rows_outside_the_band_are_the_bf16_path(drafter, on, rows):
    draft, stock, gen = drafter
    drafter_mxfp8.install(draft)
    layer, ref_layer = draft.model.layers[0], stock.model.layers[0]
    x = (torch.randn(rows, HIDDEN, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    assert torch.equal(layer.self_attn.qkv_proj(x), ref_layer.self_attn.qkv_proj(x))
    assert torch.equal(layer.mlp(x), ref_layer.mlp(x))


@pytest.mark.parametrize("rows", [42, 77, 112])
def test_graph_replay_is_the_eager_output(drafter, on, rows):
    draft, _, gen = drafter
    drafter_mxfp8.install(draft)
    layer = draft.model.layers[0]
    x = (torch.randn(rows, HIDDEN, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)

    def step():
        return layer.mlp(layer.self_attn.qkv_proj(x)[:, :HIDDEN].contiguous())

    eager = step()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        step()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = step()
    torch.cuda.synchronize()
    for _ in range(3):
        out.zero_()  # the capture already wrote the eager bits: only a replay that writes can pass
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16)), rows


def test_taken_lines_name_the_projection_and_rows(drafter, on):
    draft, _, gen = drafter
    drafter_mxfp8.install(draft)
    x = (torch.randn(77, HIDDEN, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    draft.model.layers[0].mlp(x)
    assert ("gate_up_proj", 77) in drafter_mxfp8._LOGGED and (
        "down_proj (swiglu_mxfp8)",
        77,
    ) in drafter_mxfp8._LOGGED


def test_knob_off_is_todays_bf16_bitwise(monkeypatch):
    monkeypatch.setattr(drafter_mxfp8, "DEP_MXDRAFT", False)
    gen = torch.Generator(device="cuda").manual_seed(7)
    draft = _drafter(gen)
    stock = _drafter(torch.Generator(device="cuda").manual_seed(7))
    assert drafter_mxfp8.install(draft) == (0, 0)
    for rows in (42, 77, 112):
        x = (torch.randn(rows, HIDDEN, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
        layer, ref = draft.model.layers[0], stock.model.layers[0]
        assert torch.equal(layer.self_attn.qkv_proj(x), ref.self_attn.qkv_proj(x))
        assert torch.equal(
            layer.self_attn.o_proj(x[:, : Q_HEADS * HEAD_DIM // 4].repeat(1, 4).contiguous()),
            ref.self_attn.o_proj(x[:, : Q_HEADS * HEAD_DIM // 4].repeat(1, 4).contiguous()),
        )
        assert torch.equal(layer.mlp(x), ref.mlp(x))
