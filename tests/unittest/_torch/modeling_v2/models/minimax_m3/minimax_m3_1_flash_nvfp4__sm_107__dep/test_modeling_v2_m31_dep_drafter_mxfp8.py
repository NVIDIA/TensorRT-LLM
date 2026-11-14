# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_DRAFTER_MXFP8 on CPU tensors and stub drafter layers.

The install routes each drafter projection whose BF16 weight is an exact MXFP8 dequantization to the MXFP8 GEMM for the
rows in MIN_ROWS..MAX_ROWS, and each MLP whose down_proj is routed to the swiglu_mxfp8 down chain; everything else is
the stock path. The quantize, GEMM and SwiGLU kernels are replaced by recorders here; their numerics are the GPU tests.
The family recovery helper is the TP8 target's own (shared, unchanged). CPU only.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import mxfp8_recover

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        drafter_mxfp8,
        m31_switches,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

HIDDEN = 256
INTER = 512


def _mxfp8_weight(n, k, seed=0):
    """A BF16 weight that is exactly an MXFP8 dequantization (random E4M3 codes, UE8M0 scales 2^-11..2^-7)."""
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.int16).to(torch.uint8)
    codes = codes.view(torch.float8_e4m3fn).float()
    codes = torch.nan_to_num(codes, nan=0.0)
    exp = torch.randint(-11, -6, (n, k // 32), generator=gen).float()
    w = (codes.view(n, k // 32, 32) * torch.exp2(exp).unsqueeze(-1)).view(n, k)
    return w.to(torch.bfloat16)


class _Linear(torch.nn.Module):
    """The slice of tensorrt_llm Linear the install touches: weight, bias, apply_linear, forward through it."""

    def __init__(self, weight, bias=None, tp_size=1):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = bias
        self.tp_size = tp_size

    def apply_linear(self, input, bias, lora_params=None, layer_idx=None):
        return F.linear(input, self.weight)

    def forward(self, input, all_reduce_params=None, layer_idx=None):
        return self.apply_linear(input, self.bias, None, layer_idx)


class _MLP(torch.nn.Module):
    """The slice of GatedMLP the install touches (fused gate_up, SiLU SwiGLU-OAI constants, stock forward)."""

    def __init__(self, gate_up, down):
        super().__init__()
        self.gate_up_proj, self.down_proj = gate_up, down
        self.split_gate_up = False
        self.activation = F.silu
        self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit = 1.702, 1.0, 7.0
        self.layer_idx = 0
        self.stock_calls = 0

    def _apply_activation(self, h1):
        gate, up = h1.float().chunk(2, dim=-1)
        return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(h1.dtype)

    def forward(
        self, x, all_rank_num_tokens=None, final_all_reduce_params=None, lora_params=None, **kwargs
    ):
        self.stock_calls += 1
        return self.down_proj(self._apply_activation(self.gate_up_proj(x)))


def _draft(layers=2, mxfp8=True):
    def w(n, k, seed):
        return _mxfp8_weight(n, k, seed) if mxfp8 else (torch.randn(n, k) * 0.37).to(torch.bfloat16)

    built = []
    for i in range(layers):
        attn = types.SimpleNamespace(
            qkv_proj=_Linear(w(3 * HIDDEN, HIDDEN, 10 * i)),
            o_proj=_Linear(w(HIDDEN, HIDDEN, 10 * i + 1)),
        )
        mlp = _MLP(_Linear(w(2 * INTER, HIDDEN, 10 * i + 2)), _Linear(w(HIDDEN, INTER, 10 * i + 3)))
        built.append(types.SimpleNamespace(self_attn=attn, mlp=mlp))
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=built))


@pytest.fixture
def rec(monkeypatch):
    """Recorders in place of the kernels; ``served`` admits CPU rows in the band."""
    calls = []
    monkeypatch.setattr(drafter_mxfp8, "DEP_MXDRAFT", True)
    monkeypatch.setattr(drafter_mxfp8, "_LOGGED", set())
    monkeypatch.setattr(
        drafter_mxfp8,
        "served",
        lambda x: (
            isinstance(x, torch.Tensor)
            and x.dim() == 2
            and drafter_mxfp8.MIN_ROWS <= x.shape[0] <= drafter_mxfp8.MAX_ROWS
        ),
    )

    def quantize_gemm(x, w8, sf):
        calls.append(("quantize_gemm", x.shape[0], tuple(w8.shape)))
        return torch.full((x.shape[0], w8.shape[0]), 7.0, dtype=torch.bfloat16)

    def gemm(q, sf, w8, w_sf):
        calls.append(("gemm", q.shape[0], tuple(w8.shape)))
        return torch.full((q.shape[0], w8.shape[0]), 9.0, dtype=torch.bfloat16)

    def swiglu(h1, h2, q, sf, *, alpha, beta, limit, enable_pdl):
        calls.append(("swiglu", h1.shape[0], (alpha, beta, limit)))
        h2.zero_()

    monkeypatch.setattr(drafter_mxfp8, "quantize_gemm", quantize_gemm)
    monkeypatch.setattr(drafter_mxfp8, "gemm", gemm)
    monkeypatch.setattr(drafter_mxfp8, "swiglu_kernel", lambda: swiglu)
    return calls


def test_the_switch_is_the_dep_drafter_mxfp8_item():
    assert drafter_mxfp8.DEP_MXDRAFT == (not m31_switches.m31_disabled("DEP_DRAFTER_MXFP8"))


def test_the_recovery_is_exact_or_refuses():
    w = _mxfp8_weight(64, 256, seed=3)
    codes, exp = mxfp8_recover.mxfp8_from_bf16(w)
    back = (codes.float().view(64, 8, 32) * torch.exp2(exp.float() - 127.0).unsqueeze(-1)).view(
        64, 256
    )
    assert torch.equal(back.to(torch.bfloat16), w)
    assert mxfp8_recover.mxfp8_from_bf16((torch.randn(64, 256) * 0.37).to(torch.bfloat16)) is None
    assert mxfp8_recover.mxfp8_from_bf16(w.float()) is None
    assert mxfp8_recover.mxfp8_from_bf16(w[:, :250]) is None


def test_swizzle_is_the_tp8_layout_and_round_trips():
    gen = torch.Generator().manual_seed(5)
    for rows, cols in ((1, 4), (37, 8), (128, 192), (300, 384)):
        byte = torch.randint(0, 256, (rows, cols), generator=gen, dtype=torch.int16).to(torch.uint8)
        sw = mxfp8_recover.swizzle_sf(byte)
        assert sw.numel() == (rows + 127) // 128 * 128 * cols
        assert torch.equal(mxfp8_recover.unswizzle_sf(sw, rows, cols), byte)
        # The 128x4 layout trtllm::mxfp8_quantize(x, True) writes: row r, column c at this byte.
        r, c = rows - 1, cols - 1
        off = (
            (c % 4)
            + (c // 4) * 512
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + (r // 128) * 128 * cols
        )
        assert sw[off] == byte[r, c]
    with pytest.raises(ValueError):
        mxfp8_recover.swizzle_sf(torch.zeros(4, 6, dtype=torch.uint8))


def test_install_routes_every_projection_and_mlp(rec):
    draft = _draft(layers=3)
    assert drafter_mxfp8.install(draft) == (12, 3)
    for layer in draft.model.layers:
        for lin in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            assert lin.__dict__.get(drafter_mxfp8.ATTR) is not None
        assert layer.mlp.__dict__.get(drafter_mxfp8.ATTR) is not None
    # Idempotent.
    assert drafter_mxfp8.install(draft) == (12, 3)
    assert "MXFP8 on 12 projections, swiglu_mxfp8 down chain on 3 MLPs" in drafter_mxfp8.engagement(
        (12, 3)
    )


@pytest.mark.parametrize("rows", [32, 33, 77, 128, 129])
def test_projection_rows_band(rec, rows):
    draft = _draft(layers=1)
    drafter_mxfp8.install(draft)
    lin = draft.model.layers[0].self_attn.qkv_proj
    x = torch.randn(rows, HIDDEN).to(torch.bfloat16)
    out = lin(x)
    if drafter_mxfp8.MIN_ROWS <= rows <= drafter_mxfp8.MAX_ROWS:
        assert rec == [("quantize_gemm", rows, (3 * HIDDEN, HIDDEN))]
        assert torch.all(out == 7.0)
    else:
        assert rec == []
        assert torch.equal(out, F.linear(x, lin.weight))


@pytest.mark.parametrize("rows", [32, 64, 129])
def test_mlp_down_chain_band(rec, rows):
    draft = _draft(layers=1)
    drafter_mxfp8.install(draft)
    mlp = draft.model.layers[0].mlp
    x = torch.randn(rows, HIDDEN).to(torch.bfloat16)
    out = mlp(x)
    if drafter_mxfp8.MIN_ROWS <= rows <= drafter_mxfp8.MAX_ROWS:
        assert rec == [
            ("quantize_gemm", rows, (2 * INTER, HIDDEN)),
            ("swiglu", rows, (1.702, 1.0, 7.0)),
            ("gemm", rows, (HIDDEN, INTER)),
        ]
        assert out.shape == (rows, HIDDEN) and torch.all(out == 9.0)
        assert mlp.stock_calls == 0
    else:
        assert rec == [] and mlp.stock_calls == 1


def test_bias_lora_and_declined_gemm_keep_the_stock_path(rec, monkeypatch):
    draft = _draft(layers=1)
    drafter_mxfp8.install(draft)
    lin = draft.model.layers[0].self_attn.o_proj
    x = torch.randn(64, HIDDEN).to(torch.bfloat16)
    ref = F.linear(x, lin.weight)
    assert torch.equal(lin.apply_linear(x, torch.zeros(HIDDEN)), ref)
    assert torch.equal(lin.apply_linear(x, None, {"lora": 1}), ref)
    monkeypatch.setattr(drafter_mxfp8, "quantize_gemm", lambda *a: None)
    assert torch.equal(lin(x), ref)


def test_non_mxfp8_weights_and_tp_down_are_not_routed(rec):
    assert drafter_mxfp8.install(_draft(layers=2, mxfp8=False)) == (0, 0)
    draft = _draft(layers=1)
    draft.model.layers[0].mlp.down_proj.tp_size = 2
    assert drafter_mxfp8.install(draft) == (4, 0)
    assert "down_proj is tensor-parallel" in drafter_mxfp8.mlp_decline_reason(
        draft.model.layers[0].mlp
    )


def test_knob_off_installs_nothing_and_leaves_the_stock_forwards(monkeypatch):
    monkeypatch.setattr(drafter_mxfp8, "DEP_MXDRAFT", False)
    draft = _draft(layers=1)
    assert drafter_mxfp8.install(draft) == (0, 0)
    layer = draft.model.layers[0]
    for lin in (layer.self_attn.qkv_proj, layer.mlp.down_proj):
        assert "apply_linear" not in lin.__dict__ and drafter_mxfp8.ATTR not in lin.__dict__
    assert "forward" not in layer.mlp.__dict__
    assert drafter_mxfp8.engagement((0, 0)).startswith(
        "off (TRTLLM_M31_DISABLE_DEP_DRAFTER_MXFP8=1)"
    )
    assert drafter_mxfp8.install(None) == (0, 0)


def test_the_dep_model_installs_after_native_swiglu_and_logs_the_line():
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        modeling,
    )

    src = inspect.getsource(modeling)
    assert src.index("self._native_swiglu = _swiglu.install(") < src.index(
        "self._mxdraft = _mxd.install(draft)"
    )
    assert "DEP_DRAFTER_MXFP8 drafter projections {_mxd.engagement(self._mxdraft)}" in src
