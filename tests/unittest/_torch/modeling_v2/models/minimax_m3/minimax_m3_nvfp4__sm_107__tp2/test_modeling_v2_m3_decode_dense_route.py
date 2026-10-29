# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dense layer's MLP runs its SwiGLU-OAI activation through ``_dense_swiglu``, the seam of the dense swigluoai C++
op (``m3_dense_swiglu``), in the decode-graph layer (``DecodeTarget._layer``) and in the prefill layer, eager or in a
decode graph (r4.10.1 called the module directly, so the op never ran in a decode graph). The shared expert keeps the
plain ``m3_silu_and_mul``.

CPU only: the GEMMs, the activations and the boundaries are stubbed; the test reads which seam each call takes."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling,
)


@pytest.fixture
def seams(monkeypatch):
    """Records which activation seam each MLP call takes; the GEMMs pass their input through."""
    calls = []
    monkeypatch.setattr(
        modeling,
        "_linear",
        lambda core, x, w, s, low_m_gemm, push=None: (x, push is not None),
    )
    monkeypatch.setattr(
        modeling, "_dense_swiglu", lambda core, gu: calls.append(("dense", gu.shape[0])) or gu + 1
    )
    monkeypatch.setattr(
        modeling,
        "m3_silu_and_mul",
        lambda gu, *args: calls.append(("plain", gu.shape[0])) or gu + 2,
    )
    return calls


def _core():
    return SimpleNamespace(
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        num_layers=2,
        moe_layers=set(),
        w={"l0_post_norm": None, "l1_in_norm": None},
    )


@pytest.mark.parametrize("decode_graph", [True, False])
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_decode_dense_layer_runs_through_dense_swiglu(monkeypatch, seams, rows, decode_graph):
    monkeypatch.setattr(
        modeling, "flashinfer_mxfp8_decode_graph_capture_active", lambda: decode_graph
    )
    target = SimpleNamespace(core=_core())
    target._attention = lambda i, position_ids, hidden, step, armed: (hidden, False)
    target._attn_push_target = lambda num_tokens: None
    target._boundary = lambda mc_ar, h, residual, norm, **kwargs: (h, residual)
    target._swiglu_mlp = lambda *args, **kwargs: modeling.DecodeTarget._swiglu_mlp(
        target, *args, **kwargs
    )
    x = torch.zeros(rows, 16)
    plan = SimpleNamespace(attn_mc_ar=False, moe_mc_ar=False, qkv_prequant=False)
    out, _ = modeling.DecodeTarget._layer(target, 0, plan, None, x, x.clone(), {})
    assert seams == [("dense", rows)]
    assert torch.equal(out, x + 1)


@pytest.mark.parametrize("rows", [1, 64])
def test_prefill_dense_mlp_runs_through_dense_swiglu(seams, rows):
    target = SimpleNamespace(core=_core())
    x = torch.zeros(rows, 16)
    out = modeling.PrefillTarget._swiglu_mlp(target, x, "l0_mlp", True, dense=True)
    assert seams == [("dense", rows)]
    assert torch.equal(out, x + 1)


def test_shared_expert_keeps_the_plain_activation(seams):
    target = SimpleNamespace(core=_core())
    x = torch.zeros(4, 16)
    out, pushed = modeling.DecodeTarget._swiglu_mlp(target, x, "l3_sh", True)
    assert seams == [("plain", 4)] and not pushed
    assert torch.equal(out, x + 2)
