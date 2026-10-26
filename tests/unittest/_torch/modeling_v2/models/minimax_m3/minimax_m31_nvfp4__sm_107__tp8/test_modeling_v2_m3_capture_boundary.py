# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generic layer's feed-forward exit on Eagle3 capture layers, in both targets' bodies (``PrefillTarget._layer``,
``DecodeTarget._generic_layer``).

Every layer's MoE / MLP output goes unreduced to the fused boundary (all-reduce + residual + the next layer's input
norm, the final norm after the last layer), and a capture layer captures the reduced residual sum that boundary
writes, with no residual of its own. CPU only: the attention, the feed-forward and the boundaries are stand-ins.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling,
)

LAYERS = 4


class _Spec:
    def __init__(self, layers):
        self.layers = set(layers)

    def is_layer_capture(self, i):
        return i in self.layers


@pytest.fixture(params=["prefill", "decode"])
def layer(request, monkeypatch):
    """(core, the target's generic layer bound to a target over ``core``)."""
    calls = []
    w = {f"l{i}_{n}": f"{n}{i}" for i in range(LAYERS) for n in ("in_norm", "post_norm")}
    w["final_norm"] = "final"
    c = SimpleNamespace(
        w=w,
        num_layers=LAYERS,
        moe_layers=frozenset({2, 3}),
        rms_eps=1e-6,
        calls=calls,
        # The prefill attention boundary passes a MoE layer's NVFP4 input scale to CTXNORMQ.
        moe={i: SimpleNamespace(fc31_input_scale=None) for i in (2, 3)},
    )
    monkeypatch.setattr(modeling, "_can_mc", lambda x, folded=False: False)
    # The prefill body tries the CTXNORMQ boundary first; it declines here (no op on these stand-ins).
    monkeypatch.setattr(modeling, "_ctxnormq_allreduce_norm", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        modeling,
        "_native_allreduce_norm",
        lambda core, x, r, norm: (("normed", x, norm), ("updated", x, r)),
    )

    def boundary(self, x, x2, residual, norm_w):
        calls.append((x, x2, norm_w))
        return ("next", norm_w), ("sum", x, x2)

    if request.param == "prefill":
        cls = modeling.PrefillTarget
        monkeypatch.setattr(cls, "_moe", lambda self, i, h, **k: (("moe", i), ("routed", i)))
        monkeypatch.setattr(cls, "_swiglu_mlp", lambda self, h, prefix, *a, **k: ("mlp", prefix))
    else:
        cls = modeling.DecodeTarget
        monkeypatch.setattr(
            cls, "_moe", lambda self, i, h, **k: (("moe", i), ("routed", i), False, False)
        )
        monkeypatch.setattr(
            cls, "_swiglu_mlp", lambda self, h, prefix, *a, **k: (("mlp", prefix), False)
        )
    monkeypatch.setattr(cls, "_attention", lambda self, i, *a, **k: (("attn", i), False))
    monkeypatch.setattr(cls, "_generic_boundary", boundary)
    target = cls(c)
    return c, target._layer if request.param == "prefill" else target._generic_layer


@pytest.mark.parametrize("i", range(LAYERS))
def test_a_capture_layer_captures_the_boundary_residual_sum(layer, i) -> None:
    core, run = layer
    captured = []
    out = run(
        i,
        None,
        torch.ones(2),
        torch.zeros(2),
        None,
        _Spec([i]),
        lambda spec, layer, h, r: captured.append((layer, h, r)),
    )
    norm = core.w[f"l{i + 1}_in_norm"] if i + 1 < LAYERS else "final"
    assert core.calls[-1][2] == norm, (
        "the boundary applies the next layer's norm (the final norm after the last)"
    )
    assert captured == [(i, out[1], None)]
    if i in core.moe_layers:
        assert core.calls[-1][1] == ("routed", i), "the routed part folds into the boundary"


def test_a_layer_without_capture_captures_nothing(layer) -> None:
    _, run = layer
    captured = []
    run(
        1,
        None,
        torch.ones(2),
        torch.zeros(2),
        None,
        _Spec([]),
        lambda *a: captured.append(a),
    )
    assert not captured


@pytest.fixture
def ctxnormq_prefill(monkeypatch):
    """(core, the prefill target's generic layer, the boundary calls), with CTXNORMQ taking every boundary offered.

    Only the collectives are stand-ins: the layer and its exit boundary (``PrefillTarget._generic_boundary``) are the
    target's own, so the test reads which boundary each result came from and where the layer uses it.
    """
    calls = []
    w = {f"l{i}_{n}": f"{n}{i}" for i in range(LAYERS) for n in ("in_norm", "post_norm")}
    w["final_norm"] = "final"
    c = SimpleNamespace(
        w=w,
        num_layers=LAYERS,
        moe_layers=frozenset({2, 3}),
        rms_eps=1e-6,
        moe={i: SimpleNamespace(fc31_input_scale=f"fp4_{i}") for i in (2, 3)},
    )
    monkeypatch.setattr(modeling, "_can_mc", lambda x, folded=False: False)

    def ctxnormq(core, x, residual, norm_w, fp4_scale):
        calls.append(("ctxnormq", x, residual, norm_w, fp4_scale))
        return ("ctxnormq_normed", norm_w), ("ctxnormq_updated", norm_w)

    def native(core, x, residual, norm_w):
        calls.append(("native", x, residual, norm_w))
        return ("native_normed", norm_w), ("native_updated", norm_w)

    monkeypatch.setattr(modeling, "_ctxnormq_allreduce_norm", ctxnormq)
    monkeypatch.setattr(modeling, "_native_allreduce_norm", native)
    cls = modeling.PrefillTarget
    monkeypatch.setattr(cls, "_attention", lambda self, i, *a, **k: (("attn", i), False))
    monkeypatch.setattr(cls, "_moe", lambda self, i, h, **k: (("moe", i, h), None))
    monkeypatch.setattr(cls, "_swiglu_mlp", lambda self, h, prefix, *a, **k: ("mlp", prefix, h))
    return c, cls(c)._layer, calls


@pytest.mark.parametrize("i", range(LAYERS))
def test_prefill_uses_the_ctxnormq_boundary_in_place_of_the_fused_allreduce(
    ctxnormq_prefill, i
) -> None:
    """A prefill boundary CTXNORMQ takes is used where the native fused all-reduce's would be.

    The attention boundary (with the MoE's NVFP4 input scale on a MoE layer) feeds the feed-forward, and its updated
    residual goes to the exit boundary. The exit boundary is CTXNORMQ's into every next input norm; the final norm
    keeps the native fused all-reduce. Its (normed, updated) is what the layer returns, and its updated is what a
    capture layer captures.
    """
    core, run, calls = ctxnormq_prefill
    residual = torch.zeros(2)
    captured = []
    out = run(
        i,
        None,
        torch.ones(2),
        residual,
        None,
        _Spec([i]),
        lambda spec, layer, h, r: captured.append((layer, h, r)),
    )
    post = core.w[f"l{i}_post_norm"]
    kind, x, res, norm, fp4 = calls[0]
    assert (kind, x, norm) == ("ctxnormq", ("attn", i), post), (
        "the attention boundary is offered to CTXNORMQ"
    )
    assert res is residual
    assert fp4 == (f"fp4_{i}" if i in core.moe_layers else None), "NVFP4 for the MoE's input only"
    normed = ("ctxnormq_normed", post)
    ff = ("moe", i, normed) if i in core.moe_layers else ("mlp", f"l{i}_mlp", normed)
    if i + 1 < LAYERS:
        nxt = core.w[f"l{i + 1}_in_norm"]
        assert calls[1] == ("ctxnormq", ff, ("ctxnormq_updated", post), nxt, None)
        assert out == (("ctxnormq_normed", nxt), ("ctxnormq_updated", nxt))
    else:
        assert calls[1] == ("native", ff, ("ctxnormq_updated", post), "final"), (
            "the final norm keeps the native fused all-reduce"
        )
        assert out == (("native_normed", "final"), ("native_updated", "final"))
    assert len(calls) == 2
    assert captured == [(i, out[1], None)]
