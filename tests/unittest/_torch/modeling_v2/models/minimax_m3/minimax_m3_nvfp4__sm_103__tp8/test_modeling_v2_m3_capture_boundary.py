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

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
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
        w=w, num_layers=LAYERS, moe_layers=frozenset({2, 3}), rms_eps=1e-6, calls=calls
    )
    monkeypatch.setattr(modeling, "_can_mc", lambda x, folded=False: False)
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
        monkeypatch.setattr(cls, "_swiglu_mlp", lambda self, h, prefix: ("mlp", prefix))
    else:
        cls = modeling.DecodeTarget
        monkeypatch.setattr(
            cls, "_moe", lambda self, i, h, **k: (("moe", i), ("routed", i), False, False)
        )
        monkeypatch.setattr(
            cls, "_swiglu_mlp", lambda self, h, prefix, push=None: (("mlp", prefix), False)
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
