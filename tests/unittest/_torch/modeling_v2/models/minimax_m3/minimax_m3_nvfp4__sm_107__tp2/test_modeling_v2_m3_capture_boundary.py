# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generic layer's feed-forward exit on Eagle3 capture layers, in both targets' bodies (``PrefillTarget._layer``,
``DecodeTarget._generic_layer``).

Every layer's MoE / MLP output goes unreduced to the fused boundary (all-reduce + residual + the next layer's input
norm, the final norm after the last layer), and a capture layer captures the reduced residual sum that boundary
writes, with no residual of its own. Synthetic attention, feed-forward and collectives isolate the residual routing.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)

LAYERS = 4


class _Spec:
    def __init__(self, layers):
        self.layers = set(layers)

    def is_layer_capture(self, i):
        return i in self.layers


@pytest.fixture(params=["prefill", "decode"])
def layer(request, monkeypatch, make_cuda_m3_config):
    calls = []
    core = modeling.MiniMaxM3Tp2ModelingV2(
        make_cuda_m3_config(2, moe_layers=(0, 0, 1, 1), sparse_layers=(0, 0, 1, 1))
    )
    # The stand-in boundary under test runs when the MC finalize path declines.
    core._decode = SimpleNamespace(_generic_mc_finalize=lambda i, h, r, norm_w: None)
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
        monkeypatch.setattr(
            cls, "_swiglu_mlp", lambda self, h, prefix, low_m, dense=False: ("mlp", prefix)
        )
    else:
        cls = modeling.DecodeTarget
        monkeypatch.setattr(
            cls, "_moe", lambda self, i, h, **k: (("moe", i), ("routed", i), False, False)
        )
        monkeypatch.setattr(
            cls,
            "_swiglu_mlp",
            lambda self, h, prefix, low_m, push=None, dense=False: (("mlp", prefix), False),
        )
        # The 33..48-row MC finalize declines here: the stand-in boundary is the one under test.
        monkeypatch.setattr(cls, "_generic_mc_finalize", lambda self, i, h, r, norm_w: None)
    monkeypatch.setattr(cls, "_attention", lambda self, i, *a, **k: (("attn", i), False))
    monkeypatch.setattr(cls, "_generic_boundary", boundary)
    target = cls(core)
    run_layer = target._layer if request.param == "prefill" else target._generic_layer
    return core, run_layer, calls


@pytest.mark.parametrize("i", range(LAYERS))
def test_a_capture_layer_captures_the_boundary_residual_sum(layer, i) -> None:
    core, run, calls = layer
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
    norm = core.w[f"l{i + 1}_in_norm"] if i + 1 < LAYERS else core.w["final_norm"]
    assert calls[-1][2] is norm, (
        "the boundary applies the next layer's norm (the final norm after the last)"
    )
    assert captured == [(i, out[1], None)]
    if i in core.moe_layers:
        assert calls[-1][1] == ("routed", i), "the routed part folds into the boundary"


def test_a_layer_without_capture_captures_nothing(layer) -> None:
    _, run, _ = layer
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


def test_a_prefill_moe_layer_takes_the_mc_finalize_boundary_when_it_folds(
    monkeypatch, make_cuda_m3_config
) -> None:
    """PrefillTarget at a MoE layer: the decode target's generic MC finalize (9..48 rows) ends the layer when it folds,
    its updated residual is the Eagle3 capture, and the generic boundary does not run."""
    seen = []

    def mc_finalize(i, h, r, norm_w):
        seen.append((i, h, r, norm_w))
        return ("normed", i), ("updated", i)

    core = modeling.MiniMaxM3Tp2ModelingV2(
        make_cuda_m3_config(2, moe_layers=(0, 0, 1, 1), sparse_layers=(0, 0, 1, 1))
    )
    core._decode = SimpleNamespace(_generic_mc_finalize=mc_finalize)
    monkeypatch.setattr(modeling, "_can_mc", lambda x, folded=False: False)
    monkeypatch.setattr(
        modeling, "_native_allreduce_norm", lambda core, x, r, norm: (("normed", x), ("upd", x, r))
    )
    cls = modeling.PrefillTarget
    monkeypatch.setattr(cls, "_attention", lambda self, i, *a, **k: (("attn", i), False))

    def no_boundary(*a, **k):
        raise AssertionError("the generic boundary runs after a folded MC finalize")

    monkeypatch.setattr(cls, "_generic_boundary", no_boundary)
    monkeypatch.setattr(
        cls, "_moe", lambda self, *a, **k: pytest.fail("the MoE runs outside the fold")
    )
    target = cls(core)
    captured = []
    out = target._layer(
        2,
        None,
        torch.ones(2),
        torch.zeros(2),
        None,
        _Spec([2]),
        lambda spec, layer, h, r: captured.append((layer, h, r)),
    )
    assert out == (("normed", 2), ("updated", 2))
    assert seen and seen[0][0] == 2 and seen[0][3] is core.w["l3_in_norm"]
    assert captured == [(2, ("updated", 2), None)]
