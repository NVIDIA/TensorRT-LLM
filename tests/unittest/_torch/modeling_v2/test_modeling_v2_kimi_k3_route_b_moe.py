# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE decode path of ``tp16_moetp16ep1`` (route B), host-side: which layers and steps it takes, and the engine and
latent all-reduce a step runs. Its kernels' calls are checked at 4 ranks by
``comm/test_modeling_v2_kimi_k3_route_b_decode_moe_op_matrix.py``."""

import ast
import types
from pathlib import Path

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe as _k3_fused_moe
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
    decode_moe as route_b_moe,
)


def _module_floats(path: Path) -> dict:
    """A kernel file's module-level float constants, ``NAME = <float>`` or ``NAME = float(_cfg("key", <float>))``
    (the default), read without importing it."""
    found = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        name, value = getattr(node.targets[0], "id", None), node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, float):
            found[name] = value.value
        elif (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "float"
            and value.args
            and isinstance(value.args[0], ast.Call)
            and getattr(value.args[0].func, "id", None) == "_cfg"
        ):
            found[name] = float(value.args[0].args[1].value)
    return found


def _moe(betas, top_k=16):
    """A MoE layer with the fields layout_gaps reads before its shape checks."""
    linear = types.SimpleNamespace(weight=None, bias=None)
    return types.SimpleNamespace(
        _reduce_routed_output=True,
        _situ_betas=betas,
        num_experts=896,
        top_k=top_k,
        moe_hidden_size=3584,
        hidden_size=7168,
        gate=None,
        routed_experts=types.SimpleNamespace(backend=None),
        shared_experts=types.SimpleNamespace(gate_up_proj=linear, down_proj=linear),
    )


def test_engine_situ_caps_are_the_kernels():
    """ENGINE_SITU_CAPS, against which layout_gaps checks a checkpoint's SiTU caps, are the caps k3_moe_m1, k3_moe_m2
    and k3_moe compile in."""
    kernels = Path(_k3_fused_moe.__file__).resolve().parent
    for name in ("k3_moe_m1_kernel.py", "k3_moe_m2_kernel.py", "k3_moe_kernel.py"):
        consts = _module_floats(kernels / name)
        caps = (consts["SITU_GATE_CAP"], consts["SITU_LINEAR_CAP"])
        assert caps == route_b_moe.ENGINE_SITU_CAPS, (name, caps)


def test_other_situ_caps_keep_the_generic_path():
    """A MoE layer whose SiTU caps (the checkpoint's activation_situ_beta / activation_situ_linear_beta) differ from
    the engines' gets one gap naming them, so the target leaves it on the generic path. The engines' caps pass that
    check: the gap that follows is the next check's (here the top-k)."""
    for betas in ((5.0, 25.0), (4.0, 20.0)):
        gaps = route_b_moe.layout_gaps(_moe(betas), 16, 8)
        assert len(gaps) == 1 and "SiTU caps" in gaps[0] and str(betas) in gaps[0], gaps
    gaps = route_b_moe.layout_gaps(_moe(route_b_moe.ENGINE_SITU_CAPS, top_k=8), 16, 8)
    assert len(gaps) == 1 and "experts / top-k" in gaps[0], gaps


def test_the_path_takes_at_most_8_tokens():
    """Steps of 1..8 tokens, with or without the deferred tail; no wide step (k3_moe's wide build does not fit 896
    local experts), so 9 tokens and more stay on the generic path, as does a step decode_step did not classify."""
    takes = route_b_moe.K3DecodeMoeLayer.takes
    step = types.SimpleNamespace(wide=True)

    def rows(n):
        return torch.empty(n, 7168, dtype=torch.bfloat16)

    for partial_tail in (False, True):
        for n in range(1, route_b_moe.MAX_TOKENS + 1):
            assert takes(None, rows(n), step, partial_tail)
        for n in (9, 16, 64):
            assert not takes(None, rows(n), step, partial_tail)
    assert not takes(None, rows(4), None, False)


def test_a_pushing_step_runs_the_engines_push_form(monkeypatch):
    """The latent all-reduce a MoE layer runs (forward's ``push``, the step's ``DecodeStep.latent_push``): with push
    and the state's latent exchange, the engine of the token count pushes (k3_moe_m1 at one token, k3_moe_m2 at two,
    k3_moe above) and k3_latent_reduce sums; without push, or without an exchange, the engine returns its partial to
    the routed experts' all-reduce."""
    calls = []

    def returning(name):
        return lambda x_fp8, *args: calls.append(name) or torch.zeros(x_fp8.shape[0], 3584)

    def pushing(name):
        return lambda *args: calls.append(name)

    for name in ("k3_moe_m1", "k3_moe_m2", "k3_moe"):
        monkeypatch.setattr(route_b_moe, name, returning(name))
        monkeypatch.setattr(route_b_moe, f"{name}_push", pushing(f"{name}_push"))
    monkeypatch.setattr(
        route_b_moe,
        "k3_latent_reduce",
        lambda rows, exchange: calls.append("k3_latent_reduce") or torch.zeros(rows, 3584),
    )
    monkeypatch.setattr(
        route_b_moe.K3DecodeMoeLayer,
        "_front",
        lambda self, moe, x: (
            None,
            None,
            torch.zeros(x.shape[0], 3584),
            None,
            torch.zeros(x.shape[0], 384),
        ),
    )
    moe = types.SimpleNamespace(
        routed_experts=types.SimpleNamespace(
            backend=types.SimpleNamespace(slot_start=0),
            all_reduce=lambda y: calls.append("all_reduce") or y,
        ),
        routed_expert_norm=types.SimpleNamespace(variance_epsilon=1e-5),
    )
    for exchange in (object(), None):
        layer = route_b_moe.K3DecodeMoeLayer(
            state=types.SimpleNamespace(exchange=exchange), front_weight=None, head_weight=None,
            tail_weight=None, tail_pad=None, bias=None, lo=0, width=224, shared_cols=384,
            small=None, wide=None, m1=None, m2=None,
        )  # fmt: skip
        for rows, engine in ((1, "k3_moe_m1"), (2, "k3_moe_m2"), (3, "k3_moe"), (8, "k3_moe")):
            x = torch.zeros(rows, 7168, dtype=torch.bfloat16)
            for push in (False, True):
                calls.clear()
                pending = layer.forward(moe, x, None, partial_tail=True, push=push)
                assert pending.latent.shape == (rows, 3584)
                if push and exchange is not None:
                    assert calls == [f"{engine}_push", "k3_latent_reduce"], (rows, push, calls)
                else:
                    assert calls == [engine, "all_reduce"], (rows, push, exchange, calls)
