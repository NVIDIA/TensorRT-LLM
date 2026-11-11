# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 tep8 target's opt-in routed experts: the engine's MoE layer for MEGAMOE_CUTEDSL / TRTLLM.

By default the routed experts run on the target's fused_moe. Asking for `moe_config.backend`
MEGAMOE_CUTEDSL or TRTLLM by name builds one `create_moe` layer per MoE block instead, and the
weight table stops declaring the experts and feeds those layers. What is checked here, on CPU with
`create_moe` replaced by a stand-in:

* which requests leave the default path (only an explicit MEGAMOE_CUTEDSL / TRTLLM);
* the `create_moe` arguments: Ultra's latent experts, Relu2, the router's top-22 noaux selection
  over the checkpoint's fp32 correction bias, no backend degradation;
* that construction refuses anything but the requested leaf over this rank's expert window;
* that the table declares no routed experts on that path and feeds every expert of the window.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.moe.fused_moe as fused_moe_pkg
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__tep8 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__tep8 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.moe.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    MoEWeightLoadingMode,
    SimpleActivation,
)
from tensorrt_llm._torch.utils import ActivationType

pytestmark = pytest.mark.cpu_only

TP = 8
EXPERTS, TOP_K, SCALE = 512, 22, 5.0
MOE_INTER, LATENT = 5120, 2048
KINDS = ["mamba", "moe", "attention", "mamba", "moe"]
MOE_LAYERS = [1, 4]


def _core(rank: int, moe_backend: str) -> SimpleNamespace:
    """The fields `dims` reads, at Ultra's numbers, for one rank of an 8-rank tep8 group."""
    cfg = SimpleNamespace(
        layers_block_type=list(KINDS),
        hidden_size=8192,
        vocab_size=131072,
        mamba_num_heads=256,
        mamba_head_dim=64,
        n_groups=8,
        ssm_state_size=128,
        conv_kernel=4,
        num_attention_heads=64,
        num_key_value_heads=2,
        head_dim=128,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        routed_scaling_factor=SCALE,
        moe_intermediate_size=MOE_INTER,
        moe_latent_size=LATENT,
        moe_shared_expert_intermediate_size=10240,
    )
    mapping = SimpleNamespace(tp_size=TP, tp_rank=rank, moe_ep_size=TP, moe_ep_rank=rank)
    model_config = SimpleNamespace(
        pretrained_config=cfg,
        mapping=mapping,
        torch_dtype=torch.bfloat16,
        moe_backend_requested=moe_backend,
        # A request by name resolves to itself; AUTO is resolved by the engine.
        moe_backend="CUTLASS" if moe_backend == "AUTO" else moe_backend,
    )
    return SimpleNamespace(model_config=model_config)


def _w() -> nn.ParameterDict:
    """The routers' correction biases, the one table weight the engine layers read."""
    return nn.ParameterDict(
        {
            f"l{i}_gate_b": nn.Parameter(torch.full((EXPERTS,), float(i)), requires_grad=False)
            for i in MOE_LAYERS
        }
    )


@pytest.mark.parametrize(
    "requested, engine",
    [
        ("AUTO", None),
        ("CUTLASS", None),
        ("MEGAMOE_CUTEDSL", "MEGAMOE_CUTEDSL"),
        ("megamoe_cutedsl", "MEGAMOE_CUTEDSL"),
        ("TRTLLM", "TRTLLM"),
    ],
)
def test_only_a_request_by_name_leaves_fused_moe(requested, engine):
    assert W.engine_moe_backend(SimpleNamespace(moe_backend_requested=requested)) == engine


def test_every_engine_backend_names_its_leaf():
    assert set(W.ENGINE_MOE_BACKENDS) == set(M._ENGINE_MOE_LEAVES)


@pytest.mark.parametrize("backend", ["MEGAMOE_CUTEDSL", "TRTLLM"])
def test_create_moe_kwargs(backend):
    core = _core(rank=3, moe_backend=backend)
    d = W.MODEL_WEIGHTS.dims(core)
    w = _w()
    kw = M._routed_moe_kwargs(core.model_config, d, w, 4)

    routing = kw.pop("routing_method")
    assert isinstance(routing, DeepSeekV3MoeRoutingMethod)
    impl = routing.routing_impl
    assert (impl.top_k, impl.n_group, impl.topk_group) == (TOP_K, 1, 1)
    assert impl.routed_scaling_factor == SCALE
    # The layer's own fp32 bias, looked up when routing runs, not when the layer was built.
    assert routing.e_score_correction_bias is w["l4_gate_b"]
    w["l4_gate_b"] = nn.Parameter(torch.zeros(EXPERTS), requires_grad=False)
    assert routing.e_score_correction_bias is w["l4_gate_b"]
    assert routing.e_score_correction_bias.dtype == torch.float32

    activation = kw.pop("activation")
    assert isinstance(activation, SimpleActivation)
    assert activation.kind == ActivationType.Relu2
    assert kw == dict(
        num_experts=EXPERTS,
        hidden_size=LATENT,
        intermediate_size=MOE_INTER,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=core.model_config,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        bias=False,
        layer_idx=4,
        allow_backend_degradation=False,
    )


class _FakeConfigurableMoE(nn.Module):
    """What `_routed_experts` reads off a built layer: its backend's identity and window."""

    def __init__(self, identity: str, local_expert_ids: list[int]):
        super().__init__()
        leaf = type("_Leaf", (), {"descriptor": SimpleNamespace(identity=identity)})
        self.backend = leaf()
        self.backend.initial_local_expert_ids = local_expert_ids


def _patch_create_moe(monkeypatch, make):
    calls = []

    def create_moe(**kwargs):
        calls.append(kwargs)
        return make(kwargs)

    monkeypatch.setattr(fused_moe_pkg, "create_moe", create_moe)
    monkeypatch.setattr(fused_moe_pkg, "ConfigurableMoE", _FakeConfigurableMoE)
    return calls


@pytest.mark.parametrize("backend", ["MEGAMOE_CUTEDSL", "TRTLLM"])
def test_one_engine_layer_per_moe_block(monkeypatch, backend):
    rank = 5
    window = list(range(64 * rank, 64 * rank + 64))
    calls = _patch_create_moe(
        monkeypatch, lambda kw: _FakeConfigurableMoE(M._ENGINE_MOE_LEAVES[backend], window)
    )
    core = _core(rank, backend)
    d = W.MODEL_WEIGHTS.dims(core)
    layers = M._routed_experts(core.model_config, d, _w())
    assert list(layers.keys()) == [str(i) for i in MOE_LAYERS]
    assert [c["layer_idx"] for c in calls] == MOE_LAYERS
    assert all(c["allow_backend_degradation"] is False for c in calls)


@pytest.mark.parametrize(
    "make, missed",
    [
        # MEGAMOE_CUTEDSL resolved to the NVFP4 MegaMoE leaf
        (
            lambda kw: _FakeConfigurableMoE("trtllm.cutedsl.mega_moe.nvfp4", list(range(64))),
            "runs that request only on trtllm.cutedsl.mega_moe.none",
        ),
        # an expert window other than the 64 experts the table feeds rank 0
        (
            lambda kw: _FakeConfigurableMoE("trtllm.cutedsl.mega_moe.none", list(range(1, 65))),
            "not this rank's window",
        ),
        # a layer outside ConfigurableMoE
        (lambda kw: nn.Identity(), "not ConfigurableMoE"),
    ],
    ids=["wrong-leaf", "wrong-window", "not-configurable"],
)
def test_construction_refuses_what_the_request_did_not_ask_for(monkeypatch, make, missed):
    _patch_create_moe(monkeypatch, make)
    core = _core(0, "MEGAMOE_CUTEDSL")
    d = W.MODEL_WEIGHTS.dims(core)
    with pytest.raises(RuntimeError, match=missed):
        M._routed_experts(core.model_config, d, _w())


@pytest.mark.parametrize("backend", ["AUTO", "CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"])
def test_the_table_declares_routed_experts_only_for_fused_moe(backend):
    d = W.MODEL_WEIGHTS.dims(_core(2, backend))
    declared = set(W.MODEL_WEIGHTS.manifest(_core(2, backend)))
    experts = {f"l{i}_ex_fc{n}_w" for i in MOE_LAYERS for n in (1, 2)}
    if d.engine_moe is None:
        assert experts <= declared
    else:
        assert not experts & declared
        # Everything else the MoE blocks read is still the table's.
        assert {f"l{i}_gate_b" for i in MOE_LAYERS} <= declared


@pytest.mark.parametrize("rank", [0, 7])
def test_the_engine_layers_are_fed_this_ranks_window(rank):
    """Each MoE block's engine layer gets every expert of the rank's window once, by global id,
    and nothing else, so `load`'s leftover check stays exact."""
    d = W.MODEL_WEIGHTS.dims(_core(rank, "TRTLLM"))
    window = range(64 * rank, 64 * rank + 64)
    for i in MOE_LAYERS:
        feed = W.MODEL_WEIGHTS.engine_expert_feed(d, i)
        p = f"backbone.layers.{i}.mixer.experts"
        assert feed == [
            (f"{p}.{e}.up_proj.weight", f"{p}.{e}.down_proj.weight", str(e)) for e in window
        ]
    fed = {k for i in MOE_LAYERS for up, down, _ in W.MODEL_WEIGHTS.engine_expert_feed(d, i)
           for k in (up, down)}
    every_expert = {
        f"backbone.layers.{i}.mixer.experts.{e}.{proj}_proj.weight"
        for i in MOE_LAYERS
        for e in range(EXPERTS)
        for proj in ("up", "down")
    }
    assert every_expert - fed == W.NemotronUltraBf16Tep8Weights.expected_unconsumed(
        d, dict.fromkeys(every_expert)
    )
