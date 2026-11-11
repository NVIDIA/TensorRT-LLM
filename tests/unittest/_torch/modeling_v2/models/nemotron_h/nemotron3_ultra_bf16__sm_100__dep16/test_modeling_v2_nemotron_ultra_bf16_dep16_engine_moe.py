# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 dep16 target's routed experts: the engine's MoE layer for every admitted backend.

dep16 runs attention data parallel, so a token's experts mostly live on other ranks: every MoE block
is one `create_moe` layer, whose exchange between ranks (the engine's all-to-all, or MegaMoE's fused
one) dispatches the latent rows and combines the routed sums. What is checked here, on CPU with
`create_moe` replaced by a stand-in:

* the `create_moe` arguments: Ultra's latent experts, Relu2, the router's top-22 noaux selection
  over the checkpoint's fp32 correction bias, no reduction, no backend degradation;
* that construction refuses anything but the requested leaf over this rank's 32-expert window,
  built under attention DP with an exchange between ranks;
* the load-time knobs and their bounds;
* that the pools it binds are the bf16 KV cache and the fp32 SSM state it was gated with.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.moe.fused_moe as fused_moe_pkg
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__dep16 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__dep16 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.moe.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    MoEWeightLoadingMode,
    SimpleActivation,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only

RANKS = 16
EXPERTS, TOP_K, SCALE = 512, 22, 5.0
MOE_INTER, LATENT = 5120, 2048
WINDOW = EXPERTS // RANKS
KINDS = ["mamba", "moe", "attention", "mamba", "moe"]
MOE_LAYERS = [1, 4]


def _core(rank: int, moe_backend: str, resolved: str | None = None) -> SimpleNamespace:
    """The fields `dims` reads, at Ultra's numbers, for one rank of the 16-rank dep16 group."""
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
    mapping = SimpleNamespace(
        enable_attention_dp=True, moe_tp_size=1, moe_ep_size=RANKS, moe_ep_rank=rank
    )
    model_config = SimpleNamespace(
        pretrained_config=cfg,
        mapping=mapping,
        torch_dtype=torch.bfloat16,
        moe_backend_requested=moe_backend,
        # The engine resolves AUTO to CUTLASS for this checkpoint; a request by name to itself.
        moe_backend=resolved or ("CUTLASS" if moe_backend == "AUTO" else moe_backend),
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


def test_every_engine_backend_names_its_leaf():
    assert set(W.MOE_BACKENDS.values()) == set(M._ENGINE_MOE_LEAVES)
    assert M._FUSED_COMM_BACKENDS <= set(M._ENGINE_MOE_LEAVES)


@pytest.mark.parametrize("backend", ["CUTLASS", "MEGAMOE_CUTEDSL", "TRTLLM"])
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
    """What `_routed_experts` reads off a built layer: its backend's identity and window, whether
    it was built under attention DP and its communication strategy."""

    def __init__(self, identity: str, local_expert_ids: list[int], use_dp=True, comm="A2A"):
        super().__init__()
        leaf = type("_Leaf", (), {"descriptor": SimpleNamespace(identity=identity)})
        self.backend = leaf()
        self.backend.initial_local_expert_ids = local_expert_ids
        self.use_dp = use_dp
        self.comm = None if comm is None else type(comm, (), {})()


def _patch_create_moe(monkeypatch, make):
    calls = []

    def create_moe(**kwargs):
        calls.append(kwargs)
        return make(kwargs)

    monkeypatch.setattr(fused_moe_pkg, "create_moe", create_moe)
    monkeypatch.setattr(fused_moe_pkg, "ConfigurableMoE", _FakeConfigurableMoE)
    return calls


@pytest.mark.parametrize(
    "requested, comm",
    [
        ("AUTO", "NVLinkOneSided"),
        ("CUTLASS", "NVLinkOneSided"),
        ("TRTLLM", "AllGatherReduceScatter"),
        ("MEGAMOE_CUTEDSL", None),
    ],
)
def test_one_engine_layer_per_moe_block(monkeypatch, requested, comm):
    rank = 13
    window = list(range(WINDOW * rank, WINDOW * rank + WINDOW))
    backend = W.MOE_BACKENDS[requested]
    calls = _patch_create_moe(
        monkeypatch,
        lambda kw: _FakeConfigurableMoE(M._ENGINE_MOE_LEAVES[backend], window, comm=comm),
    )
    core = _core(rank, requested)
    d = W.MODEL_WEIGHTS.dims(core)
    layers = M._routed_experts(core.model_config, d, _w())
    assert list(layers.keys()) == [str(i) for i in MOE_LAYERS]
    assert [c["layer_idx"] for c in calls] == MOE_LAYERS
    assert all(c["allow_backend_degradation"] is False for c in calls)
    assert M._exchange(layers["1"]) == (comm or "fused in the kernel")


@pytest.mark.parametrize(
    "requested, make, missed",
    [
        # MEGAMOE_CUTEDSL resolved to the NVFP4 MegaMoE leaf
        (
            "MEGAMOE_CUTEDSL",
            lambda kw: _FakeConfigurableMoE(
                "trtllm.cutedsl.mega_moe.nvfp4", list(range(WINDOW)), comm=None
            ),
            "runs that request only on trtllm.cutedsl.mega_moe.none",
        ),
        # AUTO served by the TRTLLM-Gen leaf instead of CUTLASS
        (
            "AUTO",
            lambda kw: _FakeConfigurableMoE(
                "flashinfer.trtllm_gen.fused_moe.none", list(range(WINDOW))
            ),
            "runs that request only on trtllm.cutlass.grouped_gemm.none",
        ),
        # tep8's 64-expert window instead of this rank's 32
        (
            "CUTLASS",
            lambda kw: _FakeConfigurableMoE("trtllm.cutlass.grouped_gemm.none", list(range(64))),
            "not this rank's window",
        ),
        # a layer built without attention DP
        (
            "TRTLLM",
            lambda kw: _FakeConfigurableMoE(
                "flashinfer.trtllm_gen.fused_moe.none", list(range(WINDOW)), use_dp=False
            ),
            "without attention DP",
        ),
        # an external-comm leaf with no exchange: every rank would keep its own tokens
        (
            "CUTLASS",
            lambda kw: _FakeConfigurableMoE(
                "trtllm.cutlass.grouped_gemm.none", list(range(WINDOW)), comm=None
            ),
            "no communication strategy",
        ),
        # a layer outside ConfigurableMoE
        ("CUTLASS", lambda kw: nn.Identity(), "not ConfigurableMoE"),
    ],
    ids=["wrong-leaf", "auto-not-cutlass", "wrong-window", "no-adp", "no-exchange", "not-cmoe"],
)
def test_construction_refuses_what_the_request_did_not_ask_for(
    monkeypatch, requested, make, missed
):
    _patch_create_moe(monkeypatch, make)
    core = _core(0, requested)
    d = W.MODEL_WEIGHTS.dims(core)
    with pytest.raises(RuntimeError, match=missed):
        M._routed_experts(core.model_config, d, _w())


def test_a_request_the_engine_resolved_elsewhere_is_refused(monkeypatch):
    """The engine resolves the request into `moe_backend`, which `create_moe` builds from: the
    target refuses to build from anything but the backend the request maps to."""
    _patch_create_moe(monkeypatch, lambda kw: pytest.fail("create_moe must not be reached"))
    core = _core(0, "AUTO", resolved="TRTLLM")
    d = W.MODEL_WEIGHTS.dims(core)
    with pytest.raises(AssertionError, match="resolved to 'TRTLLM', not 'CUTLASS'"):
        M._routed_experts(core.model_config, d, _w())


# --- load-time knobs ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name, raw, default, lo, hi, want",
    [
        (M.LOWM_MAX_ROWS_ENV, None, 2, 0, 2, 2),
        (M.LOWM_MAX_ROWS_ENV, "", 2, 0, 2, 2),
        (M.LOWM_MAX_ROWS_ENV, "0", 2, 0, 2, 0),
        (M.LOWM_MAX_ROWS_ENV, " 1 ", 2, 0, 2, 1),
        (M.FI_SSU_MIN_ROWS_ENV, "8", 32, 1, None, 8),
        (M.FI_SSU_MIN_ROWS_ENV, "100000", 32, 1, None, 100000),
    ],
)
def test_knobs_read_their_value(monkeypatch, name, raw, default, lo, hi, want):
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert M._env_rows(name, default, lo, hi) == want


@pytest.mark.parametrize(
    "name, raw, lo, hi, missed",
    [
        # the low-M kernel's cells cover one and two rows
        (M.LOWM_MAX_ROWS_ENV, "3", 0, 2, r"expected \[0, 2\]"),
        (M.LOWM_MAX_ROWS_ENV, "-1", 0, 2, r"expected \[0, 2\]"),
        (M.FI_SSU_MIN_ROWS_ENV, "0", 1, None, "expected >= 1"),
        (M.FI_SSU_MIN_ROWS_ENV, "auto", 1, None, "expected an integer"),
    ],
)
def test_knobs_refuse_values_out_of_bounds(monkeypatch, name, raw, lo, hi, missed):
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=missed):
        M._env_rows(name, 0, lo, hi)


def test_knob_names_survive_the_launcher():
    """trtllm-llmapi-launch drops MV2_, SLURM_, UCX_ and OMPI_ variables before the ranks start."""
    for name in (M.LOWM_MAX_ROWS_ENV, M.FI_SSU_MIN_ROWS_ENV):
        assert not name.startswith(("MV2_", "SLURM_", "UCX_", "OMPI_")), name


# --- the pools it binds --------------------------------------------------------------

# A tiny geometry: the method compares pool shapes with the dimension bundle it was given.
_DIMS = SimpleNamespace(
    attn_layers=[2],
    mamba_layers=[0, 3],
    conv_r=6,
    conv_kernel=4,
    heads_r=2,
    head_dim_ssm=4,
    ssm_state=8,
)


def _bind(kv=DataType.BF16, conv=torch.bfloat16, ssm=torch.float32):
    cm = SimpleNamespace(
        dtype=kv,
        layer_offsets={2: 0},
        get_conv_states=lambda layer: torch.zeros(3, 6, 3, dtype=conv),
        get_ssm_states=lambda layer: torch.zeros(3, 2, 4, 8, dtype=ssm),
    )
    md = SimpleNamespace(
        kv_cache_manager=cm, effective_beam_width=1, cache_indirection=None, is_cross=False
    )
    # Rank 1: the method logs the pools' depth on rank 0 only.
    core = SimpleNamespace(
        _binds=0,
        _bound=None,
        dims=_DIMS,
        model_config=SimpleNamespace(mapping=SimpleNamespace(rank=1)),
    )
    M.NemotronUltraBf16Dep16ModelingV2._bind_cache_manager(core, md)
    return core, cm


def test_the_certified_pools_bind():
    core, cm = _bind()
    assert core._bound.cm is cm and core._bound.pool_row == {2: 0}


@pytest.mark.parametrize("kv", [DataType.FP8, DataType.NVFP4, DataType.HALF])
def test_a_kv_pool_that_is_not_bf16_is_refused(kv):
    with pytest.raises(AssertionError, match="bf16 KV cache"):
        _bind(kv=kv)


@pytest.mark.parametrize(
    "conv, ssm", [(torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16)]
)
def test_a_state_pool_that_is_not_bf16_and_fp32_is_refused(conv, ssm):
    with pytest.raises(AssertionError, match="needs bf16 / fp32"):
        _bind(conv=conv, ssm=ssm)
