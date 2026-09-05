# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 fused gate_up MLP and distributed MoE tests.

The runtime dense / shared-expert ``GatedMLP`` runs a single fused
``gate_up_proj`` GEMM with K3's SiTU activation. ``KimiK3MLP`` remains the
compact reference for the same fused weight layout. These tests check that
layout against an unfused reference built from the HF checkpoint's split
``gate_proj`` / ``up_proj`` weights:

* fused ``gate_up_proj`` output matches ``two GEMMs + torch.cat`` +
  eager ``SituAndMul`` + ``down_proj`` for a decode-shaped (1 token) and
  a prefill-shaped (500 tokens) batch, with ``situ_linear_beta`` set and
  ``None`` (the two activation code paths);
* the row-concat convention is required: swapping the halves breaks
  the numerics (mutation control).
* shared-expert sharding follows the selected parallel mode;
* the complete multi-rank MoE output matches an unsharded reference with
  attention DP enabled and disabled.
"""

import os
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from _torch.moe.kimi_k3_ref_moe.kimi_k3_mlp_test_utils import KimiK3MLP
from torch import nn

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.situ import SituAndMul

cloudpickle.register_pickle_by_value(sys.modules[__name__])

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


class _UnfusedKimiMLP(nn.Module):
    """HF ``KimiMLP`` layout: separate gate/up GEMMs + torch.cat."""

    def __init__(self, hidden_size, intermediate_size, situ_beta, situ_linear_beta, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)

    def forward(self, x):
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


def _make_pair(hidden_size, intermediate_size, situ_beta, situ_linear_beta, device, seed=1234):
    """Unfused reference with random split weights + fused twin via row-concat."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    ref = _UnfusedKimiMLP(hidden_size, intermediate_size, situ_beta, situ_linear_beta, dtype).to(
        device
    )
    with torch.no_grad():
        for proj in (ref.gate_proj, ref.up_proj, ref.down_proj):
            proj.weight.copy_(torch.randn_like(proj.weight, dtype=torch.float32) * 0.05)

    fused = KimiK3MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        # The same row-concat KimiLinearForCausalLM.load_weights performs.
        inter = intermediate_size
        fused.gate_up_proj.weight[:inter].copy_(ref.gate_proj.weight)
        fused.gate_up_proj.weight[inter:].copy_(ref.up_proj.weight)
        fused.down_proj.weight.copy_(ref.down_proj.weight)
    return fused, ref


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=512,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=256,
        num_shared_experts=2,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
        num_expert_group=1,
        topk_group=1,
    )


@requires_cuda
@pytest.mark.parametrize(
    "situ_beta,situ_linear_beta",
    [
        (4.0, 25.0),  # Kimi K3 defaults
        (1.0, None),  # linear_beta disabled (identity up half)
    ],
    ids=["default", "no_linear_beta"],
)
@pytest.mark.parametrize("num_tokens", [1, 500], ids=lambda n: f"tokens{n}")
def test_fused_gate_up_matches_unfused_reference(num_tokens, situ_beta, situ_linear_beta):
    device = torch.device("cuda")
    hidden_size, intermediate_size = 512, 384
    fused, ref = _make_pair(hidden_size, intermediate_size, situ_beta, situ_linear_beta, device)

    torch.manual_seed(7)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    out_fused = fused(x)
    out_ref = ref(x)

    assert out_fused.shape == out_ref.shape and out_fused.dtype == out_ref.dtype
    # Identical weights and identical eager activation; the only difference
    # is one [H, 2I] GEMM vs two [H, I] GEMMs (kernel-tactic level).
    torch.testing.assert_close(out_fused, out_ref, rtol=1.6e-2, atol=1e-3)


@requires_cuda
def test_gate_up_half_swap_mutation_breaks_accuracy():
    """Up-first packing must break the numerics (guards the gate-first
    row-concat convention in load_weights)."""
    device = torch.device("cuda")
    hidden_size, intermediate_size = 512, 384
    fused, ref = _make_pair(hidden_size, intermediate_size, 4.0, 25.0, device)
    with torch.no_grad():
        swapped = torch.cat(
            [
                fused.gate_up_proj.weight[intermediate_size:],
                fused.gate_up_proj.weight[:intermediate_size],
            ],
            dim=0,
        )
        fused.gate_up_proj.weight.copy_(swapped)

    torch.manual_seed(13)
    x = torch.randn(64, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(fused(x), ref(x), rtol=1.6e-2, atol=1e-3)


@requires_cuda
@pytest.mark.parametrize(
    "situ_beta,situ_linear_beta",
    [(4.0, 25.0), (1.0, None)],
    ids=["default", "no_linear_beta"],
)
def test_gated_mlp_supports_fused_situ(situ_beta, situ_linear_beta):
    """The shared-expert replacement preserves K3 MLP numerics."""
    device = torch.device("cuda")
    hidden_size, intermediate_size = 512, 384
    torch.manual_seed(17)
    reference = KimiK3MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        use_fused_activation=True,
        dtype=torch.bfloat16,
        device=device,
    )
    gated = GatedMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        activation=SituAndMul(
            beta=situ_beta,
            linear_beta=situ_linear_beta,
            use_fused_activation=True,
        ),
        dtype=torch.bfloat16,
        reduce_output=True,
    ).to(device)

    with torch.no_grad():
        for projection in (reference.gate_up_proj, reference.down_proj):
            projection.weight.copy_(torch.randn_like(projection.weight, dtype=torch.float32) * 0.05)
        gated.gate_up_proj.weight.copy_(reference.gate_up_proj.weight)
        gated.down_proj.weight.copy_(reference.down_proj.weight)

    x = torch.randn(64, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    torch.testing.assert_close(gated(x), reference(x), rtol=1.6e-2, atol=1e-3)


@pytest.mark.parametrize(
    "attention_dp,tp_size,rank,expected_shared_tp,expected_shared_rank",
    [
        (True, 8, 7, 1, 0),
        (False, 1, 0, 1, 0),
        (False, 8, 7, 8, 7),
    ],
    ids=["attention_dp", "single_rank", "direct_tp"],
)
def test_kimi_k3_shared_expert_parallel_construction(
    monkeypatch,
    attention_dp,
    tp_size,
    rank,
    expected_shared_tp,
    expected_shared_rank,
):
    """Shared experts are replicated or sharded for the selected parallel mode."""
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _FakeAllReduce(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _FakeMoE(ConfigurableMoE):
        def __init__(self):
            nn.Module.__init__(self)
            from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind

            self.backend = SimpleNamespace(
                initial_local_expert_ids=[0, 1, 2, 3],
                scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
            )
            self.comm = None
            self.layer_load_balancer = None
            self.all_reduce = _FakeAllReduce()

    fake_moe = _FakeMoE()

    create_moe_kwargs = {}
    monkeypatch.setenv("KIMI_K3_ROUTER_BF16", "0")
    monkeypatch.setattr(
        modeling_kimi_linear,
        "create_moe",
        lambda **kwargs: create_moe_kwargs.update(kwargs) or fake_moe,
    )
    monkeypatch.setattr(distributed, "AllReduce", _FakeAllReduce)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())

    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        tp_size=tp_size,
        enable_attention_dp=attention_dp,
    )
    model_config = ModelConfig(
        mapping=mapping,
        quant_config=QuantConfig(),
        moe_backend="TRTLLM",
    )
    config = _runtime_config()
    moe_aux_stream_dict = {AuxStreamType.MoeChunkingOverlap: object()}
    runtime = modeling_kimi_linear.KimiK3MoERuntime(
        model_config,
        config,
        layer_idx=1,
        moe_aux_stream_dict=moe_aux_stream_dict,
    )

    shared = runtime.shared_experts
    assert create_moe_kwargs["aux_stream_dict"] is moe_aux_stream_dict
    assert isinstance(shared, GatedMLP)
    assert shared.gate_up_proj.tp_size == expected_shared_tp
    assert shared.gate_up_proj.tp_rank == expected_shared_rank
    assert shared.down_proj.tp_size == expected_shared_tp
    assert shared.down_proj.tp_rank == expected_shared_rank
    local_intermediate = (
        config.moe_intermediate_size * config.num_shared_experts // expected_shared_tp
    )
    assert shared.gate_up_proj.weight.shape == (2 * local_intermediate, config.hidden_size)
    assert shared.down_proj.weight.shape == (config.hidden_size, local_intermediate)


def _make_kimi_k3_moe_weights(config):
    generator = torch.Generator().manual_seed(2026)

    def randn(*shape, scale=0.05):
        return (
            torch.randn(
                *shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            .mul_(scale)
            .to(device="cuda", dtype=torch.bfloat16)
        )

    shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
    expert_weights = {}
    for expert_id in range(config.num_experts):
        expert_weights[f"{expert_id}.w1.weight"] = randn(
            config.moe_intermediate_size, config.routed_expert_hidden_size
        )
        expert_weights[f"{expert_id}.w2.weight"] = randn(
            config.routed_expert_hidden_size, config.moe_intermediate_size
        )
        expert_weights[f"{expert_id}.w3.weight"] = randn(
            config.moe_intermediate_size, config.routed_expert_hidden_size
        )

    return {
        "gate": randn(config.num_experts, config.hidden_size),
        "score_bias": torch.linspace(
            -0.15, 0.15, config.num_experts, dtype=torch.float32, device="cuda"
        ),
        "routed_down": randn(config.routed_expert_hidden_size, config.hidden_size),
        "routed_up": randn(config.hidden_size, config.routed_expert_hidden_size),
        "routed_norm": (
            1.0
            + randn(config.routed_expert_hidden_size, scale=0.02).to(torch.float32)
        ).to(torch.bfloat16),
        "shared_gate": randn(shared_intermediate, config.hidden_size),
        "shared_up": randn(shared_intermediate, config.hidden_size),
        "shared_down": randn(config.hidden_size, shared_intermediate),
        "experts": expert_weights,
    }


def _load_kimi_k3_moe_weights(module, weights):
    with torch.no_grad():
        module.gate.weight.copy_(weights["gate"])
        module.gate.e_score_correction_bias.copy_(weights["score_bias"])
        module.routed_expert_down_proj.weight.copy_(weights["routed_down"])
        module.routed_expert_up_proj.weight.copy_(weights["routed_up"])
        module.routed_expert_norm.weight.copy_(weights["routed_norm"])

    module.shared_experts.gate_up_proj.load_weights(
        [{"weight": weights["shared_gate"]}, {"weight": weights["shared_up"]}]
    )
    module.shared_experts.down_proj.load_weights([{"weight": weights["shared_down"]}])
    module.routed_experts.load_weights(
        [{name: value.clone() for name, value in weights["experts"].items()}]
    )
    module.routed_experts.post_load_weights()


def _kimi_k3_moe_reference(hidden_states, weights, config):
    from torch.nn import functional as F

    from tensorrt_llm._torch.moe.fused_moe import DeepSeekV3MoeRoutingMethod

    router_logits = F.linear(hidden_states.float(), weights["gate"].float())
    routing = DeepSeekV3MoeRoutingMethod(
        top_k=config.num_experts_per_token,
        n_group=config.num_expert_group,
        topk_group=config.topk_group,
        routed_scaling_factor=config.routed_scaling_factor,
        callable_e_score_correction_bias=lambda: weights["score_bias"],
        is_fused=True,
    )
    selected_experts, routing_weights = routing.apply(router_logits)
    routed_input = F.linear(hidden_states, weights["routed_down"])
    routed_latent = torch.zeros_like(routed_input, dtype=torch.float32)
    situ = SituAndMul(
        beta=config.activation_situ_beta,
        linear_beta=config.activation_situ_linear_beta,
    )
    for expert_id in range(config.num_experts):
        token_ids, topk_ids = torch.where(selected_experts == expert_id)
        if token_ids.numel() == 0:
            continue
        expert_input = routed_input[token_ids]
        gate_up = torch.cat(
            (
                F.linear(expert_input, weights["experts"][f"{expert_id}.w1.weight"]),
                F.linear(expert_input, weights["experts"][f"{expert_id}.w3.weight"]),
            ),
            dim=-1,
        )
        expert_output = F.linear(
            situ(gate_up), weights["experts"][f"{expert_id}.w2.weight"]
        )
        routed_latent.index_add_(
            0,
            token_ids,
            expert_output.float() * routing_weights[token_ids, topk_ids, None],
        )

    routed_latent = routed_latent.to(torch.bfloat16)
    variance = routed_latent.float().square().mean(dim=-1, keepdim=True)
    routed_latent = (
        routed_latent.float()
        * torch.rsqrt(variance + config.rms_norm_eps)
        * weights["routed_norm"].float()
    ).to(torch.bfloat16)
    routed_output = F.linear(routed_latent, weights["routed_up"])

    shared_gate_up = torch.cat(
        (
            F.linear(hidden_states, weights["shared_gate"]),
            F.linear(hidden_states, weights["shared_up"]),
        ),
        dim=-1,
    )
    shared_output = F.linear(situ(shared_gate_up), weights["shared_down"])
    return routed_output + shared_output


def _run_kimi_k3_moe_multi_rank_worker(attention_dp):
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    rank = mpi_rank()
    world_size = 2
    torch.cuda.set_device(rank)
    config = _runtime_config()
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = config.num_experts
    pretrained_config.hidden_size = config.routed_expert_hidden_size
    pretrained_config.intermediate_size = config.moe_intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        enable_attention_dp=attention_dp,
    )
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=QuantConfig(),
        moe_backend="CUTLASS",
        max_num_tokens=32,
    )

    previous_router_mode = os.environ.get("KIMI_K3_ROUTER_BF16")
    previous_comm_method = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
    previous_projection_mode = modeling_kimi_linear._K3_DISABLE_MIN_LATENCY_LATENT_PROJ
    previous_quant_resolver = modeling_kimi_linear.KimiK3MoERuntime._resolve_routed_quant_config
    previous_ckpt_resolver = modeling_kimi_linear._k3_expert_ckpt_spec
    runtime = None
    try:
        os.environ["KIMI_K3_ROUTER_BF16"] = "0"
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = "ALLGATHER"
        # Use unquantized CUTLASS experts so this test isolates distributed
        # K3 module numerics from checkpoint packing and quantization error.
        modeling_kimi_linear._K3_DISABLE_MIN_LATENCY_LATENT_PROJ = True
        modeling_kimi_linear.KimiK3MoERuntime._resolve_routed_quant_config = staticmethod(
            lambda *_: QuantConfig()
        )
        modeling_kimi_linear._k3_expert_ckpt_spec = lambda _: None

        with torch.device(f"cuda:{rank}"):
            runtime = modeling_kimi_linear.KimiK3MoERuntime(
                model_config,
                config,
                layer_idx=0,
                aux_stream=torch.cuda.Stream(),
            )
            weights = _make_kimi_k3_moe_weights(config)
            _load_kimi_k3_moe_weights(runtime, weights)

            input_generator = torch.Generator().manual_seed(
                9000 + rank if attention_dp else 9000
            )
            hidden_states = (
                torch.randn(
                    16,
                    config.hidden_size,
                    generator=input_generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                .mul_(0.3)
                .to(device="cuda", dtype=torch.bfloat16)
            )
            expected = _kimi_k3_moe_reference(hidden_states, weights, config)
            all_rank_num_tokens = [hidden_states.shape[0]] * world_size if attention_dp else None
            with torch.inference_mode(), with_multi_stream(True):
                actual = runtime(
                    hidden_states,
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            torch.testing.assert_close(actual, expected, rtol=8e-2, atol=8e-2)
    finally:
        if runtime is not None:
            runtime.routed_experts.destroy()
        modeling_kimi_linear._K3_DISABLE_MIN_LATENCY_LATENT_PROJ = previous_projection_mode
        modeling_kimi_linear.KimiK3MoERuntime._resolve_routed_quant_config = staticmethod(
            previous_quant_resolver
        )
        modeling_kimi_linear._k3_expert_ckpt_spec = previous_ckpt_resolver
        if previous_router_mode is None:
            os.environ.pop("KIMI_K3_ROUTER_BF16", None)
        else:
            os.environ["KIMI_K3_ROUTER_BF16"] = previous_router_mode
        if previous_comm_method is None:
            os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
        else:
            os.environ["TRTLLM_FORCE_COMM_METHOD"] = previous_comm_method


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("attention_dp", [False, True], ids=["attention_tp", "attention_dp"])
def test_kimi_k3_moe_multi_rank_output(mpi_pool_executor, attention_dp):
    results = mpi_pool_executor.map(
        _run_kimi_k3_moe_multi_rank_worker,
        [attention_dp] * mpi_pool_executor.num_workers,
    )
    assert list(results) == [None] * mpi_pool_executor.num_workers


@pytest.mark.parametrize(
    "attention_dp,tp_size,rank,gpus_per_node,intermediate_size,expected_tp_size,expected_tp_rank",
    [
        (True, 8, 7, 8, 516, 1, 0),
        (False, 8, 7, 8, 512, 8, 7),
        (False, 8, 7, 8, 516, 4, 3),
        (False, 8, 7, 8, 515, 1, 0),
        (False, 16, 15, 8, 512, 8, 7),
    ],
    ids=["attention_dp", "full_tp", "gcd_subgroup", "replicated", "single_node_cap"],
)
def test_kimi_k3_dense_layer_uses_gated_mlp(
    monkeypatch,
    attention_dp,
    tp_size,
    rank,
    gpus_per_node,
    intermediate_size,
    expected_tp_size,
    expected_tp_rank,
):
    """The first dense layer selects a block-aligned, node-local MLP TP group."""
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _IdentityAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states, attn_metadata):
            return hidden_states

    class _IdentityAllReduce(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    monkeypatch.setattr(modeling_kimi_linear, "KimiKDALinearAttention", _IdentityAttention)
    monkeypatch.setattr(distributed, "AllReduce", _IdentityAllReduce)

    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=tp_size,
        enable_attention_dp=attention_dp,
    )
    model_config = ModelConfig(mapping=mapping, quant_config=QuantConfig())
    config = SimpleNamespace(
        hidden_size=512,
        intermediate_size=intermediate_size,
        num_experts=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        linear_attn_config={"kda_layers": [1], "full_attn_layers": []},
        rms_norm_eps=1e-5,
        attn_res_block_size=1,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    layer = modeling_kimi_linear.KimiLinearDecoderLayer(model_config, config, layer_idx=0)

    assert not layer.is_moe
    assert isinstance(layer.mlp, GatedMLP)
    assert layer.mlp_tp_size == expected_tp_size
    assert layer.mlp.gate_up_proj.tp_size == expected_tp_size
    assert layer.mlp.gate_up_proj.tp_rank == expected_tp_rank
    assert layer.mlp.down_proj.tp_size == expected_tp_size
    assert layer.mlp.down_proj.tp_rank == expected_tp_rank
    assert layer.mlp.down_proj.reduce_output is (expected_tp_size > 1)
    local_intermediate = config.intermediate_size // expected_tp_size
    assert layer.mlp.gate_up_proj.weight.shape == (
        2 * local_intermediate,
        config.hidden_size,
    )
    assert layer.mlp.down_proj.weight.shape == (
        config.hidden_size,
        local_intermediate,
    )
