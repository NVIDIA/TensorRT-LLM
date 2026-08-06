# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 fused gate_up MLP tests.

The runtime dense / shared-expert ``GatedMLP`` runs a single fused
``gate_up_proj`` GEMM with K3's SiTU activation. ``KimiK3MLP`` remains the
compact reference for the same fused weight layout. These tests check that
layout against an unfused reference built from the HF checkpoint's split
``gate_proj`` / ``up_proj`` weights:

* fused ``gate_up_proj`` output matches ``two GEMMs + torch.cat`` +
  eager ``SituAndMul`` + ``down_proj`` across token counts and
  ``situ_beta`` / ``situ_linear_beta`` settings (incl. ``None``);
* the row-concat convention is load-bearing: swapping the halves breaks
  the numerics (mutation control).
* the post-load FP8 weight-read replacement preserves ``GatedMLP``'s
  callable interface and output.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _torch.modules.moe.kimi_k3_mlp_test_utils import KimiK3MLP
from torch import nn

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.kimi_k3_moe._mlp import SituAndMul

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
        (2.5, 7.0),  # asymmetric non-defaults
        (1.0, None),  # linear_beta disabled (identity up half)
    ],
    ids=["default", "asymmetric", "no_linear_beta"],
)
@pytest.mark.parametrize("num_tokens", [1, 5, 64, 500], ids=lambda n: f"tokens{n}")
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
    """Up-first packing must break the numerics (the gate-first row-concat
    convention in load_weights is load-bearing)."""
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


def test_gated_mlp_supports_kimi_fp8_weight_read(monkeypatch):
    """The K3 post-load FP8 swap remains callable through ``GatedMLP``."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _convert_moe_mlps_to_fp8_weight_read,
        _Fp8BlockScaleWeightReadLinear,
    )

    hidden_size, intermediate_size = 32, 48
    gated = GatedMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        activation=SituAndMul(beta=4.0, linear_beta=25.0),
        dtype=torch.float32,
        overridden_tp_size=1,
        reduce_output=False,
    )
    with torch.no_grad():
        gated.gate_up_proj.weight.normal_(std=0.05)
        gated.down_proj.weight.normal_(std=0.05)

    x = torch.randn(7, hidden_size)
    reference = gated(x)

    def fake_quantize_weight(weight):
        return weight.detach().clone(), torch.ones(1)

    def fake_fp8_gemm(x, weight, weight_scale, **kwargs):
        return F.linear(x, weight)

    monkeypatch.setattr(
        _Fp8BlockScaleWeightReadLinear,
        "quantize_weight",
        staticmethod(fake_quantize_weight),
    )
    monkeypatch.setattr(torch.ops.trtllm, "fp8_swap_ab_gemm", fake_fp8_gemm)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    moe = SimpleNamespace(shared_experts=gated)
    model = SimpleNamespace(layers=[SimpleNamespace(block_sparse_moe=moe)])
    assert _convert_moe_mlps_to_fp8_weight_read(model) == 2
    assert isinstance(gated.gate_up_proj, _Fp8BlockScaleWeightReadLinear)
    assert isinstance(gated.down_proj, _Fp8BlockScaleWeightReadLinear)
    torch.testing.assert_close(gated(x), reference)


@pytest.mark.parametrize(
    "attention_dp,tp_size,has_routed_comm,expected_shared_tp,expected_deferred_routed_ar",
    [
        (True, 2, True, 1, False),
        (False, 1, False, 1, False),
        (False, 2, False, 2, True),
        (False, 2, True, 2, False),
    ],
    ids=["attention_dp", "single_rank", "ordered_tp", "routed_comm"],
)
def test_kimi_k3_shared_expert_reduction_mode(
    monkeypatch,
    attention_dp,
    tp_size,
    has_routed_comm,
    expected_shared_tp,
    expected_deferred_routed_ar,
):
    """Each parallel mode selects exactly the required shared reduction."""
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _FakeAllReduce(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    fake_moe = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(fake_moe)
    fake_moe.backend = SimpleNamespace(initial_local_expert_ids=[0, 1, 2, 3])
    fake_moe.comm = object() if has_routed_comm else None
    fake_moe.layer_load_balancer = None
    fake_moe.all_reduce = _FakeAllReduce() if not attention_dp and tp_size > 1 else None

    monkeypatch.setattr(modeling_kimi_linear, "create_moe", lambda **_: fake_moe)
    monkeypatch.setattr(modeling_kimi_linear, "AllReduce", _FakeAllReduce)
    monkeypatch.setattr(distributed, "AllReduce", _FakeAllReduce)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())

    mapping = Mapping(
        world_size=tp_size,
        rank=0,
        tp_size=tp_size,
        enable_attention_dp=attention_dp,
    )
    model_config = ModelConfig(mapping=mapping, quant_config=QuantConfig())
    config = _runtime_config()
    runtime = modeling_kimi_linear.KimiK3MoERuntime(model_config, config, layer_idx=1)

    shared = runtime.shared_experts
    assert isinstance(shared, GatedMLP)
    assert shared.gate_up_proj.tp_size == expected_shared_tp
    assert shared.down_proj.tp_size == expected_shared_tp
    assert shared.down_proj.reduce_output is (expected_shared_tp > 1)
    assert runtime._defer_routed_all_reduce is expected_deferred_routed_ar
    local_intermediate = 512 // expected_shared_tp
    assert shared.gate_up_proj.weight.shape == (2 * local_intermediate, config.hidden_size)
    assert shared.down_proj.weight.shape == (config.hidden_size, local_intermediate)


@pytest.mark.parametrize("fused_finalize", [False, True], ids=["plain", "fused_finalize"])
def test_kimi_k3_routed_allreduce_waits_for_shared_allreduce(monkeypatch, fused_finalize):
    """The routed TP reduction runs only after the shared branch rejoins."""
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    events = []

    class _RoutedAllReduce(nn.Module):
        supports_moe_finalize_allreduce_rms_norm = fused_finalize

        def forward(self, hidden_states):
            assert events[-1] == "joined"
            assert hidden_states.shape[-1] == 256
            events.append("routed_all_reduce")
            return hidden_states * 2

        def moe_finalize_allreduce_rms_norm(
            self,
            fc2_output,
            expert_scale_factor,
            expanded_idx_to_permuted_idx,
            norm_weight,
            variance_epsilon,
        ):
            del expert_scale_factor, expanded_idx_to_permuted_idx, norm_weight, variance_epsilon
            assert events[-1] == "joined"
            events.extend(("routed_all_reduce", "norm"))
            return fc2_output * 2

    class _SharedAllReduce(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _FakeMoE(ConfigurableMoE):
        def __init__(self):
            nn.Module.__init__(self)
            self.backend = SimpleNamespace(initial_local_expert_ids=[0, 1, 2, 3])
            self.comm = None
            self.layer_load_balancer = None
            self.all_reduce = _RoutedAllReduce()

        def forward(self, hidden_states, *args, do_finalize=True, **kwargs):
            events.append("routed")
            if not do_finalize:
                return hidden_states, object(), object()
            return hidden_states

    class _SharedExpert(nn.Module):
        def forward(self, hidden_states):
            events.append("shared")
            events.append("shared_all_reduce")
            return hidden_states + 1

    class _LatentDownProjection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states[:, :256]

    class _LatentNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(256))
            self.variance_epsilon = 1e-5

        def forward(self, hidden_states):
            assert events[-1] == "routed_all_reduce"
            events.append("norm")
            return hidden_states

    class _LatentUpProjection(nn.Module):
        def forward(self, hidden_states):
            assert events[-1] == "norm"
            events.append("up")
            return torch.cat((hidden_states, hidden_states), dim=-1)

    class _Gate(nn.Module):
        def compute_logits(self, hidden_states):
            return hidden_states.new_zeros(hidden_states.shape[0], 8)

    def _execute_in_parallel(main, aux, *args, **kwargs):
        routed_output = main()
        shared_output = aux()
        events.append("joined")
        return routed_output, shared_output

    monkeypatch.setattr(modeling_kimi_linear, "create_moe", lambda **_: _FakeMoE())
    monkeypatch.setattr(distributed, "AllReduce", _SharedAllReduce)
    monkeypatch.setattr(modeling_kimi_linear, "maybe_execute_in_parallel", _execute_in_parallel)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", True)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_DISABLE_FUSED_LATENT_DOWN_MXFP8", True)
    monkeypatch.setattr(
        modeling_kimi_linear, "_K3_DISABLE_FUSED_MOE_FINALIZE_AR_RMS", not fused_finalize
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())

    mapping = Mapping(world_size=2, rank=0, tp_size=2)
    model_config = ModelConfig(mapping=mapping, quant_config=QuantConfig())
    runtime = modeling_kimi_linear.KimiK3MoERuntime(model_config, _runtime_config(), layer_idx=1)
    assert runtime.shared_experts.down_proj.reduce_output
    assert runtime._defer_routed_all_reduce
    runtime.gate = _Gate()
    runtime.shared_experts = _SharedExpert()
    runtime.routed_expert_down_proj = _LatentDownProjection()
    runtime.routed_expert_norm = _LatentNorm()
    runtime.routed_expert_up_proj = _LatentUpProjection()

    hidden_states = torch.ones(2, 512, dtype=torch.bfloat16)
    output = runtime(hidden_states)

    assert events == [
        "routed",
        "shared",
        "shared_all_reduce",
        "joined",
        "routed_all_reduce",
        "norm",
        "up",
    ]
    torch.testing.assert_close(output, torch.full_like(hidden_states, 4))


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

    monkeypatch.setattr(modeling_kimi_linear, "KimiKDARuntime", _IdentityAttention)
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
