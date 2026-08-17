# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 fused gate_up MLP tests.

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
* shared-expert sharding and reduction follow the selected parallel mode.
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.modules.moe.kimi_k3_ref_moe.kimi_k3_mlp_test_utils import KimiK3MLP
from torch import nn

from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.situ import SituAndMul

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
    from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _FakeAllReduce(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _FakeMoE(ConfigurableMoE):
        def __init__(self):
            nn.Module.__init__(self)
            self.backend = SimpleNamespace(initial_local_expert_ids=[0, 1, 2, 3])
            self.comm = None
            self.layer_load_balancer = None
            self.all_reduce = _FakeAllReduce()

    fake_moe = _FakeMoE()

    monkeypatch.setenv("KIMI_K3_ROUTER_BF16", "0")
    monkeypatch.setattr(modeling_kimi_linear, "create_moe", lambda **_: fake_moe)
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
    runtime = modeling_kimi_linear.KimiK3MoERuntime(model_config, config, layer_idx=1)

    shared = runtime.shared_experts
    assert isinstance(shared, GatedMLP)
    assert shared.gate_up_proj.tp_size == expected_shared_tp
    assert shared.gate_up_proj.tp_rank == expected_shared_rank
    assert shared.down_proj.tp_size == expected_shared_tp
    assert shared.down_proj.tp_rank == expected_shared_rank
    assert shared.down_proj.reduce_output is False
    local_intermediate = (
        config.moe_intermediate_size * config.num_shared_experts // expected_shared_tp
    )
    assert shared.gate_up_proj.weight.shape == (2 * local_intermediate, config.hidden_size)
    assert shared.down_proj.weight.shape == (config.hidden_size, local_intermediate)


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
