# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' routed-expert routing on the generic path: the gate's ``KimiK3MoeRoutingMethod`` computes the
top-16 outside the TRTLLM-Gen kernel.

Host-side, the method each target's gate hands ``ConfigurableMoE``. On SM 10.0, one generic MoE call as
``KimiK3MoERuntime`` builds it (W4A8_MXFP4_MXFP8 on TRTLLM-Gen, SiTU), routed outside the kernel, against the same
call on the same experts routed inside it."""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as route_a,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
    modeling as route_b,
)
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod, RoutingMethodType

TARGETS = pytest.mark.parametrize(
    "target", [route_a, route_b], ids=["tp16_moetp4ep4", "tp16_moetp16ep1"]
)

NUM_EXPERTS, TOP_K, HIDDEN, LATENT = 896, 16, 7168, 3584
# One tp16_moetp16ep1 rank's expert width: 192 of 3072, zero-padded to 256 by the loader.
INTERMEDIATE = 192
SITU_CAPS = (4.0, 25.0)


def _gate_config(routed_scaling_factor=2.5):
    """The checkpoint config fields KimiK3MoEGate reads."""
    return types.SimpleNamespace(
        num_experts_per_token=TOP_K,
        num_experts=NUM_EXPERTS,
        routed_scaling_factor=routed_scaling_factor,
        moe_router_activation_func="sigmoid",
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        hidden_size=HIDDEN,
    )


@TARGETS
def test_gate_routes_outside_the_kernel(target):
    """The gate's method routes outside the kernel and is a DeepSeek-V3 method to everything that dispatches on one:
    the backend's fused route + quant, the routing arguments the kernel receives and the routing type it is told."""
    gate = target.KimiK3MoEGate(_gate_config())
    method = gate.routing_method
    assert type(method) is target.KimiK3MoeRoutingMethod
    assert isinstance(method, DeepSeekV3MoeRoutingMethod)
    assert method.requires_separated_routing
    assert method.routing_method_type == RoutingMethodType.DeepSeekV3
    impl = method.routing_impl
    assert (impl.top_k, impl.n_group, impl.topk_group, impl.routed_scaling_factor) == (
        TOP_K,
        1,
        1,
        2.5,
    )
    assert impl.is_fused
    assert method.e_score_correction_bias is gate.e_score_correction_bias


def _sm_100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


def _gate(target):
    """The target's gate, random bf16 weights and fp32 correction bias."""
    g = torch.Generator(device="cuda").manual_seed(71)
    gate = target.KimiK3MoEGate(_gate_config(), logits_gemm_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        gate.weight.copy_(0.02 * torch.randn(gate.weight.shape, generator=g, device="cuda"))
        gate.e_score_correction_bias.copy_(
            0.05 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
        )
    return gate


def _routed_experts(routing_method):
    """The routed experts of one tp16_moetp16ep1 rank as KimiK3MoERuntime builds them (create_moe: ConfigurableMoE,
    W4A8_MXFP4_MXFP8 on TRTLLM-Gen, SiTU), every expert loaded with the same random packed MXFP4 checkpoint slice
    through the per-expert loader the target's weight load uses."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    pretrained = PretrainedConfig()
    pretrained.num_experts = NUM_EXPERTS
    pretrained.hidden_size = LATENT
    pretrained.intermediate_size = INTERMEDIATE
    pretrained.torch_dtype = torch.bfloat16
    pretrained.activation_situ_beta, pretrained.activation_situ_linear_beta = SITU_CAPS
    moe = create_moe(
        routing_method=routing_method,
        num_experts=NUM_EXPERTS,
        hidden_size=LATENT,
        intermediate_size=INTERMEDIATE,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=ModelConfig(
            pretrained_config=pretrained, mapping=Mapping(), moe_backend="TRTLLM"
        ),
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
        layer_idx=0,
        communication_method=None,
        activation=SiTuActivation(gate_softcap=SITU_CAPS[0], linear_softcap=SITU_CAPS[1]),
    ).cuda()
    assert isinstance(moe, ConfigurableMoE)
    backend = moe.backend
    assert type(backend).__name__ == "TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl", type(backend).__name__

    # The checkpoint's TP16 rank-0 slice (192 rows of 3072), as the loader slices and pads it.
    loader = types.SimpleNamespace(
        expert_size_per_partition=backend.expert_size_per_partition,
        initial_local_expert_ids=backend.initial_local_expert_ids,
        scaling_vector_size=backend.scaling_vector_size,
        intermediate_size=INTERMEDIATE * 16,
        intermediate_size_per_partition=INTERMEDIATE,
        tp_size=16,
        tp_rank=0,
        w3_w1_weight=backend.w3_w1_weight,
        w2_weight=backend.w2_weight,
        w3_w1_weight_scale=backend.w3_w1_weight_scale,
        w2_weight_scale=backend.w2_weight_scale,
    )
    g = torch.Generator(device="cuda").manual_seed(101)

    def packed(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device="cuda")

    def scales(*shape):
        return torch.randint(118, 124, shape, generator=g, dtype=torch.uint8, device="cuda")

    for e in range(NUM_EXPERTS):
        backend.quant_method.load_packed_mxfp4_expert(
            loader,
            global_expert_id=e,
            local_slot_id=e,
            w1_weight=packed(INTERMEDIATE, LATENT // 2),
            w1_weight_scale=scales(INTERMEDIATE, LATENT // 32),
            w2_weight=packed(LATENT, INTERMEDIATE // 2),
            w2_weight_scale=scales(LATENT, INTERMEDIATE // 32),
            w3_weight=packed(INTERMEDIATE, LATENT // 2),
            w3_weight_scale=scales(INTERMEDIATE, LATENT // 32),
        )
    backend._weights_transformed = False
    moe.post_load_weights()
    return moe


def _clear_rows(logits, bias, routed_scaling_factor):
    """Tokens whose routing fp32 arithmetic cannot round two ways: no two of the top 17 biased scores within 1e-6 of
    each other (the top-16 and its order are the same however fp32 rounds), and no top-16 weight within 2^-20 of a
    bf16 rounding boundary (its bf16 value is the same however the fp32 normalization rounds). Computed in fp64."""
    scores = torch.sigmoid(logits.double())
    top = torch.topk(scores + bias.double(), TOP_K + 1, dim=1)
    no_tie = (top.values[:, :-1] - top.values[:, 1:]).min(dim=1).values > 1e-6
    w = scores.gather(1, top.indices[:, :TOP_K])
    w = w / w.sum(dim=1, keepdim=True) * routed_scaling_factor
    ulp = torch.pow(2.0, torch.floor(torch.log2(w)) - 7)
    frac = w / ulp - torch.floor(w / ulp)
    no_boundary = ((frac - 0.5).abs() * ulp / w >= 2.0**-20).all(dim=1)
    return no_tie & no_boundary


@pytest.mark.skipif(
    not _sm_100(), reason="the TRTLLM-Gen MXFP4 cubins and the fused route + quant run on sm_100"
)
@TARGETS
def test_routing_outside_the_kernel_matches_inside(target, monkeypatch):
    """One generic MoE call routed outside the kernel (the gate's method: the backend's fused route + MXFP8 quant up to
    64 tokens, noaux_tc_op above) equals the same call on the same experts routed inside it
    (DeepSeekV3MoeRoutingMethod), bit for bit, on every token whose routing cannot round two ways (``_clear_rows``);
    the others within a bf16 ulp of the row's largest value."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_w4a8_mxfp4_mxfp8 import (
        TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
    )

    gate = _gate(target)
    outside = gate.routing_method
    inside = DeepSeekV3MoeRoutingMethod(
        TOP_K,
        1,
        1,
        outside.routing_impl.routed_scaling_factor,
        lambda: gate.e_score_correction_bias,
    )
    moe_outside = _routed_experts(outside)
    moe_inside = _routed_experts(inside)

    fused, applied = [], []
    fused_route_quant = TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl.try_fused_route_quant
    apply = target.KimiK3MoeRoutingMethod.apply

    def spy_fused(self, x, router_logits):
        out = fused_route_quant(self, x, router_logits)
        fused.append((self.routing_method is outside, x.shape[0], out is not None))
        return out

    def spy_apply(self, router_logits, input_ids=None):
        applied.append(router_logits.shape[0])
        return apply(self, router_logits, input_ids)

    monkeypatch.setattr(TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl, "try_fused_route_quant", spy_fused)
    monkeypatch.setattr(target.KimiK3MoeRoutingMethod, "apply", spy_apply)

    g = torch.Generator(device="cuda").manual_seed(5)
    with torch.inference_mode():
        for num_tokens in (1, 9, 64, 65, 300):
            hidden = torch.randn(num_tokens, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
            x = torch.randn(num_tokens, LATENT, generator=g, device="cuda").to(torch.bfloat16)
            logits = gate.compute_logits(hidden)
            fused.clear()
            applied.clear()
            out = moe_outside(x, logits)
            # The scheduler routed outside the kernel: the fused route + quant up to 64 tokens, the method above.
            assert fused == [(True, num_tokens, num_tokens <= 64)], fused
            assert applied == ([] if num_tokens <= 64 else [num_tokens]), applied
            fused.clear()
            ref = moe_inside(x, logits)
            assert fused == [], fused
            assert torch.isfinite(ref.float()).all()

            clear = _clear_rows(
                logits, gate.e_score_correction_bias, outside.routing_impl.routed_scaling_factor
            )
            assert clear.float().mean() > 0.9, clear.float().mean()
            assert torch.equal(out[clear].view(torch.int16), ref[clear].view(torch.int16)), (
                num_tokens
            )
            rest = ~clear
            if rest.any():
                err = (out[rest].float() - ref[rest].float()).abs().amax(dim=1)
                assert (err <= ref[rest].float().abs().amax(dim=1) * 2.0**-8).all(), (
                    num_tokens,
                    err,
                )
