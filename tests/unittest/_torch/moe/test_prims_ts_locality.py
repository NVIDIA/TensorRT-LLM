# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimsTS output partitioning, BF16 dispatch and Rubin locality domains."""

import pytest
import torch
from _torch.moe.test_kimi_k3_situ_moe import _make_nvfp4_expert_bank
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import SiTuActivation, SwigluActivation, create_moe
from tensorrt_llm._torch.moe.fused_moe.fused_moe_prims_ts import (
    PrimsTSBf16FusedMoE,
    PrimsTSNvfp4FusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEEnvironment, MoERejectReason
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, override_moe_environment
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import impl_class_for, resolve_moe_impl
from tensorrt_llm._torch.moe.fused_moe.routing import RenormalizeMoeRoutingMethod
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize("sm", [100, 103, 107])
@pytest.mark.parametrize(
    "quant,expected", [(None, PrimsTSBf16FusedMoE), (QuantAlgo.NVFP4, PrimsTSNvfp4FusedMoE)]
)
def test_prims_ts_bf16_and_rubin_resolution(sm, quant, expected):
    config = ModelConfig(moe_backend="PRIMS_TS", quant_config=QuantConfig(quant_algo=quant))
    deps = (MoEDep.PRIMS_TS.value, MoEDep.CUTEDSL_RUBIN.value)
    with override_moe_environment(MoEEnvironment(sm=sm, available_deps=deps)):
        for impl_id in (None, expected.descriptor.impl_id):
            assert impl_class_for(resolve_moe_impl(config, impl_id=impl_id)) is expected


@pytest.mark.parametrize("quant", [None, QuantAlgo.NVFP4])
def test_prims_ts_rubin_requires_rubin_dependencies(quant):
    config = ModelConfig(moe_backend="PRIMS_TS", quant_config=QuantConfig(quant_algo=quant))
    expected = PrimsTSBf16FusedMoE if quant is None else PrimsTSNvfp4FusedMoE
    with override_moe_environment(MoEEnvironment(sm=107, available_deps=(MoEDep.PRIMS_TS.value,))):
        report = resolve_moe_impl(config, impl_id=expected.descriptor.impl_id)
    assert report.winner is None
    assert report.rejected[0].reason is MoERejectReason.DEP_MISSING
    assert "requires CUTLASS DSL Rubin helpers" in report.rejected[0].detail


def make_moe_pair(
    nvfp4: bool,
    situ: bool,
    *,
    hidden: int = 512,
    intermediate: int = 256,
    experts: int = 8,
    top_k: int = 2,
    expert_bank=None,
    localize: bool = True,
):
    """Load identical checkpoint tensors through both production module paths."""
    generator = torch.Generator().manual_seed(37)
    if expert_bank is not None:
        bank = expert_bank
    elif nvfp4:
        bank = _make_nvfp4_expert_bank(experts, intermediate, hidden)
        # Keep SwiGLU's intermediate values above the NVFP4 scale-factor
        # underflow range. Agreement between two all-zero outputs would not
        # exercise the partitioned output or scale-factor addressing.
        for expert in bank:
            for name in ("w1", "w2", "w3"):
                expert[f"{name}.weight_scale_2"].fill_(0.01)
    else:
        bank = [
            {
                "w1.weight": torch.randn(
                    intermediate, hidden, generator=generator, dtype=torch.bfloat16
                )
                * 0.02,
                "w3.weight": torch.randn(
                    intermediate, hidden, generator=generator, dtype=torch.bfloat16
                )
                * 0.02,
                "w2.weight": torch.randn(
                    hidden, intermediate, generator=generator, dtype=torch.bfloat16
                )
                * 0.02,
            }
            for _ in range(experts)
        ]
    checkpoint = {
        f"{index}.{key}": value
        for index, expert in enumerate(bank)
        for key, value in expert.items()
    }
    modules = []
    for localized in (False, localize):
        config = ModelConfig(
            pretrained_config=PretrainedConfig(
                hidden_size=hidden, intermediate_size=intermediate, num_experts=experts
            ),
            moe_backend="PRIMS_TS",
            quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4) if nvfp4 else None,
            locality_domain_policy=LocalityDomainPolicy(enabled=localized),
        )
        module = create_moe(
            routing_method=RenormalizeMoeRoutingMethod(top_k=top_k),
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            dtype=torch.bfloat16,
            reduce_results=True,
            model_config=config,
            layer_idx=0,
            activation=SiTuActivation(gate_softcap=4.0, linear_softcap=25.0)
            if situ
            else SwigluActivation(),
        ).cuda()
        module.backend.load_weights([checkpoint])
        module.post_load_weights()
        assert module.backend.uses_locality_domain is localized
        if localized:
            assert module.backend.w3_w1_weight.numel() == 0
            assert module.backend.w2_weight.numel() == 0
            assert len(module.backend._locality_domain_weight_shards) == 2
        modules.append(module)
    return modules, bank


def _bf16_reference(x, logits, routing_method, bank):
    ids, scales = routing_method.apply(logits)
    reference = torch.zeros_like(x, dtype=torch.float32)
    for expert_id, weights in enumerate(bank):
        gate = x.float() @ weights["w1.weight"].cuda().float().T
        up = x.float() @ weights["w3.weight"].cuda().float().T
        expert = (torch.nn.functional.silu(gate) * up) @ weights["w2.weight"].cuda().float().T
        scale = ((ids == expert_id) * scales.float()).sum(dim=-1)
        reference += expert * scale[:, None]
    return reference


rubin = pytest.mark.skipif(get_sm_version() != 107, reason="requires Rubin locality domains")


@rubin
@pytest.mark.parametrize("nvfp4,situ", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tokens", [1, 8, 64, 257])
def test_prims_ts_locality_matches_unpartitioned(nvfp4, situ, tokens, monkeypatch):
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    assert is_locality_domain_enabled()
    (baseline, candidate), bank = make_moe_pair(nvfp4, situ)
    torch.manual_seed(71)
    x = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(tokens, 8, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        expected = baseline(x, logits)
        actual = candidate(x, logits)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        if not nvfp4:
            reference = _bf16_reference(x, logits, baseline.backend.routing_method, bank)
            relative_error = torch.linalg.vector_norm(
                actual.float() - reference
            ) / torch.linalg.vector_norm(reference)
            assert relative_error < 0.02

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = candidate(x, logits)
        for _ in range(5):
            x.normal_(std=0.5)
            logits.normal_()
            expected = baseline(x, logits)
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="requires Blackwell")
@pytest.mark.parametrize("tokens", [1, 8, 257])
def test_prims_ts_bf16_blackwell_reference(tokens):
    (_, candidate), bank = make_moe_pair(False, False, localize=False)
    torch.manual_seed(73)
    x = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(tokens, 8, device="cuda")
    with torch.inference_mode():
        actual = candidate(x, logits)
        reference = _bf16_reference(x, logits, candidate.backend.routing_method, bank)
    relative_error = torch.linalg.vector_norm(
        actual.float() - reference
    ) / torch.linalg.vector_norm(reference)
    assert relative_error < 0.02


@rubin
@pytest.mark.parametrize("nvfp4", [False, True])
def test_prims_ts_locality_reuses_compiled_gemm(nvfp4, monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.flashinfer.prims_ts.moe import compile_cache

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    # A local intermediate width of 64 is one complete NVFP4 SF group.
    # Multiple token blocks also exercise the full shared SF leading stride.
    (baseline, candidate), _ = make_moe_pair(nvfp4, False, intermediate=128)
    torch.manual_seed(83)
    x = torch.randn(257, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(257, 8, device="cuda")
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        expected = baseline(x, logits)
        candidate(x, logits)
    context = capture._captured_contexts[-1]
    runner, inputs = context["runners"][0], context["inputs"]
    launches = {"fc1": [], "fc2": []}
    original = compile_cache.get_compiled_gemm

    def record_compiled(config_hash, stage, io, stream):
        compiled = original(config_hash, stage, io, stream)
        launches[stage].append((config_hash, compiled))
        return compiled

    monkeypatch.setattr(compile_cache, "get_compiled_gemm", record_compiled)
    with torch.inference_mode():
        actual = torch.empty_like(x)
        runner(inputs, output=actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
    for children in launches.values():
        assert len(children) == 2
        assert children[0][0] == children[1][0]
        assert children[0][1] is children[1][1]


@rubin
@pytest.mark.parametrize("nvfp4", [False, True])
def test_prims_ts_locality_empty_and_inactive_experts(nvfp4, monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.custom_ops.prims_ts_partitioned_moe import prims_ts_partitioned_moe

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    (_, candidate), _ = make_moe_pair(nvfp4, False)
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(8, 8, device="cuda")
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        candidate(x, logits)
    context = capture._captured_contexts[-1]
    runner, inputs = context["runners"][0], list(context["inputs"])
    shards = runner.locality_weights
    # Test the operator's zero-token contract directly: the shared external
    # scheduler requires a positive global chunk count before backend dispatch.
    with torch.inference_mode():
        empty, empty_map, _ = prims_ts_partitioned_moe(
            inputs[2][:0],
            inputs[3][:0] if nvfp4 else None,
            [s[0] for s in shards],
            [s[1] for s in shards] if nvfp4 else [],
            [s[2] for s in shards],
            [s[3] for s in shards] if nvfp4 else [],
            inputs[7],
            inputs[8],
            inputs[9],
            inputs[13],
            inputs[14],
            inputs[15],
            inputs[-1][:0],
            inputs[-2][:0],
            8,
            0,
            runner.activation_type,
            True,
            x.new_empty((0, 512)),
        )
    assert empty.shape == (0, 512) and empty_map.shape == (0, 2)
    # Keep eight local experts but route every token to remote experts.
    runner.num_experts = 16
    inputs[-1] = torch.tensor([8, 9], device="cuda", dtype=torch.int32).expand(8, 2).clone()
    inputs[-2] = torch.full((8, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    output = torch.full_like(x, float("nan"))
    with torch.inference_mode():
        runner(inputs, output=output)
        assert torch.count_nonzero(output) == 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner(inputs, output=output)
        # Retain and update captured input tensors so both nonempty and empty
        # local routing are exercised in the same graph.
        for expert_ids in ([0, 1], [8, 9], [2, 8], [8, 9]):
            inputs[-1].copy_(torch.tensor(expert_ids, device="cuda", dtype=torch.int32))
            expected = torch.empty_like(x)
            runner(inputs, output=expected)
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            if expert_ids == [8, 9]:
                assert torch.count_nonzero(output) == 0
            else:
                assert torch.count_nonzero(output) > 0


@rubin
@pytest.mark.parametrize("nvfp4", [False, True])
def test_prims_ts_locality_concurrent_autotune(nvfp4, monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner, TuningConfig, autotune
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.custom_ops.prims_ts_moe import PrimsTSMoERunner

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    # Bound the correctness test to two distinct routing tiles. Both weight
    # partitions still execute concurrently during each measured tactic.
    monkeypatch.setattr(FP4BlockScaleMoERunner, "get_tuning_config", lambda *a: TuningConfig())
    monkeypatch.setattr(PrimsTSMoERunner, "get_valid_tactics", lambda *a, **kw: [[8, 0], [16, 0]])
    (_, candidate), _ = make_moe_pair(nvfp4, False)
    x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(16, 8, device="cuda")
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        expected = candidate(x, logits)
    context = capture._captured_contexts[-1]
    runner = context["runners"][0]
    assert runner.locality_runtime is not None
    assert len(runner.locality_weights) == 2
    assert not context["tuning_config"].use_cold_l2_cache
    assert "locality_v2" in runner.unique_id()
    with torch.inference_mode(), autotune():
        actual = candidate(x, logits)
    torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.1)


@rubin
@pytest.mark.parametrize("hidden,intermediate", [(512, 256), (3584, 3072)])
def test_prims_ts_locality_nvfp4_situ_reference(monkeypatch, hidden, intermediate):
    from _torch.moe.test_kimi_k3_situ_moe import _quantize_expert_to_nvfp4, _situ_reference_moe

    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    torch.manual_seed(91)
    x = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(8, 8, device="cuda")
    w1 = [
        torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
        for _ in range(8)
    ]
    w3 = [
        torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
        for _ in range(8)
    ]
    w2 = [
        torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.05
        for _ in range(8)
    ]
    bank = [_quantize_expert_to_nvfp4(w1[e], w2[e], w3[e], 1.0) for e in range(8)]
    (_, candidate), _ = make_moe_pair(
        True, True, hidden=hidden, intermediate=intermediate, expert_bank=bank
    )
    shards = candidate.backend._locality_domain_weight_shards
    # The full hook after staged loading must preserve the localized tensors.
    candidate.backend.post_load_weights()
    assert candidate.backend._locality_domain_weight_shards is shards
    with torch.inference_mode():
        actual = candidate(x, logits).float()
        reference = _situ_reference_moe(
            x, logits, candidate.backend.routing_method, w1, w2, w3, beta=4.0, linear_beta=25.0
        )
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
    relative_error = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(
        reference
    )
    # Match the existing K3 NVFP4 reference gate, which includes weight and
    # activation quantization error relative to the original BF16 weights.
    assert cosine > 0.95
    assert relative_error < 0.35


@rubin
def test_prims_ts_rubin_nvfp4_large_batch_tactics(monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    modules, _ = make_moe_pair(True, True)
    x = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(2048, 8, device="cuda")
    outputs = []
    with torch.inference_mode():
        for module in modules:
            with AutoTuner.get().capture() as capture:
                expected = module(x, logits)
            context = capture._captured_contexts[-1]
            runner, inputs = context["runners"][0], context["inputs"]
            tactics = runner.get_valid_tactics(inputs, None)
            assert tactics and {tactic[0] for tactic in tactics} == {128}
            assert "rubin_k128_v1" in runner.unique_id()
            actual = torch.empty_like(x)
            runner(inputs, tactic=(128, 0), output=actual)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            with pytest.raises(ValueError, match="token tile of at most 128"):
                runner(inputs, tactic=(256, 0))
            outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
