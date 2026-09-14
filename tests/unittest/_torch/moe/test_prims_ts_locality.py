# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimsTS output partitioning, BF16 dispatch and Rubin locality domains."""

from copy import copy
from functools import partial

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
    routing_method=None,
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
            routing_method=routing_method or RenormalizeMoeRoutingMethod(top_k=top_k),
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


def _assert_locality_close(actual, expected, *, fused=True):
    if not fused:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    # Fused finalize rounds each weighted contribution to BF16 and performs
    # unordered BF16 bulk adds, matching the CuTe DSL finalize arithmetic.
    error = actual.float() - expected.float()
    reference_norm = torch.linalg.vector_norm(expected.float())
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(error) <= 0.02 * reference_norm
    assert error.abs().max() <= 0.02 * expected.float().abs().max()


@pytest.mark.skipif(get_sm_version() not in (100, 103, 107), reason="requires Blackwell or Rubin")
@pytest.mark.parametrize("nvfp4,tactic", [(False, (64, 0)), (True, (8, 18)), (True, (16, 12))])
@pytest.mark.parametrize("reuse", [2, 4])
def test_prims_ts_partitioned_persistent_graph(nvfp4, tactic, reuse, monkeypatch):
    """Exercise multiple expert tiles per resident cluster on ordinary streams."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.moe.custom_ops import prims_ts_moe as ops

    class StreamPartitions:
        def __init__(self):
            self.streams = [torch.cuda.Stream() for _ in range(2)]

        def fork(self):
            for stream in self.streams:
                stream.wait_stream(torch.cuda.current_stream())

        def join(self):
            for stream in self.streams:
                torch.cuda.current_stream().wait_stream(stream)

        def partition_context(self, index):
            return torch.cuda.stream(self.streams[index])

    # One resident cluster guarantees work-tile iteration even for tiny batches.
    monkeypatch.setattr(ops, "_full_device_cluster_budget", lambda *args: 1)
    (_, baseline), _ = make_moe_pair(nvfp4, False, hidden=1024, intermediate=1024, localize=False)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(16, 8, device="cuda")
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        baseline(x, logits)
    context = next(
        c for c in capture._captured_contexts if isinstance(c["runners"][0], ops.PrimsTSMoERunner)
    )
    runner, inputs = context["runners"][0], context["inputs"]
    candidate = copy(runner)
    candidate.locality_runtime = StreamPartitions()
    shards = []
    for partition in range(2):
        shard = []
        for index in (4, 5, 10, 11):
            tensor = inputs[index]
            if tensor is None:
                shard.append(None)
            else:
                axis = 2 if tensor.ndim == 4 else 1
                width = tensor.shape[axis] // 2
                shard.append(tensor.narrow(axis, partition * width, width).clone())
        shards.append(tuple(shard))
    candidate.locality_weights = tuple(shards)
    partitioned_inputs = list(inputs)
    for index, tensor in zip((4, 5, 10, 11), shards[0]):
        partitioned_inputs[index] = tensor
    expected, actual = torch.empty_like(x), torch.empty_like(x)
    with torch.inference_mode():
        runner(inputs, tactic=tactic, output=expected)
        for fused in (False, True):
            candidate.fuse_finalize = fused
            candidate(partitioned_inputs, tactic=(*tactic, reuse), output=actual)
            _assert_locality_close(actual, expected, fused=fused)
            assert torch.count_nonzero(actual) > 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate(partitioned_inputs, tactic=(*tactic, reuse), output=actual)
        for _ in range(5):
            inputs[-1].add_(1).remainder_(8)
            inputs[-2].uniform_(0.1, 0.9)
            runner(inputs, tactic=tactic, output=expected)
            graph.replay()
            _assert_locality_close(actual, expected)


@pytest.mark.skipif(get_sm_version() != 107, reason="requires Rubin locality domains")
@pytest.mark.parametrize("nvfp4,mma_group", [(False, 2), (True, 1), (True, 2)])
@pytest.mark.parametrize("reuse", [2, 4])
def test_prims_ts_locality_activation_multicast(nvfp4, mma_group, reuse, monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.custom_ops.prims_ts_moe import PrimsTSMoERunner
    from tensorrt_llm._torch.moe.flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_bf16_moe_tactic,
        map_trtllm_nvfp4_moe_tactic,
    )
    from tensorrt_llm._torch.moe.flashinfer.tllm_enums import WeightLayout

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    (baseline, candidate), _ = make_moe_pair(nvfp4, False, hidden=1024, intermediate=1024)
    x = torch.randn(16, 1024, dtype=torch.bfloat16, device="cuda") * 0.5
    logits = torch.randn(16, 8, device="cuda")
    tuner = AutoTuner.get()
    with torch.inference_mode(), tuner.capture() as capture:
        candidate(x, logits)
    context = next(
        context
        for context in capture._captured_contexts
        if isinstance(context["runners"][0], PrimsTSMoERunner)
    )
    runner, inputs = context["runners"][0], context["inputs"]
    mapper = map_trtllm_nvfp4_moe_tactic if nvfp4 else map_trtllm_bf16_moe_tactic
    selected = None
    for tactic in runner.get_valid_tactics(inputs, None):
        if len(tactic) != 3 or tactic[2] != reuse:
            continue
        pair = mapper(
            tactic[:2],
            num_tokens=16,
            top_k=2,
            num_local_experts=8,
            enable_pdl=runner.enable_pdl,
            weight_layout=int(
                WeightLayout.BlockMajorK if inputs[10].ndim == 4 else WeightLayout.MajorK
            ),
        )
        if pair.fc1.cfg.kwargs["cluster_m"] == mma_group:
            selected = tactic
            break
    assert selected is not None, "autotuning must expose every supported multicast cluster"
    choose_one = tuner.choose_one

    def choose_multicast(*args, **kwargs):
        chosen_runner, tactic = choose_one(*args, **kwargs)
        if (
            isinstance(chosen_runner, PrimsTSMoERunner)
            and chosen_runner.locality_runtime is not None
        ):
            return chosen_runner, selected
        return chosen_runner, tactic

    monkeypatch.setattr(tuner, "choose_one", choose_multicast)
    with torch.inference_mode():
        expected = baseline(x, logits)
        actual = candidate(x, logits)
        _assert_locality_close(actual, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = candidate(x, logits)
        for _ in range(3):
            x.normal_(std=0.5)
            logits.normal_()
            expected = baseline(x, logits)
            graph.replay()
            _assert_locality_close(captured, expected)


rubin = pytest.mark.skipif(get_sm_version() != 107, reason="requires Rubin locality domains")


@rubin
def test_prims_ts_locality_releases_stale_primary_pool_storage(monkeypatch):
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    is_locality_domain_enabled.cache_clear()
    config = ModelConfig(
        pretrained_config=PretrainedConfig(hidden_size=512, intermediate_size=256),
        moe_backend="PRIMS_TS",
        locality_domain_policy=LocalityDomainPolicy(enabled=True),
    )
    module = create_moe(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=8,
        hidden_size=512,
        intermediate_size=256,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=config,
        layer_idx=0,
        activation=SwigluActivation(),
    ).cuda()
    # Finish the primary weight layout before simulating storage left behind
    # by previously migrated layers. The locality transform is still pending.
    TRTLLMGenFusedMoE.transform_weights(module.backend)
    torch.cuda.empty_cache()
    cached_bytes = 512 * 1024**2
    old_layer_storage = torch.empty(cached_bytes, device="cuda", dtype=torch.uint8)
    del old_layer_storage
    torch.cuda.synchronize()
    unused_before = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert unused_before >= cached_bytes

    module.post_load_weights()
    torch.cuda.synchronize()
    assert module.backend.uses_locality_domain
    assert len(module.backend._locality_domain_weight_shards) == 2
    unused_after = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert unused_after < cached_bytes // 2


@rubin
@pytest.mark.parametrize("nvfp4,situ", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tokens", [1, 8, 64, 257])
@pytest.mark.parametrize("fuse_finalize", [False, True])
def test_prims_ts_locality_matches_unpartitioned(nvfp4, situ, tokens, fuse_finalize, monkeypatch):
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.custom_ops import prims_ts_partitioned_moe as ops

    monkeypatch.setattr(
        ops,
        "prims_ts_partitioned_moe",
        partial(ops.prims_ts_partitioned_moe, fuse_finalize=fuse_finalize),
    )

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
        _assert_locality_close(actual, expected, fused=fuse_finalize)
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
            _assert_locality_close(captured, expected, fused=fuse_finalize)


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
@pytest.mark.parametrize("tokens", [1, 16, 1024])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_prims_ts_locality_fused_routing_graph(tokens, enable_pdl, monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
    from tensorrt_llm._torch.moe.fused_moe import fused_moe_prims_ts as backend
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

    monkeypatch.delenv("DISABLE_LOCALITY_DOMAINS", raising=False)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", str(int(enable_pdl)))
    is_locality_domain_enabled.cache_clear()
    # Exercise the production logits path with a small expert bank. Full K3
    # geometry is covered separately by the model benchmark and SiTU test.
    monkeypatch.setattr(backend, "_supports_kimi_routing", lambda *args: True)
    bias = torch.randn(8, device="cuda", dtype=torch.float32) * 0.1
    routing = DeepSeekV3MoeRoutingMethod(2, 1, 1, 1.0, lambda: bias)
    (baseline, candidate), _ = make_moe_pair(True, True, routing_method=routing)
    x = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(tokens, 8, device="cuda", dtype=torch.float32)
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        expected = baseline(x, logits)
        actual = candidate(x, logits)
    context = capture._captured_contexts[-1]
    runner, inputs = context["runners"][0], context["inputs"]
    assert runner.use_hybrid_routing and runner.fuse_finalize
    assert runner.enable_pdl is enable_pdl
    assert inputs[0] is not None and inputs[-1] is None and inputs[-2] is None
    _assert_locality_close(actual, expected)
    assert torch.count_nonzero(actual) > 0
    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = candidate(x, logits)
        for _ in range(5):
            x.normal_(std=0.5)
            logits.normal_()
            bias.normal_(std=0.1)
            expected = baseline(x, logits)
            graph.replay()
            _assert_locality_close(captured, expected)


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
        _assert_locality_close(actual, expected)
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
            _assert_locality_close(output, expected)
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
    assert "locality_execution_v4" in runner.unique_id()
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
            _assert_locality_close(actual, expected)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            with pytest.raises(ValueError, match="token tile of at most 128"):
                runner(inputs, tactic=(256, 0))
            outputs.append(actual)
    _assert_locality_close(outputs[0], outputs[1])
