# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PrimTS selection, routing ABI, Kimi numerics and CUDA Graph coverage."""

from dataclasses import replace

import pytest
import torch
from _torch.moe.test_kimi_k3_situ_moe import (
    _TP_HIDDEN,
    _TP_INTERMEDIATE,
    _load_bank,
    _load_nvfp4_bank_for,
    _make_nvfp4_expert_bank,
    _make_nvfp4_moe,
    _make_packed_expert_bank,
    _make_routed_moe,
    _make_test_gate,
)

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.fused_moe_prims_ts import (
    PrimsTSMxfp4Mxfp8FusedMoE,
    PrimsTSNvfp4FusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, override_moe_environment
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import impl_class_for, resolve_moe_impl
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _environment(sm=100):
    return MoEEnvironment(sm=sm, available_deps=(MoEDep.PRIMS_TS.value,))


@pytest.mark.parametrize("tokens", [1, 8, 16])
@pytest.mark.parametrize("hidden", [2048, 2576, 3584])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("float_scales", [False, True])
def test_small_token_finalization_matches_generic(tokens, hidden, dtype, float_scales):
    """Padding to 17 tokens selects the generic kernel as an exact reference."""
    top_k, reference_tokens = 16, 17
    permuted = torch.randn(reference_tokens * top_k, hidden, device="cuda", dtype=dtype)
    indices = torch.randint(
        0, permuted.shape[0], (reference_tokens, top_k), device="cuda", dtype=torch.int32
    )
    weights = torch.randn(
        reference_tokens, top_k, device="cuda", dtype=torch.float32 if float_scales else dtype
    )
    # Cover dense routing, remote experts, and a token with no local expert.
    for remote_stride in (0, 2, 1):
        if remote_stride:
            indices[:, ::remote_stride] = -1
        expected = torch.empty(reference_tokens, hidden, device="cuda", dtype=dtype)
        actual = torch.empty(tokens, hidden, device="cuda", dtype=dtype)
        torch.ops.trtllm.moe_unpermute_inplace(permuted, expected, indices, weights)
        torch.ops.trtllm.moe_unpermute_inplace(permuted, actual, indices[:tokens], weights[:tokens])
        torch.testing.assert_close(actual, expected[:tokens], rtol=0, atol=0)


@pytest.mark.parametrize(
    "quant,cls",
    [
        (QuantAlgo.NVFP4, PrimsTSNvfp4FusedMoE),
        (QuantAlgo.W4A8_MXFP4_MXFP8, PrimsTSMxfp4Mxfp8FusedMoE),
    ],
)
def test_backend_and_identity_select_same_implementation(quant, cls):
    config = ModelConfig(moe_backend="PRIMS_TS", quant_config=QuantConfig(quant_algo=quant))
    with override_moe_environment(_environment()):
        for impl_id in (None, cls.descriptor.impl_id):
            report = resolve_moe_impl(config, impl_id=impl_id)
            assert impl_class_for(report) is cls
            assert not report.degraded


def test_pinned_prims_ts_does_not_fall_back_without_dependencies():
    config = ModelConfig(
        moe_backend="PRIMS_TS", quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4)
    )
    with override_moe_environment(MoEEnvironment(sm=100)):
        report = resolve_moe_impl(config, impl_id=PrimsTSNvfp4FusedMoE.descriptor.impl_id)
    assert report.winner is None
    assert report.rejected[0].reason == MoERejectReason.DEP_MISSING


@pytest.mark.parametrize(
    "problem_change,sm,reason",
    [
        ({}, 90, MoERejectReason.SM_UNSUPPORTED),
        ({"dtype_act": torch.float16}, 100, MoERejectReason.DTYPE_UNSUPPORTED),
        ({"quant": QuantAlgo.FP8_BLOCK_SCALES.value}, 100, MoERejectReason.QUANT_UNSUPPORTED),
        ({"hidden_size": 129}, 100, MoERejectReason.SHAPE_UNALIGNED),
        ({"activation": "Relu2"}, 100, MoERejectReason.ACTIVATION_UNSUPPORTED),
    ],
)
def test_prims_ts_eligibility_rejects_unsupported_inputs(problem_change, sm, reason):
    problem = MoEProblem(
        quant=QuantAlgo.NVFP4.value,
        dtype_act=torch.bfloat16,
        hidden_size=512,
        intermediate_size=256,
        num_experts=8,
        top_k=2,
    )
    deployment = MoEDeployment(
        ep_size=1, tp_size=1, use_dp=False, num_slots=8, parallel_size=1, env=_environment(sm)
    )
    eligibility = PrimsTSNvfp4FusedMoE.can_implement(replace(problem, **problem_change), deployment)
    assert eligibility.reject_reason == reason


blackwell = pytest.mark.skipif(get_sm_version() not in (100, 103), reason="requires Blackwell")


@blackwell
@pytest.mark.parametrize("is_nvfp4", [False, True])
@pytest.mark.parametrize("use_dp", [False, True])
def test_autotune_profiles_regenerate_expanded_routes(is_nvfp4, use_dp):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import prepare_dummy_topk_and_hook
    from tensorrt_llm._torch.moe.custom_ops.prims_ts_moe import (
        _tuning_config,
        _with_routing_profile_hook,
    )
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType

    torch.manual_seed(37)
    x = torch.zeros(
        1,
        256 if is_nvfp4 else 512,
        dtype=torch.uint8 if is_nvfp4 else torch.float8_e4m3fn,
        device="cuda",
    )
    weights = torch.ones(1, 16, dtype=torch.bfloat16, device="cuda")
    ids = torch.arange(16, dtype=torch.int32, device="cuda").view(1, 16)
    with autotune():
        _, dummy_weights, dummy_ids, config = prepare_dummy_topk_and_hook(
            weights,
            ids,
            x,
            None,
            int(RoutingMethodType.DeepSeekV3),
            _tuning_config(is_nvfp4, 8, 128, use_dp),
            16,
            896,
            112,
            1,
            1,
            1.0,
            local_expert_offset=112,
            use_dp=use_dp,
        )
        inputs = [None] * 18
        inputs[2] = x
        inputs[3] = torch.zeros(32 if is_nvfp4 else 16, dtype=torch.uint8, device="cuda")
        inputs[-2:] = [dummy_weights, dummy_ids]
        config = _with_routing_profile_hook(config, dummy_weights, dummy_ids)
        tuner = AutoTuner.get()
        profile = next(
            p
            for p in tuner._optimization_profiles(config, inputs)
            if p.get_opt_shapes()[2][0] == 128
        )
        resized = tuner._prepare_input_tensors(profile, inputs)
        # This is the resize that used to hide the token-count change.
        assert torch.unique(resized[-1], dim=0).shape[0] == 1
        prepared = config.inputs_pre_hook(resized)
    assert prepared[-1].shape == (128, 16)
    assert torch.unique(prepared[-1], dim=0).shape[0] > 100
    assert prepared[-1].unique().numel() > 16
    assert prepared[-1].min() >= (112 if use_dp else 0)
    assert prepared[-1].max() < (224 if use_dp else 896)
    assert dummy_ids.shape == (1, 16)


@blackwell
@pytest.mark.parametrize("is_nvfp4", [False, True])
def test_autotune_profiles_preserve_model_routing_bias(is_nvfp4):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import prepare_dummy_topk_and_hook
    from tensorrt_llm._torch.moe.custom_ops.prims_ts_moe import (
        _tuning_config,
        _with_routing_profile_hook,
    )
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType

    x = torch.zeros(
        1,
        256 if is_nvfp4 else 512,
        dtype=torch.uint8 if is_nvfp4 else torch.float8_e4m3fn,
        device="cuda",
    )
    weights = torch.ones(1, 16, dtype=torch.bfloat16, device="cuda")
    ids = torch.arange(16, dtype=torch.int32, device="cuda").view(1, 16)
    # A model bias can outweigh sigmoid scores and consistently select a
    # specific expert subset, which a fresh random bias would not preserve.
    bias = torch.full((896,), -8.0, dtype=torch.float32, device="cuda")
    bias[112:128] = 8.0
    with autotune():
        _, dummy_weights, dummy_ids, config = prepare_dummy_topk_and_hook(
            weights,
            ids,
            x,
            None,
            int(RoutingMethodType.DeepSeekV3),
            _tuning_config(is_nvfp4, 8, 128, False),
            16,
            896,
            112,
            1,
            1,
            1.0,
            local_expert_offset=112,
            routing_bias=bias,
        )
        inputs = [None] * 18
        inputs[1:4] = [
            bias,
            x,
            torch.zeros(32 if is_nvfp4 else 16, dtype=torch.uint8, device="cuda"),
        ]
        inputs[-2:] = [dummy_weights, dummy_ids]
        config = _with_routing_profile_hook(config, dummy_weights, dummy_ids)
        tuner = AutoTuner.get()
        profile = next(
            p
            for p in tuner._optimization_profiles(config, inputs)
            if p.get_opt_shapes()[2][0] == 128
        )
        prepared = config.inputs_pre_hook(tuner._prepare_input_tensors(profile, inputs))
    expected = torch.arange(112, 128, dtype=torch.int32, device="cuda").expand(128, -1)
    torch.testing.assert_close(prepared[-1].sort(dim=-1).values, expected, rtol=0, atol=0)


@blackwell
def test_moe_sort_token_map_matches_expanded_map():
    ids = torch.tensor([[0, 2], [1, 3], [3, 2], [0, 1]], dtype=torch.int32, device="cuda")
    scales = torch.full(ids.shape, 0.5, dtype=torch.bfloat16, device="cuda")
    tile, limits, expanded, token_map, total, count = torch.ops.trtllm.moe_sort(
        ids, scales, 4, 2, 2, 2, 8, True
    )
    expanded_cpu = expanded.cpu()
    token_cpu = token_map.cpu()
    for token in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            if ids[token, slot] >= 2:
                assert token_cpu[expanded_cpu[token, slot]] == token
    # The default schema must keep the old expanded-index representation.
    _, _, old_expanded, old_map, _, _ = torch.ops.trtllm.moe_sort(ids, scales, 4, 2, 2, 2, 8)
    for token in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            if ids[token, slot] >= 2:
                assert old_map[old_expanded[token, slot]] == token * 2 + slot


@blackwell
@pytest.mark.parametrize("num_tokens,experts,top_k", [(1, 8, 2), (16, 32, 16), (128, 8, 2)])
def test_kimi_mxfp4_matches_trtllm_gen(num_tokens, experts, top_k):
    bank = _make_packed_expert_bank(experts, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate(num_experts=experts, top_k=top_k)
    baseline = _load_bank(_make_routed_moe(_TP_INTERMEDIATE, gate, num_experts=experts), bank)
    candidate = _load_bank(
        _make_routed_moe(_TP_INTERMEDIATE, gate, num_experts=experts, moe_backend="PRIMS_TS"), bank
    )
    assert isinstance(candidate.backend, PrimsTSMxfp4Mxfp8FusedMoE)
    torch.manual_seed(37)
    x = torch.randn(num_tokens, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    logits = gate.compute_logits(x)
    with torch.inference_mode():
        expected = baseline(x, logits)
        actual = candidate(x, logits)
    assert torch.isfinite(actual).all()
    relative_l2 = torch.linalg.vector_norm(
        actual.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float())
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    print(f"PrimsTS vs TRTLLM: cosine={cosine.item():.6f}, relative_l2={relative_l2.item():.6f}")
    assert cosine > 0.99
    assert relative_l2 < 0.1

    # Captured kernels must consume updated tokens AND expert assignments.
    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = candidate(x, logits)
        x.normal_(std=0.5)
        logits.copy_(gate.compute_logits(x))
        expected_replay = candidate(x, logits)
        graph.replay()
    torch.testing.assert_close(captured, expected_replay, atol=0, rtol=0)


@blackwell
def test_tuned_moe_workspace_and_external_finalize(monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner, TuningConfig, autotune
    from tensorrt_llm._torch.moe.custom_ops import prims_ts_moe as ops

    # Exercise real tuning over different routing tiles without sweeping the
    # full performance search space in a correctness test.
    monkeypatch.setattr(ops, "_tuning_config", lambda *args: TuningConfig())
    monkeypatch.setattr(
        ops.PrimsTSMoERunner, "get_valid_tactics", lambda *args, **kwargs: [[8, 0], [16, 0]]
    )
    bank = _make_packed_expert_bank(8, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()
    moe = _load_bank(_make_routed_moe(_TP_INTERMEDIATE, gate, moe_backend="PRIMS_TS"), bank)
    x = torch.randn(16, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    logits = gate.compute_logits(x)
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        expected = moe(x, logits)
    context = capture._captured_contexts[-1]
    runner, inputs = context["runners"][0], context["inputs"]
    with torch.inference_mode(), autotune():
        actual = moe(x, logits)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    # The communication scheduler may perform the weighted combine itself.
    runner.do_finalize = False
    with torch.inference_mode():
        a, expanded_a, unused_a = runner(inputs, tactic=[8, 0])
        b, expanded_b, unused_b = runner(inputs, tactic=[16, 0])
        assert unused_a.numel() == unused_b.numel() == 0
        assert a.shape == b.shape
        combined_a, combined_b = torch.empty_like(x), torch.empty_like(x)
        torch.ops.trtllm.moe_unpermute_inplace(a, combined_a, expanded_a, inputs[-2])
        torch.ops.trtllm.moe_unpermute_inplace(b, combined_b, expanded_b, inputs[-2])
    torch.testing.assert_close(combined_a, expected, atol=0, rtol=0)
    torch.testing.assert_close(combined_b, expected, atol=0.1, rtol=0.1)


@blackwell
def test_prefill_tma_stride_does_not_overflow_int32():
    from tensorrt_llm._torch.autotuner import AutoTuner

    bank = _make_packed_expert_bank(32, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate(num_experts=32, top_k=16)
    moe = _load_bank(
        _make_routed_moe(_TP_INTERMEDIATE, gate, num_experts=32, moe_backend="PRIMS_TS"), bank
    )
    x = torch.randn(8192, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    logits = gate.compute_logits(x)
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        expected = moe(x, logits)
    context = capture._captured_contexts[-1]
    runner, inputs = context["runners"][0], context["inputs"]
    actual = torch.empty_like(expected)
    with torch.inference_mode():
        # The clustered LDGSTS variant constructs a 3D activation TensorMap.
        # Its large expert stride exceeds int32 when expressed in bits.
        runner(inputs, tactic=[128, 8], output=actual)
    assert torch.isfinite(actual).all()
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.01


@blackwell
@pytest.mark.parametrize("tokens", [1, 16, 1024])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("return_token_map", [False, True])
def test_logits_routing_bias_dtype_and_map_contract(tokens, bias_dtype, return_token_map):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType

    experts, top_k, offset, local_experts, tile_n = 128, 8, 64, 64, 8
    torch.manual_seed(37)
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="cuda")
    # The selected set straddles the local shard boundary. Integer-spaced
    # biases avoid near-ties while still exercising nonuniform sigmoid weights.
    bias = (torch.arange(experts, device="cuda", dtype=torch.float32) * 2).roll(70).to(bias_dtype)
    args = (
        logits,
        bias,
        experts,
        top_k,
        1,
        1,
        offset,
        local_experts,
        1.75,
        tile_n,
        int(RoutingMethodType.DeepSeekV3),
    )
    options = {"return_token_map": True} if return_token_map else {}
    result = torch.ops.trtllm.moe_topk_sort(*args, **options)
    tiles, _, expanded, route_map, _, _, weights = result
    assert weights.dtype == torch.bfloat16
    assert route_map.numel() == tiles.numel() * tile_n + int(return_token_map)

    scores = logits.sigmoid()
    expected_ids = (scores + bias.float()).topk(top_k, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.75
    torch.testing.assert_close(
        weights.sort(dim=-1).values.float(),
        expected_weights.to(torch.bfloat16).sort(dim=-1).values.float(),
        rtol=0.01,
        atol=0.001,
    )
    valid = expanded >= 0
    tokens_grid = torch.arange(tokens, device="cuda")[:, None].expand(-1, top_k)
    slots_grid = torch.arange(top_k, device="cuda")[None, :].expand(tokens, -1)
    expected_map = tokens_grid if return_token_map else tokens_grid * top_k + slots_grid
    torch.testing.assert_close(
        route_map[expanded[valid].long()].long(), expected_map[valid], rtol=0, atol=0
    )
    local_ids = tiles[(expanded[valid] // tile_n).long()].long() + offset
    expected_local = expected_ids[
        (expected_ids >= offset) & (expected_ids < offset + local_experts)
    ]
    # Each token selects the same six local experts, potentially in another order.
    torch.testing.assert_close(
        local_ids.reshape(tokens, -1).sort(dim=-1).values,
        expected_local.reshape(tokens, -1).sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    local_weights = (
        scores[tokens_grid[valid], local_ids]
        / scores.gather(1, expected_ids).sum(dim=-1)[tokens_grid[valid]]
        * 1.75
    )
    torch.testing.assert_close(
        weights[valid].float(), local_weights.to(torch.bfloat16).float(), rtol=0.01, atol=0.001
    )
    with FakeTensorMode() as mode:
        fake_args = tuple(
            mode.from_tensor(value) if isinstance(value, torch.Tensor) else value for value in args
        )
        fake_result = torch.ops.trtllm.moe_topk_sort(*fake_args, **options)
    assert [(value.shape, value.dtype) for value in fake_result] == [
        (value.shape, value.dtype) for value in result
    ]


@blackwell
@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("use_hybrid_routing", [False, True])
@pytest.mark.parametrize("is_nvfp4", [False, True])
def test_fused_routing_moe_tuning_graph_and_external_finalize(
    monkeypatch, tokens, use_hybrid_routing, is_nvfp4
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoEInputs
    from tensorrt_llm._torch.moe.custom_ops import prims_ts_moe as ops
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType

    monkeypatch.setattr(
        ops.PrimsTSMoERunner, "get_valid_tactics", lambda *args, **kwargs: [[8, 0], [16, 0]]
    )
    gate = _make_test_gate()
    if is_nvfp4:
        bank = _make_nvfp4_expert_bank(8, _TP_INTERMEDIATE, _TP_HIDDEN)
        for expert in bank:
            # Keep the routing comparison nontrivial with a static activation
            # scale of 1.0: tiny fixture weights quantize FC1 outputs to zero.
            for name in ("w1.weight_scale", "w2.weight_scale", "w3.weight_scale"):
                expert[name] = (expert[name].float() * 64).to(torch.float8_e4m3fn)
        moe = _make_nvfp4_moe(gate, moe_backend="PRIMS_TS")
        _load_nvfp4_bank_for(moe, bank, "PRIMS_TS")
    else:
        bank = _make_packed_expert_bank(8, _TP_INTERMEDIATE, _TP_HIDDEN)
        moe = _load_bank(_make_routed_moe(_TP_INTERMEDIATE, gate, moe_backend="PRIMS_TS"), bank)
    x = torch.randn(tokens, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    logits = gate.compute_logits(x)
    with torch.inference_mode(), AutoTuner.get().capture() as capture:
        expected = moe(x, logits)
    assert torch.count_nonzero(expected) > 0
    inputs = capture._captured_contexts[-1]["inputs"]
    kwargs = vars(FP4BlockScaleMoEInputs(*inputs)).copy()
    for key in ("routing_logits", "gemm1_bias", "gemm2_bias"):
        kwargs.pop(key)
    kwargs.update(
        router_logits=logits,
        routing_bias=gate.e_score_correction_bias,
        topk_ids=None,
        topk_weights=None,
        top_k=gate.routing_method.top_k,
        num_experts=8,
        local_expert_offset=0,
        activation_type=10,
        do_finalize=True,
        output=torch.empty_like(x),
        enable_pdl=True,
        tune_max_num_tokens=16,
        routing_method_type=int(RoutingMethodType.DeepSeekV3),
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        use_hybrid_routing=use_hybrid_routing,
    )
    # Model startup warms up one token and profiles all requested buckets.
    # Medium-batch inference must then hit that same cache, even when its
    # routing implementation differs from the single-token warmup.
    warmup = dict(kwargs)
    warmup.update(
        hidden_states=kwargs["hidden_states"][:1],
        router_logits=logits[:1],
        output=kwargs["output"][:1],
    )
    with torch.inference_mode(), autotune():
        ops.prims_ts_moe(**warmup)
    tuner = AutoTuner.get()
    with torch.inference_mode(), tuner.capture() as capture:
        result = ops.prims_ts_moe(**kwargs)
    context = capture._captured_contexts[-1]
    cache_hit, _, _, _ = tuner.profiling_cache.search_cache(
        context["custom_op"],
        context["runners"],
        tuple(tuner._get_input_sizes(context["inputs"])),
        context["tuning_config"],
        apply_map_to_tuning_buckets=True,
    )
    assert cache_hit, "Single-token warmup must tune the medium-batch routing path."
    torch.testing.assert_close(kwargs["output"], expected, rtol=0.01, atol=0.02)
    assert result[2].shape == (tokens, gate.routing_method.top_k)
    assert result[2].dtype == torch.bfloat16

    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.prims_ts_moe(**kwargs)
        x.normal_(std=0.5)
        logits.copy_(gate.compute_logits(x))
        quantized, scales = moe.backend.quantize_input(x)
        kwargs["hidden_states"].copy_(quantized)
        kwargs["hidden_states_scale"].copy_(scales.reshape_as(kwargs["hidden_states_scale"]))
        ops.prims_ts_moe(**kwargs)
        expected_replay = kwargs["output"].clone()
        graph.replay()
        torch.testing.assert_close(kwargs["output"], expected_replay, rtol=0, atol=0)
        kwargs["do_finalize"] = False
        permuted, expanded, weights = ops.prims_ts_moe(**kwargs)
        combined = torch.empty_like(x)
        torch.ops.trtllm.moe_unpermute_inplace(permuted, combined, expanded, weights)
        torch.testing.assert_close(combined, expected_replay, rtol=0.01, atol=0.02)

    with FakeTensorMode() as mode:
        fake_kwargs = {
            key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        fake_result = ops.prims_ts_moe(**fake_kwargs)
    assert [(value.shape, value.dtype) for value in fake_result] == [
        (value.shape, value.dtype) for value in (permuted, expanded, weights)
    ]
    kwargs.update(
        hidden_states=kwargs["hidden_states"][:0],
        hidden_states_scale=kwargs["hidden_states_scale"][:0],
        router_logits=logits[:0],
        output=torch.empty_like(x[:0]),
    )
    with torch.inference_mode():
        empty = ops.prims_ts_moe(**kwargs)
    assert [value.shape for value in empty] == [
        (0, _TP_HIDDEN),
        (0, gate.routing_method.top_k),
        (0, gate.routing_method.top_k),
    ]
