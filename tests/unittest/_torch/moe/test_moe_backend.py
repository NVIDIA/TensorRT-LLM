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
"""MoE backend unit tests."""

import collections
import dataclasses
import importlib
import itertools
import logging
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
from _torch.moe.moe_test_utils import (
    IS_CI_MODE,
    MoeBackendType,
    MoeModelConfig,
    build_test_activation,
    create_test_param,
    get_backend_class,
    iter_base_test_configs,
    replay_tactics_and_check,
    should_skip_to_accelerate_ci,
    skip_if_insufficient_gpu_memory,
    supports_autotuner_capture,
)
from _torch.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig
from utils.util import check_accuracy, skip_pre_blackwell

from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, autotune
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import _select_explicit_fallback_tactic
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
)
from tensorrt_llm._torch.moe.fused_moe.activation import (
    DEFAULT_MOE_ACTIVATION,
    MoEActivation,
    SiTuActivation,
    SwigluActivation,
    SwigluBiasActivation,
    materialize_activation_params,
)
from tensorrt_llm._torch.moe.fused_moe.communication import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.communication.deep_ep_low_latency import DeepEPLowLatency
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe_backend
from tensorrt_llm._torch.moe.fused_moe.cutedsl import (
    TrtllmCutedslDenseGemmNvfp4Impl,
    TrtllmCutedslGroupedGemmBf16Impl,
    TrtllmCutedslGroupedGemmNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.cutedsl.fused_fc12 import CuteDslFc12FusedMoENvfp4Runner
from tensorrt_llm._torch.moe.fused_moe.cutedsl.runners import CuteDslFusedMoENvfp4Runner
from tensorrt_llm._torch.moe.fused_moe.cutlass import (
    CUTLASS_LEAVES,
    TrtllmCutlassFp8Impl,
    TrtllmCutlassNvfp4Impl,
    TrtllmCutlassUnquantizedImpl,
    TrtllmCutlassW4a16Mxfp4Impl,
    TrtllmCutlassW4a16Nvfp4Impl,
    TrtllmTritonFp8BlockScalesImpl,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import (
    CuteDslB12xNvfp4FusedMoE,
    CuteDslB12xW4a16Nvfp4FusedMoE,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_densegemm import DenseGEMMFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_marlin import MarlinFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import (
    TRTLLMGenFusedMoE,
    trtllm_gen_leaf,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoECommPlan,
    MoEDeployment,
    MoEEnvironment,
    MoEInputRequirement,
    MoEProblem,
    MoERejection,
    MoERejectReason,
    MoEReplicaPlan,
    MoEResolutionReport,
    MoERunContext,
    MoEStaticCapability,
    canonical_activation,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import (
    MoEDep,
    collect_moe_environment,
    override_moe_environment,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoE, MoESchedulerKind, MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.mega_moe import (
    MegaMoECuteDsl,
    MegaMoEDeepGemm,
    TrtllmCutedslMegaMoeNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    IMPL_PRIORITY,
    _reject_unsupported_activation,
    build_moe_deployment,
    impl_class_for,
    resolve_moe_impl,
)
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import (
    ExternalCommMoEScheduler,
    FusedCommMoEScheduler,
)
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.planner import (
    RebalanceSlotSchedulerGroup,
    _LiveBankLeaseProvider,
)
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    FusedMoEMethodBase,
    NVFP4FusedMoEMethod,
    NVFP4MarlinFusedMoEMethod,
    NVFP4MegaMoECuteDslMethod,
    NVFP4TRTLLMGenFusedMoEBaseMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
    UnquantizedFusedMoEMethod,
    W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
    W4A16NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import (
    FlashinferTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenNvfp4Impl,
    TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
)
from tensorrt_llm._torch.utils import (
    ActivationType,
    Fp4QuantizedTensor,
    MxFp8QuantizedTensor,
    is_gated_activation,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f, mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cutlass_moe_routing_round_trip(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUTLASS routing requires CUDA")
    torch.manual_seed(17)
    num_experts, hidden_size = 4, 128
    x = torch.randn(8, hidden_size, device="cuda", dtype=dtype)
    selected = torch.arange(8, device="cuda", dtype=torch.int32).remainder(num_experts).unsqueeze(1)
    scores = torch.ones(8, 1, device="cuda", dtype=torch.float32)
    w1 = torch.empty(num_experts, hidden_size * 2, hidden_size, device="cuda", dtype=dtype)
    w2 = torch.empty(num_experts, hidden_size, hidden_size, device="cuda", dtype=dtype)

    p2u, _, permuted, offsets, _, u2p = torch.ops.trtllm.moe_permute_op(
        x, selected, scores, w1, w2, None, None, num_experts, 1, 0, 1, 0, 1, 0, False, False, False
    )
    torch.testing.assert_close(permuted, x[p2u.long()], atol=0, rtol=0)
    restored = torch.ops.trtllm.moe_finalize_scale_op(
        permuted,
        None,
        scores,
        u2p,
        p2u,
        selected,
        offsets,
        False,
        x.shape[0],
        hidden_size,
        hidden_size,
        1,
        num_experts,
        1,
        0,
        1,
        0,
    )
    torch.testing.assert_close(restored, x, atol=0, rtol=0)


@pytest.mark.parametrize(
    "activation_type,clamp_limit,alpha_only,capture_first",
    [
        (ActivationType.Identity, None, False, False),
        (ActivationType.Gelu, None, False, False),
        (ActivationType.Relu, None, False, False),
        (ActivationType.Silu, None, False, False),
        (ActivationType.Swiglu, None, False, False),
        (ActivationType.Geglu, None, False, False),
        (ActivationType.SwigluBias, None, False, False),
        (ActivationType.Swiglu, 0.2, False, False),
        (ActivationType.SwigluBias, 0.2, False, False),
        pytest.param(ActivationType.Swiglu, 0.2, True, False, id="alpha-only-clamp"),
        pytest.param(ActivationType.Swiglu, 0.2, False, True, id="capture-before-eager"),
    ],
)
def test_cutlass_moe_public_activation_ids(
    activation_type: ActivationType,
    clamp_limit: Optional[float],
    alpha_only: bool,
    capture_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available() or get_sm_version() < 80:
        pytest.skip("BF16 CUTLASS MoE requires CUDA and SM80 or newer")

    torch.manual_seed(17)
    dtype = torch.bfloat16
    num_experts, hidden_size, inter_size = 4, 128, 128
    gated = is_gated_activation(activation_type)
    x = torch.randn(8, hidden_size, device="cuda", dtype=dtype) * 0.5
    w1 = (
        torch.randn(
            num_experts, inter_size * (2 if gated else 1), hidden_size, device="cuda", dtype=dtype
        )
        * 0.1
    )
    w2 = torch.randn(num_experts, hidden_size, inter_size, device="cuda", dtype=dtype) * 0.1
    selected = torch.arange(8, device="cuda", dtype=torch.int32).remainder(num_experts).unsqueeze(1)
    scores = torch.ones(8, 1, device="cuda", dtype=torch.float32)
    alpha = (
        torch.ones(num_experts, device="cuda")
        if activation_type == ActivationType.SwigluBias or alpha_only
        else None
    )
    beta = (
        torch.zeros(num_experts, device="cuda")
        if activation_type == ActivationType.SwigluBias
        else None
    )
    limit = (
        torch.full((num_experts,), clamp_limit, device="cuda") if clamp_limit is not None else None
    )

    def run_moe():
        return torch.ops.trtllm.fused_moe(
            x,
            selected,
            scores,
            w1,
            None,
            w2,
            None,
            dtype,
            [],
            activation_type=int(activation_type),
            swiglu_alpha=alpha,
            swiglu_beta=beta,
            swiglu_limit=limit,
        )[0]

    if capture_first:
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

        # Isolate the native runner from earlier parameterized cases. Construct
        # it before capture, with no eager forward or cached neutral defaults.
        key = (dtype, dtype, dtype, False, False, False, False, False)
        runner = torch.classes.trtllm.FusedMoeRunner(*key, True)
        monkeypatch.setattr(MoERunner, "runner_dict", {key: runner})
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            captured = run_moe()
        # Same stream, before replay: the eager call must not read defaults
        # whose initialization was only recorded in the captured graph.
        with torch.cuda.stream(stream):
            actual = run_moe()
        torch.cuda.current_stream().wait_stream(stream)
    else:
        actual = run_moe()

    expected = []
    for row, expert in zip(x, selected[:, 0]):
        fc1 = torch.nn.functional.linear(row.float(), w1[expert].float())
        if gated:
            up, gate = fc1.chunk(2)
            if clamp_limit is not None:
                up = up.clamp(min=-clamp_limit, max=clamp_limit)
                gate = gate.clamp(max=clamp_limit)
            activated = (
                torch.nn.functional.gelu(gate, approximate="tanh")
                if activation_type == ActivationType.Geglu
                else torch.nn.functional.silu(gate)
            ) * up
        elif activation_type == ActivationType.Gelu:
            activated = torch.nn.functional.gelu(fc1, approximate="tanh")
        elif activation_type == ActivationType.Relu:
            activated = torch.nn.functional.relu(fc1)
        elif activation_type == ActivationType.Silu:
            activated = torch.nn.functional.silu(fc1)
        else:
            activated = fc1
        expected.append(
            torch.nn.functional.linear(activated.to(dtype).float(), w2[expert].float()).to(dtype)
        )

    torch.testing.assert_close(actual, torch.stack(expected), atol=0.01, rtol=0.05)
    if clamp_limit is not None:
        if not capture_first:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run_moe()
        graph.replay()
        torch.testing.assert_close(captured, torch.stack(expected), atol=0.01, rtol=0.05)


_MEGAMOE_BACKEND_TYPES = {
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
}


def test_fused_comm_scheduler_accepts_singleton_non_dp_token_metadata():
    mapping = SimpleNamespace(moe_ep_rank=3, moe_ep_size=4)
    scheduler = FusedCommMoEScheduler(SimpleNamespace(mapping=mapping))
    x = torch.randn(6, 8)
    router_logits = torch.randn(6, 4)

    x_real, router_logits_real, _, all_rank_num_tokens, ep_rank, had_meta = (
        scheduler._strip_adp_padding(x, router_logits, None, [6])
    )

    assert x_real.shape == x.shape
    assert router_logits_real.shape == router_logits.shape
    assert all_rank_num_tokens == [6]
    assert ep_rank == 0
    assert not had_meta


def test_fused_comm_scheduler_indexes_full_ep_token_metadata():
    mapping = SimpleNamespace(moe_ep_rank=3, moe_ep_size=4)
    scheduler = FusedCommMoEScheduler(SimpleNamespace(mapping=mapping))
    x = torch.randn(6, 8)
    router_logits = torch.randn(6, 4)

    x_real, router_logits_real, _, all_rank_num_tokens, ep_rank, had_meta = (
        scheduler._strip_adp_padding(x, router_logits, None, [6, 5, 4, 3])
    )

    assert x_real.shape[0] == 3
    assert router_logits_real.shape[0] == 3
    assert all_rank_num_tokens == [6, 5, 4, 3]
    assert ep_rank == 3
    assert had_meta


def _split_chunk(tokens, chunks):
    return [tokens // chunks + 1] * (tokens % chunks) + [tokens // chunks] * (
        chunks - tokens % chunks
    )


@pytest.mark.cpu_only
def test_fused_comm_scheduler_passes_unpadded_single_chunk_inputs_through():
    moe = SimpleNamespace(
        mapping=SimpleNamespace(moe_ep_rank=1, moe_ep_size=2),
        moe_max_num_tokens=8,
        split_chunk=_split_chunk,
    )
    scheduler = FusedCommMoEScheduler(moe)
    x = torch.randn(6, 8)
    router_logits = torch.randn(6, 4)
    input_ids = torch.arange(6)

    x_real, rl_real, ids_real, counts, ep_rank, had_meta = scheduler._strip_adp_padding(
        x, router_logits, input_ids, [4, 6]
    )
    # This rank carries no ADP padding and fits one chunk: no view is taken.
    assert x_real is x and rl_real is router_logits and ids_real is input_ids
    assert (counts, ep_rank, had_meta) == ([4, 6], 1, True)
    num_chunks, x_chunks, rl_chunks, chunk_sizes = scheduler._compute_chunk_layout(
        x_real, rl_real, counts, ep_rank
    )
    assert num_chunks == 1
    assert x_chunks[0] is x and rl_chunks[0] is router_logits
    assert chunk_sizes == [[4], [6]]

    # Padding is still stripped and several chunks are still split.
    x_real, rl_real, _, counts, ep_rank, _ = scheduler._strip_adp_padding(
        x, router_logits, None, [4, 5]
    )
    assert x_real.shape[0] == 5 and torch.equal(x_real, x[:5])
    moe.moe_max_num_tokens = 2
    num_chunks, x_chunks, rl_chunks, chunk_sizes = scheduler._compute_chunk_layout(
        x_real, rl_real, counts, ep_rank
    )
    assert num_chunks == 3
    assert [chunk.shape[0] for chunk in x_chunks] == [2, 2, 1]
    assert torch.equal(torch.cat(x_chunks), x[:5])
    assert torch.equal(torch.cat(rl_chunks), router_logits[:5])
    assert chunk_sizes == [[2, 1, 1], [2, 2, 1]]


def test_fused_comm_scheduler_rejects_partial_ep_token_metadata():
    mapping = SimpleNamespace(moe_ep_rank=1, moe_ep_size=4)
    scheduler = FusedCommMoEScheduler(SimpleNamespace(mapping=mapping))
    x = torch.randn(6, 8)
    router_logits = torch.randn(6, 4)

    with pytest.raises(ValueError, match="got 2 counts for EP size 4"):
        scheduler._strip_adp_padding(x, router_logits, None, [6, 5])


def test_import_deep_gemm_rejects_pre_situ_mega_moe_api(monkeypatch):
    import tensorrt_llm
    import tensorrt_llm._torch.moe.fused_moe.quantization as quantization_module

    def fp8_fp4_mega_moe():
        pass

    def per_token_cast_to_fp8(*, use_packed_ue8m0=False):
        pass

    deep_gemm = SimpleNamespace(
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        get_symm_buffer_for_mega_moe=lambda: None,
        transform_sf_into_required_layout=lambda: None,
        transform_weights_for_mega_moe=lambda: None,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
        set_barrier_timeout_seconds=lambda seconds: None,
    )
    monkeypatch.setattr(tensorrt_llm, "deep_gemm", deep_gemm)

    with pytest.raises(
        quantization_module._MegaMoEUnavailable,
        match="fp8_fp4_mega_moe does not accept.*situ_beta",
    ):
        quantization_module._import_deep_gemm()


def test_import_deep_gemm_rejects_mega_moe_without_barrier_timeout(monkeypatch):
    import tensorrt_llm
    import tensorrt_llm._torch.moe.fused_moe.quantization as quantization_module

    def fp8_fp4_mega_moe(*, situ_beta=None, situ_linear_beta=None):
        pass

    def per_token_cast_to_fp8(*, use_packed_ue8m0=False):
        pass

    deep_gemm = SimpleNamespace(
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        get_symm_buffer_for_mega_moe=lambda: None,
        transform_sf_into_required_layout=lambda: None,
        transform_weights_for_mega_moe=lambda: None,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
    )
    monkeypatch.setattr(tensorrt_llm, "deep_gemm", deep_gemm)

    with pytest.raises(
        quantization_module._MegaMoEUnavailable,
        match="lacks set_barrier_timeout_seconds",
    ):
        quantization_module._import_deep_gemm()

    deep_gemm.set_barrier_timeout_seconds = lambda seconds: None
    assert quantization_module._import_deep_gemm() is deep_gemm


def test_fp8_block_scale_moe_fallback_tactic_is_explicit_and_deterministic():
    valid_tactics = [
        [8, 0],
        [32, 4],
        [16, 0],
        [32, 0],
    ]

    assert _select_explicit_fallback_tactic(valid_tactics) == [32, 0]

    with pytest.raises(RuntimeError, match="no valid fallback tactic"):
        _select_explicit_fallback_tactic([])


@pytest.mark.parametrize("use_direct_expert_metadata", [False, True])
def test_cutedsl_count_native_runner_keeps_both_tactics_and_threads_counts(
    monkeypatch: pytest.MonkeyPatch, use_direct_expert_metadata: bool
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.moe.fused_moe.cutedsl.runners.get_sm_version", lambda: 100
    )
    forward_impl = MagicMock(return_value=torch.empty(1))
    runner = CuteDslFusedMoENvfp4Runner(
        forward_impl=forward_impl,
        num_experts=256,
        top_k=1,
        num_local_experts=8,
        local_expert_offset=0,
        use_direct_expert_metadata=use_direct_expert_metadata,
    )
    counts = torch.tensor([0, 1, 31, 32, 7, 0, 16, 2], dtype=torch.int32)
    inputs = [torch.empty(8 * 32, 1) for _ in range(6)]
    forward_inputs = inputs + [counts] if use_direct_expert_metadata else inputs

    assert runner.forward(forward_inputs, tactic=256) is forward_impl.return_value

    assert runner.get_valid_tactics([], OptimizationProfile()) == [128, 256]
    expected_kwargs = {"enable_alltoall": False, "tile_size": 256}
    if use_direct_expert_metadata:
        assert runner.unique_id()[-1] == "direct_expert_metadata"
        assert runner.get_tuning_config().inputs_pre_hook is None
        expected_kwargs.update(
            recv_expert_count=counts,
            deep_ep_expert_capacity=32,
            use_count_native_expert_metadata=True,
        )
    else:
        assert runner.get_tuning_config().inputs_pre_hook is not None
    forward_impl.assert_called_once_with(*inputs, **expected_kwargs)


def test_cutedsl_direct_metadata_tuning_buckets_capacity_and_legacy_uses_tokens() -> None:
    common_kwargs = {
        "forward_impl": MagicMock(),
        "num_experts": 256,
        "top_k": 1,
        "num_local_experts": 3,
        "local_expert_offset": 13,
    }
    direct_runner = CuteDslFusedMoENvfp4Runner(
        **common_kwargs,
        use_direct_expert_metadata=True,
    )

    direct_config = direct_runner.get_tuning_config()
    assert len(direct_config.dynamic_tensor_specs) == 1
    assert len(direct_config.constraint_specs) == 4
    assert direct_config.inputs_pre_hook is None
    assert direct_runner.unique_id()[-1] == "direct_expert_metadata"
    direct_dynamic_spec = direct_config.dynamic_tensor_specs[0]
    expert_capacity_rows = 3 * 257
    assert direct_dynamic_spec.gen_tuning_buckets(expert_capacity_rows) == (384, 768)
    assert direct_dynamic_spec.map_to_tuning_buckets(expert_capacity_rows) == 768

    legacy_runner = CuteDslFusedMoENvfp4Runner(**common_kwargs)
    legacy_config = legacy_runner.get_tuning_config()
    assert len(legacy_config.dynamic_tensor_specs) == 1
    assert legacy_config.inputs_pre_hook is not None
    legacy_dynamic_spec = legacy_config.dynamic_tensor_specs[0]
    assert legacy_dynamic_spec.gen_tuning_buckets(expert_capacity_rows)[-1] == 512
    assert legacy_dynamic_spec.map_to_tuning_buckets(expert_capacity_rows) == 512


@pytest.mark.parametrize("decode_tuning", [False, True])
def test_fp4_block_scale_moe_decode_tuning(decode_tuning: bool) -> None:
    """Only a model that declares decode tuning gets the decode-step buckets and rank 0's tactic on every
    rank; otherwise every rank tunes on its own over the power-of-2 ladder."""
    from tensorrt_llm._torch.autotuner import DistributedTuningStrategy
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    config = FP4BlockScaleMoERunner.get_tuning_config(4, decode_tuning=decode_tuning)
    assert config.distributed_tuning_strategy == (
        DistributedTuningStrategy.BROADCAST
        if decode_tuning
        else DistributedTuningStrategy.INDEPENDENT
    )
    (spec,) = config.dynamic_tensor_specs
    pow2 = (1, 2, 4, 8, 16, 32, 64, 128)
    if decode_tuning:
        assert {3, 7, 24, 120} <= set(spec.gen_tuning_buckets)
        assert [spec.map_to_tuning_buckets(m) for m in (7, 14, 28, 130)] == [7, 8, 24, 128]
    else:
        assert set(spec.gen_tuning_buckets) & set(range(1, 129)) == set(pow2)
        assert [spec.map_to_tuning_buckets(m) for m in (7, 14, 28, 130)] == [4, 8, 16, 128]


@pytest.mark.parametrize(
    "disable_value,expected_disabled",
    [
        (None, False),
        ("0", False),
        ("1", True),
    ],
)
def test_cutedsl_deep_ep_direct_metadata_disable_env(
    monkeypatch: pytest.MonkeyPatch,
    disable_value: Optional[str],
    expected_disabled: bool,
) -> None:
    env_name = "TRTLLM_DISABLE_CUTEDSL_DEEP_EP_DIRECT_METADATA"
    if disable_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, disable_value)

    monkeypatch.setattr(torch.cuda, "Stream", MagicMock(return_value=object()))
    monkeypatch.setattr(torch.cuda, "Event", MagicMock(return_value=object()))

    constructor_kwargs = {
        "routing_method": MagicMock(),
        "num_experts": 8,
        "hidden_size": 64,
        "intermediate_size": 128,
        "model_config": ModelConfig(skip_create_weights_in_init=True),
    }

    backend = TrtllmCutedslGroupedGemmNvfp4Impl(**constructor_kwargs)
    assert backend.disable_deep_ep_direct_metadata is expected_disabled


def test_deep_ep_adapter_free_output_matches_schema_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    is_capturing = MagicMock(side_effect=AssertionError("CPU path queried CUDA capture state"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", is_capturing)
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    comm.mapping = SimpleNamespace(moe_ep_rank=1, moe_ep_size=2)
    comm.num_slots = 4
    comm.deep_ep_max_num_tokens = 8
    comm._adapter_free_placeholder_cache = {}
    comm.deep_ep_buffer = object()

    hidden_states = torch.arange(24, dtype=torch.bfloat16).view(2, 3, 4)
    hidden_states_sf = torch.arange(12, dtype=torch.uint8).view(2, 3, 2)
    recv_expert_count = torch.tensor([2, 1], dtype=torch.int32)

    legacy = comm._modify_output_to_adapt_fused_moe(
        hidden_states,
        hidden_states_sf,
        recv_expert_count,
        torch.float32,
    )
    adapter_free = comm._modify_output_to_adapt_fused_moe(
        hidden_states,
        hidden_states_sf,
        recv_expert_count,
        torch.float32,
        remove_adapter=True,
    )
    adapter_free_reused = comm._modify_output_to_adapt_fused_moe(
        hidden_states,
        hidden_states_sf,
        recv_expert_count,
        torch.float32,
        remove_adapter=True,
    )

    torch.testing.assert_close(adapter_free[0], legacy[0])
    torch.testing.assert_close(adapter_free[1], legacy[1])
    assert adapter_free[2].shape == legacy[2].shape
    assert adapter_free[2].shape == (hidden_states.shape[0] * hidden_states.shape[1], 1)
    assert adapter_free[2].dtype == legacy[2].dtype
    torch.testing.assert_close(adapter_free[3], legacy[3])
    assert (
        adapter_free_reused[2].untyped_storage().data_ptr()
        == adapter_free[2].untyped_storage().data_ptr()
    )
    assert (
        adapter_free_reused[3].untyped_storage().data_ptr()
        == adapter_free[3].untyped_storage().data_ptr()
    )
    assert torch.all(adapter_free[2] == -1)
    assert len(comm._adapter_free_placeholder_cache) == 1

    smaller_hidden_states = torch.arange(16, dtype=torch.bfloat16).view(2, 2, 4)
    smaller_hidden_states_sf = torch.arange(8, dtype=torch.uint8).view(2, 2, 2)
    smaller = comm._modify_output_to_adapt_fused_moe(
        smaller_hidden_states,
        smaller_hidden_states_sf,
        recv_expert_count,
        torch.float32,
        remove_adapter=True,
    )
    assert smaller[2].shape == (4, 1)
    assert smaller[2].untyped_storage().data_ptr() == adapter_free[2].untyped_storage().data_ptr()
    assert len(comm._adapter_free_placeholder_cache) == 1
    is_capturing.assert_not_called()

    comm.destroy()
    assert comm._adapter_free_placeholder_cache == {}
    assert comm.deep_ep_buffer is None


def test_deep_ep_adapter_free_cache_miss_rejected_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", MagicMock(return_value=True))
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    comm._adapter_free_placeholder_cache = {}
    hidden_states = MagicMock(
        shape=(2, 3, 4),
        device=torch.device("cuda:0"),
        is_cuda=True,
    )

    with pytest.raises(RuntimeError, match="must be warmed up before CUDA graph capture"):
        comm._modify_output_to_adapt_fused_moe(
            hidden_states,
            None,
            torch.tensor([2, 1], dtype=torch.int32),
            torch.float32,
            remove_adapter=True,
        )
    assert comm._adapter_free_placeholder_cache == {}


@pytest.mark.parametrize(
    "disabled,supports_post_quant,expected",
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_scheduler_selects_cutedsl_deep_ep_direct_metadata(
    disabled: bool, supports_post_quant: bool, expected: bool
) -> None:
    backend = TrtllmCutedslGroupedGemmNvfp4Impl.__new__(TrtllmCutedslGroupedGemmNvfp4Impl)
    backend.disable_deep_ep_direct_metadata = disabled
    backend.use_fused_finalize = True
    backend._weights_created = True
    assert backend.can_use_deep_ep_direct_metadata(supports_post_quant) is expected


def test_other_backend_rejects_deep_ep_direct_metadata() -> None:
    backend = TrtllmCutlassNvfp4Impl.__new__(TrtllmCutlassNvfp4Impl)
    assert backend.can_use_deep_ep_direct_metadata(True) is False


def test_cutedsl_rejects_direct_metadata_without_fused_finalize() -> None:
    backend = TrtllmCutedslGroupedGemmNvfp4Impl.__new__(TrtllmCutedslGroupedGemmNvfp4Impl)
    backend.disable_deep_ep_direct_metadata = False
    backend.use_fused_finalize = False
    backend._weights_created = True
    assert backend.can_use_deep_ep_direct_metadata(True) is False


def test_cutedsl_direct_metadata_request_fails_without_fused_finalize() -> None:
    backend = TrtllmCutedslGroupedGemmNvfp4Impl.__new__(TrtllmCutedslGroupedGemmNvfp4Impl)
    backend.activation_type = ActivationType.Swiglu
    backend._locality_domain_runtime = None
    backend.use_fused_finalize = False
    backend.hidden_size = 4
    backend._weights_created = True
    token_selected_experts = torch.full((2, 1), -1, dtype=torch.int32)
    token_final_scales = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires fused finalize"):
        backend.run_moe_nvfp4(
            x=torch.empty((2, 2), dtype=torch.uint8),
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            moe_output=torch.empty((2, 4), dtype=torch.bfloat16),
            weight_view=MagicMock(),
            recv_expert_count=torch.tensor([2], dtype=torch.int32),
            deep_ep_expert_capacity=2,
            use_deep_ep_direct_metadata=True,
        )


@pytest.mark.parametrize("disabled,expected", [(False, True), (True, False)])
def test_scheduler_threads_deep_ep_expert_metadata_to_cutedsl(
    disabled: bool, expected: bool
) -> None:
    recv_expert_count = torch.tensor([3, 0, 5], dtype=torch.int32)
    expert_capacity = 8
    comm = DeepEPLowLatency.__new__(DeepEPLowLatency)
    comm._dispatch_state = {
        "recv_expert_count": recv_expert_count,
        "expert_capacity": expert_capacity,
    }
    comm.enable_postquant_alltoall = False
    backend = TrtllmCutedslGroupedGemmNvfp4Impl.__new__(TrtllmCutedslGroupedGemmNvfp4Impl)
    backend.disable_deep_ep_direct_metadata = disabled
    backend.use_fused_finalize = True
    backend._weights_created = True
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=backend,
        comm=comm,
        enable_alltoall=False,
        routing_method=SimpleNamespace(experts_per_token=1),
    )

    use_direct_metadata = backend.can_use_deep_ep_direct_metadata(True)
    assert use_direct_metadata is expected

    plan = scheduler._build_comm_plan(
        all_rank_num_tokens=None,
        output_dtype=torch.bfloat16,
        do_finalize=True,
        use_deep_ep_direct_metadata=use_direct_metadata,
    )

    if expected:
        assert plan.recv_expert_count is recv_expert_count
        assert plan.deep_ep_expert_capacity == expert_capacity
    else:
        assert plan.recv_expert_count is None
        assert plan.deep_ep_expert_capacity is None
    assert plan.use_deep_ep_direct_metadata is expected


@pytest.mark.parametrize(
    "multi_stream,capturing,has_comm,router_weight_on_input,expected",
    [
        # Eager forward: host-bound, quantize on the main stream.
        (False, False, False, False, False),
        # CUDA graph runner (multi-stream) or any active capture: fork it.
        (True, False, False, False, True),
        (False, True, False, False, True),
        # A comm strategy owns the quantize/dispatch order.
        (True, False, True, False, False),
        # x is scaled by the router weight before quantization.
        (True, False, False, True, False),
    ],
)
def test_scheduler_quantizes_input_async_only_under_multi_stream(
    monkeypatch,
    multi_stream: bool,
    capturing: bool,
    has_comm: bool,
    router_weight_on_input: bool,
    expected: bool,
) -> None:
    from tensorrt_llm._torch.moe.fused_moe import moe_scheduler

    monkeypatch.setattr(moe_scheduler, "do_multi_stream", lambda: multi_stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=TrtllmCutedslGroupedGemmNvfp4Impl.__new__(TrtllmCutedslGroupedGemmNvfp4Impl),
        comm=object() if has_comm else None,
        apply_router_weight_on_input=router_weight_on_input,
    )

    assert scheduler._should_quantize_input_async(torch.empty(2, 4)) is expected


def test_scheduler_skips_async_quantization_without_backend_support(monkeypatch) -> None:
    from tensorrt_llm._torch.moe.fused_moe import moe_scheduler

    monkeypatch.setattr(moe_scheduler, "do_multi_stream", lambda: True)
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=TrtllmCutlassNvfp4Impl.__new__(TrtllmCutlassNvfp4Impl),
        comm=None,
        apply_router_weight_on_input=False,
    )

    assert scheduler._should_quantize_input_async(torch.empty(2, 4)) is False


# =====================================================================
# do_finalize=False: construction-time validation, the model-owned
# finalize the scheduler runs before the combine, and the one-sided
# workspace plan for that path.
# =====================================================================

_HOOK_HIDDEN = 5
_HOOK_TOP_K = 4
_HOOK_DISPATCHED_ROWS = 12


class _HookBackend:
    """Named only so refusal messages have something to report."""

    def supports_unfinalized_output(self) -> bool:
        return True

    def validate_configurable_moe(self, moe) -> None:
        del moe


class _HookRefusingBackend(_HookBackend):
    def supports_unfinalized_output(self) -> bool:
        return False


class _HookComm:
    def supports_finalize_before_combine(self) -> bool:
        return True


class _HookRefusingComm(_HookComm):
    def supports_finalize_before_combine(self) -> bool:
        return False


def _validating_moe(
    *,
    hook,
    backend=None,
    comm=None,
    apply_router_weight_on_input: bool = False,
) -> ConfigurableMoE:
    """A ``ConfigurableMoE`` carrying exactly what ``validate_backend`` reads.

    ``validate_backend`` runs from ``__init__`` after every wrapper-owned
    attribute is assigned; these tests pin those attributes directly so each
    refusal is exercised without building a real layer.
    """
    moe = ConfigurableMoE.__new__(ConfigurableMoE)
    moe.unfinalized_combine_fn = hook
    moe.comm = comm
    moe.apply_router_weight_on_input = apply_router_weight_on_input
    moe._using_load_balancer = lambda: False
    return moe


def test_unfinalized_combine_fn_construction_accepts_a_capable_configuration():
    moe = _validating_moe(hook=lambda **kwargs: None, comm=_HookComm())
    moe.validate_backend(_HookBackend())  # does not raise


def test_a_backend_without_the_unfinalized_triple_is_refused_at_construction():
    moe = _validating_moe(hook=lambda **kwargs: None, comm=_HookComm())
    with pytest.raises(ValueError, match="cannot return the unfinalized"):
        moe.validate_backend(_HookRefusingBackend())


def test_a_combine_that_is_not_an_unweighted_row_sum_is_refused_at_construction():
    """``DeepEPLowLatency.combine`` consumes expert-major rows and applies the
    routing weights itself; partials finalized per token cannot feed it."""
    moe = _validating_moe(hook=lambda **kwargs: None, comm=_HookRefusingComm())
    with pytest.raises(ValueError, match="unweighted row sum"):
        moe.validate_backend(_HookBackend())


def test_router_weights_folded_into_the_activations_are_refused_at_construction():
    moe = _validating_moe(
        hook=lambda **kwargs: None, comm=_HookComm(), apply_router_weight_on_input=True
    )
    with pytest.raises(ValueError, match="apply_router_weight_on_input"):
        moe.validate_backend(_HookBackend())


def test_a_non_callable_hook_is_refused_at_construction():
    moe = _validating_moe(hook=object())
    with pytest.raises(ValueError, match="callable"):
        moe.validate_backend(_HookBackend())


def test_no_hook_skips_the_hook_validation_entirely():
    """Registered-hook checks must not reject the common hook-less layer."""
    moe = _validating_moe(hook=None, comm=_HookRefusingComm())
    moe.validate_backend(_HookRefusingBackend())  # does not raise


@pytest.mark.cpu_only
def test_the_capability_declarations_match_the_kernels():
    """The leaves that plumb ``do_finalize`` through say so; the default does not."""
    assert TrtllmTrtllmGenNvfp4Impl.__new__(TrtllmTrtllmGenNvfp4Impl).supports_unfinalized_output()
    assert not TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl.__new__(
        TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
    ).supports_unfinalized_output()
    for leaf in CUTLASS_LEAVES:
        assert not leaf.__new__(leaf).supports_unfinalized_output(), leaf.__name__
    assert not DeepEPLowLatency.__new__(DeepEPLowLatency).supports_finalize_before_combine()


def _hook_scheduler(*, finalize_fn, comm=True) -> ExternalCommMoEScheduler:
    moe = SimpleNamespace(
        backend=_HookBackend(),
        comm=_HookComm() if comm else None,
        unfinalized_combine_fn=finalize_fn,
        hidden_size=_HOOK_HIDDEN,
    )
    return ExternalCommMoEScheduler(moe)


def _unfinalized_triple(
    *, rows: int = _HOOK_DISPATCHED_ROWS, flattened: bool = False
) -> List[torch.Tensor]:
    """What ``run_moe`` returns for ``do_finalize=False``.

    ``expanded_idx_to_permuted_idx`` covers ``num_tokens * top_k`` slots.
    FlashInfer returns it flattened; neither its leading dimension nor the
    expert-major ``gemm2_output`` height is necessarily the token count.
    """
    return [
        torch.randn(rows * _HOOK_TOP_K, _HOOK_HIDDEN),
        torch.empty(rows, _HOOK_TOP_K),  # the kernel leaves this buffer uninitialized
        torch.zeros((rows * _HOOK_TOP_K,) if flattened else (rows, _HOOK_TOP_K), dtype=torch.int32),
    ]


@pytest.mark.parametrize("flattened", [False, True], ids=["matrix", "flat-flashinfer"])
@pytest.mark.parametrize("rows", [0, 8, _HOOK_DISPATCHED_ROWS])
def test_the_finalize_runs_on_the_dispatched_rows(flattened: bool, rows: int) -> None:
    seen = {}

    def finalize(
        *,
        gemm2_output,
        expanded_idx_to_permuted_idx,
        token_selected_slots,
        routing_weights,
        num_tokens,
    ):
        seen.update(
            gemm2=gemm2_output,
            index=expanded_idx_to_permuted_idx,
            slots=token_selected_slots,
            weights=routing_weights,
            num_tokens=num_tokens,
        )
        return torch.ones(num_tokens, _HOOK_HIDDEN, dtype=torch.float32)

    triple = _unfinalized_triple(rows=rows, flattened=flattened)
    slots = torch.zeros(rows, _HOOK_TOP_K, dtype=torch.int32)
    scales = torch.rand(rows, _HOOK_TOP_K, dtype=torch.bfloat16)

    out = _hook_scheduler(finalize_fn=finalize)._finalize_before_combine(
        triple,
        token_selected_slots=slots,
        token_final_scales=scales,
        output_dtype=torch.bfloat16,
    )

    assert seen["gemm2"] is triple[0]
    assert seen["index"] is triple[2]
    # A per-expert transform needs the slot each row belongs to; the index map
    # alone cannot answer that.
    assert seen["slots"] is slots
    # The POST-dispatch weights, not the model's own: under attention DP this
    # rank holds rows for tokens it does not own, and only the dispatched copy
    # covers them.
    assert seen["weights"] is scales
    assert seen["num_tokens"] == rows
    assert out.shape == (rows, _HOOK_HIDDEN)


def test_the_result_is_cast_down_before_the_combine_not_after():
    """The one-sided combine region is sized from ``(hidden_size, act_dtype)``.

    The finalize accumulates in FP32; handing that to ``combine`` would need
    twice the symmetric-memory region the workspace reserved.
    """
    scheduler = _hook_scheduler(
        finalize_fn=lambda **kwargs: torch.ones(
            kwargs["num_tokens"], _HOOK_HIDDEN, dtype=torch.float32
        )
    )
    out = scheduler._finalize_before_combine(
        _unfinalized_triple(),
        token_selected_slots=torch.zeros(_HOOK_DISPATCHED_ROWS, _HOOK_TOP_K, dtype=torch.int32),
        token_final_scales=torch.rand(_HOOK_DISPATCHED_ROWS, _HOOK_TOP_K),
        output_dtype=torch.bfloat16,
    )
    assert out.dtype == torch.bfloat16


def test_a_model_that_registered_no_finalize_is_refused_by_name():
    """The one refusal left at runtime: ``do_finalize`` is a forward argument.

    Every other model keeps ``do_finalize=False`` and a comm strategy apart, so
    this combination is unreachable for them today. If one ever reaches it, the
    message has to say which layer and which strategy -- not an
    ``AttributeError`` four frames down inside the comm layer.
    """
    scheduler = _hook_scheduler(finalize_fn=None)
    with pytest.raises(NotImplementedError, match="unfinalized_combine_fn"):
        scheduler._finalize_before_combine(
            _unfinalized_triple(),
            token_selected_slots=None,
            token_final_scales=torch.rand(_HOOK_DISPATCHED_ROWS, _HOOK_TOP_K),
            output_dtype=torch.bfloat16,
        )


def test_a_backend_that_ignored_do_finalize_is_refused():
    """A dense tensor here means the backend already ran its own finalize.

    That finalize is the weighted sum *without* the model's per-expert
    transform. Running the model's finalize on top of it would apply the
    transform to the combined result, which is a different function.
    """
    scheduler = _hook_scheduler(finalize_fn=lambda **kwargs: None)
    with pytest.raises(NotImplementedError, match="triple"):
        scheduler._finalize_before_combine(
            torch.randn(_HOOK_DISPATCHED_ROWS, _HOOK_HIDDEN),
            token_selected_slots=None,
            token_final_scales=torch.rand(_HOOK_DISPATCHED_ROWS, _HOOK_TOP_K),
            output_dtype=torch.bfloat16,
        )


def test_missing_dispatched_routing_weights_are_refused():
    """``token_final_scales`` has no post-dispatch copy to hand the finalize."""
    scheduler = _hook_scheduler(finalize_fn=lambda **kwargs: None)
    with pytest.raises(NotImplementedError, match="token_final_scales"):
        scheduler._finalize_before_combine(
            _unfinalized_triple(),
            token_selected_slots=None,
            token_final_scales=None,
            output_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("flattened", [False, True], ids=["matrix", "flat-flashinfer"])
@pytest.mark.parametrize(
    "shape", [(_HOOK_DISPATCHED_ROWS - 1, _HOOK_TOP_K), (_HOOK_DISPATCHED_ROWS, _HOOK_TOP_K - 1)]
)
def test_the_weights_and_the_index_map_have_to_cover_the_same_rows(
    flattened: bool, shape: Tuple[int, int]
) -> None:
    """The two come from different places and must agree.

    ``token_final_scales`` comes back from ``comm.dispatch``; the index map
    comes out of the MoE kernel. A mismatch means one of them is the
    pre-dispatch copy, which would weight each token's experts with another
    token's routing weights.
    """
    scheduler = _hook_scheduler(finalize_fn=lambda **kwargs: None)
    with pytest.raises(AssertionError, match="index map"):
        scheduler._finalize_before_combine(
            _unfinalized_triple(flattened=flattened),
            token_selected_slots=None,
            token_final_scales=torch.rand(shape),
            output_dtype=torch.bfloat16,
        )


def test_a_hook_that_returns_the_padded_hidden_size_is_caught():
    """``gemm2_output`` is hidden-padded; the hook owns the slice back down."""
    scheduler = _hook_scheduler(
        finalize_fn=lambda **kwargs: torch.ones(kwargs["num_tokens"], _HOOK_HIDDEN + 3)
    )
    with pytest.raises(AssertionError, match="hidden-padded"):
        scheduler._finalize_before_combine(
            _unfinalized_triple(),
            token_selected_slots=None,
            token_final_scales=torch.rand(_HOOK_DISPATCHED_ROWS, _HOOK_TOP_K),
            output_dtype=torch.bfloat16,
        )


def _onesided_scheduler(*, workspace_tensor=None):
    comm = MagicMock(spec=NVLinkOneSided)
    comm.get_combine_payload_tensor_in_workspace.return_value = workspace_tensor
    backend = MagicMock()
    backend.supports_moe_output_in_alltoall_workspace.return_value = True
    backend.input_requirement.onesided_workspace_dtype = None
    moe = SimpleNamespace(comm=comm, backend=backend, hidden_size=_HOOK_HIDDEN)
    return ExternalCommMoEScheduler(moe), comm


def test_the_combine_payload_is_not_claimed_to_be_in_the_workspace():
    """``do_finalize=False`` means the backend writes no combine payload.

    ``combine()`` passes ``payload_in_workspace`` to ``moe_a2a_combine`` as a
    flag, independent of the tensor argument, so a stale ``True`` makes the
    kernel reduce an unwritten workspace region and return garbage without
    raising. The finalize that runs before the combine produces a fresh
    tensor, which is staged like any other caller-owned payload.
    """
    scheduler, comm = _onesided_scheduler()
    moe_output, payload_in_workspace = scheduler._plan_onesided_workspace(
        all_rank_num_tokens=[4, 4], output_dtype=torch.bfloat16, do_finalize=False
    )
    assert moe_output is None
    assert payload_in_workspace is False
    assert not comm.get_combine_payload_tensor_in_workspace.called


def test_a_backend_that_does_finalize_still_gets_the_workspace_payload():
    """The opposite case, so the guard above cannot quietly disable the fast path."""
    sentinel = torch.zeros(8, _HOOK_HIDDEN)
    scheduler, _ = _onesided_scheduler(workspace_tensor=sentinel)
    moe_output, payload_in_workspace = scheduler._plan_onesided_workspace(
        all_rank_num_tokens=[4, 4], output_dtype=torch.bfloat16, do_finalize=True
    )
    assert moe_output is sentinel
    assert payload_in_workspace is True


def _ensure_single_proc_dist_for_megamoe(backend_type: MoeBackendType, rank: int) -> None:
    """Initialize the process group required by MegaMoE constructors."""
    if backend_type not in _MEGAMOE_BACKEND_TYPES:
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MegaMoE tests")
    if dist.is_initialized():
        return
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        store=dist.HashStore(),
        rank=0,
        world_size=1,
    )


def should_skip_gptoss(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    swiglu_gptoss_style: bool,
) -> Optional[str]:
    """
    Check if swiglu_gptoss_style test should be skipped for this backend.

    Only CUTLASS and TRTLLM backends support swiglu_gptoss_style (SwiGlu with custom
    alpha/beta/limit parameters and bias).

    Args:
        backend_type: The MoE backend type
        quant_algo: The quantization algorithm
        swiglu_gptoss_style: Whether swiglu_gptoss_style is enabled

    Returns:
        Skip reason string if test should be skipped, None otherwise
    """
    if not swiglu_gptoss_style:
        return None

    # Only CUTLASS and TRTLLM backends support swiglu_gptoss_style
    supported_backends = {MoeBackendType.CUTLASS, MoeBackendType.TRTLLM}
    if backend_type not in supported_backends:
        return (
            f"swiglu_gptoss_style is only supported by CUTLASS and TRTLLM backends "
            f"(got backend_type={backend_type.value})"
        )

    return None


def test_kimi_fused_route_quant_skips_prequantized_input(monkeypatch) -> None:
    """An upstream fused down projection owns quantization on this path."""
    monkeypatch.delenv("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", raising=False)
    # Patched in the module that owns the override, so the assertion fires if
    # the pre-quantized short-circuit stops running before the SM probe.
    monkeypatch.setattr(
        "tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_w4a8_mxfp4_mxfp8.is_sm_100f",
        MagicMock(side_effect=AssertionError("SM probe must be short-circuited")),
    )
    # The MXFP8-activation leaf: it is the one that overrides
    # ``try_fused_route_quant`` at all, so declining has to be shown on it and
    # not on a leaf that inherits the base's unconditional ``None``.
    backend = TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl.__new__(TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl)
    hidden_states = MxFp8QuantizedTensor(
        fp8_tensor=torch.empty(1, 3584, dtype=torch.float8_e4m3fn),
        scaling_factor=torch.empty(1, 112, dtype=torch.uint8),
    )

    assert (
        backend.try_fused_route_quant(hidden_states, torch.empty(1, 896, dtype=torch.float32))
        is None
    )


def test_kimi_mxfp8_quantized_tensor_handoff() -> None:
    """The NVFP4 communication path preserves an upstream MXFP8 payload."""
    fp8_tensor = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
    scaling_factor = torch.empty(2, 2, dtype=torch.uint8)
    hidden_states = MxFp8QuantizedTensor(fp8_tensor, scaling_factor)
    # The NVFP4 leaf owns this path by identity now, so there is no quant-mode
    # mock to set up: which branch runs is which class was constructed.
    backend = TrtllmTrtllmGenNvfp4Impl.__new__(TrtllmTrtllmGenNvfp4Impl)
    backend._weights_created = True

    quantized, scales = backend.quantize_input(hidden_states)

    assert quantized is fp8_tensor
    assert scales.data_ptr() == scaling_factor.data_ptr()
    assert torch.equal(scales, scaling_factor)


def create_test_backend(
    backend_type: MoeBackendType,
    routing_method: RenormalizeMoeRoutingMethod,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    quant_config,
    mapping: Mapping,
    bias: bool = False,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
    activation_type: ActivationType = ActivationType.Swiglu,
    locality_domain_policy: Optional[LocalityDomainPolicy] = None,
    n_shared_experts: int = 0,
    moe_cls: Optional[type] = None,
) -> MoE:
    """Create a MoE backend for testing.

    ``moe_cls`` pins the leaf class directly, for the providers the default
    lookup never selects (resolution prefers the native TRTLLM-Gen leaf
    whenever one exists for the format).
    """
    backend_cls = moe_cls or get_backend_class(
        backend_type, None if quant_config is None else quant_config.quant_algo
    )
    if locality_domain_policy is None:
        locality_domain_policy = LocalityDomainPolicy(enabled=False)

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype
    if n_shared_experts > 0:
        # TRTLLMGenFusedMoE reads n_shared_experts off the pretrained config to
        # decide num_fused_shared_expert (shared-expert fusion, opt-in via
        # TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1).
        pretrained_config.n_shared_experts = n_shared_experts

    # CUTE_DSL_B12X is internal-only: the user-facing API selects it on the
    # CUTEDSL path when SM120/121 + NVFP4 + flashinfer is importable. Route
    # through "CUTEDSL" so the test exercises the same code path users hit.
    moe_backend_value = (
        "CUTEDSL" if backend_type == MoeBackendType.CUTE_DSL_B12X else backend_type.value
    )
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=mapping,
        moe_backend=moe_backend_value,
        locality_domain_policy=locality_domain_policy,
    )
    if n_shared_experts > 0:
        # The shared-expert-fusion gate runs after the eager create_weights()
        # in __init__, so weight creation must be deferred (as the real model
        # engine does) for the fused trailing slots to be allocated.
        model_config.skip_create_weights_in_init = True

    backend = create_moe_backend(
        moe_cls=backend_cls,
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
        bias=bias,
        weight_loading_mode=weight_loading_mode,
        activation=build_test_activation(activation_type, swiglu_alpha, swiglu_beta, swiglu_limit),
    )
    if n_shared_experts > 0:
        backend.create_weights()
    return backend


# The quantized FlashInfer provider is otherwise never run: resolution prefers
# the native TRTLLM-Gen leaf whenever one exists for the format (only the bf16
# leaf is FlashInfer-exclusive), so without pinning the class the FlashInfer
# NVFP4 kernels have zero coverage. The real provider is exercised here, which
# needs Blackwell: FlashInfer's FP4 path has no pre-Blackwell backend.
@skip_pre_blackwell
def test_flashinfer_provider_nvfp4_matches_reference(tmp_path: Path):
    pytest.importorskip("flashinfer")

    model_config = MoeModelConfig(8, 2, 512, 512)
    seq_len = 8
    dtype = torch.bfloat16
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)

        routing_method = RenormalizeMoeRoutingMethod(top_k=model_config.top_k)
        x = torch.randn((seq_len, model_config.hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, model_config.num_experts), dtype=dtype, device="cuda")

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, x, MoeBackendType.TRTLLM
        )
        quantize_util = quantize_util_cls(
            num_experts=model_config.num_experts,
            dtype=dtype,
            intermediate_size=model_config.intermediate_size,
            hidden_size=model_config.hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            activation_type=ActivationType.Swiglu,
        )
        NVFP4TRTLLMGenFusedMoEBaseMethod._cache_permute_indices.clear()

        backend = create_test_backend(
            backend_type=MoeBackendType.TRTLLM,
            routing_method=routing_method,
            num_experts=model_config.num_experts,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            weight_loading_mode=getattr(
                quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
            ),
            moe_cls=FlashinferTrtllmGenNvfp4Impl,
        )
        assert backend.use_flashinfer

        ref_cls = quant_kwargs.pop("ref_cls", None)
        weights = quantize_util.create_weights(**quant_kwargs)
        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        if ref_cls is not None:
            ref_fused_moe = quantize_util.create_ref_module(routing_method, ref_cls=ref_cls)
        else:
            ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        def run_moe():
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            return run_backend_moe(
                backend,
                MoeBackendType.TRTLLM,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype,
                router_logits,
            )

        autotuner = AutoTuner.get()
        autotuner.warmup = 0
        autotuner.repeat = 1
        autotuner.stream_delay_micro_secs = 10
        autotuner.clear_cache()
        with (
            torch.inference_mode(),
            autotune(cache_path=str(tmp_path / "moe_autotuner_cache.json")),
        ):
            _ = run_moe()
        with torch.inference_mode():
            output = run_moe()
        ref_fused_moe.check_accuracy(output, ref_output)


# =====================================================================
# Staged post-load hook lifecycle tests
# =====================================================================
# These tests cover staged hook contracts rather than the common backend matrix
# below. Keep them grouped so they can move to a dedicated file if the MoE test
# layout is split later.
def test_moe_post_load_weights_uses_idempotent_transform_hook():
    class HookTestMoE(MoE):
        def create_weights(self):
            raise NotImplementedError

        def load_weights(self, weights, allow_partial_loading=False):
            raise NotImplementedError

        def quantize_input(self, x, **kwargs):
            return x, None

        def run_moe(self, ctx, *, workspace=None):
            raise NotImplementedError

    moe = HookTestMoE.__new__(HookTestMoE)
    torch.nn.Module.__init__(moe)
    quant_method = SimpleNamespace(
        transform_weights=MagicMock(),
        cache_derived_state=MagicMock(),
    )
    moe.quant_method = quant_method

    moe.post_load_weights()
    moe.transform_weights()

    quant_method.transform_weights.assert_called_once_with(moe)
    quant_method.cache_derived_state.assert_called_once_with(moe)
    assert moe._weights_transformed is True

    moe.cache_derived_state()
    assert quant_method.cache_derived_state.call_count == 2

    moe._weights_transformed = False
    moe.transform_weights()
    assert quant_method.transform_weights.call_count == 2


def test_fused_moe_load_weights_invalidates_transform_guard():
    class GuardResetMethod(UnquantizedFusedMoEMethod):
        def load_expert_weights_to_dst(
            self,
            module,
            weights,
            weight_loading_mode,
            load_expert_ids,
            dst_w3_w1_weight,
            dst_w2_weight,
            dst_w3_w1_bias,
            dst_w2_bias,
            allow_partial_loading=False,
        ):
            module.loaded_allow_partial = allow_partial_loading

        def load_quant_scales(self, module, weights):
            module.loaded_scales = bool(weights)

        def setup_quant_scales(self, module):
            module.quant_scales = ()

    method = GuardResetMethod()
    module = SimpleNamespace(
        initial_local_expert_ids=[0],
        w3_w1_weight=torch.empty(1, 2, 2),
        w2_weight=torch.empty(1, 2, 2),
        bias=False,
        _weights_transformed=True,
    )

    method.load_weights(
        module,
        {"0.w1.weight": torch.ones(1)},
        MoEWeightLoadingMode.VANILLA,
        allow_partial_loading=True,
    )

    assert module.loaded_allow_partial is True
    assert module.loaded_scales is True
    assert module._weights_transformed is False


def test_configurable_moe_post_load_weights_uses_backend_staged_hooks():
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    class HookTestConfigurableMoE(ConfigurableMoE):
        def quantize_input(self, x, **kwargs):
            return x, None

        def run_moe(self, ctx, *, workspace=None):
            raise NotImplementedError

    configurable_moe = HookTestConfigurableMoE.__new__(HookTestConfigurableMoE)
    torch.nn.Module.__init__(configurable_moe)
    backend = torch.nn.Module()
    backend.transform_weights = MagicMock()
    backend.cache_derived_state = MagicMock()
    configurable_moe.backend = backend

    configurable_moe.post_load_weights()
    configurable_moe.transform_weights()

    backend.transform_weights.assert_called_once_with()
    backend.cache_derived_state.assert_called_once_with()
    assert configurable_moe._weights_transformed is True

    configurable_moe.cache_derived_state()
    assert backend.cache_derived_state.call_count == 2


def test_configurable_moe_load_weights_invalidates_wrapper_transform_guard():
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    configurable_moe = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(configurable_moe)
    backend = torch.nn.Module()
    backend.load_weights = MagicMock(return_value="loaded")
    configurable_moe.backend = backend
    configurable_moe._weights_transformed = True

    weights = [{"0.w1.weight": torch.ones(1)}]
    result = configurable_moe.load_weights(weights, allow_partial_loading=True)

    assert result == "loaded"
    backend.load_weights.assert_called_once_with(weights, True)
    assert configurable_moe._weights_transformed is False


def test_moe_nvfp4_activation_quantization_capability():
    assert NVFP4FusedMoEMethod.quantizes_nvfp4_activations
    assert not W4A16NVFP4CutlassFusedMoEMethod.quantizes_nvfp4_activations


def test_marlin_moe_repack_is_transform_stage():
    assert "transform_weights" in NVFP4MarlinFusedMoEMethod.__dict__
    assert "post_load_weights" not in NVFP4MarlinFusedMoEMethod.__dict__
    assert NVFP4MarlinFusedMoEMethod.post_load_weights is FusedMoEMethodBase.post_load_weights


@pytest.mark.parametrize("act_scale", [None, torch.ones(1, 8)], ids=["absent", "present"])
def test_marlin_refuses_a_pre_quant_activation_scale(act_scale):
    """An AWQ-style checkpoint must be refused, not served with the scale dropped.

    ``canonical_quant`` folds NVFP4_AWQ / NVFP4_ARC into ``nvfp4`` before the
    ``MoEProblem`` is built, so no eligibility gate can see the distinction --
    the guard has to sit where the evidence appears, which is after
    ``load_quant_scales`` has materialized ``fc31_act_scale``.

    ``__new__`` without ``__init__``: the check reads one attribute off the
    module and a real constructor would need a GPU.
    """
    method = NVFP4MarlinFusedMoEMethod.__new__(NVFP4MarlinFusedMoEMethod)
    module = SimpleNamespace(fc31_act_scale=act_scale)

    if act_scale is None:
        # Nothing to refuse; it falls through to the real repack, which needs
        # loaded weights. Reaching past the guard is the assertion here.
        with pytest.raises(AttributeError):
            method.transform_weights(module)
    else:
        with pytest.raises(ValueError, match="pre-quant activation scale"):
            method.transform_weights(module)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "leaf_cls,run_fn",
    [
        (TrtllmCutedslGroupedGemmNvfp4Impl, "run_moe_nvfp4"),
        (TrtllmCutedslGroupedGemmBf16Impl, "run_moe_bf16"),
        (CuteDslFc12FusedMoE, "run_moe_nvfp4"),
    ],
    ids=["grouped_gemm_nvfp4", "grouped_gemm_bf16", "fused_fc12"],
)
def test_cutedsl_finalize_weights_once_with_the_router_weight_on_the_input(leaf_cls, run_fn):
    """The scheduler folds the top-1 weight into ``x`` and passes no scales.

    The finalize epilogue multiplies by per-token scales regardless, so it gets ones.
    """
    captured = {}
    backend = SimpleNamespace(
        apply_router_weight_on_input=True,
        _build_local_weight_view=lambda: None,
        **{run_fn: lambda **kwargs: captured.update(kwargs)},
    )
    ctx = SimpleNamespace(
        x=torch.zeros(2, 8),
        x_sf=None,
        x_ready_event=None,
        token_selected_experts=torch.tensor([[3], [1]], dtype=torch.int32),
        token_final_scales=None,
        comm_plan=SimpleNamespace(
            moe_output=None,
            enable_alltoall=False,
            recv_expert_count=None,
            deep_ep_expert_capacity=None,
            use_deep_ep_direct_metadata=False,
        ),
    )
    leaf_cls.run_moe(backend, ctx)
    torch.testing.assert_close(captured["token_final_scales"], torch.ones(2, 1))

    backend.apply_router_weight_on_input = False
    with pytest.raises(AssertionError, match="apply_router_weight_on_input"):
        leaf_cls.run_moe(backend, ctx)


# The CuteDSL NVFP4 leaves all bind the shared ``quantize_input_nvfp4``.
_CUTEDSL_NVFP4_LEAVES = pytest.mark.parametrize(
    "leaf_cls",
    [TrtllmCutedslGroupedGemmNvfp4Impl, CuteDslFc12FusedMoE, TrtllmCutedslDenseGemmNvfp4Impl],
    ids=["grouped_gemm", "fused_fc12", "dense_gemm"],
)


def _nvfp4_input_quantizer_state(
    act_scale: torch.Tensor | None, hidden_size: int = 32
) -> SimpleNamespace:
    """The backend attributes ``quantize_input`` reads, and nothing else."""
    return SimpleNamespace(
        hidden_size=hidden_size,
        scaling_vector_size=16,
        fc31_input_scale=torch.ones(()),
        fc31_act_scale=act_scale,
    )


def _fake_fp4_quantize(x, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled):
    """A CPU ``fp4_quantize`` with the real op's output shapes."""
    num_rows, k = x.shape
    return (
        torch.zeros(num_rows, k // 2, dtype=torch.uint8),
        torch.zeros(num_rows * k // sf_vec_size, dtype=torch.uint8),
    )


@pytest.mark.cpu_only
@_CUTEDSL_NVFP4_LEAVES
def test_cutedsl_nvfp4_scale_width_survives_an_empty_micro_batch(leaf_cls, monkeypatch):
    """``view(0, -1)`` cannot infer a width from zero rows, so it is spelled out."""
    hidden_size = 32
    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", _fake_fp4_quantize)
    _, x_sf = leaf_cls.quantize_input(
        _nvfp4_input_quantizer_state(None, hidden_size),
        torch.empty(0, hidden_size, dtype=torch.bfloat16),
        post_quant_comm=False,
    )

    assert x_sf.shape == (0, hidden_size // 16)


@pytest.mark.cpu_only
def test_cutedsl_dense_gemm_returns_an_empty_output_for_an_empty_micro_batch():
    """The leaf returns before ``swizzle_sf``, which cannot lay out zero rows."""
    hidden_size, num_experts = 32, 4
    backend = SimpleNamespace(has_nvfp4=True, hidden_size=hidden_size)
    output = TrtllmCutedslDenseGemmNvfp4Impl.run_moe_nvfp4(
        backend,
        x=torch.empty(0, hidden_size // 2, dtype=torch.uint8),
        token_selected_experts=torch.empty(0, num_experts, dtype=torch.int32),
        token_final_scales=torch.empty(0, num_experts, dtype=torch.float32),
        x_sf=torch.empty(0, hidden_size // 16, dtype=torch.uint8),
    )
    assert output.shape == (0, hidden_size)
    assert output.dtype == torch.bfloat16


@skip_pre_blackwell
def test_cutedsl_nvfp4_pre_quant_scale_matches_cutlass():
    """CuteDSL and CUTLASS agree on an NVFP4_AWQ layer, and both match the reference.

    The checkpoint is calibrated on ``x * pre_quant_scale``, which is what the
    reference runs FC1 on. The last check is the control: the same output
    against the reference on the unsmoothed input must fail, or this test could
    not tell a dropped scale apart.
    """
    model_config = MoeModelConfig(8, 2, 512, 512)
    seq_len = 16
    dtype = torch.bfloat16
    verdict = TrtllmCutedslGroupedGemmNvfp4Impl.can_implement(
        MoEProblem(
            quant=QuantAlgo.NVFP4.value,
            dtype_act=dtype,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_experts=model_config.num_experts,
            top_k=model_config.top_k,
        ),
        MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=model_config.num_experts,
            env=collect_moe_environment(),
        ),
    )
    if not verdict.eligible:
        pytest.skip(verdict.detail)

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)

        routing_method = RenormalizeMoeRoutingMethod(top_k=model_config.top_k)
        x = torch.randn((seq_len, model_config.hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, model_config.num_experts), dtype=dtype, device="cuda")
        # Per-channel factors in [1/e, e]: dropping them moves FC1's input far
        # past the comparison tolerance.
        pre_quant_scale = (
            torch.empty(model_config.hidden_size, device="cuda").uniform_(-1, 1).exp().to(dtype)
        )
        x_smoothed = x * pre_quant_scale

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, x_smoothed, MoeBackendType.CUTEDSL
        )
        quantize_util = quantize_util_cls(
            num_experts=model_config.num_experts,
            dtype=dtype,
            intermediate_size=model_config.intermediate_size,
            hidden_size=model_config.hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            activation_type=ActivationType.Swiglu,
        )
        weights = quantize_util.create_weights(**quant_kwargs)
        for expert_id in range(model_config.num_experts):
            weights[f"{expert_id}.w1.pre_quant_scale"] = pre_quant_scale
            weights[f"{expert_id}.w3.pre_quant_scale"] = pre_quant_scale

        outputs = {}
        for backend_type in (MoeBackendType.CUTLASS, MoeBackendType.CUTEDSL):
            backend = create_test_backend(
                backend_type=backend_type,
                routing_method=routing_method,
                num_experts=model_config.num_experts,
                hidden_size=model_config.hidden_size,
                intermediate_size=model_config.intermediate_size,
                dtype=dtype,
                quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4_AWQ),
                mapping=mapping,
            )
            backend.load_weights([weights])
            backend.post_load_weights()
            backend.cuda()
            assert backend.fc31_act_scale is not None
            with torch.inference_mode():
                token_selected_experts, token_final_scales = routing_method.apply(router_logits)
                x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
                outputs[backend_type] = run_backend_moe(
                    backend,
                    backend_type,
                    x_quantized,
                    x_sf,
                    token_selected_experts,
                    token_final_scales,
                    dtype,
                    router_logits,
                )

        ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()
        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x_smoothed, router_logits)
            unsmoothed_ref_output = ref_fused_moe.forward(x, router_logits)

        cutlass_output = outputs[MoeBackendType.CUTLASS]
        cutedsl_output = outputs[MoeBackendType.CUTEDSL]
        ref_fused_moe.check_accuracy(cutlass_output, ref_output)
        ref_fused_moe.check_accuracy(cutedsl_output, ref_output)
        ref_fused_moe.check_accuracy(cutedsl_output, cutlass_output)
        with pytest.raises(Exception, match="Mismatch percentage"):
            ref_fused_moe.check_accuracy(cutedsl_output, unsmoothed_ref_output)


def _marlin_model_config(quant_algo=QuantAlgo.NVFP4):
    cfg = ModelConfig()
    cfg.moe_backend = "MARLIN"
    cfg.quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo else None
    return cfg


def _marlin_environment(sm: int = 90) -> MoEEnvironment:
    """Marlin's own SM window, so quantization stays the only variable."""
    return MoEEnvironment(sm=sm)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "quant_algo, expected",
    [
        pytest.param(None, TrtllmCutlassUnquantizedImpl, id="unquantized"),
        pytest.param(QuantAlgo.FP8, TrtllmCutlassFp8Impl, id="fp8"),
    ],
)
def test_marlin_degrades_to_cutlass_on_non_nvfp4(quant_algo, expected):
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(_marlin_model_config(quant_algo))
    assert impl_class_for(report) is expected
    assert report.degraded
    # The cause is Marlin's own rejection, not a fallback leaf's.
    assert issubclass(_class_named(report.degraded_from.legacy_backend), MarlinFusedMoE)
    assert report.degraded_from.reason is MoERejectReason.QUANT_UNSUPPORTED


@pytest.mark.cpu_only
def test_marlin_override_quant_config_degrades_per_layer():
    cfg = _marlin_model_config()
    with override_moe_environment(_marlin_environment()):
        report = resolve_moe_impl(
            cfg,
            override_quant_config=QuantConfig(quant_algo=None),
            layer_idx=52,
        )
    assert impl_class_for(report) is TrtllmCutlassUnquantizedImpl
    assert issubclass(_class_named(report.degraded_from.legacy_backend), MarlinFusedMoE)
    assert report.degraded_from.reason is MoERejectReason.QUANT_UNSUPPORTED


def _class_named(legacy_backend: str) -> type:
    return next(impl for impl in IMPL_PRIORITY if impl.__name__ == legacy_backend)


# =====================================================================
# TRTLLM-Gen SiTu backend contract
# =====================================================================
# SiTu rides the generic SwiGLU geometry, so the host-side wiring is easy to
# get wrong in ways no shape check catches: it reaches the cubin through the
# same ``gemm1_alpha`` / ``gemm1_beta`` slots SwiGLU's constants use, and only
# the activation kind separates them. Kernel-level coverage -- tactic
# availability, launch evidence and SiTu-vs-SwiGLU numerics -- lives in
# test_kimi_k3_situ_moe.py (and moe/kernels/serial/test_moe.py for the runner). These
# are the contracts that hold without running a cubin.

_situ_supported = pytest.mark.skipif(
    not is_sm_100f(),
    reason="TRTLLM-Gen SiTu cubins are sm_100f: the SM100 family only",
)

# Kimi K3 defaults. The gate-side SiTu beta is the cubin's ``alpha``, the
# linear-side beta its ``beta``. See modeling_kimi_linear.py.
_SITU_GATE_ALPHA = 4.0
_SITU_LINEAR_BETA = 25.0


def _make_trtllm_gen_moe(
    *,
    quant_config: Optional[QuantConfig],
    situ: bool,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 512,
    intermediate_size: int = 256,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> TRTLLMGenFusedMoE:
    """A TRTLLM-Gen MoE, with or without the SiTu override."""
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_rank),
        moe_backend="TRTLLM",
    )
    # The leaf for this checkpoint's format. ``TRTLLMGenFusedMoE`` itself is
    # abstract now, so the class has to come from the quantization the caller
    # asked for.
    leaf_cls = trtllm_gen_leaf(None if quant_config is None else quant_config.quant_algo)
    return leaf_cls(
        routing_method=RenormalizeMoeRoutingMethod(top_k=top_k),
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        # The soft-caps go out as scalars; the backend's declared
        # PER_EXPERT_TENSOR shape is what broadcasts them per slot.
        activation=(
            SiTuActivation(
                gate_softcap=_SITU_GATE_ALPHA,
                linear_softcap=_SITU_LINEAR_BETA,
            )
            if situ
            else DEFAULT_MOE_ACTIVATION
        ),
    )


def _make_loaded_nvfp4_trtllm_gen_moe(situ: bool) -> TRTLLMGenFusedMoE:
    """The same MoE with NVFP4 weights and quant scales actually loaded.

    The scale contracts below are written by ``load_quant_scales``, so they
    only exist after a real load; the stock quantize utils supply the weights.
    """
    num_experts, hidden_size, intermediate_size = 8, 512, 256
    torch.manual_seed(0)
    x = torch.randn((8, hidden_size), dtype=torch.bfloat16, device="cuda") * 0.5
    util_cls, quant_config, quant_kwargs = get_test_quant_params(QuantAlgo.NVFP4, x, "TRTLLM")
    quant_kwargs.pop("ref_cls", None)
    quantize_util = util_cls(
        num_experts=num_experts,
        dtype=torch.bfloat16,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        quant_config=quant_config,
        bias=False,
        activation_type=ActivationType.Swiglu,
    )
    backend = _make_trtllm_gen_moe(
        quant_config=quant_config,
        situ=situ,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    backend.load_weights([quantize_util.create_weights(**quant_kwargs)])
    backend.post_load_weights()
    return backend.cuda()


@_situ_supported
def test_trtllm_gen_nvfp4_situ_selects_padded_quant_method() -> None:
    """``_get_quant_method`` must key off ``is_situ_activation``.

    SiTu fills the act_alpha/act_beta slots from ``create_weights``, i.e.
    after ``_get_quant_method`` has already run, so keying off
    ``act_alpha is not None`` would make the selected method depend on
    *when* it is resolved. Plain SwiGLU is the control: it still gets the
    unpadded base method, so this is not asserting a constant.
    """
    situ = _make_loaded_nvfp4_trtllm_gen_moe(situ=True)
    assert situ.is_situ_activation
    assert isinstance(situ.quant_method, NVFP4TRTLLMGenFusedMoEMethod)
    assert situ.scaling_vector_size == 16

    plain = _make_loaded_nvfp4_trtllm_gen_moe(situ=False)
    assert not plain.is_situ_activation
    assert type(plain.quant_method) is NVFP4TRTLLMGenFusedMoEBaseMethod


@_situ_supported
def test_trtllm_gen_nvfp4_situ_fc31_scale_c_drops_dequant_scale() -> None:
    """SiTuGlu is nonlinear in x0, so scaleC must be quantScaleC alone.

    Pins the trtllm-gen host rule (``!isLinearInX0(mActType) && mFusedAct`` in
    ``BatchedGemm/BatchedGemmTestUtils.h``) at the level where TRT-LLM
    computes it, without needing to run a kernel. SwiGLU is the control: it
    *is* linear in x0 and keeps the combined ``quantScaleC * dequantScaleAb``.

    Getting this wrong is not a small numeric error. The extra dequantScaleAb
    factor (~6e-8) drives the FC1 output's per-block E4M3 scale factors below
    their smallest subnormal, so the NVFP4 intermediate quantizes to exactly
    zero and the whole MoE returns zeros -- an output that still passes a
    ``rtol=0.1, atol=0.15`` tolerance band, because these MoE outputs have a
    standard deviation (~0.04) well below ``atol``.
    """
    situ = _make_loaded_nvfp4_trtllm_gen_moe(situ=True)
    swiglu = _make_loaded_nvfp4_trtllm_gen_moe(situ=False)

    # fc2_input_scale is quantScaleC; fc31_alpha is dequantScaleAb (the
    # scaleGate the kernel already folds into the tanh/sigmoid arguments).
    torch.testing.assert_close(
        situ.fc31_scale_c.data.float(),
        situ.fc2_input_scale.data.float().expand_as(situ.fc31_scale_c.data),
        msg="SiTu fc31_scale_c must not carry the dequantScaleAb factor",
    )
    torch.testing.assert_close(
        swiglu.fc31_scale_c.data.float(),
        (swiglu.fc2_input_scale.data * swiglu.fc31_alpha.data).float(),
        msg="SwiGLU fc31_scale_c must keep the combined scale",
    )
    # The two conventions must actually differ, otherwise this test would pass
    # vacuously on a build where dequantScaleAb happens to be 1.
    assert not torch.allclose(situ.fc31_scale_c.data.float(), swiglu.fc31_scale_c.data.float())


@_situ_supported
@pytest.mark.parametrize(
    "hidden_size,intermediate_size,expected",
    [
        pytest.param(512, 256, (32, 32), id="no_padding"),
        pytest.param(2880, 2880, (256, 256), id="hidden_not_256_aligned"),
        pytest.param(1536, 192, (128, 32), id="intermediate_needs_128"),
    ],
)
def test_nvfp4_trtllm_gen_resolve_alignments_matches_create_weights(
    hidden_size, intermediate_size, expected
) -> None:
    """``resolve_alignments`` must predict what ``create_weights`` selects.

    ``TRTLLMGenFusedMoE`` validates its MoE-TP shard from ``__init__``, i.e.
    before ``create_weights`` has shadowed the class attributes with
    shape-resolved instance ones, so the check has to predict them. If the two
    ever drift apart the shard contract is validated against the wrong number.
    """
    predicted = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(hidden_size, intermediate_size)
    assert predicted == expected

    backend = _make_trtllm_gen_moe(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        situ=True,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    assert (
        backend.quant_method.weight_alignment,
        backend.quant_method.input_hidden_alignment,
    ) == predicted


@_situ_supported
def test_nvfp4_trtllm_gen_tp_accepts_logical_group_aligned_shard() -> None:
    """Logical TP alignment is independent of the padded kernel layout."""
    backend = _make_trtllm_gen_moe(
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        situ=True,
        hidden_size=7168,
        intermediate_size=3072,
        tp_size=16,
    )

    assert backend.intermediate_size_per_partition == 192
    assert backend.intermediate_size_per_partition % 16 == 0
    assert backend.quant_method.weight_alignment == 128
    assert backend.intermediate_size_per_partition % backend.quant_method.weight_alignment != 0


@pytest.mark.cpu_only
def test_megamoe_quant_method_defines_canonical_live_plane_spec():
    method = NVFP4MegaMoECuteDslMethod()
    module = SimpleNamespace(
        hidden_size=7168,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
    )

    spec = method.live_weight_plane_spec(module)

    assert (spec.hidden, spec.intermediate, spec.expand_intermediate) == (7168, 3072, 6144)
    assert tuple(plane.name for plane in spec.planes) == (
        "mega_fc1_weight",
        "mega_fc1_weight_sf",
        "mega_fc2_weight",
        "mega_fc2_weight_sf",
        "fc31_alpha",
        "fc2_alpha",
        "fc1_norm_const",
    )

    # Exercise the storage-to-kernel dtype/view bridge on a small legal
    # geometry; production metadata is covered by the independent goldens.
    bridge = method._live_weight_bundle_layout(160, 80, 160)
    for plane in bridge.planes:
        shape, stride, dtype_name, element_size, pointer_alignment = plane.storage_metadata(3)
        tensor = torch.empty(shape, dtype=getattr(torch, dtype_name))
        assert tuple(tensor.stride()) == stride
        assert tensor.element_size() == element_size
        assert tensor.data_ptr() % pointer_alignment == 0

        kernel_shape, kernel_stride, _, kernel_element_size, _ = plane.kernel_metadata(3)
        kernel_view = tensor
        if plane.name in ("mega_fc1_weight", "mega_fc2_weight"):
            kernel_view = kernel_view.transpose(1, 2)
        assert tuple(kernel_view.shape) == kernel_shape
        assert tuple(kernel_view.stride()) == kernel_stride
        assert kernel_view.element_size() == kernel_element_size


@pytest.mark.cpu_only
def test_megamoe_live_plane_spec_rejects_expand_before_allocation():
    method = NVFP4MegaMoECuteDslMethod()
    module = SimpleNamespace(
        hidden_size=160,
        intermediate_size_per_partition=80,
        expand_intermediate_size_per_partition=176,
    )

    with pytest.raises(ValueError, match="expand_intermediate == 2"):
        method.live_weight_plane_spec(module)


@pytest.mark.cpu_only
def test_megamoe_rejects_expand_before_arena_allocation():
    allocate_arena = MagicMock()
    module = SimpleNamespace(
        hidden_size=160,
        intermediate_size_per_partition=80,
        expand_intermediate_size_per_partition=176,
        layer_load_balancer=SimpleNamespace(
            produces_replica_plan=True, allocate_arena=allocate_arena
        ),
    )

    with pytest.raises(ValueError, match="expand_intermediate == 2"):
        NVFP4MegaMoECuteDslMethod().create_weights(module)

    allocate_arena.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("produces_replica_plan", [True, False, None])
def test_megamoe_quant_method_allocates_and_binds_the_arena_through_the_layer_handle(
    monkeypatch, produces_replica_plan
):
    method = NVFP4MegaMoECuteDslMethod()
    bundle = method._live_weight_bundle_layout(160, 80, 160)
    resident, helpers = 2, 1 if produces_replica_plan else 0
    arena = object()
    events = []

    def create_parent_weights(self, module, *args):
        events.append("parent")
        for name in NVFP4MegaMoECuteDslMethod._STREAMED_SOURCE_PARAMS:
            module.register_parameter(
                name, torch.nn.Parameter(torch.empty(1, dtype=torch.uint8), requires_grad=False)
            )
        for name in ("fc31_alpha", "fc2_alpha"):
            shape, _, dtype_name, _, _ = method._live_weight_plane(bundle, name).storage_metadata(
                resident
            )
            module.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.ones(shape, dtype=getattr(torch, dtype_name)), requires_grad=False
                ),
            )

    def allocate_arena(**kwargs):
        events.append(("allocate", kwargs))
        return arena

    monkeypatch.setattr(NVFP4FusedMoEMethod, "create_weights", create_parent_weights)
    monkeypatch.setattr(NVFP4MegaMoECuteDslMethod, "setup_quant_scales", lambda self, module: None)
    monkeypatch.setattr(
        NVFP4MegaMoECuteDslMethod,
        "_bind_live_weight_planes",
        lambda self, module, bound: events.append(("bind", bound)),
    )
    module = torch.nn.Module()
    module.hidden_size = 160
    module.intermediate_size_per_partition = 80
    module.expand_intermediate_size_per_partition = 160
    module.bias = False
    module.expert_size_per_partition = resident
    module.compute_slot_count = resident + helpers
    module.rebuild_tensor_metadata = {}
    module.ep_process_group = "ep_group"
    module.mapping = SimpleNamespace(local_rank=3)
    module.layer_idx = 5
    module.layer_load_balancer = (
        None
        if produces_replica_plan is None
        else SimpleNamespace(
            produces_replica_plan=produces_replica_plan, allocate_arena=allocate_arena
        )
    )

    method.create_weights(module)

    assert module.mega_fc1_weight.shape[0] == resident + helpers
    if not produces_replica_plan:
        assert events == ["parent"]
        return
    # Every rank allocates the arena before any Parameter exists and binds the
    # seven planes once all of them are registered.
    assert events == [
        (
            "allocate",
            dict(
                bundle=bundle,
                ep_process_group="ep_group",
                mapping=module.mapping,
                device=3,
                layer_idx=5,
            ),
        ),
        "parent",
        ("bind", arena),
    ]


@pytest.mark.cpu_only
def test_megamoe_quant_method_builds_the_producer_after_checking_the_planes(monkeypatch):
    events = []
    monkeypatch.setattr(
        NVFP4FusedMoEMethod, "cache_derived_state", lambda self, module: events.append("parent")
    )
    monkeypatch.setattr(
        NVFP4MegaMoECuteDslMethod,
        "assert_live_weight_aliases",
        lambda self, module, arena: events.append(("aliases", arena)),
    )
    arena = SimpleNamespace(assert_identity=lambda: events.append("identity"))
    handle = SimpleNamespace(
        produces_replica_plan=True,
        live_weight_arena=arena,
        bind_backend=lambda module: events.append(("bind_backend", module)),
    )
    module = SimpleNamespace(layer_load_balancer=handle)
    method = NVFP4MegaMoECuteDslMethod()

    method.cache_derived_state(module)

    assert events == ["parent", "identity", ("aliases", arena), ("bind_backend", module)]

    # Standard EPLB and EPLB-off layers only refresh the parent state.
    events.clear()
    method.cache_derived_state(SimpleNamespace(layer_load_balancer=None))
    method.cache_derived_state(
        SimpleNamespace(layer_load_balancer=SimpleNamespace(produces_replica_plan=False))
    )
    assert events == ["parent", "parent"]

    handle.live_weight_arena = None
    with pytest.raises(RuntimeError, match="no live-weight arena"):
        method.cache_derived_state(module)


@pytest.mark.cpu_only
def test_megamoe_live_plane_identity_guard_detects_parameter_replacement():
    class Tensor:
        def __init__(self, pointer):
            self.shape = (2, 3)
            self.dtype = "u8"
            self.device = "cuda:0"
            self.is_meta = False
            self._pointer = pointer

        def stride(self):
            return (3, 1)

        def data_ptr(self):
            return self._pointer

    arena_view = Tensor(17)
    alias = Tensor(17)
    plane = SimpleNamespace(name="weight")
    bundle = SimpleNamespace(planes=(plane,))
    arena = SimpleNamespace(
        bundle=bundle,
        local_plane_views=(arena_view,),
        tekit_alias_views=(alias,),
    )
    module = SimpleNamespace(weight=Tensor(17))
    method = NVFP4MegaMoECuteDslMethod()
    method.live_weight_plane_spec = MagicMock(return_value=bundle)

    method.assert_live_weight_aliases(module, arena)

    module.weight = Tensor(23)
    with pytest.raises(RuntimeError, match="changed storage"):
        method.assert_live_weight_aliases(module, arena)


def _per_layer_dynamic_handle(helper_slots_per_rank):
    return SimpleNamespace(produces_replica_plan=True, helper_slots_per_rank=helper_slots_per_rank)


@pytest.mark.cpu_only
def test_megamoe_compute_slots_come_from_the_layer_handle():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe.expert_size_per_partition = 8
    moe.layer_load_balancer = None
    assert moe.compute_slot_count == 8

    # Standard EPLB produces no replica plan and adds no helper slot.
    moe.layer_load_balancer = SimpleNamespace(produces_replica_plan=False)
    assert moe.compute_slot_count == 8

    moe.layer_load_balancer = _per_layer_dynamic_handle(3)
    assert moe.compute_slot_count == 11
    assert moe.expert_size_per_partition == 8


@pytest.mark.cpu_only
def test_megamoe_replica_plan_supplies_launch_facts_and_route_wait():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe.expert_size_per_partition = 8
    moe.layer_load_balancer = _per_layer_dynamic_handle(3)
    wait_for_routes = MagicMock()
    ready_flags = torch.empty(8, dtype=torch.int64)
    plan = MoEReplicaPlan(
        resident_experts_per_rank=8,
        compute_slots_per_rank=11,
        ready_flags=ready_flags,
        ready_generation=7,
        reserved_sms=4,
        _wait_for_routes=wait_for_routes,
    )

    compute_slots, helper_slots, actual_flags, generation, reserved_sms = (
        moe._replica_plan_launch_args(plan)
    )
    assert (compute_slots, helper_slots, generation, reserved_sms) == (11, 3, 7, 4)
    assert actual_flags is ready_flags
    moe._wait_rebalance_routes(plan)
    wait_for_routes.assert_called_once_with()

    moe._wait_rebalance_routes()
    wait_for_routes.assert_called_once_with()


@pytest.mark.cpu_only
def test_megamoe_staging_widens_int32_routes_after_the_route_wait():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe._last_staged_T = {8: 6}
    trace = []
    moe._wait_rebalance_routes = lambda replica_plan=None: trace.append(("wait", replica_plan))
    bufs = SimpleNamespace(
        activation=torch.zeros(8, 16, dtype=torch.uint8),
        activation_sf=torch.zeros(8, 4, dtype=torch.uint8),
        topk_weights=torch.ones(8, 4, dtype=torch.float32),
        topk_idx_local=torch.full((8, 4), 5, dtype=torch.int64),
    )
    routes = torch.tensor([[0, 3, -1, 2**31 - 1], [7, 6, 5, 4], [1, 1, 1, 1]], dtype=torch.int32)
    weights = torch.full((3, 4), 0.25, dtype=torch.float32)

    moe._stage_inputs(
        bufs=bufs,
        x=torch.ones(3, 16, dtype=torch.uint8),
        x_sf=torch.ones(3, 4, dtype=torch.uint8),
        topk_idx=routes,
        topk_weights=weights,
        num_tokens=3,
        top_k=4,
        stage_activation=True,
        replica_plan="plan",
    )

    # One copy widens the int32 routes into the int64 kernel index buffer.
    assert bufs.topk_idx_local.dtype == torch.int64
    assert torch.equal(bufs.topk_idx_local[:3], routes.to(torch.int64))
    # Rows a previous launch wrote, and only those, are reset to -1.
    assert torch.equal(bufs.topk_idx_local[3:6], torch.full((3, 4), -1, dtype=torch.int64))
    assert torch.equal(bufs.topk_idx_local[6:], torch.full((2, 4), 5, dtype=torch.int64))
    assert torch.equal(bufs.topk_weights[:3], weights)
    assert not bufs.topk_weights[3:].any()
    assert trace == [("wait", "plan")]


@pytest.mark.cpu_only
def test_megamoe_run_moe_hands_routes_to_staging_unconverted():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe.ep_size = 1
    moe._weights_created = True
    moe.quant_method = object()
    moe.hidden_size = 16
    moe.dtype = torch.bfloat16
    moe._topk_idx_dtype = torch.int64
    moe._build_weight_view = lambda: "weights"
    moe._wait_rebalance_routes = MagicMock()
    captured = {}
    moe._run_moe = lambda **kwargs: captured.update(kwargs) or "out"
    routes = torch.zeros(3, 4, dtype=torch.int32)
    scales = torch.ones(3, 4, dtype=torch.float32)

    out = moe.run_moe(
        MoERunContext(
            token_selected_experts=routes,
            token_final_scales=scales,
            x=torch.zeros(3, 8, dtype=torch.uint8),
            x_sf=torch.zeros(3, 4, dtype=torch.uint8),
            output_dtype=torch.bfloat16,
        )
    )

    assert out == "out"
    # No conversion kernel and no early route wait: staging reads the routes.
    assert captured["topk_idx"] is routes
    assert captured["topk_weights"] is scales
    moe._wait_rebalance_routes.assert_not_called()

    half_scales = scales.to(torch.float16)
    moe.run_moe(
        MoERunContext(
            token_selected_experts=routes,
            token_final_scales=half_scales,
            x=torch.zeros(3, 8, dtype=torch.uint8),
            x_sf=torch.zeros(3, 4, dtype=torch.uint8),
            output_dtype=torch.bfloat16,
        )
    )
    assert captured["topk_weights"].dtype == torch.float32
    assert torch.equal(captured["topk_weights"], scales)


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_sm_budget", [True, False])
def test_megamoe_quantize_reserves_sms_only_for_a_replica_plan(monkeypatch, request, has_sm_budget):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl

    calls = []

    def outputs(x):
        rows = x.shape[0]
        return (
            torch.zeros(rows, x.shape[1] // 2, dtype=torch.uint8),
            torch.zeros(rows * x.shape[1] // 16, dtype=torch.uint8),
        )

    def fp4_quantize(x, scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled):
        calls.append(("default", sf_vec_size, sf_use_ue8m0, is_sf_swizzled))
        return outputs(x)

    def sm_budget(x, scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled, reserved_sms):
        calls.append(("sm_budget", sf_vec_size, sf_use_ue8m0, is_sf_swizzled, reserved_sms))
        return outputs(x)

    if has_sm_budget:
        fp4_quantize.sm_budget = sm_budget
    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", fp4_quantize, raising=False)
    mega_moe_cute_dsl._fp4_quantize_has_sm_budget.cache_clear()
    request.addfinalizer(mega_moe_cute_dsl._fp4_quantize_has_sm_budget.cache_clear)
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe.fc31_input_scale = torch.ones(1)
    x = torch.zeros(3, 64, dtype=torch.bfloat16)

    moe.quantize_input(x)
    moe.quantize_input(x, reserved_sms=4)

    default = ("default", 16, False, False)
    budget = ("sm_budget", 16, False, False, 4) if has_sm_budget else default
    assert calls == [default, budget]


@pytest.mark.cpu_only
def test_megamoe_replica_plan_rejects_unallocated_geometry():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe.expert_size_per_partition = 8
    moe.layer_load_balancer = _per_layer_dynamic_handle(3)
    plan = MoEReplicaPlan(
        resident_experts_per_rank=8,
        compute_slots_per_rank=12,
        ready_flags=torch.empty(8, dtype=torch.int64),
        ready_generation=7,
        reserved_sms=8,
        _wait_for_routes=lambda: None,
    )

    with pytest.raises(RuntimeError, match="geometry does not match"):
        moe._replica_plan_launch_args(plan)


@pytest.mark.cpu_only
def test_megamoe_cutedsl_post_load_weights_uses_staged_hooks():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    quant_method = SimpleNamespace(
        transform_weights=MagicMock(),
        cache_derived_state=MagicMock(),
    )
    moe.quant_method = quant_method

    moe.post_load_weights()
    moe.transform_weights()

    quant_method.transform_weights.assert_called_once_with(moe)
    quant_method.cache_derived_state.assert_called_once_with(moe)
    assert moe._weights_transformed is True


def test_megamoe_cutedsl_mpi_bootstrap_keeps_executor_cuda_device(monkeypatch):
    """The bootstrap must bind NCCL to the device the executor already chose.

    The executor binds global_mpi_rank() % device_count before the model is
    built; re-selecting a device here would move later layers to another GPU.
    """
    import tensorrt_llm._utils as utils

    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    moe.ep_size = 8
    mpi_comm = MagicMock()
    mpi_comm.bcast.return_value = ("127.0.0.1", "29500")
    local_mpi_comm = MagicMock()
    init_process_group = MagicMock()
    set_device = MagicMock()

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(utils, "mpi_world_size", lambda: 8)
    monkeypatch.setattr(utils, "mpi_rank", lambda: 3)
    monkeypatch.setattr(utils, "mpi_comm", lambda: mpi_comm)
    monkeypatch.setattr(utils, "local_mpi_comm", local_mpi_comm)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)

    moe._maybe_init_torch_dist_under_mpi()

    set_device.assert_not_called()
    # Split_type is a collective; it must not run in this bootstrap.
    local_mpi_comm.assert_not_called()
    init_process_group.assert_called_once_with(
        backend="cuda:nccl,cpu:gloo",
        rank=3,
        world_size=8,
        device_id=torch.device("cuda", 2),
    )


@pytest.mark.cpu_only
def test_megamoe_cutedsl_cache_derived_state_reuses_rebalance_binding(
    monkeypatch,
):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
        PerLayerDynamicMoeLoadBalancer,
        planner,
    )
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    config = SimpleNamespace(
        num_slots=24,
        auxiliary_sms=8,
        validate_expert_capacity=lambda num_experts: None,
    )
    manager = PerLayerDynamicMoeLoadBalancer(
        ep_rank=0,
        ep_size=4,
        config=config,
    )
    layer = manager.add_layer(
        expert_count=16,
        top_k=2,
        slot_count_per_rank=6,
    )
    bundle = NVFP4MegaMoECuteDslMethod._live_weight_bundle_layout(160, 80, 160)
    arena = MagicMock(home_experts=4, helper_slots=2, bundle=bundle)
    arena.assert_identity = MagicMock()
    allocate = MagicMock(return_value=arena)
    monkeypatch.setattr(arena_module, "allocate_rebalance_arena", allocate)
    monkeypatch.setattr(manager, "get_ep_comm", lambda process_group: "ep_comm")
    monkeypatch.setattr(manager, "create_arena_provider", lambda: "provider")
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe._ep_pg = object()
    moe._maxt_buckets = [64, 128]
    moe.mapping = SimpleNamespace(local_rank=0)
    moe.layer_idx = 5
    moe.layer_load_balancer = layer
    moe.initial_global_assignments = list(range(16))
    # The quant method's create_weights allocates through the handle; a second
    # call with the same plane spec returns the same arena.
    for _ in range(2):
        assert (
            layer.allocate_arena(
                bundle=bundle,
                ep_process_group=moe.ep_process_group,
                mapping=moe.mapping,
                device=0,
                layer_idx=5,
            )
            is arena
        )
    allocate.assert_called_once()
    moe.quant_method = NVFP4MegaMoECuteDslMethod()
    monkeypatch.setattr(NVFP4MegaMoECuteDslMethod, "setup_quant_scales", lambda self, module: None)
    assert_aliases = MagicMock()
    monkeypatch.setattr(NVFP4MegaMoECuteDslMethod, "assert_live_weight_aliases", assert_aliases)
    builder = MagicMock(return_value=SimpleNamespace(home_experts=4, helper_slots=2))
    monkeypatch.setattr(
        planner,
        "build_rebalance_slot_scheduler_group",
        builder,
    )

    moe.cache_derived_state()
    moe.cache_derived_state()

    assert arena.assert_identity.call_count == 2
    assert assert_aliases.call_count == 2
    assert layer.live_weight_arena is arena
    builder.assert_called_once_with(
        arena=arena,
        manager=manager,
        mapping=moe.mapping,
        ep_process_group=moe._ep_pg,
        home_experts=4,
        helper_slots=2,
        topk=2,
        max_tokens_per_rank=128,
        layer_idx=5,
    )


@pytest.mark.cpu_only
def test_rebalance_discard_plan_orders_release_and_allows_next_reuse():
    trace = []
    outputs = SimpleNamespace(physical_slot_ids=[3, 4])
    part = (outputs, 2)
    copy_stream = object()

    group = RebalanceSlotSchedulerGroup.__new__(RebalanceSlotSchedulerGroup)
    group._check_owner = MagicMock()
    group._lent_rows = -1
    group._lent_routes = None
    group._plan_part = part
    group._generation = 1
    group._finished_generation = 0
    group._scheduled_generation = 1
    group._route_wait_generation = 0
    group._pending_release = None
    group.plan_calls = 0
    group._execution_stream_handle = 23
    group.copy_stream = copy_stream
    group.scheduler_stream = copy_stream

    def wait_for_routes():
        trace.append("wait_routes")
        group._route_wait_generation = group._generation

    def release_generation_after(generation, event):
        # Without an event, the release follows the MAIN work queued so far.
        assert (generation, event) == (1, None)
        trace.append("release_generation")

    broadcaster = SimpleNamespace(
        release_generation_after=MagicMock(side_effect=release_generation_after),
        mark_collective_reuse_safe=MagicMock(
            side_effect=lambda authority, generation: trace.append("prove_reuse")
        ),
    )
    group.wait_for_routes = wait_for_routes
    group.broadcaster = broadcaster

    group.discard_plan(part)

    assert group._plan_part is None
    assert group.plan_calls == 1
    assert group._finished_generation == 1
    assert group._pending_release == (1, None)
    assert trace == ["wait_routes", "release_generation"]

    # The next schedule advances this counter before proving that generation 1
    # can be reused. Exercise the real lease check to show cleanup left a valid
    # transition rather than a permanently poisoned group.
    group._scheduled_generation = 2
    lease = _LiveBankLeaseProvider(group)
    lease.prove(1)

    assert lease.proved_generation == 1
    assert trace[-1] == "prove_reuse"


def test_megamoe_load_weights_invalidates_cached_deepgemm_views():
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    hidden_size = 128
    intermediate_size = 128
    module = SimpleNamespace(
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        initial_local_expert_ids=[0],
        w3_w1_weight=torch.empty(1, intermediate_size * 2, hidden_size // 2, dtype=torch.uint8),
        w3_w1_weight_scale=torch.empty(
            1, intermediate_size * 2, hidden_size // 32, dtype=torch.uint8
        ),
        w2_weight=torch.empty(1, hidden_size, intermediate_size // 2, dtype=torch.uint8),
        w2_weight_scale=torch.empty(1, hidden_size, intermediate_size // 32, dtype=torch.uint8),
        _t_l1=(torch.empty(1), torch.empty(1)),
        _t_l2=(torch.empty(1), torch.empty(1)),
        _t_l1_weight=torch.empty(1),
        _t_l1_scale=torch.empty(1),
        _t_l1_scale_slot=torch.empty(1),
        _t_l2_weight=torch.empty(1),
        _t_l2_scale=torch.empty(1),
        _t_l2_scale_slot=torch.empty(1),
    )
    weights = {
        "0.w1.weight": torch.full((intermediate_size, hidden_size // 2), 1, dtype=torch.uint8),
        "0.w3.weight": torch.full((intermediate_size, hidden_size // 2), 2, dtype=torch.uint8),
        "0.w2.weight": torch.full((hidden_size, intermediate_size // 2), 3, dtype=torch.uint8),
        "0.w1.weight_scale": torch.full(
            (intermediate_size, hidden_size // 32), 4, dtype=torch.uint8
        ),
        "0.w3.weight_scale": torch.full(
            (intermediate_size, hidden_size // 32), 5, dtype=torch.uint8
        ),
        "0.w2.weight_scale": torch.full(
            (hidden_size, intermediate_size // 32), 6, dtype=torch.uint8
        ),
    }

    method.load_weights(module, [weights])

    assert module.w3_w1_weight[0, 0, 0].item() == 1
    assert module.w3_w1_weight[0, intermediate_size, 0].item() == 2
    assert module._weights_loaded is True
    for attr in (
        "_t_l1",
        "_t_l2",
        "_t_l1_weight",
        "_t_l1_scale",
        "_t_l1_scale_slot",
        "_t_l2_weight",
        "_t_l2_scale",
        "_t_l2_scale_slot",
    ):
        assert getattr(module, attr) is None


def test_megamoe_streaming_reload_resets_slot_claims():
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    method._clear_transformed_weight_cache = MagicMock()
    module = SimpleNamespace(
        rebuild_tensor_metadata={},
        _packed_mxfp4_loaded_slots={0},
        expert_size_per_partition=1,
        initial_local_expert_ids=[0],
        w3_w1_weight=torch.empty(1, 2, 1, dtype=torch.uint8),
        w3_w1_weight_scale=torch.empty(1, 2, 1, dtype=torch.uint8),
        w2_weight=torch.empty(1, 1, 1, dtype=torch.uint8),
        w2_weight_scale=torch.empty(1, 1, 1, dtype=torch.uint8),
    )

    method.pre_reload_weights(module)

    assert module._packed_mxfp4_loaded_slots == set()

    weight = torch.ones(1, 1, dtype=torch.uint8)
    method.load_packed_mxfp4_expert(
        module,
        global_expert_id=0,
        local_slot_id=0,
        w1_weight=weight,
        w1_weight_scale=weight,
        w2_weight=weight,
        w2_weight_scale=weight,
        w3_weight=weight,
        w3_weight_scale=weight,
    )

    assert module._packed_mxfp4_loaded_slots == {0}
    method._clear_transformed_weight_cache.assert_called_once_with(module)


def test_megamoe_cache_derived_state_sets_initial_assignments_once():
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()
    method.setup_quant_scales = MagicMock()
    load_balancer = MagicMock()
    module = SimpleNamespace(
        layer_load_balancer=load_balancer,
        initial_global_assignments=[0],
    )

    method.cache_derived_state(module)

    load_balancer.set_initial_weight_assignments.assert_called_once_with([0])
    method.setup_quant_scales.assert_called_once_with(module)


def test_megamoe_deepgemm_cache_derived_state_allocates_symm_buffer():
    moe = MegaMoEDeepGemm.__new__(MegaMoEDeepGemm)
    torch.nn.Module.__init__(moe)
    quant_method = SimpleNamespace(cache_derived_state=MagicMock())
    moe.quant_method = quant_method
    moe._alloc_symm_buffer = MagicMock()

    moe.cache_derived_state()

    moe._alloc_symm_buffer.assert_called_once_with()
    quant_method.cache_derived_state.assert_called_once_with(moe)


def test_megamoe_deepgemm_mpi_bootstrap_replaces_outer_launcher_world(monkeypatch):
    """A launcher env describing another world must not seed this MPI world.

    A disaggregated launcher can export its outer RANK/WORLD_SIZE into each
    server, whose model runs in its own MPI world. The bootstrap must then
    rendezvous over MPI and overwrite the stale variables.
    """
    mpi_comm = MagicMock()
    mpi_comm.Get_rank.return_value = 2
    mpi_comm.Get_size.return_value = 4
    mpi_comm.bcast.return_value = ("model-host", 29501)
    local_mpi_comm = MagicMock()
    local_mpi_comm.Get_rank.return_value = 2
    mpi_comm.Split_type.return_value = local_mpi_comm
    mpi = SimpleNamespace(COMM_WORLD=mpi_comm, COMM_TYPE_SHARED=1)
    init_process_group = MagicMock()

    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=mpi))
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_ADDR", "outer-host")
    monkeypatch.setenv("MASTER_PORT", "29400")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    MegaMoEDeepGemm._maybe_init_dist_from_mpi()

    mpi_comm.bcast.assert_called_once_with(None, root=0)
    assert os.environ["MASTER_ADDR"] == "model-host"
    assert os.environ["MASTER_PORT"] == "29501"
    assert os.environ["RANK"] == "2"
    assert os.environ["WORLD_SIZE"] == "4"
    init_process_group.assert_called_once_with(
        backend="nccl",
        rank=2,
        world_size=4,
        device_id=None,
    )


def test_megamoe_cache_derived_state_survives_the_read_only_reader_walk():
    """The GMS read-only reader reaches the override through the wrapper.

    ``ConfigurableMoE`` marks itself ``_weights_removed`` so the loader does not
    run the wrapper's hooks on top of the backend's, which leaves the backend
    reachable only as a registered submodule. Losing the override there is
    silent -- the SymmBuffer simply never gets allocated and the failure
    surfaces as the assert in ``run_moe`` -- so the walk itself is asserted,
    not just the override in isolation.
    """
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    # ``__new__`` rather than the constructor: the real ``__init__`` would want
    # MPI, a process group, and a DG SymmBuffer, none of which this walk needs.
    backend = MegaMoEDeepGemm.__new__(MegaMoEDeepGemm)
    torch.nn.Module.__init__(backend)
    backend.quant_method = SimpleNamespace(cache_derived_state=MagicMock())
    backend._alloc_symm_buffer = MagicMock()

    wrapper = torch.nn.Module()
    wrapper._weights_removed = True
    wrapper.cache_derived_state = MagicMock()
    wrapper.backend = backend

    model = torch.nn.Module()
    model.moe = wrapper

    ModelLoader._walk_cache_state(model)

    backend._alloc_symm_buffer.assert_called_once_with()
    backend.quant_method.cache_derived_state.assert_called_once_with(backend)
    wrapper.cache_derived_state.assert_not_called()


def test_megamoe_cutedsl_cache_derived_state_survives_the_read_only_reader_walk():
    """The CuteDSL leaf's override has to be the one the walk reaches.

    Same wrapper geometry as the DeepGEMM sibling above, but this override
    guards a different thing: the base hook dereferences ``self.quant_method``
    unguarded, and a weights-removed reader never reaches
    ``post_load_weights``, so losing the override here leaves the derived
    MegaMoE-format state silently never recomputed.
    """
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    # ``__new__`` rather than the constructor: the real ``__init__`` would want
    # a process group and the cuMem symmetric-memory rendezvous.
    backend = TrtllmCutedslMegaMoeNvfp4Impl.__new__(TrtllmCutedslMegaMoeNvfp4Impl)
    torch.nn.Module.__init__(backend)
    backend.quant_method = None
    quant_method = SimpleNamespace(cache_derived_state=MagicMock())
    backend.create_weights = MagicMock(
        side_effect=lambda: setattr(backend, "quant_method", quant_method)
    )

    wrapper = torch.nn.Module()
    wrapper._weights_removed = True
    wrapper.cache_derived_state = MagicMock()
    wrapper.backend = backend

    model = torch.nn.Module()
    model.moe = wrapper

    ModelLoader._walk_cache_state(model)

    backend.create_weights.assert_called_once_with()
    quant_method.cache_derived_state.assert_called_once_with(backend)
    wrapper.cache_derived_state.assert_not_called()


def test_megamoe_bakes_situ_softcaps_as_uniform_scalars():
    # MegaMoE declares UNIFORM_SCALAR for alpha/beta because the kernels bake
    # them at codegen time, so a per-expert tensor is reduced here.
    params = materialize_activation_params(
        SiTuActivation(gate_softcap=torch.full((8,), 4.0), linear_softcap=25.0),
        MegaMoEDeepGemm.activation_support,
        num_local_experts=8,
        owner="MegaMoEDeepGemm",
    )

    assert params.activation_type is ActivationType.SiTu
    assert params.alpha == 4.0
    assert params.beta == 25.0


def test_megamoe_plain_swiglu_carries_no_constants():
    params = materialize_activation_params(
        SwigluActivation(),
        MegaMoEDeepGemm.activation_support,
        num_local_experts=8,
        owner="MegaMoEDeepGemm",
    )

    assert (params.alpha, params.beta) == (None, None)
    assert params.clamp is None
    assert params.clamp_after_silu is False


def test_cutlass_materializes_post_silu_clamp_mode():
    params = materialize_activation_params(
        SwigluActivation(clamp=5.0, clamp_after_silu=True),
        CutlassFusedMoE.activation_support,
        num_local_experts=2,
        device="cpu",
        owner="CutlassFusedMoE",
    )

    assert torch.equal(params.clamp, torch.full((2,), 5.0))
    assert params.clamp_after_silu is True


def test_deepgemm_rejects_post_silu_clamp_materialization():
    with pytest.raises(ValueError, match="does not implement post-SiLU clamping"):
        materialize_activation_params(
            SwigluActivation(clamp=5.0, clamp_after_silu=True),
            DeepGemmFusedMoE.activation_support,
            num_local_experts=2,
            owner="DeepGemmFusedMoE",
        )


@pytest.mark.cpu_only
def test_cutedsl_bf16_refuses_a_clamp_at_materialization():
    """The BF16 leaf's declaration refuses the clamp its FC1 op would otherwise drop."""
    with pytest.raises(ValueError, match="take no activation clamp"):
        materialize_activation_params(
            SwigluActivation(clamp=7.0),
            TrtllmCutedslGroupedGemmBf16Impl.activation_support,
            num_local_experts=2,
            owner="TrtllmCutedslGroupedGemmBf16Impl",
        )


def test_post_silu_clamp_mode_requires_limit():
    with pytest.raises(ValueError, match="requires a clamp value"):
        SwigluActivation(clamp_after_silu=True)


def test_fused_moe_appends_post_silu_mode_to_positional_schema():
    registered_schema = torch.ops.trtllm.fused_moe.default._schema

    assert registered_schema.arguments[-1].name == "swiglu_clamp_after_silu"


def test_create_moe_forwards_situ_activation_as_one_carrier(monkeypatch):
    create_moe_module = importlib.import_module("tensorrt_llm._torch.moe.fused_moe.create_moe")
    configurable_moe = MagicMock(return_value=object())
    monkeypatch.setattr(create_moe_module, "ConfigurableMoE", configurable_moe)
    monkeypatch.setattr(
        create_moe_module,
        "resolve_moe_cls",
        MagicMock(return_value=MegaMoEDeepGemm),
    )
    activation = SiTuActivation(gate_softcap=4.0, linear_softcap=25.0)

    result = create_moe_module.create_moe(
        routing_method=MagicMock(),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=ModelConfig(),
        activation=activation,
    )

    assert result is configurable_moe.return_value
    assert configurable_moe.call_args.kwargs["activation"] is activation


def test_create_moe_backend_rejects_apply_router_weight_on_input_by_declaration():
    """The gate runs ahead of every ``moe_cls`` branch, so a class the factory
    has never heard of still reaches it -- which is the point of reading a
    declaration instead of matching a class."""

    class _Undeclared:
        capabilities = MoEStaticCapability()

    with pytest.raises(ValueError, match="apply_router_weight_on_input"):
        create_moe_backend(
            moe_cls=_Undeclared,
            routing_method=MagicMock(),
            num_experts=8,
            hidden_size=512,
            intermediate_size=512,
            apply_router_weight_on_input=True,
        )


def test_apply_router_weight_on_input_support_is_not_inherited():
    """The NVFP4 B12x leaf has a CUTLASS parent, and this is a field where the
    two disagree: only the NVFP4 prefill chunk reaches the Cutlass ``run_moe``,
    while the decode path hands ``token_final_scales`` to the flashinfer
    wrapper."""
    assert TrtllmCutlassNvfp4Impl.capabilities.supports_apply_router_weight_on_input
    assert MarlinFusedMoE.capabilities.supports_apply_router_weight_on_input
    for b12x_leaf in (CuteDslB12xNvfp4FusedMoE, CuteDslB12xW4a16Nvfp4FusedMoE):
        assert not b12x_leaf.capabilities.supports_apply_router_weight_on_input
    assert not TRTLLMGenFusedMoE.capabilities.supports_apply_router_weight_on_input


def test_megamoe_init_rejects_uneven_num_slots_with_value_error():
    routing_method = RenormalizeMoeRoutingMethod(top_k=1)
    model_config = ModelConfig(
        mapping=Mapping(
            world_size=4,
            rank=0,
            tp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
            enable_attention_dp=True,
        ),
        moe_backend=MoeBackendType.MEGAMOE_DEEPGEMM.value,
    )

    # The check runs inside ``__init__``, before any distributed setup, so the
    # rejection is reachable without a process group.
    with pytest.raises(
        ValueError,
        match=r"MegaMoEDeepGemm requires num_slots \(10\) divisible by ep_size \(4\)",
    ):
        MegaMoEDeepGemm(
            routing_method=routing_method,
            num_experts=10,
            hidden_size=512,
            intermediate_size=512,
            dtype=torch.bfloat16,
            model_config=model_config,
            init_load_balancer=False,
        )


def test_megamoe_post_load_rejects_uneven_num_slots_with_value_error(monkeypatch):
    import tensorrt_llm._torch.moe.fused_moe.quantization as quantization_module

    class DummyModule:
        _weights_loaded = True
        num_slots = 10
        ep_size = 4

    monkeypatch.setattr(quantization_module, "_import_deep_gemm", lambda: object())
    method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()

    with pytest.raises(
        ValueError,
        match=r"MegaMoEDeepGemm requires num_slots \(10\) divisible by ep_size \(4\)",
    ):
        method.post_load_weights(DummyModule())


def _make_megamoe_cutedsl_for_ctor_test(
    *,
    activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
) -> MegaMoECuteDsl:
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
    )
    return MegaMoECuteDsl(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
        activation=activation,
    )


def _make_megamoe_cutedsl_minimax_problem(
    *,
    bias: bool,
    activation: ActivationType,
) -> MoEProblem:
    return MoEProblem(
        quant=QuantAlgo.NVFP4.value,
        dtype_act=torch.bfloat16,
        hidden_size=512,
        intermediate_size=512,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=True,
        bias=bias,
        activation=activation.name,
        activation_constants=frozenset({"alpha", "beta", "clamp"}),
    )


def _make_megamoe_cutedsl_test_deployment(
    *, ep_size: int = 1, parallel_size: int = 1, use_dp: bool = False
) -> MoEDeployment:
    return MoEDeployment(
        ep_size=ep_size,
        tp_size=1,
        parallel_size=parallel_size,
        use_dp=use_dp,
        num_slots=8,
        env=MoEEnvironment(
            sm=100,
            available_deps=(MoEDep.MEGAMOE_CUTEDSL_RUNTIME, MoEDep.MEGAMOE_CUTEDSL_OP),
        ),
    )


@pytest.mark.cpu_only
def test_megamoe_cutedsl_accepts_minimax_m3_problem() -> None:
    verdict = MegaMoECuteDsl.can_implement(
        _make_megamoe_cutedsl_minimax_problem(
            bias=False,
            activation=ActivationType.SwigluBias,
        ),
        _make_megamoe_cutedsl_test_deployment(),
    )

    assert verdict.eligible, verdict.detail


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "bias, activation",
    [
        (True, ActivationType.SwigluBias),
        (False, ActivationType.Swiglu),
    ],
    ids=["expert_bias", "wrong_activation"],
)
def test_megamoe_cutedsl_rejects_non_minimax_swiglu_bias_package(
    bias: bool,
    activation: ActivationType,
) -> None:
    verdict = MegaMoECuteDsl.can_implement(
        _make_megamoe_cutedsl_minimax_problem(bias=bias, activation=activation),
        _make_megamoe_cutedsl_test_deployment(),
    )

    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


def _make_megamoe_cutedsl_tep_problem(
    activation: ActivationType, constants: set[str]
) -> MoEProblem:
    return MoEProblem(
        quant=QuantAlgo.NVFP4.value,
        dtype_act=torch.bfloat16,
        hidden_size=512,
        intermediate_size=512,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=activation is ActivationType.SwigluBias,
        bias=False,
        activation=activation.name,
        activation_constants=frozenset(constants),
    )


_MEGAMOE_CUTEDSL_TEP_ACTIVATIONS = [
    pytest.param(ActivationType.SwigluBias, {"alpha", "beta", "clamp"}, id="minimax"),
    pytest.param(ActivationType.SiTu, {"alpha", "beta"}, id="kimi"),
    pytest.param(ActivationType.Swiglu, set(), id="plain-swiglu"),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("parallel_size", [2, 8])
@pytest.mark.parametrize("activation,constants", _MEGAMOE_CUTEDSL_TEP_ACTIVATIONS)
def test_megamoe_cutedsl_accepts_tep_for_every_model(
    parallel_size: int, activation: ActivationType, constants: set[str]
) -> None:
    verdict = MegaMoECuteDsl.can_implement(
        _make_megamoe_cutedsl_tep_problem(activation, constants),
        _make_megamoe_cutedsl_test_deployment(ep_size=parallel_size, parallel_size=parallel_size),
    )

    assert verdict.eligible, verdict.detail


@pytest.mark.cpu_only
@pytest.mark.parametrize("activation,constants", _MEGAMOE_CUTEDSL_TEP_ACTIVATIONS)
def test_megamoe_cutedsl_rejects_moe_tensor_parallelism(
    activation: ActivationType, constants: set[str]
) -> None:
    deployment = dataclasses.replace(
        _make_megamoe_cutedsl_test_deployment(ep_size=2, parallel_size=4), tp_size=2
    )
    verdict = MegaMoECuteDsl.can_implement(
        _make_megamoe_cutedsl_tep_problem(activation, constants), deployment
    )

    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED


@pytest.mark.cpu_only
def test_strict_megamoe_cutedsl_request_reports_rejection_detail() -> None:
    """A strict MEGAMOE_CUTEDSL request names why the backend was rejected."""
    model_config = ModelConfig()
    model_config.moe_backend = "MEGAMOE_CUTEDSL"
    deployment = dataclasses.replace(
        _make_megamoe_cutedsl_test_deployment(ep_size=2, parallel_size=4), tp_size=2
    )

    with pytest.raises(ValueError, match="EP-only"):
        resolve_moe_impl(
            model_config,
            problem=_make_megamoe_cutedsl_tep_problem(ActivationType.Swiglu, set()),
            deployment=deployment,
            allow_degradation=False,
        )


def _fused_comm_scheduler(
    *,
    rank: int,
    tp_size: int = 4,
    enable_attention_dp: bool = False,
    reduce_results: bool = False,
    accepts_global_routed_output: bool = False,
    run_moe_reuses_buffers: bool = False,
) -> FusedCommMoEScheduler:
    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        tp_size=tp_size,
        moe_tp_size=1,
        moe_ep_size=tp_size,
        enable_attention_dp=enable_attention_dp,
    )
    moe = Mock(
        spec_set=[
            "parallel_size",
            "use_dp",
            "reduce_results",
            "accepts_global_routed_output",
            "mapping",
            "backend",
        ]
    )
    moe.parallel_size = tp_size
    moe.use_dp = enable_attention_dp
    moe.reduce_results = reduce_results
    moe.accepts_global_routed_output = accepts_global_routed_output
    moe.mapping = mapping
    # ``forward`` reads the declaration off the backend before it chunks.
    moe.backend = SimpleNamespace(
        input_requirement=MoEInputRequirement(run_moe_reuses_buffers=run_moe_reuses_buffers)
    )
    return FusedCommMoEScheduler(moe)


@pytest.mark.cpu_only
def test_fused_comm_tep_output_is_a_rank_partial_for_external_all_reduce() -> None:
    """The caller's TP all-reduce of the per-rank outputs restores the global
    routed output exactly once, as it does for external-comm backends."""
    tp_size = 4
    global_output = torch.randn(5, 16, dtype=torch.bfloat16)
    partials = [
        _fused_comm_scheduler(rank=rank, tp_size=tp_size)._as_rank_partial_if_reduced_externally(
            global_output.clone()
        )
        for rank in range(tp_size)
    ]

    assert torch.equal(partials[0], global_output)
    for partial in partials[1:]:
        assert partial.shape == global_output.shape
        assert partial.dtype == global_output.dtype
        assert not partial.any()
    assert torch.equal(torch.stack(partials).sum(dim=0), global_output)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"reduce_results": True}, id="layer-reduces"),
        pytest.param({"accepts_global_routed_output": True}, id="caller-takes-global"),
        pytest.param({"enable_attention_dp": True}, id="attention-dp"),
        pytest.param({"tp_size": 1}, id="single-rank"),
    ],
)
def test_fused_comm_output_stays_global_when_nobody_reduces_it_again(overrides: dict) -> None:
    tp_size = overrides.get("tp_size", 4)
    global_output = torch.randn(5, 16, dtype=torch.bfloat16)
    for rank in range(tp_size):
        scheduler = _fused_comm_scheduler(rank=rank, **overrides)
        assert scheduler._as_rank_partial_if_reduced_externally(global_output) is global_output


@pytest.mark.cpu_only
@pytest.mark.parametrize("accepts_global_routed_output", [False, True])
def test_fused_comm_forward_returns_what_the_caller_reduces(
    monkeypatch: pytest.MonkeyPatch, accepts_global_routed_output: bool
) -> None:
    """``forward`` itself must hand back a rank partial under TEP, not only the
    helper; multi-chunk outputs are concatenated before the conversion."""
    tp_size = 4
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    router_logits = torch.randn(5, 8)
    global_output = torch.randn(5, 16, dtype=torch.bfloat16)
    outputs = []
    for rank in range(tp_size):
        scheduler = _fused_comm_scheduler(
            rank=rank,
            tp_size=tp_size,
            accepts_global_routed_output=accepts_global_routed_output,
        )
        monkeypatch.setattr(
            scheduler,
            "_strip_adp_padding",
            lambda x, rl, ids, counts: (x, rl, ids, [x.shape[0]], 0, False),
        )
        monkeypatch.setattr(
            scheduler,
            "_compute_chunk_layout",
            lambda x, rl, counts, ep_rank: (
                2,
                list(x.split([3, 2])),
                list(rl.split([3, 2])),
                [[3, 2]],
            ),
        )
        monkeypatch.setattr(
            scheduler,
            "_run_chunks",
            lambda *args, **kwargs: list(global_output.clone().split([3, 2])),
        )
        outputs.append(
            scheduler.forward(
                x,
                router_logits,
                do_finalize=True,
                output_dtype=torch.bfloat16,
                all_rank_num_tokens=None,
                use_dp_padding=None,
                input_ids=None,
            )
        )

    assert torch.equal(outputs[0], global_output)
    for output in outputs[1:]:
        if accepts_global_routed_output:
            assert torch.equal(output, global_output)
        else:
            assert not output.any()


@pytest.mark.cpu_only
def test_fused_comm_forward_refuses_a_backend_that_reuses_its_buffers() -> None:
    """``_run_chunks`` keeps the tensor each chunk returned, so a backend whose
    next ``run_moe`` overwrites it would collect the same storage for every
    chunk. Only ``ExternalCommMoEScheduler`` copies chunks out; here the
    combination is a named refusal rather than the last chunk's rows repeated."""
    scheduler = _fused_comm_scheduler(rank=0, run_moe_reuses_buffers=True)

    with pytest.raises(NotImplementedError, match="run_moe_reuses_buffers"):
        scheduler.forward(
            torch.randn(5, 16, dtype=torch.bfloat16),
            torch.randn(5, 8),
            do_finalize=True,
            output_dtype=torch.bfloat16,
            all_rank_num_tokens=None,
            use_dp_padding=None,
            input_ids=None,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "scheduler_kind, accepts_global_routed_output, expected",
    [
        (MoESchedulerKind.FUSED_COMM, True, True),
        (MoESchedulerKind.FUSED_COMM, False, False),
        (MoESchedulerKind.EXTERNAL_COMM, True, False),
        (MoESchedulerKind.EXTERNAL_COMM, False, False),
    ],
)
def test_configurable_moe_reports_global_routed_output_only_when_opted_in(
    scheduler_kind: MoESchedulerKind, accepts_global_routed_output: bool, expected: bool
) -> None:
    """Callers decide whether to skip their routed all-reduce from this
    property, so it must agree with what ``FusedCommMoEScheduler`` returns."""
    moe = ConfigurableMoE.__new__(ConfigurableMoE)
    # The class-level default is what ``FusedCommMoEScheduler`` reads when a
    # caller does not pass the flag.
    assert moe.accepts_global_routed_output is False
    moe.accepts_global_routed_output = accepts_global_routed_output
    moe.backend = SimpleNamespace(scheduler_kind=scheduler_kind)

    assert moe.returns_global_routed_output is expected
    assert MoE.returns_global_routed_output.fget(moe) is False


@pytest.mark.cpu_only
def test_megamoe_cutedsl_accepts_uniform_minimax_m3_swiglu_bias() -> None:
    moe = _make_megamoe_cutedsl_for_ctor_test(
        activation=SwigluBiasActivation(
            gate_sigmoid_scale=torch.full((8,), 1.702),
            linear_offset=torch.ones(8),
            clamp=torch.full((8,), 7.0),
        )
    )

    assert moe.act_alpha == pytest.approx(1.702)
    assert moe.act_beta == pytest.approx(1.0)
    assert moe.act_clamp == pytest.approx(7.0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("parameter_name", "register_name"),
    [
        ("gate_sigmoid_scale", "alpha"),
        ("clamp", "clamp"),
    ],
)
def test_megamoe_cutedsl_rejects_nonuniform_swiglu_bias(
    parameter_name: str, register_name: str
) -> None:
    parameters = {
        "gate_sigmoid_scale": torch.full((8,), 1.702),
        "linear_offset": torch.ones(8),
        "clamp": torch.full((8,), 7.0),
    }
    parameters[parameter_name][-1] += 1.0

    with pytest.raises(ValueError, match=rf"uniform .*activation {register_name}"):
        _make_megamoe_cutedsl_for_ctor_test(
            activation=SwigluBiasActivation(**parameters),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_pre_quant_scale", [False, True], ids=["nvfp4", "nvfp4_awq"])
def test_megamoe_cutedsl_quantize_input_applies_pre_quant_scale(
    monkeypatch: pytest.MonkeyPatch, has_pre_quant_scale: bool
) -> None:
    """An NVFP4_AWQ / NVFP4_ARC pre-quant scale reaches the activation quantizer.

    ``canonical_quant`` folds both recipes into ``nvfp4``, so ``can_implement``
    admits them, and ``load_quant_scales`` loads their per-channel pre-quant
    scale into ``fc31_act_scale``. The fused kernel consumes only the quantized
    activation, which leaves ``quantize_input`` as the one place to apply it.

    ``__new__`` without ``__init__``: the real constructor wants a GPU and a
    symmetric-memory provider, and ``quantize_input`` reads two parameters.
    """
    hidden_size = 64
    backend = TrtllmCutedslMegaMoeNvfp4Impl.__new__(TrtllmCutedslMegaMoeNvfp4Impl)
    torch.nn.Module.__init__(backend)
    backend.fc31_input_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    # Weight creation registers the slot as None, and only a checkpoint that
    # carries ``*.pre_quant_scale`` fills it, as ``NVFP4FusedMoEMethod`` does.
    backend.register_parameter("fc31_act_scale", None)
    if has_pre_quant_scale:
        backend.register_parameter(
            "fc31_act_scale",
            torch.nn.Parameter(
                torch.linspace(0.25, 4.0, hidden_size, dtype=torch.bfloat16).unsqueeze(0),
                requires_grad=False,
            ),
        )
    quantizer_calls = []

    def fake_fp4_quantize(x, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled):
        quantizer_calls.append((x.clone(), global_scale))
        num_rows, num_cols = x.shape
        return (
            torch.zeros(num_rows, num_cols // 2, dtype=torch.uint8),
            torch.zeros(num_rows * num_cols // sf_vec_size, dtype=torch.uint8),
        )

    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", fake_fp4_quantize, raising=False)
    x = torch.randn(3, hidden_size, dtype=torch.bfloat16)

    backend.quantize_input(x)

    [(quantized, global_scale)] = quantizer_calls
    expected = x * backend.fc31_act_scale if has_pre_quant_scale else x
    torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
    assert global_scale is backend.fc31_input_scale


def _make_cutedsl_sibling_for_quantize_input_test(
    backend_cls: type, hidden_size: int, pre_quant_scale: Optional[torch.Tensor]
):
    """A CuteDSL sibling backend carrying only what ``quantize_input`` reads.

    ``__new__`` without ``__init__``: the real constructors want a GPU (and,
    for the FC12 one, Rubin), while ``quantize_input`` reads the quant config,
    the two scales and the scale geometry. The DenseGEMM backend reads the
    same four, so it shares this fake.
    """
    backend = backend_cls.__new__(backend_cls)
    torch.nn.Module.__init__(backend)
    # ``has_nvfp4`` asserts weights exist and reads the layer quant mode;
    # NVFP4_AWQ folds onto the NVFP4 quant mode (quantization/mode.py), so
    # either recipe clears that gate. Declare plain NVFP4 when no scale is
    # given, so that case exercises a non-AWQ layer rather than an AWQ one
    # whose checkpoint happened to carry no ``pre_quant_scale``.
    backend._weights_created = True
    backend.quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4_AWQ if pre_quant_scale is not None else QuantAlgo.NVFP4
    )
    backend.hidden_size = hidden_size
    backend.scaling_vector_size = 16
    backend.fc31_input_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    # ``create_weights`` registers the slot as None, and only a checkpoint
    # that carries ``*.pre_quant_scale`` fills it, as
    # ``NVFP4FusedMoEMethod.load_quant_scales`` does.
    backend.register_parameter("fc31_act_scale", None)
    if pre_quant_scale is not None:
        backend.register_parameter(
            "fc31_act_scale", torch.nn.Parameter(pre_quant_scale, requires_grad=False)
        )
    return backend


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "backend_cls",
    [TrtllmCutedslGroupedGemmNvfp4Impl, CuteDslFc12FusedMoE, DenseGEMMFusedMoE],
    ids=["grouped_gemm", "fused_fc12", "dense_gemm"],
)
@pytest.mark.parametrize("has_pre_quant_scale", [False, True], ids=["nvfp4", "nvfp4_awq"])
def test_cutedsl_quantize_input_applies_pre_quant_scale(
    monkeypatch: pytest.MonkeyPatch, backend_cls: type, has_pre_quant_scale: bool
) -> None:
    """The CuteDSL siblings apply an NVFP4_AWQ / NVFP4_ARC pre-quant scale.

    All three select ``NVFP4CuteDslFusedMoEMethod``, which inherits the CUTLASS
    NVFP4 pipeline and therefore loads a checkpoint's per-channel
    ``pre_quant_scale`` into ``fc31_act_scale``, and all three calibrate
    ``fc31_input_scale`` on the scaled activation. Their kernels consume only
    the quantized activation, which leaves ``quantize_input`` as the one place
    to apply it; before this it was dropped and the output silently wrong.
    """
    hidden_size = 64
    pre_quant_scale = (
        torch.linspace(0.25, 4.0, hidden_size, dtype=torch.bfloat16).unsqueeze(0)
        if has_pre_quant_scale
        else None
    )
    backend = _make_cutedsl_sibling_for_quantize_input_test(
        backend_cls, hidden_size, pre_quant_scale
    )
    quantizer_calls = []

    def fake_fp4_quantize(x, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled):
        quantizer_calls.append((x.clone(), global_scale))
        num_rows, num_cols = x.shape
        return (
            torch.zeros(num_rows, num_cols // 2, dtype=torch.uint8),
            torch.zeros(num_rows * num_cols // sf_vec_size, dtype=torch.uint8),
        )

    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", fake_fp4_quantize, raising=False)
    x = torch.randn(3, hidden_size, dtype=torch.bfloat16)

    backend.quantize_input(x, post_quant_comm=False)

    [(quantized, global_scale)] = quantizer_calls
    expected = x * pre_quant_scale if has_pre_quant_scale else x
    torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
    assert global_scale is backend.fc31_input_scale


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "backend_cls",
    [TrtllmCutedslGroupedGemmNvfp4Impl, CuteDslFc12FusedMoE, DenseGEMMFusedMoE],
    ids=["grouped_gemm", "fused_fc12", "dense_gemm"],
)
def test_cutedsl_quantize_input_rejects_prequantized_awq_input(backend_cls: type) -> None:
    """An already-quantized input can no longer take the pre-quant scale.

    Each backend also accepts an ``Fp4QuantizedTensor`` produced upstream (a
    fused norm+quant, or a post-quant dispatch) that never saw
    ``fc31_act_scale``. Refuse that combination rather than serve the
    checkpoint wrong, as ``CutlassFusedMoE.quantize_input`` does.
    """
    hidden_size = 64
    backend = _make_cutedsl_sibling_for_quantize_input_test(
        backend_cls,
        hidden_size,
        torch.linspace(0.25, 4.0, hidden_size, dtype=torch.bfloat16).unsqueeze(0),
    )
    x = Fp4QuantizedTensor(
        fp4_tensor=torch.zeros(3, hidden_size // 2, dtype=torch.uint8),
        scaling_factor=torch.zeros(3 * hidden_size // 16, dtype=torch.uint8),
        is_sf_swizzled=False,
    )

    with pytest.raises(AssertionError, match="not expected for AWQ"):
        backend.quantize_input(x, post_quant_comm=False)


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_pre_quant_scale", [False, True], ids=["nvfp4", "nvfp4_awq"])
def test_w4a16_nvfp4_quantize_input_applies_pre_quant_scale(has_pre_quant_scale: bool) -> None:
    """The CUTLASS W4A16_NVFP4 leaf applies the pre-quant scale too.

    This leaf keeps activations in high precision -- ``run_moe`` dequantizes the
    NVFP4 weights and runs the unquantized kernel -- so unlike the ``nvfp4``
    leaf it does no FP4 activation quantization. A pre-quant scale is not part
    of that quantization, though: ``NVFP4FusedMoEMethod.load_quant_scales``
    fills ``fc31_act_scale`` from the checkpoint's ``*.pre_quant_scale`` keys
    without consulting ``quant_algo``, and the dequantized ``W / s`` fed into
    the BF16 MoE expect the scaled activation. The dense counterpart,
    ``W4A16NVFP4LinearMethod``, applies ``pre_quant_scale`` the same way.
    """
    hidden_size = 64
    moe = TrtllmCutlassW4a16Nvfp4Impl.__new__(TrtllmCutlassW4a16Nvfp4Impl)
    torch.nn.Module.__init__(moe)
    # ``create_weights`` registers the slot as None, and only a checkpoint
    # that carries ``*.pre_quant_scale`` fills it.
    moe.register_parameter("fc31_act_scale", None)
    pre_quant_scale = None
    if has_pre_quant_scale:
        pre_quant_scale = torch.linspace(0.25, 4.0, hidden_size, dtype=torch.bfloat16).unsqueeze(0)
        moe.register_parameter(
            "fc31_act_scale", torch.nn.Parameter(pre_quant_scale, requires_grad=False)
        )
    x = torch.randn(3, hidden_size, dtype=torch.bfloat16)

    out, out_sf = moe.quantize_input(x)

    assert out_sf is None
    expected = x * pre_quant_scale if has_pre_quant_scale else x
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# NVFP4_AWQ on the per-expert STREAMING loader.
#
# Kimi K3 never calls the MoE backend's whole-checkpoint ``load_weights``, so
# ``load_quant_scales`` -- the only thing that read ``*.pre_quant_scale`` --
# never runs for it. The scale therefore has to travel through
# ``load_streaming_nvfp4_expert``, or an AWQ checkpoint served over that path
# gets ``fc31_act_scale is None`` and the fix above does nothing for it.
# Nothing raises in that state: the loads are silent and only accuracy moves.
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_kimi_k3_nvfp4_expert_spec_covers_the_awq_recipes() -> None:
    """The streaming layout serves NVFP4 and both of its calibration recipes.

    ``_k3_expert_ckpt_spec`` used to know only plain ``NVFP4``, so an
    NVFP4_AWQ / NVFP4_ARC checkpoint could not reach this loader at all. The
    weights are byte-identical across the three; only the extra per-channel
    scale differs, so one layout covers them.

    ``pre_quant_scale`` must stay OPTIONAL: ``nvidia/Kimi-K3-NVFP4`` stores
    none, and requiring it would reject the one checkpoint on this path today.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import _k3_expert_ckpt_spec

    nvfp4 = _k3_expert_ckpt_spec(QuantAlgo.NVFP4)
    for quant_algo in (QuantAlgo.NVFP4_AWQ, QuantAlgo.NVFP4_ARC):
        assert _k3_expert_ckpt_spec(quant_algo) is nvfp4, quant_algo
    assert "pre_quant_scale" not in nvfp4.kinds
    assert nvfp4.optional_suffixes == ("w1.pre_quant_scale", "w3.pre_quant_scale")


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_pre_quant_scale", [False, True], ids=["nvfp4", "nvfp4_awq"])
def test_kimi_k3_nvfp4_expert_loader_forwards_pre_quant_scale(
    has_pre_quant_scale: bool,
) -> None:
    """The per-expert loader passes a stored pre-quant scale on, and only then.

    The two getters are the seam every streaming route shares (file-grouped,
    split-file and the shared-dict fallback): the required one raises on a
    missing key, the optional one answers ``None``. A plain NVFP4 checkpoint
    must still load, which is why ``None`` has to reach the quant method
    rather than the key being demanded.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import _load_nvfp4_expert

    pre_quant_scale = torch.linspace(0.5, 2.0, 8)
    stored = (
        {
            "experts.5.w1.pre_quant_scale": pre_quant_scale,
            "experts.5.w3.pre_quant_scale": pre_quant_scale,
        }
        if has_pre_quant_scale
        else {}
    )
    captured = {}
    backend = SimpleNamespace(
        quant_method=SimpleNamespace(
            load_streaming_nvfp4_expert=lambda module, **kwargs: captured.update(
                module=module, **kwargs
            )
        )
    )

    _load_nvfp4_expert(backend, "experts", 5, 2, lambda key: key, stored.get)

    assert captured["module"] is backend
    assert captured["global_expert_id"] == 5
    assert captured["local_slot_id"] == 2
    # Required tensors still come from the required getter, unchanged.
    assert captured["w1_weight"] == "experts.5.w1.weight"
    assert captured["w2_input_scale"] == "experts.5.w2.input_scale"
    for w in ("w1", "w3"):
        got = captured[f"{w}_pre_quant_scale"]
        if has_pre_quant_scale:
            assert got is pre_quant_scale, w
        else:
            assert got is None, w


def _streaming_module_for_pre_quant_scale_test(hidden_size: int) -> torch.nn.Module:
    """A module carrying only what the pre-quant-scale staging touches."""
    module = torch.nn.Module()
    module.dtype = torch.bfloat16
    module.hidden_size = hidden_size
    module.tp_size = 1
    module.tp_rank = 0
    module.is_gated_activation = True
    # ``create_weights`` registers the slot as None; on the whole-checkpoint
    # path ``load_quant_scales`` replaces it the moment it sees the keys.
    module.register_parameter("fc31_act_scale", None)
    return module


@pytest.mark.cpu_only
def test_nvfp4_streaming_stages_pre_quant_scales_for_finalization() -> None:
    """A streamed AWQ expert's scale reaches ``fc31_act_scale``.

    Two halves, asserted together because either alone passes trivially: the
    staging container has to exist before the (multi-threaded) loads, and
    ``_finalize_pre_quant_scales`` has to consume it and allocate the
    parameter, which on this path no ``load_quant_scales`` has created.

    The real allocation is ``device='cuda'``, so it is substituted here; what
    is asserted is that the finalizer allocates at all and lands
    ``max(w3, w1)``, which is the reduction the whole-checkpoint path uses.
    """
    hidden_size = 8
    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    module = _streaming_module_for_pre_quant_scale_test(hidden_size)

    # The base implementation, not MegaMoE's override: that one additionally
    # rematerializes the raw source parameters, which needs real weights.
    NVFP4FusedMoEMethod.prepare_streaming_expert_load(method, module)
    assert module.tmp_pre_quant_scales == {}, "staging must predate the loader threads"

    w1 = torch.linspace(0.5, 2.0, hidden_size)
    w3 = torch.linspace(2.0, 0.5, hidden_size)
    for global_expert_id in (3, 7):
        module.tmp_pre_quant_scales[global_expert_id] = {"w1": w1.clone(), "w3": w3.clone()}

    def allocate_on_cpu(mod: torch.nn.Module) -> None:
        mod.register_parameter(
            "fc31_act_scale",
            torch.nn.Parameter(
                torch.empty(1, mod.hidden_size, dtype=mod.dtype), requires_grad=False
            ),
        )

    method._ensure_fc31_act_scale = allocate_on_cpu
    method._finalize_pre_quant_scales(module)

    assert not hasattr(module, "tmp_pre_quant_scales")
    torch.testing.assert_close(
        module.fc31_act_scale.data,
        torch.max(w1, w3).to(torch.bfloat16).unsqueeze(0),
        rtol=0,
        atol=0,
    )


@pytest.mark.cpu_only
def test_nvfp4_streaming_plain_checkpoint_leaves_act_scale_unset() -> None:
    """A plain NVFP4 streamed load must stay exactly as it was.

    The container is now pre-created for every streamed NVFP4 load, AWQ or
    not, so an empty one means "this checkpoint has no pre-quant scale" and
    not "the scales went missing". Without this the ``nvidia/Kimi-K3-NVFP4``
    load would raise ``Missing pre_quant_scale data``.
    """
    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    module = _streaming_module_for_pre_quant_scale_test(8)

    # Base implementation, for the reason given in the test above.
    NVFP4FusedMoEMethod.prepare_streaming_expert_load(method, module)
    method._finalize_pre_quant_scales(module)

    assert module.fc31_act_scale is None
    assert not hasattr(module, "tmp_pre_quant_scales")


def _streaming_expert_kwargs() -> dict:
    """The required ``load_streaming_nvfp4_expert`` arguments, as packed bytes.

    Only the three weights are inspected before the pre-quant-scale checks, so
    the scales can be placeholders: the call under test never reaches them.
    """
    packed = torch.zeros(2, 2, dtype=torch.uint8)
    return {
        f"{w}_{kind}": packed if kind == "weight" else None
        for w in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("supplied", ["w1", "w3"])
def test_nvfp4_streaming_rejects_half_a_pre_quant_scale_pair(supplied: str) -> None:
    """One half of the pair cannot produce ``max(w3, w1)``.

    Accepting it would stage an entry the finalizer then rejects with a
    message about a missing expert, one layer deep and pointing at the wrong
    thing. Refused at the argument instead.
    """
    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    module = SimpleNamespace(
        expert_size_per_partition=1,
        initial_local_expert_ids=[0],
        is_gated_activation=True,
        _streamed_expert_slots=set(),
    )

    with pytest.raises(ValueError, match="must be loaded together"):
        method.load_streaming_nvfp4_expert(
            module,
            global_expert_id=0,
            local_slot_id=0,
            **_streaming_expert_kwargs(),
            **{f"{supplied}_pre_quant_scale": torch.ones(4)},
        )


@pytest.mark.cpu_only
def test_nvfp4_streaming_refuses_pre_quant_scale_without_gating() -> None:
    """``fc31_act_scale`` is a gated-activation concept, as on the other path.

    ``load_quant_scales`` asserts the same thing: without a gate there is no
    ``max(w3, w1)`` to take, so a checkpoint in that shape is refused rather
    than loaded with the scale dropped.
    """
    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)
    module = SimpleNamespace(
        expert_size_per_partition=1,
        initial_local_expert_ids=[0],
        is_gated_activation=False,
        _streamed_expert_slots=set(),
    )

    with pytest.raises(ValueError, match="non-gated activations"):
        method.load_streaming_nvfp4_expert(
            module,
            global_expert_id=0,
            local_slot_id=0,
            **_streaming_expert_kwargs(),
            w1_pre_quant_scale=torch.ones(4),
            w3_pre_quant_scale=torch.ones(4),
        )


def test_megamoe_cutedsl_tuning_mode_forces_top_maxt_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Profiling scratch is sized for the largest adaptive bucket.
    monkeypatch.setenv("MEGAMOE_TACTIC_AUTOTUNE", "1")
    moe = _make_megamoe_cutedsl_for_ctor_test()
    buckets = moe._maxt_buckets
    assert len(buckets) >= 2, f"expected a multi-bucket ladder, got {buckets}"
    small_hint = buckets[0]
    assert moe._select_launch_max_tokens(small_hint) == buckets[0]
    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
    assert moe._select_launch_max_tokens(small_hint) == buckets[-1]


def test_megamoe_cutedsl_tactic_autotune_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Standard serving must not pay for the tactic sweep by default.
    monkeypatch.delenv("MEGAMOE_TACTIC_AUTOTUNE", raising=False)
    moe = _make_megamoe_cutedsl_for_ctor_test()
    assert moe.tactic_autotune is False


@pytest.mark.parametrize(
    "sm_version,decode_count,prefill_count", [(100, 36, 40), (103, 36, 40), (107, 11, 13)]
)
def test_enumerate_megamoe_candidate_tactics_curated_space(
    sm_version: int, decode_count: int, prefill_count: int
) -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    decode = megamoe_op.enumerate_megamoe_candidate_tactics(1024, sm_version=sm_version)
    prefill = megamoe_op.enumerate_megamoe_candidate_tactics(16384, sm_version=sm_version)
    assert (len(decode), len(prefill)) == (decode_count, prefill_count)
    assert all(len(t) == 11 for t in decode + prefill)
    assert {t[-2] for t in decode} == {(1, 1)}
    assert {t[-2] for t in prefill} == {(2, 4)}
    assert {t[-1] for t in decode + prefill} == {"expert"}
    for tactic in decode + prefill:
        megamoe_op.validate_megamoe_tactic(tactic, sm_version=sm_version)
    default_tactics = [
        megamoe_op.default_megamoe_tactic(num_tokens) for num_tokens in (64, 4096, 16384)
    ]
    assert all(len(tactic) == 11 for tactic in default_tactics)
    assert {tactic[-1] for tactic in default_tactics} == {"expert"}
    for tactic in default_tactics:
        megamoe_op.validate_megamoe_tactic(tactic, sm_version=sm_version)
    if sm_version == 107:
        for bucket, tactic in megamoe_op._SM107_GENPHASE_TACTICS.items():
            megamoe_op.validate_megamoe_tactic(tactic, sm_version=sm_version)
            selected = megamoe_op._default_megamoe_tactic_for_problem(
                sm_version=sm_version,
                max_tokens_per_rank=bucket,
                num_tokens=bucket,
                apply_topk_in_fc1=False,
                in_kernel_fc2_reduce=False,
                combine_format="bf16",
            )
            assert len(selected) == 11
            assert selected == megamoe_op._unpack_tactic(tactic)
    invalid_tactic = list(megamoe_op.default_megamoe_tactic(64))
    invalid_tactic[3] = ("grouped", 0)
    with pytest.raises(ValueError, match=r"schedule_policy hint must be a positive int or None"):
        megamoe_op.validate_megamoe_tactic(tuple(invalid_tactic), sm_version=sm_version)
    legacy = ([256, 128, 256], [2, 1, 1], 512, "static", "epi_warps", True, 1, (1, 1))
    megamoe_op.validate_megamoe_tactic(legacy, sm_version=sm_version)
    assert megamoe_op._unpack_tactic(legacy) == megamoe_op.default_megamoe_tactic(64)
    v4 = megamoe_op.default_megamoe_tactic(64)[:-1]
    megamoe_op.validate_megamoe_tactic(v4, sm_version=sm_version)
    assert megamoe_op._unpack_tactic(v4) == megamoe_op.default_megamoe_tactic(64)


@pytest.mark.gpu
def test_megamoe_cutedsl_runner_cache_identity_includes_swiglu_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

    if not megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
        pytest.skip(str(megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON))
    monkeypatch.setattr(megamoe_op, "get_sm_version", lambda: 100)
    runner_kwargs = {
        "world_size": 1,
        "local_rank": 0,
        "num_topk": 2,
        "num_experts_per_rank": 8,
        "hidden_size": 512,
        "intermediate_size_per_partition": 512,
        "expand_intermediate_size_per_partition": 1024,
        "max_tokens_per_rank": 64,
        "output_dtype": torch.bfloat16,
        "gate_up_clamp": 7.0,
    }
    runner = megamoe_op.Sm100MegaMoENvfp4Runner(
        **runner_kwargs,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    different_alpha = megamoe_op.Sm100MegaMoENvfp4Runner(
        **runner_kwargs,
        swiglu_alpha=1.0,
        swiglu_beta=1.0,
    )
    different_beta = megamoe_op.Sm100MegaMoENvfp4Runner(
        **runner_kwargs,
        swiglu_alpha=1.702,
        swiglu_beta=0.0,
    )
    tactic = megamoe_op.default_megamoe_tactic(64)

    assert runner.unique_id() != different_alpha.unique_id()
    assert runner.unique_id() != different_beta.unique_id()
    assert runner._tactic_cache_key(tactic) != different_alpha._tactic_cache_key(tactic)
    assert runner._tactic_cache_key(tactic) != different_beta._tactic_cache_key(tactic)


@pytest.mark.gpu
def test_megamoe_cutedsl_minimax_m3_swiglu_bias_numerics() -> None:
    """Run bias-free MiniMax SwiGLU through the B200 kernel and its reference."""
    backend_type = MoeBackendType.MEGAMOE_CUTEDSL
    dtype = torch.bfloat16
    num_experts = 8
    top_k = 2
    hidden_size = 512
    intermediate_size = 512
    seq_len = 16
    swiglu_alpha = 1.702
    swiglu_beta = 1.0
    swiglu_limit = 7.0

    verdict = MegaMoECuteDsl.can_implement(
        _make_megamoe_cutedsl_minimax_problem(
            bias=False,
            activation=ActivationType.SwigluBias,
        ),
        MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=num_experts,
            env=collect_moe_environment(),
        ),
    )
    if not verdict.eligible:
        pytest.skip(verdict.detail)

    mapping = Mapping(rank=mpi_rank())
    _ensure_single_proc_dist_for_megamoe(backend_type, mapping.rank)
    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        # The larger input range makes the dequantized FC1 gate/up values cross
        # MiniMax's clamp boundary in both directions.
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda") * 8.0
        router_logits = torch.full((seq_len, num_experts), -100.0, dtype=dtype, device="cuda")
        router_logits[:, 0] = 100.0
        router_logits[:, 1] = 90.0
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, x, backend_type
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=True,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_type=ActivationType.SwigluBias,
        )
        swiglu_tensors = quantize_util.get_swiglu_tensors()
        assert swiglu_tensors is not None
        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            bias=False,
            swiglu_alpha=swiglu_tensors["swiglu_alpha"],
            swiglu_beta=swiglu_tensors["swiglu_beta"],
            swiglu_limit=swiglu_tensors["swiglu_limit"],
            activation_type=ActivationType.SwigluBias,
        )
        ref_cls = quant_kwargs.pop("ref_cls")
        weights = quantize_util.create_weights(**quant_kwargs)
        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()
        ref_fused_moe = quantize_util.create_ref_module(
            routing_method,
            ref_cls=ref_cls,
        )
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            gate_up = ref_fused_moe.experts[0].gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            assert torch.any(gate > swiglu_limit)
            assert torch.any(gate < swiglu_limit)
            assert torch.any(up < -swiglu_limit)
            assert torch.any((up < 0) & (up > -swiglu_limit))
            ref_output = ref_fused_moe(x, router_logits)
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            output = run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype,
                router_logits,
            )

        # This clamp-heavy input is intentionally wider than the generic
        # MegaMoE sweep and needs one percentage point of NVFP4 headroom.
        check_accuracy(output, ref_output, rtol=0.1, atol=0.1, percent=0.94)


@pytest.mark.gpu
def test_megamoe_cutedsl_nvfp4_awq_matches_cutlass() -> None:
    """MegaMoE and CUTLASS agree on an NVFP4_AWQ layer, and both match the reference.

    The checkpoint carries the per-channel ``w1`` / ``w3`` ``pre_quant_scale``
    an AWQ calibration writes, and its input scale is calibrated on
    ``x * pre_quant_scale``, which is what the reference runs FC1 on. The last
    check is the control: MegaMoE against the reference on the unscaled input
    must fail, or this test could not tell a dropped scale apart.
    """
    dtype = torch.bfloat16
    num_experts = 8
    top_k = 2
    hidden_size = 512
    intermediate_size = 512
    seq_len = 16

    verdict = MegaMoECuteDsl.can_implement(
        MoEProblem(
            quant=QuantAlgo.NVFP4.value,
            dtype_act=dtype,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        ),
        MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=num_experts,
            env=collect_moe_environment(),
        ),
    )
    if not verdict.eligible:
        pytest.skip(verdict.detail)

    mapping = Mapping(rank=mpi_rank())
    _ensure_single_proc_dist_for_megamoe(MoeBackendType.MEGAMOE_CUTEDSL, mapping.rank)
    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")
        # One vector for every expert, as AWQ writes it, with per-channel
        # factors in [1/e, e]: dropping it moves FC1's input far past tolerance.
        pre_quant_scale = torch.empty(hidden_size, device="cuda").uniform_(-1, 1).exp().to(dtype)
        x_smoothed = x * pre_quant_scale
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, x_smoothed, MoeBackendType.MEGAMOE_CUTEDSL
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
        )
        ref_cls = quant_kwargs.pop("ref_cls")
        weights = quantize_util.create_weights(**quant_kwargs)
        for expert_id in range(num_experts):
            weights[f"{expert_id}.w1.pre_quant_scale"] = pre_quant_scale
            weights[f"{expert_id}.w3.pre_quant_scale"] = pre_quant_scale
        token_selected_experts, token_final_scales = routing_method.apply(router_logits)

        outputs = {}
        for backend_type in (MoeBackendType.CUTLASS, MoeBackendType.MEGAMOE_CUTEDSL):
            backend = create_test_backend(
                backend_type=backend_type,
                routing_method=routing_method,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4_AWQ),
                mapping=mapping,
            )
            backend.load_weights([weights])
            backend.post_load_weights()
            backend.cuda()
            assert backend.fc31_act_scale is not None
            with torch.inference_mode():
                x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
                outputs[backend_type] = run_backend_moe(
                    backend,
                    backend_type,
                    x_quantized,
                    x_sf,
                    token_selected_experts,
                    token_final_scales,
                    dtype,
                    router_logits,
                )

        ref_fused_moe = quantize_util.create_ref_module(routing_method, ref_cls=ref_cls)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()
        with torch.inference_mode():
            ref_output = ref_fused_moe(x_smoothed, router_logits)
            unscaled_ref_output = ref_fused_moe(x, router_logits)

    cutlass_output = outputs[MoeBackendType.CUTLASS]
    megamoe_output = outputs[MoeBackendType.MEGAMOE_CUTEDSL]
    ref_fused_moe.check_accuracy(cutlass_output, ref_output)
    ref_fused_moe.check_accuracy(megamoe_output, ref_output)
    ref_fused_moe.check_accuracy(megamoe_output, cutlass_output)
    with pytest.raises(Exception, match="Mismatch percentage"):
        ref_fused_moe.check_accuracy(megamoe_output, unscaled_ref_output)


@pytest.fixture(scope="module")
def _run_megamoe_prologue_helper() -> Callable[[int, bool], torch.Tensor]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("The scheduler helper regression requires an SM107 CUDA GPU")

    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("cutlass-dsl is not available")

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import Boolean, Int32
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import ImplDesc, ProblemDesc
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.device_workspace import (
        DeviceWorkspace,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.smem_workspace import SmemWorkspace
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.schedulers.fc12_mapping import (
        BlockPhase,
        map_phase_interleaved_fc12_work_id,
        resolve_phase_interleaved_fc1_claim_target,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.schedulers.fc12_scheduler import (
        PhaseInterleavedFc12Scheduler,
    )

    class PrologueProbe:
        def __init__(self) -> None:
            self.scheduler = PhaseInterleavedFc12Scheduler(
                ProblemDesc(dict(expert_count=1, intermediate_gateup_size=128, hidden_size=128)),
                ImplDesc(
                    dict(
                        num_scheduler_consumer_threads=32,
                        mma_tiler_mnk=(128, 128, 64),
                        cluster_shape_mn=(1, 1),
                        use_2cta_instrs=False,
                        hint=1,
                        token_padding_block=128,
                        sf_padding_block=128,
                        work_id_mode="atomic_counter",
                        is_swap_ab=False,
                        launch_cluster_count=3,
                    )
                ),
            )
            assert self.scheduler.minimum_global_fc1_claims == 3
            self.smem = SmemWorkspace()
            self.device = DeviceWorkspace()
            self.scheduler.register_smem_regions(self.smem)
            self.scheduler.register_device_workspace(self.device)
            self.smem.finalize()
            self.device.finalize()
            alignment = self.smem.base_alignment
            self.smem_launch_bytes = (
                (self.smem.total_bytes + alignment - 1) // alignment * alignment
            )
            assert self.smem_launch_bytes >= self.smem.storage_class().size_in_bytes()

        @cute.jit
        def __call__(
            self,
            counts: cute.Tensor,
            workspace: cute.Tensor,
            exhausted: cute.Tensor,
            output: cute.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            self.kernel(counts, workspace, exhausted, output).launch(
                grid=(1, 1, 1), block=(32, 1, 1), smem=self.smem_launch_bytes, stream=stream
            )

        @cute.kernel
        def kernel(
            self,
            counts: cute.Tensor,
            workspace: cute.Tensor,
            exhausted: cute.Tensor,
            output: cute.Tensor,
        ) -> None:
            storage = utils.SmemAllocator().allocate(self.smem.storage_class())
            self.device.assign_device_members(workspace.iterator)
            self.scheduler.assign_device_members(
                expert_token_sizes=counts,
                expert_token_prefix_sum=None,
                actual_expert_shape=None,
                block_idx=cute.arch.block_idx(),
                smem_workspace=self.smem,
                smem_base=storage.buffer.data_ptr(),
                device_workspace=self.device,
            )
            pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=(1, 1))
            worker = self.scheduler._work_id_worker
            state = self.scheduler._task_mapping_state
            _, fc1_valid, state = map_phase_interleaved_fc12_work_id(
                Int32(0), Int32(BlockPhase.Linear1), (Int32(0), Int32(0), Int32(0)), state
            )
            _, fc2_valid, state = map_phase_interleaved_fc12_work_id(
                Int32(0), Int32(BlockPhase.Linear2), (Int32(0), Int32(0), Int32(0)), state
            )
            target, short_stream = resolve_phase_interleaved_fc1_claim_target(
                Int32(self.scheduler.minimum_global_fc1_claims), state
            )
            input_exhausted = Boolean(exhausted[0])
            pointer = worker.get_atomic_counter_pointer(Int32(0))
            result = self.scheduler._wait_for_fc1_prologue_claims(input_exhausted, worker, state)
            tid, _, _ = cute.arch.thread_idx()
            output[tid, 0] = Int32(result)
            output[tid, 1] = cute.arch.load(pointer, Int32, sem="acquire", scope="gpu")
            output[tid, 2] = target
            output[tid, 3] = Int32(short_stream)
            output[tid, 4] = Int32(fc1_valid)
            output[tid, 5] = Int32(fc2_valid)

    probe = PrologueProbe()
    counts = torch.tensor([1], device="cuda", dtype=torch.int32)
    workspace = torch.zeros(probe.device.total_bytes("local"), device="cuda", dtype=torch.uint8)
    exhausted = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.full((32, 6), -1, device="cuda", dtype=torch.int32)
    tensors = tuple(
        from_dlpack(t, assumed_align=16) for t in (counts, workspace, exhausted, output)
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(probe, *tensors, stream)
    counter_index = probe.device.offset(probe.scheduler._work_id_worker.work_id_counter_region) // 4

    def run(counter: int, input_exhausted: bool) -> torch.Tensor:
        workspace.zero_()
        workspace.view(torch.int32)[counter_index] = counter
        workspace.view(torch.int32)[counter_index + 1] = 1
        exhausted.fill_(int(input_exhausted))
        output.fill_(-1)
        compiled(*tensors, stream)
        torch.cuda.synchronize()
        return output.cpu()

    return run


@pytest.mark.parametrize(
    "counter,input_exhausted",
    [(1, False), (3, False), (2, True)],
    ids=["slow-peek", "fast-peek", "already-exhausted"],
)
def test_fc1_prologue_wait_preserves_exhaustion(
    _run_megamoe_prologue_helper: Callable[[int, bool], torch.Tensor],
    counter: int,
    input_exhausted: bool,
) -> None:
    # One observer warp represents a three-cluster snapshot with one real FC1
    # tile. Counter 1 precedes peer arrivals; counter 3 includes two invalid peer
    # claims. Already-exhausted controls include an invalid local claim.
    actual = _run_megamoe_prologue_helper(counter, input_exhausted)
    expected = torch.tensor(
        [int(input_exhausted), counter, 1, 1, 1, 1], dtype=torch.int32
    ).expand_as(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def run_backend_moe(
    backend: MoE,
    backend_type: MoeBackendType,
    x_quantized: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    dtype: torch.dtype,
    router_logits: torch.Tensor = None,
    trtllm_use_router_logits: bool = True,
) -> torch.Tensor:
    """
    Run MoE computation with backend-specific parameters.

    Each backend has different requirements:
    - CUTLASS: output_dtype, token_final_scales=float32
    - TRTLLM: token_final_scales=bfloat16, optionally router_logits
    - CUTEDSL: token_final_scales=float32
    - DEEPGEMM: workspace, token_final_scales=float32
    - MegaMoE backends: token_selected_experts=int64, output_dtype

    Args:
        trtllm_use_router_logits: If True, TRTLLM backend uses router_logits for routing,
            unless the backend routes outside its kernel. Otherwise it uses
            token_selected_experts and token_final_scales.
            Note: When both are provided, TRTLLM only uses (topk_ids and topk_weights).
    """
    # Common args for all backends (default: token_final_scales=float32)
    args = dict(
        x=x_quantized,
        token_selected_experts=token_selected_experts.to(torch.int32),
        token_final_scales=token_final_scales.to(torch.float32),
        x_sf=x_sf,
    )
    workspace = None

    # Backend-specific overrides
    if backend_type == MoeBackendType.CUTLASS:
        args["output_dtype"] = dtype
    elif backend_type == MoeBackendType.TRTLLM:
        args["token_final_scales"] = token_final_scales.to(torch.bfloat16)
        # A leaf whose kernel cannot route (e.g. FlashInfer BF16) takes top-k
        # from the host, as the scheduler supplies it in production.
        if trtllm_use_router_logits and not backend._routes_outside_the_kernel():
            # Use router_logits for routing (TRTLLM will compute topk internally)
            args["router_logits"] = router_logits
            args["token_selected_experts"] = None
            args["token_final_scales"] = None
        # else: use token_selected_experts and token_final_scales (already set)
    elif backend_type == MoeBackendType.DEEPGEMM:
        import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils

        m_max = fp8_utils.align(x_quantized.shape[0], 128)
        workspace = backend.get_workspace(m_max, 128)
    elif backend_type in _MEGAMOE_BACKEND_TYPES:
        args["token_selected_experts"] = token_selected_experts.to(torch.int64)
        args["output_dtype"] = dtype

    # Mirror what each scheduler hands the backend: ExternalCommMoEScheduler
    # builds a plan on every path, FusedCommMoEScheduler never does because the
    # fused kernel owns the exchange. Single GPU with no comm strategy, so this
    # is the no-comm plan: quantize_input ran locally and left the scale factors
    # swizzled, no alltoall, and no workspace-backed output buffer.
    if backend.scheduler_kind == MoESchedulerKind.EXTERNAL_COMM:
        args["comm_plan"] = MoECommPlan(
            input_sf_swizzled=True,
            enable_alltoall=False,
            moe_output=None,
            payload_in_workspace=False,
        )

    return backend.run_moe(MoERunContext(**args), workspace=workspace)


# =====================================================================
# Test Parameters
# =====================================================================

# Quantization algorithms to test
QUANT_ALGOS_TO_TEST = [
    None,  # Unquantized
    QuantAlgo.FP8,
    QuantAlgo.NVFP4,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.W4A8_NVFP4_FP8,
    QuantAlgo.W4A16_MXFP4,
    QuantAlgo.W4A8_MXFP4_FP8,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.MXFP8,
    QuantAlgo.W8A16,
    QuantAlgo.W4A8_AWQ,
]

# Backend types to test
BACKEND_TYPES_TO_TEST = [
    MoeBackendType.CUTLASS,
    MoeBackendType.TRTLLM,
    MoeBackendType.CUTEDSL,
    MoeBackendType.DEEPGEMM,
    MoeBackendType.DENSEGEMM,
    MoeBackendType.MEGAMOE_DEEPGEMM,
    MoeBackendType.MEGAMOE_CUTEDSL,
    MoeBackendType.CUTE_DSL_B12X,
    MoeBackendType.MARLIN,
    MoeBackendType.CUTEDSL_FC12,
]

# Data types to test
DTYPES_TO_TEST = [
    torch.float16,
    torch.bfloat16,
]

# Format: (num_experts, top_k, hidden_size, intermediate_size)
#
# Default runs the CI subset (TRTLLM_TEST_MOE_CI=1).
# Set TRTLLM_TEST_MOE_CI=0 for the full local config matrix.
CI_MOE_MODEL_CONFIGS = [
    # Real models (small/medium — tactic replay is model-size-independent,
    # e256 is covered by test_moe_module integration tests)
    MoeModelConfig(60, 4, 2048, 1408),  # retained 60-expert backend shape
    MoeModelConfig(128, 4, 2880, 2880),  # GPT-OSS-120B
    MoeModelConfig(8, 1, 512, 512),  # boundary: top_k=1, single expert activated
    # Boundary tests for tactic correctness
    MoeModelConfig(4, 4, 512, 512),  # top_k=num_experts, all experts activated
    MoeModelConfig(7, 2, 256, 512),  # prime num_experts
    MoeModelConfig(13, 3, 256, 512),  # prime num_experts, odd top_k
]

LOCAL_MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS + [
    MoeModelConfig(256, 8, 7168, 2048),  # DeepSeek-V3
    MoeModelConfig(256, 6, 4096, 2048),  # DeepSeek-V4-Flash
    MoeModelConfig(384, 6, 7168, 3072),  # DeepSeek-V4-Pro
    MoeModelConfig(8, 2, 4096, 14336),  # Mixtral-8x7B
    MoeModelConfig(64, 6, 2048, 1408),  # DeepSeek-MoE-16B / DeepSeek-V2-Lite
    MoeModelConfig(8, 2, 6144, 32768),  # Grok-1
    # === Boundary Tests: small sizes ===
    MoeModelConfig(4, 2, 64, 128),  # very small hidden_size
    MoeModelConfig(4, 2, 128, 64),  # intermediate < hidden
]

MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS if IS_CI_MODE else LOCAL_MOE_MODEL_CONFIGS

# Sequence lengths to test
SEQ_LENS_TO_TEST = [1, 8]

# SwiGLU parameters for swiglu_gptoss_style testing
SWIGLU_ALPHAS = [1, 1.702]  # default, GPT-OSS (modeling_gpt_oss.py)
SWIGLU_BETAS = [0, 1.0]  # default, GPT-OSS
SWIGLU_LIMITS = [float("inf"), 7.0]  # default, GPT-OSS

# Full product of all SwiGLU combos (local exhaustive testing only)
LOCAL_SWIGLU_COMBOS = list(itertools.product(SWIGLU_ALPHAS, SWIGLU_BETAS, SWIGLU_LIMITS))

# CI: only non-gptoss (default) and one gptoss combo
# All non-default combos trigger the same swiglu_gptoss_style=True code path;
# different alpha/beta/limit values are just kernel parameters, not code branches.
CI_SWIGLU_COMBOS = [
    (1, 0, float("inf")),  # non-gptoss (default SwiGLU)
    (1.702, 1.0, 7.0),  # gptoss style (GPT-OSS real values)
]

SWIGLU_COMBOS = CI_SWIGLU_COMBOS if IS_CI_MODE else LOCAL_SWIGLU_COMBOS


def should_skip_locality_domain_param(
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    activation_type: ActivationType,
    swiglu_gptoss_style: bool,
    dtype: torch.dtype,
) -> Optional[str]:
    """Return a static skip reason for locality domain MoE backend params."""
    if backend_type != MoeBackendType.CUTEDSL:
        return "locality domain MoE backend test only supports CuteDSL"
    if quant_algo not in (QuantAlgo.NVFP4, None):
        return "locality domain MoE backend test only supports NVFP4 or BF16"
    # plan_moe only enables the unquantized path for bfloat16 activations.
    if quant_algo is None and dtype != torch.bfloat16:
        return "unquantized locality domain MoE requires bfloat16"
    if activation_type != ActivationType.Swiglu:
        return "locality domain MoE backend test only supports SwiGLU"
    if swiglu_gptoss_style:
        return "locality domain MoE backend test does not cover GPT-OSS SwiGLU style"
    return None


def should_skip_locality_domain_runtime(enable_locality_domains: bool) -> Optional[str]:
    """Return a runtime skip reason for locality domain MoE backend params."""
    if not enable_locality_domains:
        return None
    if not torch.cuda.is_available():
        return "CUDA is not available"
    sm_version = get_sm_version()
    if sm_version != 107:
        return f"Rubin (SM 107) required, got SM {sm_version}"
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        return "public CuteDSL Rubin kernels are not available"
    is_locality_domain_enabled.cache_clear()
    if not is_locality_domain_enabled():
        return "locality domain is not enabled/supported on this system"
    return None


def test_ci_acceleration_keeps_only_locality_domain_cutedsl_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _torch.moe import moe_test_utils

    monkeypatch.setattr(moe_test_utils, "IS_CI_MODE", True)
    model_config = MoeModelConfig(60, 4, 2048, 1408)
    common_kwargs = {
        "backend_type": MoeBackendType.CUTEDSL,
        "quant_algo": None,
        "model_config": model_config,
        "routing_method_cls": RenormalizeMoeRoutingMethod,
        "activation_type": ActivationType.Swiglu,
    }

    assert should_skip_to_accelerate_ci(dtype=torch.bfloat16, **common_kwargs) is not None
    assert (
        should_skip_to_accelerate_ci(
            dtype=torch.bfloat16,
            enable_locality_domains=True,
            **common_kwargs,
        )
        is None
    )
    assert (
        should_skip_to_accelerate_ci(
            dtype=torch.float16,
            enable_locality_domains=True,
            **common_kwargs,
        )
        is not None
    )


def generate_test_params() -> List:
    """
    Generate test parameter combinations, filtering out unsupported configurations.

    Unsupported combinations (those with a skip_reason from get_quick_skip_reason)
    are excluded entirely so they never appear in pytest collection output.

    Returns:
        List of pytest.param objects for runnable test configurations only
    """
    params: List = []
    for (
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        model_config,
        seq_len,
        dtype,
        backend_type,
        quant_algo,
        routing_method_cls,
        skip_reason,
        test_id,
    ) in iter_base_test_configs(
        SWIGLU_COMBOS,
        MOE_MODEL_CONFIGS,
        SEQ_LENS_TO_TEST,
        DTYPES_TO_TEST,
        BACKEND_TYPES_TO_TEST,
        QUANT_ALGOS_TO_TEST,
    ):
        if skip_reason:
            continue
        param_values = (
            dtype,
            backend_type,
            quant_algo,
            seq_len,
            model_config,
            routing_method_cls,
            ActivationType.Swiglu,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            False,
        )
        params.append(create_test_param(param_values, test_id))

        if quant_algo in (QuantAlgo.NVFP4, None):
            swiglu_gptoss_style = (
                swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")
            )
            locality_domain_skip_reason = should_skip_locality_domain_param(
                backend_type,
                quant_algo,
                ActivationType.Swiglu,
                swiglu_gptoss_style,
                dtype,
            )
            locality_domain_param_values = (
                dtype,
                backend_type,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                ActivationType.Swiglu,
                swiglu_alpha,
                swiglu_beta,
                swiglu_limit,
                True,
            )
            params.append(
                create_test_param(
                    locality_domain_param_values,
                    f"locality_domain=enabled-{test_id}",
                    locality_domain_skip_reason,
                )
            )

    return params


# Pre-generate test parameters at module load time
TEST_PARAMS = generate_test_params()


def generate_element_wise_test_params() -> List:
    params: List = []
    for activation_type in [ActivationType.Silu, ActivationType.Relu2]:
        for (
            _,  # swiglu_alpha  (ignored)
            _,  # swiglu_beta   (ignored)
            _,  # swiglu_limit  (ignored)
            model_config,
            seq_len,
            dtype,
            backend_type,
            quant_algo,
            routing_method_cls,
            skip_reason,
            base_test_id,
        ) in iter_base_test_configs(
            [(1, 0, float("inf"))],  # swiglu parameters are irrelevant
            MOE_MODEL_CONFIGS,
            SEQ_LENS_TO_TEST,
            DTYPES_TO_TEST,
            [MoeBackendType.CUTLASS, MoeBackendType.TRTLLM],
            [None, QuantAlgo.NVFP4, QuantAlgo.W8A16],
        ):
            if skip_reason:
                continue
            if backend_type == MoeBackendType.CUTLASS and activation_type == ActivationType.Silu:
                continue
            if backend_type == MoeBackendType.TRTLLM and quant_algo is None:
                continue
            # INT8 weight-only per-channel non-gated MoE is CUTLASS-path only.
            if quant_algo == QuantAlgo.W8A16 and backend_type != MoeBackendType.CUTLASS:
                continue
            test_id = f"act={activation_type.name}-{base_test_id}"
            param_values = (
                dtype,
                backend_type,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                activation_type,
                None,
                None,
                None,
                False,
            )
            params.append(create_test_param(param_values, test_id))
    return params


TEST_PARAMS += generate_element_wise_test_params()

# INT8 weight-only per-channel packs mixed-GEMM weights in row tiles of 64
# (preprocess_weights_for_mixed_gemm: rows_per_tile = 128 * 8 // BITS_PER_ELT_A,
# BITS_PER_ELT_A = 16 for W8A16; also num_rows % B_ROWS_PER_MMA == 0, = 16).
# Every intermediate_size in MOE_MODEL_CONFIGS is a multiple of 64, so the path
# that pads a non-aligned per-partition intermediate is otherwise uncovered.
# 200 is a multiple of neither 16 nor 64 and rounds up to 256.
W8A16_NON_ALIGNED_MOE_MODEL_CONFIGS = [
    MoeModelConfig(4, 2, 512, 200),
]


def generate_int8_woq_non_aligned_test_params() -> List:
    """W8A16 CUTLASS coverage for a non-64-aligned intermediate size."""
    params: List = []
    # Cover both the non-gated (single up-projection) and gated (gate+up
    # concatenated) layouts: the gated path pads w3 as well as w1 before the
    # concatenation, so it exercises a branch the non-gated case does not.
    for activation_type in (ActivationType.Relu2, ActivationType.Swiglu):
        for (
            _,  # swiglu_alpha  (ignored)
            _,  # swiglu_beta   (ignored)
            _,  # swiglu_limit  (ignored)
            model_config,
            seq_len,
            dtype,
            backend_type,
            quant_algo,
            routing_method_cls,
            skip_reason,
            base_test_id,
        ) in iter_base_test_configs(
            [(1, 0, float("inf"))],  # swiglu parameters are irrelevant
            W8A16_NON_ALIGNED_MOE_MODEL_CONFIGS,
            SEQ_LENS_TO_TEST,
            DTYPES_TO_TEST,
            [MoeBackendType.CUTLASS],
            [QuantAlgo.W8A16],
        ):
            # get_quick_skip_reason() rejects every quantized config whose sizes are
            # not 128-aligned. That gate is exactly what this case exists to cover:
            # the INT8-woq path must pad a non-64-aligned per-partition intermediate
            # up to 64. Honour every other skip reason.
            if skip_reason and "Non-128-aligned" not in skip_reason:
                continue
            test_id = f"act={activation_type.name}-nonaligned-{base_test_id}"
            gated = is_gated_activation(activation_type)
            param_values = (
                dtype,
                backend_type,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                activation_type,
                # Gated activations read these; None would be misread as
                # gptoss-style SwiGLU. Non-gated ignores them.
                1 if gated else None,
                0 if gated else None,
                float("inf") if gated else None,
                False,
            )
            params.append(create_test_param(param_values, test_id))
    # CPU-only and pre-SM80 runners legitimately yield no cases; elsewhere an empty
    # list means the 128-alignment skip message was reworded upstream.
    if not params and get_sm_version() >= 80:
        raise ValueError("non-aligned W8A16 coverage generated no test cases")
    return params


TEST_PARAMS += generate_int8_woq_non_aligned_test_params()


# =====================================================================
# Test Implementation
# =====================================================================
#
# This file provides a UNIFIED TEST FRAMEWORK for testing all MoE backend
# implementations through their backend-level interfaces.
#
# ======================================================================
# Purpose & Scope
# ======================================================================
# - Test MoE backends via: routing_method.apply -> quantize_input -> run_moe
# - Single GPU execution (no multi-GPU/distributed testing)
# - Accuracy validation against reference implementations
#
# ======================================================================
# Test Coverage Matrix
# ======================================================================
# 1. BACKENDS: CUTLASS, TRTLLM, CUTEDSL, DEEPGEMM
#    - When using element wise activations (Relu2, Silu), only CUTLASS and TRTLLM
#      are supported
#
# 2. QUANTIZATION ALGORITHMS:
#    - When using Swiglu:
#      - Unquantized (None)
#      - FP8, FP8_BLOCK_SCALES
#      - NVFP4, W4A8_NVFP4_FP8
#      - W4A16_MXFP4, W4A8_MXFP4_MXFP8
#      - W8A16, W4A8_AWQ
#    - When using element-wise activations
#      - Unquantized (CUTLASS)
#      - NVFP4 (TRTLLM, CUTLASS)
#
# 3. ACTIVATION DTYPES: float16, bfloat16
#
# 4. AUTOTUNER TACTICS:
#    - Autotune phase: find optimal tactics via AutoTuner
#    - Capture phase: record all tactics used
#    - Replay phase: verify each tactic produces correct results
#
# 5. GPTOSS_STYLE (SwiGLU with custom parameters):
#    - swiglu_alpha: scaling factor (default=1)
#    - swiglu_beta: bias term (default=0)
#    - swiglu_limit: clipping limit (default=inf)
#    - Supported by: CUTLASS (W4A8_MXFP4_MXFP8), TRTLLM (W4A8_MXFP4_MXFP8)
#
# 6. MODEL CONFIGURATIONS:
#    - Real models: Mixtral, DeepSeek, Grok, GPT-OSS
#    - Boundary cases: prime num_experts, small sizes, top_k=1, top_k=num_experts
#
# ======================================================================
# Skip Logic
# ======================================================================
# Tests are automatically skipped for unsupported configurations using:
# - backend.can_implement(p, d): declared quant / dtype / SM / dependency support
# - should_skip_trtllm(): TRTLLM-specific constraints (num_experts % 4, etc.)
# - should_skip_cutedsl(): CuteDSL-specific accuracy issues
# - 128-alignment requirements for quantization
#
# ======================================================================
@pytest.mark.parametrize(
    "dtype_activation,backend_type,quant_algo,seq_len,model_config,"
    "routing_method_cls,activation_type,swiglu_alpha,swiglu_beta,swiglu_limit,"
    "enable_locality_domains",
    TEST_PARAMS,
)
def test_moe_backend(
    dtype_activation: torch.dtype,
    backend_type: MoeBackendType,
    quant_algo: Optional[QuantAlgo],
    seq_len: int,
    model_config: MoeModelConfig,
    routing_method_cls,
    activation_type: ActivationType,
    swiglu_alpha: Optional[float],
    swiglu_beta: Optional[float],
    swiglu_limit: Optional[float],
    monkeypatch: pytest.MonkeyPatch,
    enable_locality_domains: bool,
    tmp_path: Path,
):
    """
    Test MoE backend with autotune to capture all tactics.

    This test verifies:
    1. Autotune works correctly with the backend
    2. All tactics are captured properly
    3. Different sequence lengths use appropriate tactics
    4. swiglu_gptoss_style (SwiGlu with custom parameters) works correctly
    """
    # DENSEGEMM: disable fused fc2_alpha path for backend-level testing.
    if backend_type == MoeBackendType.DENSEGEMM:
        monkeypatch.setenv("TRTLLM_MOE_FUSED_FC2_ALPHA", "0")

    # MEGAMOE_CUTEDSL threads per-expert fc31_alpha / fc2_alpha /
    # fc1_norm_const through the kernel ABI, so NVFP4QuantizeUtil's non-1
    # weight_scale_2 values compute correctly without a test bypass.

    is_gated = is_gated_activation(activation_type)
    swiglu_gptoss_style = False
    if is_gated:
        # Determine swiglu_gptoss_style based on swiglu parameters
        # swiglu_gptoss_style is True when any swiglu parameter deviates from default
        # Default values: alpha=1, beta=0, limit=inf
        swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")
    is_minimax_megamoe = backend_type == MoeBackendType.MEGAMOE_CUTEDSL and swiglu_gptoss_style
    expert_bias = swiglu_gptoss_style and not is_minimax_megamoe

    locality_domain_runtime_skip = should_skip_locality_domain_runtime(enable_locality_domains)
    if locality_domain_runtime_skip:
        pytest.skip(locality_domain_runtime_skip)
    locality_domain_policy = LocalityDomainPolicy(enabled=enable_locality_domains)

    ci_skip = should_skip_to_accelerate_ci(
        backend_type=backend_type,
        quant_algo=quant_algo,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
        dtype=dtype_activation,
        seq_len=seq_len,
        swiglu_gptoss_style=swiglu_gptoss_style,
        activation_type=activation_type,
        enable_locality_domains=enable_locality_domains,
    )
    if ci_skip:
        pytest.skip(ci_skip)

    # Extract model parameters
    num_experts = model_config.num_experts
    top_k = model_config.top_k
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype_activation)

    # Create mapping
    mapping = Mapping()
    mapping.rank = mpi_rank()
    _ensure_single_proc_dist_for_megamoe(backend_type, mapping.rank)

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Setup autotuner distributed state
        AutoTuner.get().setup_distributed_state(mapping)

        # Create routing method from parametrized class
        routing_method = routing_method_cls(top_k=top_k)

        # Create test inputs
        x = torch.randn((seq_len, hidden_size), dtype=dtype_activation, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype_activation, device="cuda")

        # Get quantization parameters
        # Pass backend_type to determine scale format (DEEPGEMM/TRTLLM need E8M0 scale)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )

        # Create quantize utility with swiglu_gptoss_style parameters
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype_activation,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=expert_bias,
            swiglu_gptoss_style=swiglu_gptoss_style,
            swiglu_alpha=swiglu_alpha if swiglu_gptoss_style else None,
            swiglu_beta=swiglu_beta if swiglu_gptoss_style else None,
            swiglu_limit=swiglu_limit if swiglu_gptoss_style else None,
            activation_type=activation_type,
        )

        # Get swiglu tensors if swiglu_gptoss_style is enabled
        swiglu_tensors = quantize_util.get_swiglu_tensors()

        # Determine weight loading mode based on quantization algorithm
        weight_loading_mode = MoEWeightLoadingMode.VANILLA
        if hasattr(quantize_util, "weight_loading_mode"):
            weight_loading_mode = quantize_util.weight_loading_mode

        # Clear class-level permute indices cache between parametrized test cases
        # to work around a B200-specific kernel bug (tactic [32,5] illegal memory access)
        from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEBaseMethod

        NVFP4TRTLLMGenFusedMoEBaseMethod._cache_permute_indices.clear()

        # Create backend first (needed for MXFP4_MXFP8 to get shapes)
        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype_activation,
            quant_config=quant_config,
            mapping=mapping,
            bias=expert_bias,
            swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
            swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
            swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
            weight_loading_mode=weight_loading_mode,
            activation_type=activation_type,
            locality_domain_policy=locality_domain_policy,
        )

        # W4A8_MXFP4_MXFP8 / W4A8_MXFP4_FP8 require backend-layout-aware
        # weights. CUTLASS and MegaMoE use 128 hidden alignment; TRTLLMGen
        # pads FC1 input to 512. MXFP4FP8QuantizeUtil inherits
        # prepare_weights_from_backend from MXFP4MXFP8QuantizeUtil so the
        # backend-vs-reference weight split applies to both variants.
        ref_cls = quant_kwargs.pop("ref_cls", None)
        ref_module_kwargs = {}
        if quant_algo in (QuantAlgo.W4A8_MXFP4_MXFP8, QuantAlgo.W4A8_MXFP4_FP8):
            weights, ref_weights, ref_module_kwargs = quantize_util.prepare_weights_from_backend(
                backend, **quant_kwargs
            )
        else:
            weights = quantize_util.create_weights(**quant_kwargs)
            ref_weights = weights

        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()
        if enable_locality_domains:
            assert backend._locality_domain_runtime is not None
            assert backend._locality_domain_weight_shards is not None

        # Create reference
        if ref_cls is not None:
            ref_fused_moe = quantize_util.create_ref_module(
                routing_method, ref_cls=ref_cls, **ref_module_kwargs
            )
        else:
            ref_fused_moe = quantize_util.create_ref_module(routing_method, **ref_module_kwargs)
        ref_fused_moe.load_weights([ref_weights])
        ref_fused_moe.cuda()

        # Clear autotuner cache before autotune phase
        AutoTuner.get().clear_cache()
        if enable_locality_domains:
            AutoTuner.get().reset_statistics()

        # Get reference output first
        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        # Helper to run MoE computation
        def run_moe():
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            return run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype_activation,
                router_logits,
            )

        # Configure AutoTuner for faster profiling (reduce warmup/repeat for unit tests)
        autotuner = AutoTuner.get()
        autotuner.warmup = 0  # default: 2
        autotuner.repeat = 1  # default: 10
        autotuner.stream_delay_micro_secs = 10  # default: 1000

        # Autotune phase: tune kernels to find best tactics
        # Use cache_path to speed up subsequent runs by reusing tuning results
        cache_path = (
            str(tmp_path / "moe_autotuner_cache.json")
            if enable_locality_domains
            else "/tmp/moe_autotuner_cache.json"
        )
        with torch.inference_mode(), autotune(cache_path=cache_path):
            _ = run_moe()
        if enable_locality_domains:
            quant_name = "nvfp4" if quant_algo == QuantAlgo.NVFP4 else "bf16"
            expected_tuning_ops = (
                f"CuteDslFusedMoE::run_moe_{quant_name}::locality_domain_end_to_end",
                f"trtllm::cute_dsl_{quant_name}_gather_grouped_gemm_"
                f"{'act_fusion' if quant_algo == QuantAlgo.NVFP4 else 'swiglu'}"
                "_rubin::locality_domain_concurrent",
                f"trtllm::cute_dsl_{quant_name}_grouped_gemm_finalize_"
                "inplace_rubin::locality_domain_concurrent",
            )
            for op_name in expected_tuning_ops:
                assert autotuner.stats.tuned_op_profiled_configs.get(op_name, 0) > 0
                assert not autotuner.stats.failed_profiling_count.get(op_name, set())

        # flashinfer has no capture and replay mechanisms, so we skip test_all_kernels
        use_flashinfer = getattr(backend, "use_flashinfer", False)

        # Check if this backend+quant_algo combination supports autotuner capture/replay
        if supports_autotuner_capture(backend_type, quant_algo, use_flashinfer):
            # Capture phase: record which tactics are used (requires actual execution)
            with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
                _ = run_moe()

            # Replaying every outer tile is deliberately exhaustive and would
            # multiply the inner FC tactic replay for every matrix member. One
            # representative production shape per locality domain path covers that
            # outer-tile contract; all matrix members still validate their
            # tuned/failed statistics and replay the tactics selected for their
            # own shape below.
            representative_outer_tile_replay = (
                enable_locality_domains
                and quant_algo in (QuantAlgo.NVFP4, None)
                and seq_len == 1
                and (num_experts, top_k, hidden_size, intermediate_size) == (60, 4, 2048, 1408)
            )
            if representative_outer_tile_replay:
                # The regular Cartesian replay contains inner FC tactics only
                # for the selected outer tile. Exercise every outer tile
                # directly after tuning, when all corresponding FC caches have
                # been prepared, so a non-winning tile cannot silently regress.
                outer_context = all_tactics._captured_contexts[0]
                outer_runner = outer_context["runners"][0]
                outer_tactics = outer_runner.get_valid_tactics(
                    outer_context["inputs"], OptimizationProfile()
                )
                expected_outer_tactics = (
                    {128, 256, 512} if quant_algo == QuantAlgo.NVFP4 else {64, 128, 256}
                )
                assert set(outer_tactics) == expected_outer_tactics
                for outer_tactic in outer_tactics:
                    # Direct runner replay reuses the captured inplace output;
                    # reset it to the fresh-output baseline used by run_moe().
                    with torch.inference_mode():
                        outer_context["inputs"][-1].zero_()
                        output = outer_runner(outer_context["inputs"], tactic=outer_tactic)
                    ref_fused_moe.check_accuracy(output, ref_output)

            # Replay phase: test each tactic for correctness
            # Set fail_fast=True to stop on first failure, False to run all and report summary
            replay_tactics_and_check(
                all_tactics=all_tactics,
                run_moe_fn=run_moe,
                check_accuracy_fn=ref_fused_moe.check_accuracy,
                ref_output=ref_output,
                backend_type=backend_type,
                quant_algo=quant_algo,
                fail_fast=False,  # Change to True to fail on first error
            )
        else:
            # For backends that don't support autotuner capture/replay,
            # just run a simple accuracy check
            with torch.inference_mode():
                output = run_moe()
                ref_fused_moe.check_accuracy(output, ref_output)


# =====================================================================
# BF16 (unquantized) TRTLLM-Gen MoE: DeepSeekV3 / Renormalize routing
# =====================================================================
# Covers the BF16 FlashInfer path beyond the Renormalize/SwiGLU cases of the
# main test_moe_backend matrix (Nemotron-H enablement): DeepSeekV3/Renormalize
# routing x Relu2/Swiglu, via both fused and separated routing.

# DeepSeekV3 trtllm-gen routing requires num_experts >= 22, multiple of 4.
_BF16_UNQUANT_NUM_EXPERTS = 72
_BF16_UNQUANT_TOP_K = 6
_BF16_UNQUANT_HIDDEN = 1024
_BF16_UNQUANT_INTERMEDIATE = 512


def _make_bf16_routing_method(routing_kind: str, top_k: int, num_experts: int, device: str):
    if routing_kind == "renormalize":
        return RenormalizeMoeRoutingMethod(top_k=top_k)
    # DeepSeekV3 (noaux_tc): sigmoid scores + correction bias, single group.
    bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    return DeepSeekV3MoeRoutingMethod(
        top_k=top_k,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        callable_e_score_correction_bias=lambda: bias,
    )


@pytest.mark.parametrize(
    "trtllm_use_router_logits", [True, False], ids=["fused_routing", "separated_routing"]
)
@pytest.mark.parametrize("seq_len", [8, 256])
@pytest.mark.parametrize(
    "activation_type", [ActivationType.Relu2, ActivationType.Swiglu], ids=["relu2", "swiglu"]
)
@pytest.mark.parametrize("routing_kind", ["deepseekv3", "renormalize"])
def test_trtllm_bf16_unquantized_moe(
    routing_kind,
    activation_type,
    seq_len,
    trtllm_use_router_logits,
    num_experts=_BF16_UNQUANT_NUM_EXPERTS,
    top_k=_BF16_UNQUANT_TOP_K,
):
    """TRTLLM-Gen BF16 (unquantized) MoE accuracy vs the reference impl."""
    backend_type = MoeBackendType.TRTLLM
    dtype = torch.bfloat16

    num_experts = _BF16_UNQUANT_NUM_EXPERTS
    top_k = _BF16_UNQUANT_TOP_K
    hidden_size = _BF16_UNQUANT_HIDDEN
    intermediate_size = _BF16_UNQUANT_INTERMEDIATE

    # This test constructs the backend directly, so query it directly. The
    # parametrized activation has to go into the problem: this leaf declares
    # which kinds it reaches, so leaving it out would ask the Relu2 cases for
    # the SwiGLU verdict.
    verdict = get_backend_class(backend_type).can_implement(
        MoEProblem(
            quant=None,
            dtype_act=dtype,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation=canonical_activation(activation_type),
        ),
        MoEDeployment(
            ep_size=1,
            tp_size=1,
            parallel_size=1,
            use_dp=False,
            num_slots=num_experts,
            env=collect_moe_environment(),
        ),
    )
    if not verdict.eligible:
        pytest.skip(verdict.detail)

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype)

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)

        routing_method = _make_bf16_routing_method(routing_kind, top_k, num_experts, "cuda")

        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")

        # Unquantized path: get_test_quant_params returns BaseQuantizeUtil.
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(None, x, backend_type)
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            activation_type=activation_type,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            activation_type=activation_type,
        )
        if trtllm_use_router_logits and backend._routes_outside_the_kernel():
            # This config only routes outside the kernel, so run_moe drops
            # router_logits and the scheduler never pairs the two. Asking for
            # fused routing here tests a combination production cannot reach.
            pytest.skip("routing happens outside the kernel; fused routing is unreachable")

        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        AutoTuner.get().clear_cache()

        def run_moe():
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            return run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype,
                router_logits=router_logits,
                trtllm_use_router_logits=trtllm_use_router_logits,
            )

        # Autotune, then verify accuracy against the reference.
        with torch.inference_mode(), autotune(cache_path="/tmp/moe_autotuner_cache.json"):
            _ = run_moe()
        with torch.inference_mode():
            output = run_moe()
            ref_fused_moe.check_accuracy(output, ref_output)


@pytest.mark.parametrize("seq_len", [1, 8])
def test_trtllm_bf16_dsv3_routing_kimi_k3_shape(seq_len):
    """Kimi-K3 routing shape: 896 experts / top_k 16 without expert groups.

    Exercises the (1024, 32) tier of the trtllm-gen DeepSeekV3 routing
    kernels, including the cooperative small-batch kernel at num_tokens 1,
    via the fused-routing path (router logits consumed in-kernel).
    """
    test_trtllm_bf16_unquantized_moe(
        routing_kind="deepseekv3",
        activation_type=ActivationType.Swiglu,
        seq_len=seq_len,
        trtllm_use_router_logits=True,
        num_experts=896,
        top_k=16,
    )


# =====================================================================
# TRTLLM-Gen shared-expert fusion (migrated from deprecated
# tests/unittest/_torch/moe/kernels/serial/test_moe.py::TestMoeFP8 fusion coverage)
# =====================================================================
# TRTLLMGenFusedMoE can fold n_shared_experts into the routed grouped GEMM as
# always-selected experts (opt-in via TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1;
# requires FP8_BLOCK_SCALES + dp_size==1 + DeepSeekV3 routing). The fused
# experts occupy trailing weight slots [num_experts, num_experts+n_fused) and
# receive routing weight 1.0, applied after routed_scaling_factor.

# (num_experts, n_group, topk_group, top_k, n_fused) — shapes taken from the
# deprecated thop test's expert_info parametrization (n_fused > 0 variants).
FUSED_SHARED_EXPERT_INFOS = [
    (32, 8, 4, 8, 1),
    (256, 8, 4, 8, 2),
    (72, 1, 1, 6, 1),
    (72, 1, 1, 6, 2),
]


@pytest.fixture
def shared_expert_fusion_enabled(monkeypatch: pytest.MonkeyPatch):
    """Turn the flag on somewhere the backend will actually read it.

    ``setenv`` alone is not enough. The flag reaches the backend through the
    selection environment, which is collected once per process and cached, so
    any earlier test that resolved a MoE has already frozen it -- leaving the
    write visible to ``os.environ`` and to nothing else. Dropping the cache on
    the way out rather than re-collecting there lets the next reader collect
    after ``monkeypatch`` has restored the variable.
    """
    from tensorrt_llm._torch.moe.fused_moe.impl_environment import (
        MoEEnvFlag,
        collect_moe_environment,
        reset_moe_environment_cache,
    )

    monkeypatch.setenv(MoEEnvFlag.SHARED_EXPERT_FUSION.value, "1")
    collect_moe_environment(force=True)
    yield
    reset_moe_environment_cache()


class _AppendSharedExpertsRouting:
    """Reference-side routing wrapper: appends the fused shared experts as
    always-selected entries (ids [num_experts, num_experts+n_fused), weight
    1.0 after routed scaling) to the wrapped routing method's output."""

    def __init__(self, routing_method, num_experts: int, n_fused: int) -> None:
        self._routing_method = routing_method
        self._num_experts = num_experts
        self._n_fused = n_fused

    def apply(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ids, weights = self._routing_method.apply(router_logits)
        num_tokens = ids.shape[0]
        shared_ids = torch.arange(
            self._num_experts,
            self._num_experts + self._n_fused,
            dtype=ids.dtype,
            device=ids.device,
        ).expand(num_tokens, -1)
        shared_weights = torch.ones(
            (num_tokens, self._n_fused), dtype=weights.dtype, device=weights.device
        )
        return torch.cat([ids, shared_ids], dim=1), torch.cat([weights, shared_weights], dim=1)


def _write_fused_shared_expert_slots(
    backend: MoE, weights: dict, num_experts: int, n_fused: int
) -> None:
    """Copy the shared experts' quantized tensors into the trailing weight
    slots, mirroring DeepSeekFP8BlockScalesFusedMoEMethod's routed-slot layout
    (w3_w1 slot = [w3; w1] along dim 0; scales likewise)."""
    for i in range(n_fused):
        expert_id = num_experts + i
        slot = num_experts + i
        dst_w3, dst_w1 = backend.w3_w1_weight.data[slot].chunk(2, dim=0)
        dst_w3.copy_(weights[f"{expert_id}.w3.weight"].view(dst_w3.dtype))
        dst_w1.copy_(weights[f"{expert_id}.w1.weight"].view(dst_w1.dtype))
        backend.w2_weight.data[slot].copy_(
            weights[f"{expert_id}.w2.weight"].view(backend.w2_weight.dtype)
        )
        dst_w3_scale, dst_w1_scale = backend.w3_w1_weight_scaling_factor.data[slot].chunk(2, dim=0)
        dst_w3_scale.copy_(weights[f"{expert_id}.w3.weight_scale"])
        dst_w1_scale.copy_(weights[f"{expert_id}.w1.weight_scale"])
        backend.w2_weight_scaling_factor.data[slot].copy_(weights[f"{expert_id}.w2.weight_scale"])


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="TRTLLM-Gen FP8 block scales requires SM100/103",
)
@pytest.mark.parametrize("seq_len", [16, 1024])
@pytest.mark.parametrize(
    "expert_info",
    FUSED_SHARED_EXPERT_INFOS,
    ids=lambda info: f"e{info[0]}g{info[1]}tg{info[2]}k{info[3]}fused{info[4]}",
)
@pytest.mark.usefixtures("shared_expert_fusion_enabled")
def test_trtllm_fp8_block_scales_fused_shared_experts(
    expert_info,
    seq_len: int,
):
    """Fused-shared-expert accuracy for the TRTLLM backend (FP8 block scales).

    Loads routed experts through the regular path, writes the shared experts
    into the trailing fused slots, and checks the fused kernel against a
    reference that treats the shared experts as always-selected with weight
    1.0. Replays every captured tactic; also asserts the small-tile WAR
    (fused path restricted to tileN >= 32) holds in the autotuner candidates.
    """
    num_experts, n_group, topk_group, top_k, n_fused = expert_info
    hidden_size = 512
    intermediate_size = 512
    dtype = torch.bfloat16
    backend_type = MoeBackendType.TRTLLM

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        AutoTuner.get().setup_distributed_state(mapping)

        e_score_correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device="cuda")
        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=2.5,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )

        router_logits = torch.randn((seq_len, num_experts), dtype=torch.float32, device="cuda")

        placeholder_x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.FP8_BLOCK_SCALES, placeholder_x, backend_type
        )
        # The quantize util covers routed + fused shared experts so create_weights
        # emits per-expert tensors for all of them and the reference module gets
        # a GatedMLP per fused slot as well.
        quantize_util = quantize_util_cls(
            num_experts=num_experts + n_fused,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
        )
        weights = quantize_util.create_weights(**quant_kwargs)
        x = quantize_util.create_input(seq_len)

        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            n_shared_experts=n_fused,
        )
        assert backend.num_fused_shared_expert == n_fused, (
            "shared-expert fusion gate did not activate"
        )
        assert backend.w3_w1_weight.shape[0] == num_experts + n_fused

        # Mirror the real flow (modeling_deepseekv3): load routed experts,
        # post-process, then fill the trailing fused slots.
        routed_weights = {k: v for k, v in weights.items() if int(k.split(".")[0]) < num_experts}
        backend.load_weights([routed_weights])
        backend.post_load_weights()
        backend.cuda()
        _write_fused_shared_expert_slots(backend, weights, num_experts, n_fused)

        ref_routing = _AppendSharedExpertsRouting(routing_method, num_experts, n_fused)
        ref_fused_moe = quantize_util.create_ref_module(ref_routing)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        AutoTuner.get().clear_cache()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

        def run_moe():
            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            return run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype,
                router_logits=router_logits,
            )

        autotuner = AutoTuner.get()
        autotuner.warmup = 0
        autotuner.repeat = 1
        autotuner.stream_delay_micro_secs = 10

        with (
            torch.inference_mode(),
            autotune(cache_path="/tmp/moe_autotuner_cache_fused_shared.json"),
        ):
            _ = run_moe()

        with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
            _ = run_moe()

        # WAR regression check: with num_fused_shared_experts > 0 the C++ side
        # (fp8BlockScaleMoe.cpp fusedMinTileN, default 32) must exclude the
        # small-tile (tileN 8/16) dynB cubins from the candidate tactics.
        moe_tile_ns = []
        for combo in all_tactics:
            for _, tactic in combo:
                if isinstance(tactic, (list, tuple)) and len(tactic) == 2:
                    moe_tile_ns.append(int(tactic[0]))
        assert moe_tile_ns, "expected [tileN, config] tactics from the fused MoE runner"
        assert all(tile_n >= 32 for tile_n in moe_tile_ns), (
            f"fused path produced small-tile tactics (tileN < 32): {sorted(set(moe_tile_ns))}"
        )

        replay_tactics_and_check(
            all_tactics=all_tactics,
            run_moe_fn=run_moe,
            check_accuracy_fn=ref_fused_moe.check_accuracy,
            ref_output=ref_output,
            backend_type=backend_type,
            quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
        )


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="TRTLLM-Gen FP8 block scales requires SM100/103",
)
@pytest.mark.usefixtures("shared_expert_fusion_enabled")
def test_trtllm_fp8_block_scales_fuse_shared_expert_layout():
    """Layout-only check of fuse_shared_expert (no kernel launch).

    Builds a shared GatedMLP of intermediate n_fused*I, fuses it, and asserts
    each trailing slot holds the expected per-expert sub-tensors by
    definition: slot i's [w3; w1] = gate_up rows [i*I, (i+1)*I) of the up/gate
    halves, slot i's w2 = down_proj columns [i*I, (i+1)*I) — including the
    block-scale tensors (the w2 scale split is a historically bug-prone spot).
    """
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    num_experts = 8
    top_k = 2
    hidden_size = 256
    intermediate_size = 256
    n_fused = 2
    dtype = torch.bfloat16
    scale_block = 128

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)

        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=top_k,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            callable_e_score_correction_bias=lambda: torch.zeros(
                num_experts, dtype=dtype, device="cuda"
            ),
        )
        placeholder_x = torch.randn((1, hidden_size), dtype=dtype, device="cuda")
        _, quant_config, _ = get_test_quant_params(
            QuantAlgo.FP8_BLOCK_SCALES, placeholder_x, MoeBackendType.TRTLLM
        )

        backend = create_test_backend(
            backend_type=MoeBackendType.TRTLLM,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            quant_config=quant_config,
            mapping=mapping,
            n_shared_experts=n_fused,
        )
        assert backend.num_fused_shared_expert == n_fused

        shared_mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=n_fused * intermediate_size,
            bias=False,
            dtype=dtype,
            config=ModelConfig(quant_config=quant_config),
        )
        shared_mlp.cuda()
        # Fill weights/scales with distinct random data (fp8 weights are filled
        # via a bf16 sample viewed as fp8 bytes; values only need to be unique).
        gate_up = shared_mlp.gate_up_proj
        down = shared_mlp.down_proj
        gate_up.weight.data.copy_(
            torch.randn(gate_up.weight.shape, dtype=torch.bfloat16, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(gate_up.weight.dtype)
        )
        down.weight.data.copy_(
            torch.randn(down.weight.shape, dtype=torch.bfloat16, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(down.weight.dtype)
        )
        gate_up.weight_scale.data.copy_(
            torch.rand(gate_up.weight_scale.shape, dtype=torch.float32, device="cuda")
        )
        down.weight_scale.data.copy_(
            torch.rand(down.weight_scale.shape, dtype=torch.float32, device="cuda")
        )

        backend.fuse_shared_expert(shared_mlp)
        torch.cuda.synchronize()

        # gate_up_proj rows are [gate(w1); up(w3)]; per-expert i owns rows
        # [i*I, (i+1)*I) within each half. down_proj columns [i*I, (i+1)*I).
        w1_all, w3_all = gate_up.weight.data.chunk(2, dim=0)
        w1_scale_all, w3_scale_all = gate_up.weight_scale.data.chunk(2, dim=0)
        inter_blocks = intermediate_size // scale_block
        for i in range(n_fused):
            slot = num_experts + i
            rows = slice(i * intermediate_size, (i + 1) * intermediate_size)
            scale_rows = slice(i * inter_blocks, (i + 1) * inter_blocks)

            dst_w3, dst_w1 = backend.w3_w1_weight.data[slot].chunk(2, dim=0)
            torch.testing.assert_close(dst_w3, w3_all[rows].view(dst_w3.dtype))
            torch.testing.assert_close(dst_w1, w1_all[rows].view(dst_w1.dtype))
            torch.testing.assert_close(
                backend.w2_weight.data[slot],
                down.weight.data[:, rows].view(backend.w2_weight.dtype),
            )

            dst_w3_scale, dst_w1_scale = backend.w3_w1_weight_scaling_factor.data[slot].chunk(
                2, dim=0
            )
            torch.testing.assert_close(dst_w3_scale, w3_scale_all[scale_rows])
            torch.testing.assert_close(dst_w1_scale, w1_scale_all[scale_rows])
            torch.testing.assert_close(
                backend.w2_weight_scaling_factor.data[slot],
                down.weight_scale.data[:, scale_rows],
            )


@contextmanager
def fine_grained_sync_env(enabled: bool):
    """Toggle TLLM_USE_FINE_GRAINED_SYNC for the duration of a test case.

    The C++ side reads this env var fresh on each kernel-option construction
    (envUtils.cpp getEnvUseFineGrainedSync), so per-test scoping works within
    a single process.
    """
    prev = os.environ.get("TLLM_USE_FINE_GRAINED_SYNC")
    os.environ["TLLM_USE_FINE_GRAINED_SYNC"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TLLM_USE_FINE_GRAINED_SYNC", None)
        else:
            os.environ["TLLM_USE_FINE_GRAINED_SYNC"] = prev


@pytest.mark.parametrize("num_tokens", [1, 1024])
def test_moe_backend_trtllm_nvfp4_fine_grained(num_tokens: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    sm_version = get_sm_version()
    if sm_version != 107:
        pytest.skip(f"fine-grained sync requires SM107, got SM{sm_version}")

    dtype_activation = torch.bfloat16
    backend_type = MoeBackendType.TRTLLM
    quant_algo = QuantAlgo.NVFP4
    activation_type = ActivationType.Relu2

    num_experts = 2048
    top_k = 32
    hidden_size = 1024
    intermediate_size = 1024

    skip_if_insufficient_gpu_memory(num_experts, hidden_size, intermediate_size, dtype_activation)

    mapping = Mapping()
    mapping.rank = mpi_rank()

    # The C++ runner reads TLLM_USE_FINE_GRAINED_SYNC at kernel-option
    # construction time, so the env must be set for backend creation and the
    # forward pass alike.
    with fine_grained_sync_env(True), torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        AutoTuner.get().setup_distributed_state(mapping)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        x = torch.randn((num_tokens, hidden_size), dtype=dtype_activation, device="cuda")
        router_logits = torch.randn(
            (num_tokens, num_experts), dtype=dtype_activation, device="cuda"
        )

        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype_activation,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            activation_type=activation_type,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        backend = create_test_backend(
            backend_type=backend_type,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype_activation,
            quant_config=quant_config,
            mapping=mapping,
            activation_type=activation_type,
        )
        backend.load_weights([weights])
        backend.post_load_weights()
        backend.cuda()

        ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)

            token_selected_experts, token_final_scales = routing_method.apply(router_logits)
            x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
            output = run_backend_moe(
                backend,
                backend_type,
                x_quantized,
                x_sf,
                token_selected_experts,
                token_final_scales,
                dtype_activation,
                router_logits,
            )

            ref_fused_moe.check_accuracy(output, ref_output)


def test_build_moe_deployment_carries_finalize_and_locality_domain():
    """The two fields SM107 eligibility reads must survive ``ModelConfig``."""
    from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy

    default = build_moe_deployment(ModelConfig(), num_experts=8)
    assert default.fused_finalize_enabled is True
    assert default.locality_domain_requested is False

    disabled = build_moe_deployment(ModelConfig(moe_disable_finalize_fusion=True), num_experts=8)
    assert disabled.fused_finalize_enabled is False

    requested = build_moe_deployment(
        ModelConfig(locality_domain_policy=LocalityDomainPolicy(enabled=True)),
        num_experts=8,
    )
    assert requested.locality_domain_requested is True


def _nvfp4_problem(intermediate_size: Optional[int], activation: str) -> MoEProblem:
    return MoEProblem(
        quant=QuantAlgo.NVFP4.value,
        dtype_act=torch.bfloat16,
        hidden_size=7168,
        intermediate_size=intermediate_size,
        num_experts=256,
        top_k=8,
        activation=activation,
    )


def _deployment_at_moe_tp(moe_tp_size: int) -> MoEDeployment:
    return MoEDeployment(
        ep_size=1,
        tp_size=moe_tp_size,
        parallel_size=moe_tp_size,
        use_dp=False,
        num_slots=256,
        env=MoEEnvironment(sm=100),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "backend_cls",
    [TrtllmCutlassNvfp4Impl, TrtllmCutedslGroupedGemmNvfp4Impl],
    ids=["cutlass", "cutedsl"],
)
@pytest.mark.parametrize(
    "intermediate_size,activation,moe_tp_size,rejected",
    [
        # DeepSeek-R1-0528 NVFP4. moe_tp_size=64 is what tp_size=8 + cp_size=8
        # HELIX resolves to, and it shards FC1 to 64 rows: CuteDSL dies in
        # unswizzle_sf, Cutlass returns an all-zero output. NVBUG 5859751.
        (2048, "Swiglu", 32, False),
        (2048, "Swiglu", 64, True),
        # Unknown is not false: an absent shape must abstain, not reject.
        (None, "Swiglu", 64, False),
        # Nemotron-3 Nano NVFP4: 128-unaligned, but non-gated FC1 has no
        # gate/up split so the padding is a zero tail and it loads fine.
        (1856, "Relu2", 1, False),
    ],
    ids=["gated_aligned", "gated_unaligned", "unknown_shape", "non_gated"],
)
def test_nvfp4_fc1_row_alignment_gate(
    backend_cls, intermediate_size, activation, moe_tp_size, rejected
):
    """A gated NVFP4 FC1 buffer rounded up to the 128-row block-scale tile
    splits gate/up on padding, so those layers must be turned down here.
    """
    verdict = backend_cls.can_implement(
        _nvfp4_problem(intermediate_size, activation), _deployment_at_moe_tp(moe_tp_size)
    )
    if rejected:
        assert not verdict.eligible
        assert verdict.reject_reason is MoERejectReason.SHAPE_UNALIGNED
        assert "moe_expert_parallel_size" in verdict.detail
    else:
        # Other gates may still turn the layer down; this one must not.
        assert verdict.reject_reason is not MoERejectReason.SHAPE_UNALIGNED


@pytest.mark.cpu_only
@pytest.mark.parametrize("moe_tp_size, rejected", [(4, False), (8, True)], ids=["tp4", "tp8"])
def test_w4a16_nvfp4_fc1_row_alignment_gate(moe_tp_size, rejected):
    """W4A16 NVFP4 weights go through the same padded CUTLASS NVFP4 loader, and
    the op splits the dequantized FC1 the same way. Qwen3-30B-A3B shape: 768
    shards to 192 FC1 rows at moe_tp_size=8."""
    problem = dataclasses.replace(_nvfp4_problem(768, "Swiglu"), quant=QuantAlgo.W4A16_NVFP4.value)
    verdict = TrtllmCutlassW4a16Nvfp4Impl.can_implement(problem, _deployment_at_moe_tp(moe_tp_size))
    assert verdict.eligible is not rejected
    if rejected:
        assert verdict.reject_reason is MoERejectReason.SHAPE_UNALIGNED


@pytest.mark.parametrize(
    "backend_cls",
    [TrtllmCutlassNvfp4Impl, TrtllmCutedslGroupedGemmNvfp4Impl, CuteDslFc12FusedMoE],
    ids=["cutlass", "cutedsl", "cutedsl_fc12"],
)
def test_situ_survives_resolution_not_just_construction(backend_cls):
    """A SiTU layer must be admitted by the *resolver*, not only build.

    Every other SiTU test constructs a backend directly and so never consults
    ``activation_support``. Resolution does, and it reads the **class**
    attribute, because it judges candidates before any instance exists. A
    backend that declared its alpha/beta shape per instance instead passed
    every unit test and then resolved away to CUTLASS on real hardware with
    "kernels take no activation alpha, which this layer's SiTu supplies" --
    silently, because Kimi K3 permits degradation for this backend.
    """
    problem = dataclasses.replace(
        _nvfp4_problem(2048, "SiTu"),
        activation_constants=frozenset({"alpha", "beta"}),
    )
    rejection = _reject_unsupported_activation(backend_cls, problem)
    assert rejection is None, f"{backend_cls.__name__} refuses SiTU at resolution: {rejection}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", [100, 103, 107])
def test_cutedsl_nvfp4_situ_admitted_on_blackwell_and_rubin(sm):
    """The NVFP4 grouped-GEMM leaf admits SiTU on SM100/SM103 and on Rubin (SM107).

    Kimi K3 runs NVFP4 SiTU on ``CUTEDSL`` without degradation, so a gate here
    would fail it at construction rather than fall back.
    """
    deps = (MoEDep.CUTEDSL_RUBIN,) if sm == 107 else ()
    deployment = MoEDeployment(
        ep_size=1,
        tp_size=1,
        parallel_size=1,
        use_dp=False,
        num_slots=256,
        env=MoEEnvironment(sm=sm, available_deps=deps),
    )
    verdict = TrtllmCutedslGroupedGemmNvfp4Impl.can_implement(
        _nvfp4_problem(2048, "SiTu"), deployment
    )
    assert verdict.eligible, verdict.detail


def test_cutedsl_fc12_refuses_relu2_at_resolution():
    """Relu2 is not gated; the fused FC12 epilogue would compile it as SwiGLU."""
    rejection = _reject_unsupported_activation(CuteDslFc12FusedMoE, _nvfp4_problem(2048, "Relu2"))
    assert rejection is not None
    assert rejection.reason is MoERejectReason.ACTIVATION_UNSUPPORTED


def test_cutedsl_fc12_outer_runner_unique_id_includes_epilogue():
    """The outer AutoTuner key must not share a tile tactic across epilogues."""

    def _forward(*args, **kwargs):
        raise AssertionError("not launched")

    def _runner(identity):
        return CuteDslFc12FusedMoENvfp4Runner(
            forward_impl=_forward,
            num_experts=8,
            top_k=2,
            num_local_experts=8,
            local_expert_offset=0,
            enable_finalize_fusion=True,
            enable_alltoall=False,
            workload_identity=identity,
        )

    swiglu = _runner((int(ActivationType.Swiglu), -1.0, -1.0))
    situ = _runner((int(ActivationType.SiTu), 4.0, 25.0))
    assert swiglu.unique_id() != situ.unique_id()

    # A non-None workload_identity is a cache key only. It must not open the
    # locality-domain inner-tactic priming branch, which calls forward_impl with
    # overlap_moe_output_memset -- a kwarg the fused single-op impl does not
    # take. _forward raises if reached; no GPU needed, the failure is at kwarg
    # binding.
    assert situ.prime_inner_tactics is False
    inputs = [torch.empty(0) for _ in range(5)]
    assert situ(inputs, tactic=-1, do_preparation=True) is inputs[4]


def test_unresolvable_layer_error_carries_rejection_details():
    """describe() prints reason codes only, so impl_class_for has to add the
    details -- without them a shape rejection reaches the operator as a bare
    ``shape_unaligned`` with no shapes and no way forward.
    """
    report = MoEResolutionReport(
        problem=_nvfp4_problem(2048, "Swiglu"),
        deployment=_deployment_at_moe_tp(64),
        winner=None,
        selected_by="failed",
        rejected=(
            MoERejection(
                "CUTLASS", MoERejectReason.SHAPE_UNALIGNED, "raise moe_expert_parallel_size"
            ),
        ),
        requested="CUTLASS",
    )
    with pytest.raises(ValueError, match="raise moe_expert_parallel_size"):
        impl_class_for(report)


# ============================================================================
# CUTLASS leaves: what run_moe hands the op, on CPU
# ============================================================================
# ``torch.ops.trtllm.fused_moe`` is replaced by a recorder, so a leaf reaching
# for state it does not carry fails here without a GPU, and the kernel flags,
# the weight view and the LoRA operands each leaf passes must follow its own
# declarations, which ``test_moe_impl.test_cutlass_leaf_declarations`` pins.

_CUTLASS_FUSED_MOE_OP_LEAVES = tuple(
    leaf
    for leaf in CUTLASS_LEAVES
    # The W4A16 NVFP4 leaf hands the op dequantized weights instead of its
    # buffers, and the Triton leaf runs another kernel; each has its own test.
    if leaf not in (TrtllmCutlassW4a16Nvfp4Impl, TrtllmTritonFp8BlockScalesImpl)
)

_LORA_SENTINEL = {"fc1_lora_ranks": "fc1-ranks"}


def _cutlass_leaf_on_cpu(leaf_cls):
    """A leaf carrying only the state its ``run_moe`` reads; no ``__init__``."""
    leaf = leaf_cls.__new__(leaf_cls)
    # Mock weights: the op is replaced, and a mock records the per-format
    # ``.view(dtype)`` reinterpretation that a real CPU tensor may refuse.
    leaf.w3_w1_weight = MagicMock(dtype=torch.bfloat16)
    leaf.w2_weight = MagicMock(dtype=torch.bfloat16)
    leaf.w3_w1_bias = None
    leaf.w2_bias = None
    leaf.quant_scales = ()
    leaf.act_alpha = None
    leaf.act_beta = None
    leaf.act_clamp = None
    leaf.act_clamp_after_silu = False
    leaf.tp_size, leaf.tp_rank, leaf.ep_size, leaf.ep_rank = 1, 0, 1, 0
    leaf.cluster_size, leaf.cluster_rank = 1, 0
    leaf.use_fused_finalize = True
    leaf.tune_max_num_tokens = 16
    leaf.activation_type = ActivationType.Swiglu
    leaf.hidden_size = 8
    leaf.unpadded_hidden_size = 8
    leaf.force_dynamic_quantization = False
    # Stubbed on every leaf, so only whether run_moe calls it decides.
    leaf._extract_moe_lora_tensors = lambda lora_params: dict(_LORA_SENTINEL)
    return leaf


def _cpu_run_context(num_tokens: int = 2, lora_params: Optional[dict] = None) -> MoERunContext:
    return MoERunContext(
        token_selected_experts=torch.zeros((num_tokens, 1), dtype=torch.int32),
        token_final_scales=torch.ones((num_tokens, 1), dtype=torch.float32),
        x=torch.zeros((num_tokens, 8), dtype=torch.bfloat16),
        x_sf=None,
        output_dtype=torch.bfloat16,
        lora_params=lora_params,
        comm_plan=MoECommPlan(
            input_sf_swizzled=False,
            enable_alltoall=False,
            moe_output=None,
            payload_in_workspace=False,
        ),
    )


def _record_fused_moe(monkeypatch) -> list:
    """Replace the op with a recorder; returns the keyword sets it saw, each
    with the positional operands under ``"args"``."""
    calls = []

    def fused_moe(*args, **kwargs):
        calls.append(dict(kwargs, args=args))
        return [torch.zeros_like(args[0])]

    monkeypatch.setattr(torch.ops.trtllm, "fused_moe", fused_moe, raising=False)
    return calls


# The op's keyword names, spelled out so that renaming a flag fails here.
_FLAG_KWARGS = (
    "use_deepseek_fp8_block_scale",
    "use_w4_group_scaling",
    "use_int8_woq_per_channel",
    "use_mxfp8_act_scaling",
    "use_mxfp8_weight_scaling",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("leaf_cls", _CUTLASS_FUSED_MOE_OP_LEAVES, ids=lambda cls: cls.__name__)
def test_cutlass_leaf_run_moe_hands_the_op_its_declarations(leaf_cls, monkeypatch):
    calls = _record_fused_moe(monkeypatch)
    leaf = _cutlass_leaf_on_cpu(leaf_cls)
    out = leaf.run_moe(_cpu_run_context(lora_params={"num_seqs": 1}))

    assert len(calls) == 1
    assert out.shape == (2, 8)
    flags = leaf_cls.FUSED_MOE_OP_FLAGS
    assert flags.op_kwargs() == {name: calls[0][name] for name in _FLAG_KWARGS}
    leaf.w3_w1_weight.view.assert_called_once_with(flags.weight_dtype or torch.bfloat16)
    leaf.w2_weight.view.assert_called_once_with(flags.weight_dtype or torch.bfloat16)
    # Only the leaves that fuse LoRA hand lora_params to the op.
    fuses_lora = leaf_cls.capabilities.supports_moe_lora
    assert (calls[0].get("fc1_lora_ranks") == "fc1-ranks") is fuses_lora


@pytest.mark.cpu_only
def test_cutlass_w4a16_nvfp4_runs_the_op_unquantized_on_dequantized_weights(monkeypatch):
    calls = _record_fused_moe(monkeypatch)
    leaf = _cutlass_leaf_on_cpu(TrtllmCutlassW4a16Nvfp4Impl)
    leaf.expert_size_per_partition = 4
    leaf.slot_start = 4  # this rank owns slots 4..7
    w3_w1_hp, w2_hp = torch.zeros(4, 16, 8), torch.zeros(4, 8, 8)
    seen = {}

    def dequant_active_experts_to_hp(module, ids, out_dtype):
        seen["ids"], seen["out_dtype"] = ids, out_dtype
        return w3_w1_hp, w2_hp

    leaf.quant_method = SimpleNamespace(dequant_active_experts_to_hp=dequant_active_experts_to_hp)
    leaf.quant_scales = ("fc31-scale",)
    ctx = dataclasses.replace(
        _cpu_run_context(num_tokens=2),
        token_selected_experts=torch.tensor([[5], [9]], dtype=torch.int32),
        x_sf=torch.ones(1),
        output_dtype=None,
    )
    leaf.run_moe(ctx)

    # The dequant scatter indexes local experts; the op filters on global ids.
    assert seen["ids"].tolist() == [[1], [3]]
    assert seen["out_dtype"] is torch.bfloat16
    args = calls[0]["args"]
    assert args[1] is ctx.token_selected_experts
    assert args[3] is w3_w1_hp and args[5] is w2_hp
    assert calls[0]["quant_scales"] == []
    assert calls[0]["input_sf"] is None
    assert not any(calls[0][name] for name in _FLAG_KWARGS)
    leaf.w3_w1_weight.view.assert_not_called()


@pytest.mark.cpu_only
def test_cutlass_nvfp4_dynamic_fc2_scale_reaches_the_op(monkeypatch):
    calls = _record_fused_moe(monkeypatch)
    leaf = _cutlass_leaf_on_cpu(TrtllmCutlassNvfp4Impl)
    leaf.force_dynamic_quantization = True
    leaf.fc2_weight_scale_2 = torch.ones(1)
    leaf.run_moe(_cpu_run_context())

    assert calls[0]["use_dynamic_fc2_scale"] is True
    assert calls[0]["quant_scales"][-1] is leaf.fc2_weight_scale_2


@pytest.mark.cpu_only
def test_cutlass_w4a16_mxfp4_pads_the_activation_to_its_hidden_size(monkeypatch):
    calls = _record_fused_moe(monkeypatch)
    leaf = _cutlass_leaf_on_cpu(TrtllmCutlassW4a16Mxfp4Impl)
    leaf.hidden_size = 16  # padded at construction; the activation stays 8 wide
    leaf.run_moe(_cpu_run_context())

    x = calls[0]["args"][0]
    assert x.shape == (2, 16)
    assert not x[:, 8:].any()


# Spelled out, so that moving the threshold is a deliberate test change.
_B12X_PREFILL_THRESHOLD = 64


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "leaf_cls, num_tokens, reaches_cutlass",
    [
        pytest.param(CuteDslB12xNvfp4FusedMoE, _B12X_PREFILL_THRESHOLD, True, id="nvfp4-prefill"),
        pytest.param(
            CuteDslB12xNvfp4FusedMoE, _B12X_PREFILL_THRESHOLD - 1, False, id="nvfp4-decode"
        ),
        pytest.param(
            CuteDslB12xW4a16Nvfp4FusedMoE, _B12X_PREFILL_THRESHOLD, False, id="w4a16-prefill"
        ),
    ],
)
def test_b12x_reaches_the_cutlass_op_only_for_an_nvfp4_prefill_chunk(
    leaf_cls, num_tokens, reaches_cutlass, monkeypatch
):
    calls = _record_fused_moe(monkeypatch)
    leaf = _cutlass_leaf_on_cpu(leaf_cls)
    leaf._run_b12x_decode = MagicMock(return_value=torch.zeros(num_tokens, 8))
    out = leaf.run_moe(_cpu_run_context(num_tokens=num_tokens))

    assert len(calls) == int(reaches_cutlass)
    assert leaf._run_b12x_decode.called is not reaches_cutlass
    assert out.shape == (num_tokens, 8)


# =====================================================================
# Multi-chunk forwards: a backend whose run_moe reuses its buffers
# across calls.
# =====================================================================

_MAIN_STREAM = "main"
_AUX_STREAM = "aux"


def _record_streams(monkeypatch) -> list:
    """Replace ``torch.cuda.stream`` with a CPU stand-in; the returned stack's
    top is the stream work is being issued on."""
    stack = [_MAIN_STREAM]

    @contextmanager
    def stream(s):
        stack.append(s)
        try:
            yield
        finally:
            stack.pop()

    monkeypatch.setattr(torch.cuda, "stream", stream)
    return stack


class _OneBufferB12xWrapper:
    """``B12xMoEWrapper`` as built with ``use_cuda_graph=True``: every call
    writes into one preallocated buffer and returns its leading rows, so
    consecutive calls hand back the same storage."""

    def __init__(self, max_num_tokens: int, hidden_size: int, stream_stack: list) -> None:
        self._moe_output = torch.empty((max_num_tokens, hidden_size), dtype=torch.bfloat16)
        self._stream_stack = stream_stack
        self.streams = []

    def run(self, *, x, token_selected_experts, token_final_scales, **weights) -> torch.Tensor:
        self.streams.append(self._stream_stack[-1])
        out = self._moe_output[: x.shape[0]]
        out.copy_(x * 2)  # any per-token function of the input
        return out


class _FreshOutputBackend:
    """A backend whose ``run_moe`` allocates its output, as the CUTLASS op does."""

    capabilities = MoEStaticCapability()
    input_requirement = MoEInputRequirement()

    def __init__(self, stream_stack: list) -> None:
        self._stream_stack = stream_stack
        self.streams = []

    def _supports_load_balancer(self) -> bool:
        return True

    def try_fused_route_quant(self, x, router_logits):
        return None

    def quantize_input(self, x, post_quant_comm=True):
        return x, None

    def run_moe(self, ctx: MoERunContext, *, workspace=None) -> torch.Tensor:
        self.streams.append(self._stream_stack[-1])
        return ctx.x * 2


def _chunking_moe(backend, num_chunks: int) -> SimpleNamespace:
    """What ``ExternalCommMoEScheduler`` reads off a single-rank layer with no
    communication strategy and no EPLB, whose ``moe_max_num_tokens`` is below
    the batch so it has an aux stream for chunk overlap."""

    def no_op(*args, **kwargs):
        return None

    def route(router_logits, input_ids=None):
        num_tokens = router_logits.shape[0]
        return (
            torch.zeros((num_tokens, 1), dtype=torch.int32),
            torch.ones((num_tokens, 1), dtype=torch.float32),
        )

    return SimpleNamespace(
        backend=backend,
        comm=None,
        use_dp=False,
        enable_alltoall=False,
        parallel_size=1,
        reduce_results=False,
        apply_router_weight_on_input=False,
        layer_idx=0,
        num_slots=4,
        repeat_idx=0,
        repeat_count=1,
        layer_load_balancer=None,
        aux_stream=_AUX_STREAM,
        event_dict=collections.defaultdict(Mock),
        routing_method=SimpleNamespace(
            experts_per_token=1, requires_separated_routing=True, apply=route
        ),
        calculate_num_chunks=lambda all_rank_num_tokens: num_chunks,
        split_chunk=lambda n, k: [n // k + (i < n % k) for i in range(k)],
        determine_communication_method=no_op,
        _using_load_balancer=lambda: False,
        _load_balancer_start_wait_gpu_stage=no_op,
        _load_balancer_start_set_cpu_stage=no_op,
        _load_balancer_done_set_cpu_stage=no_op,
        _maybe_get_perfect_router_expert_ids=lambda token_selected_experts: token_selected_experts,
    )


def _forward_in_two_chunks(backend) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eight tokens through the external-comm scheduler, split into two chunks
    of four. Integer-valued, so the expected ``x * 2`` is exact in bfloat16."""
    x = torch.arange(8 * 8, dtype=torch.bfloat16).reshape(8, 8)
    out = ExternalCommMoEScheduler(_chunking_moe(backend, num_chunks=2)).forward(
        x,
        torch.zeros((8, 4)),
        do_finalize=True,
        output_dtype=torch.bfloat16,
        all_rank_num_tokens=None,
        use_dp_padding=None,
    )
    return x, out


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "leaf_cls", [CuteDslB12xNvfp4FusedMoE, CuteDslB12xW4a16Nvfp4FusedMoE], ids=["nvfp4", "w4a16"]
)
def test_b12x_multi_chunk_forward_keeps_every_chunk(leaf_cls, monkeypatch):
    """Every b12x call returns the same storage, so a two-chunk forward has to
    copy the first chunk out before the second overwrites it. The two calls
    also share the wrapper's workspace, so they must not be in flight on two
    streams at once. Chunks of four tokens stay below the NVFP4 leaf's
    CUTLASS threshold, so both leaves run the b12x kernel."""
    stream_stack = _record_streams(monkeypatch)
    leaf = leaf_cls.__new__(leaf_cls)
    leaf.fc31_act_scale = None
    leaf.b12x_wrapper = _OneBufferB12xWrapper(4, 8, stream_stack)
    leaf._b12x_weights = {}

    x, out = _forward_in_two_chunks(leaf)

    torch.testing.assert_close(out, x * 2, rtol=0, atol=0)
    assert leaf.b12x_wrapper.streams == [_MAIN_STREAM, _MAIN_STREAM]


@pytest.mark.cpu_only
def test_multi_chunk_forward_overlaps_chunks_of_a_backend_that_allocates_its_output(monkeypatch):
    """The serialization is the declaration's doing: a backend that does not
    declare ``run_moe_reuses_buffers`` keeps the aux-stream overlap."""
    backend = _FreshOutputBackend(_record_streams(monkeypatch))

    x, out = _forward_in_two_chunks(backend)

    torch.testing.assert_close(out, x * 2, rtol=0, atol=0)
    assert backend.streams == [_AUX_STREAM, _MAIN_STREAM]


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_alltoall", [False, True], ids=["allgather", "alltoall"])
def test_triton_fp8_block_scales_leaf_localizes_global_slot_ids(enable_alltoall, monkeypatch):
    """Dispatch keeps the global slot numbering with or without alltoall, and
    marks padding with -1 (num_slots under DeepEP low latency); the Triton
    kernel indexes this rank's experts, so every slot outside them weighs 0."""
    from tensorrt_llm._torch.moe.fused_moe.cutlass import fp8_block_scales_triton

    seen = {}

    def run_triton(x, ids, scales, *args, **kwargs):
        seen["ids"], seen["scales"] = ids, scales
        return torch.zeros_like(x)

    monkeypatch.setattr(fp8_block_scales_triton, "run_triton_fp8_block_scale_moe", run_triton)
    leaf = TrtllmTritonFp8BlockScalesImpl.__new__(TrtllmTritonFp8BlockScalesImpl)
    leaf.expert_size_per_partition = 4
    leaf.slot_start = 4  # this rank owns slots 4..7
    leaf.w3_w1_weight = leaf.w2_weight = None
    leaf.quant_scales = SimpleNamespace(fc_weight_scales=None, proj_weight_scales=None)
    leaf.activation_type = ActivationType.Swiglu
    leaf.act_clamp = None
    ctx = dataclasses.replace(
        _cpu_run_context(num_tokens=2),
        # 1 is below this rank's slots, 8 (num_slots) above them.
        token_selected_experts=torch.tensor([[5, 1], [-1, 8]], dtype=torch.int32),
        token_final_scales=torch.full((2, 2), 0.5),
        comm_plan=MoECommPlan(
            input_sf_swizzled=False,
            enable_alltoall=enable_alltoall,
            moe_output=None,
            payload_in_workspace=False,
        ),
    )
    leaf.run_moe(ctx)

    assert seen["ids"].tolist() == [[1, 0], [0, 3]]
    assert seen["scales"].tolist() == [[0.5, 0.0], [0.0, 0.0]]
    # The kernel returns its own tensor, so the scheduler must not hand it the
    # alltoall workspace to fill.
    assert not leaf.supports_moe_output_in_alltoall_workspace()


@pytest.mark.cpu_only
def test_configurable_moe_exposes_the_backend_lora_state():
    """``MoE.forward``'s torch.compile guard reads the layer, so the wrapper has
    to answer for the backend that holds the state."""
    moe = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(moe)
    moe.backend = SimpleNamespace(_moe_lora_enabled=True)
    assert getattr(moe, "_moe_lora_enabled", False) is True
    moe.backend = SimpleNamespace()
    assert getattr(moe, "_moe_lora_enabled", False) is False
