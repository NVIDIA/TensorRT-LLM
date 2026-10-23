# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic warmup respects a context-only cache's supported request phases."""

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


def _warmup_fixture(
    context_only: bool, available: int = 64, *, prefill_only: bool = False
) -> tuple[DummyRequestFactory, SimpleNamespace, SimpleNamespace]:
    manager = SimpleNamespace(
        context_swa_layer_limit=20 if context_only else None,
        supports_generation=not prefill_only,
        tokens_per_block=128,
        num_extra_kv_tokens=0,
        _has_cp_helix=False,
        get_num_available_tokens=lambda **kwargs: min(available, kwargs["token_num_upper_bound"]),
        get_num_free_blocks=lambda: 64,
    )

    def add_dummy_requests(request_ids, **kwargs):
        # Use production request construction/state initialization. Only the
        # GPU cache allocation is excluded from these CPU phase-contract tests.
        return KVCacheManagerV2.add_dummy_requests(
            manager, request_ids, prepare_resource=False, **kwargs
        )

    manager.add_dummy_requests = Mock(side_effect=add_dummy_requests)
    resource_manager = SimpleNamespace(
        get_resource_manager=lambda key: (
            manager if key == ResourceManagerType.KV_CACHE_MANAGER else None
        )
    )
    requests = DummyRequestFactory(
        config=SimpleNamespace(
            max_num_tokens=64,
            max_seq_len=17,
            max_batch_size=8,
            max_total_draft_tokens=0,
            spec_config=None,
            max_draft_len=0,
            max_draft_loop_tokens=0,
            max_beam_width=1,
            use_mrope=False,
            enable_autotuner=False,
        ),
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        model=SimpleNamespace(model_config=SimpleNamespace(pretrained_config=SimpleNamespace())),
        get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
        release_request_state=None,
    )
    return requests, manager, resource_manager


def _cuda_graphs(requests: DummyRequestFactory, warmup_only: bool) -> DecoderCudaGraphs:
    return DecoderCudaGraphs(
        cuda_graph_runner=SimpleNamespace(enabled=True, is_warmup_only=warmup_only),
        breakable_cuda_graph_runner=None,
        config=requests._config,
        mapping=None,
        model=None,
        sparse_attention_config=None,
        metrics={},
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        dummy_requests=requests,
        agreement=None,
        metadata=None,
        get_runtime_tokens_per_gen_step=requests.get_runtime_tokens_per_gen_step,
        dynamic_draft_len_mapping=None,
        lora=SimpleNamespace(cuda_graph_manager=None),
        forward_warmup=None,
        torch_compile_bypass=contextlib.nullcontext,
        context_promotion_supported=True,
    )


@pytest.mark.parametrize(
    "context_only,prefill_only", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("tokens,generation_requests", [(1, 1), (8, 8), (2, 1), (33, 1), (64, 0)])
def test_warmup_preserves_token_and_batch_bounds_with_context_cache(
    context_only: bool, prefill_only: bool, tokens: int, generation_requests: int
) -> None:
    requests, manager, resources = _warmup_fixture(context_only, prefill_only=prefill_only)
    batch = requests.create_warmup_request(resources, tokens, generation_requests)
    assert batch is not None
    config = requests._config
    contexts, generations = batch.context_requests, batch.generation_requests
    assert sum(request.context_chunk_size for request in contexts) + len(generations) == tokens
    assert len(contexts) + len(generations) <= config.max_batch_size
    assert all(request.context_chunk_size <= config.max_seq_len - 1 for request in contexts)
    assert all(request.state == LlmRequestState.CONTEXT_INIT for request in contexts)
    assert all(request.is_dummy_request for request in contexts + generations)
    assert all(request.context_current_position == 0 for request in contexts)
    assert all(request.context_chunk_size == request.prompt_len for request in contexts)
    if context_only or prefill_only:
        assert not generations
        if tokens == generation_requests:
            assert len(contexts) == generation_requests
            assert all(request.context_chunk_size == 1 for request in contexts)
        assert all(not call.kwargs["is_gen"] for call in manager.add_dummy_requests.call_args_list)
    else:
        assert len(generations) == generation_requests
        assert all(
            request.state == LlmRequestState.GENERATION_IN_PROGRESS for request in generations
        )
    if generation_requests == 0:
        # Long pure-prefill warmup keeps the original fewest-requests packing.
        assert [request.context_chunk_size for request in contexts] == [16] * 4


@pytest.mark.parametrize(
    "context_only,prefill_only", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("tokens,generations,available", [(65, 1, 100), (8, 9, 100), (8, 8, 7)])
def test_context_warmup_preserves_admission_failures(
    context_only: bool, prefill_only: bool, tokens: int, generations: int, available: int
) -> None:
    requests, manager, resources = _warmup_fixture(
        context_only, available=available, prefill_only=prefill_only
    )
    assert requests.create_warmup_request(resources, tokens, generations) is None
    manager.add_dummy_requests.assert_not_called()


@pytest.mark.parametrize("warmup_only", [False, True])
@pytest.mark.parametrize("prefill_only", [False, True])
@pytest.mark.parametrize(
    "prefill_backend", [PrefillCudaGraphBackend.DISABLED, PrefillCudaGraphBackend.PIECEWISE]
)
def test_context_only_skips_decode_graphs_and_keeps_prefill_capture(
    warmup_only: bool, prefill_only: bool, prefill_backend: PrefillCudaGraphBackend
) -> None:
    requests, _, resources = _warmup_fixture(not prefill_only, prefill_only=prefill_only)
    requests._config.prefill_cuda_graph_backend = prefill_backend
    graphs = _cuda_graphs(requests, warmup_only)
    with (
        patch.object(graphs, "_capture_prefill_cuda_graphs") as prefill,
        patch.object(graphs, "_capture_generation_cuda_graphs") as generation,
        patch.object(graphs, "_capture_additional_cuda_graphs") as additional,
    ):
        graphs.run_cuda_graph_warmup(resources)
    generation.assert_not_called()
    additional.assert_not_called()
    if warmup_only:
        prefill.assert_not_called()
    else:
        prefill.assert_called_once_with(resources)
    assert requests.create_cuda_graph_warmup_request(resources, 8, 0) is None


@pytest.mark.parametrize(
    "context_only,prefill_only", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize(
    "tokens,batch_size,expected",
    [(5, 2, [3, 2]), (7, 3, [3, 2, 2]), (9, 8, [2] + [1] * 7)],
)
def test_many_context_warmup_keeps_each_chunk_inside_sequence_limit(
    context_only: bool, prefill_only: bool, tokens: int, batch_size: int, expected: list[int]
) -> None:
    requests, _, resources = _warmup_fixture(context_only, prefill_only=prefill_only)
    config = requests._config
    config.max_batch_size = batch_size
    config.max_seq_len = 4
    batch = requests.create_warmup_request(
        resources, tokens, 1 if context_only or prefill_only else 0, least_requests=False
    )
    assert batch is not None
    assert not batch.generation_requests
    assert [request.context_chunk_size for request in batch.context_requests] == expected
    assert sum(expected) == tokens
    assert max(expected) <= config.max_seq_len - 1
    assert len(expected) == min(tokens, config.max_batch_size)


@pytest.mark.parametrize(
    "context_only,prefill_only", [(False, False), (True, False), (False, True)]
)
def test_context_only_avoids_generation_padding_and_promotion(
    context_only: bool, prefill_only: bool
) -> None:
    requests, manager, resources = _warmup_fixture(context_only, prefill_only=prefill_only)
    graphs = _cuda_graphs(requests, warmup_only=False)
    graphs.cuda_graph_runner.preallocate_padding_dummies = Mock()
    graphs.preallocate_padding_dummies(resources)
    if not context_only and not prefill_only:
        graphs.cuda_graph_runner.preallocate_padding_dummies.assert_called_once_with(resources)
        return
    graphs.cuda_graph_runner.preallocate_padding_dummies.assert_not_called()
    batch = SimpleNamespace(num_context_requests=1, attention_dp_phase=None)
    promoted, request_ids = graphs.promote_final_contexts(
        batch, manager, enable_spec_decode=False, runtime_draft_len=0, ced_enabled=False
    )
    assert promoted is batch
    assert not request_ids


@pytest.mark.parametrize("local_layers", [0, 1])
def test_prefill_only_mamba_omits_speculative_state_buffers(local_layers: int) -> None:
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.supports_generation = False
    manager.local_num_mamba_layers = local_layers
    manager._use_replay_state_update = False
    manager._mamba_ssm_stochastic_rounding = False
    spec_config = SimpleNamespace(tokens_per_gen_step=4)
    with (
        patch("torch.zeros", side_effect=AssertionError("unexpected state allocation")),
        patch("torch.arange", side_effect=AssertionError("unexpected state allocation")),
    ):
        manager._setup_mtp_intermediate_states(spec_config, max_batch_size=8)
        assert not manager._allocate_pool_replay_buffers(spec_config, cache_size=8, device=None)
    assert manager.intermediate_ssm_states is None
    assert manager.intermediate_conv_states is None
    assert manager.intermediate_state_indices is None
    assert manager.prev_num_accepted_tokens is None
    assert manager.old_x is None
    assert manager.old_B is None
    assert manager.kda_k_cache is None
    assert manager.kda_g_cache is None
