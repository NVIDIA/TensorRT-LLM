# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic warmup respects an encoder-only cache's actual request phases."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


def _warmup_fixture(context_only: bool, available: int = 64):
    manager = SimpleNamespace(
        context_swa_layer_limit=20 if context_only else None,
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
    engine = SimpleNamespace(
        _metrics={},
        maybe_autotune_lora=nullcontext,
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        _get_draft_kv_cache_manager=lambda _: None,
        max_num_tokens=64,
        max_seq_len=17,
        batch_size=8,
        max_total_draft_tokens=0,
        spec_config=None,
        max_draft_len=0,
        max_draft_loop_tokens=0,
        max_beam_width=1,
        use_mrope=False,
    )
    return engine, manager, resource_manager


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("tokens,generation_requests", [(1, 1), (8, 8), (2, 1), (33, 1), (64, 0)])
def test_warmup_preserves_token_and_batch_bounds_with_context_cache(
    context_only, tokens, generation_requests
):
    engine, manager, resources = _warmup_fixture(context_only)
    batch = PyTorchModelEngine._create_warmup_request(
        engine, resources, tokens, generation_requests
    )
    assert batch is not None
    contexts, generations = batch.context_requests, batch.generation_requests
    assert sum(request.context_chunk_size for request in contexts) + len(generations) == tokens
    assert len(contexts) + len(generations) <= engine.batch_size
    assert all(request.context_chunk_size <= engine.max_seq_len - 1 for request in contexts)
    assert all(request.state == LlmRequestState.CONTEXT_INIT for request in contexts)
    assert all(request.is_dummy_request for request in contexts + generations)
    assert all(request.context_current_position == 0 for request in contexts)
    assert all(request.context_chunk_size == request.prompt_len for request in contexts)
    if context_only:
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


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("tokens,generations,available", [(65, 1, 100), (8, 9, 100), (8, 8, 7)])
def test_context_warmup_preserves_admission_failures(context_only, tokens, generations, available):
    engine, manager, resources = _warmup_fixture(context_only, available=available)
    assert PyTorchModelEngine._create_warmup_request(engine, resources, tokens, generations) is None
    manager.add_dummy_requests.assert_not_called()


@pytest.mark.parametrize("warmup_only", [False, True])
@pytest.mark.parametrize(
    "prefill_backend", [PrefillCudaGraphBackend.DISABLED, PrefillCudaGraphBackend.PIECEWISE]
)
def test_context_only_skips_decode_graphs_and_keeps_prefill_capture(warmup_only, prefill_backend):
    engine, _, resources = _warmup_fixture(True)
    engine.cuda_graph_runner = SimpleNamespace(enabled=True, is_warmup_only=warmup_only)
    engine.prefill_cuda_graph_backend = prefill_backend
    engine._capture_prefill_cuda_graphs = Mock()
    engine._capture_generation_cuda_graphs = Mock()
    engine._capture_mixed_encoder_decoder_cuda_graphs = Mock()
    PyTorchModelEngine._run_cuda_graph_warmup(engine, resources)
    engine._capture_generation_cuda_graphs.assert_not_called()
    engine._capture_mixed_encoder_decoder_cuda_graphs.assert_not_called()
    if warmup_only:
        engine._capture_prefill_cuda_graphs.assert_not_called()
    else:
        engine._capture_prefill_cuda_graphs.assert_called_once_with(resources)
    assert PyTorchModelEngine._create_cuda_graph_warmup_request(engine, resources, 8, 0) is None


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize(
    "tokens,batch_size,expected",
    [(5, 2, [3, 2]), (7, 3, [3, 2, 2]), (9, 8, [2] + [1] * 7)],
)
def test_many_context_warmup_keeps_each_chunk_inside_sequence_limit(
    context_only, tokens, batch_size, expected
):
    engine, _, resources = _warmup_fixture(context_only)
    engine.batch_size = batch_size
    engine.max_seq_len = 4
    batch = PyTorchModelEngine._create_warmup_request(
        engine, resources, tokens, 1 if context_only else 0, least_requests=False
    )
    assert batch is not None
    assert not batch.generation_requests
    assert [request.context_chunk_size for request in batch.context_requests] == expected
    assert sum(expected) == tokens
    assert max(expected) <= engine.max_seq_len - 1
    assert len(expected) == min(tokens, engine.batch_size)
