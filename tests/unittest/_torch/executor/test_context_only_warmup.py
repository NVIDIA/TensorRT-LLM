# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic warmup respects an encoder-only cache's actual request phases."""

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _warmup_fixture(context_only: bool, available: int = 64):
    manager = SimpleNamespace(
        context_swa_layer_limit=20 if context_only else None,
        tokens_per_block=128,
        num_extra_kv_tokens=0,
        _has_cp_helix=False,
        _stream=SimpleNamespace(cuda_stream=0),
        _create_kv_cache=Mock(
            side_effect=lambda *_args, **_kwargs: Mock(
                num_committed_tokens=0,
                capacity=0,
                resume=Mock(return_value=True),
                resize=Mock(return_value=True),
            )
        ),
        _ensure_generation_beam_width=Mock(return_value=True),
        get_num_available_tokens=lambda **kwargs: min(available, kwargs["token_num_upper_bound"]),
        get_num_free_blocks=lambda: 64,
    )

    def add_dummy_requests(request_ids, **kwargs):
        # Exercise production request initialization and allocation control flow
        # with CPU stand-ins for the GPU cache objects.
        return KVCacheManagerV2.add_dummy_requests(manager, request_ids, **kwargs)

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
        mapping=Mapping(),
        model=None,
        sparse_attention_config=None,
        metrics={},
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


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("tokens,generation_requests", [(1, 1), (8, 8), (2, 1), (33, 1), (64, 0)])
def test_warmup_preserves_token_and_batch_bounds_with_context_cache(
    context_only, tokens, generation_requests
):
    requests, manager, resources = _warmup_fixture(context_only)
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
    requests, manager, resources = _warmup_fixture(context_only, available=available)
    assert requests.create_warmup_request(resources, tokens, generations) is None
    manager.add_dummy_requests.assert_not_called()


@pytest.mark.parametrize("warmup_only", [False, True])
@pytest.mark.parametrize(
    "prefill_backend", [PrefillCudaGraphBackend.DISABLED, PrefillCudaGraphBackend.PIECEWISE]
)
def test_context_only_skips_decode_graphs_and_keeps_prefill_capture(warmup_only, prefill_backend):
    requests, _, resources = _warmup_fixture(True)
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


@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize(
    "tokens,batch_size,expected",
    [(5, 2, [3, 2]), (7, 3, [3, 2, 2]), (9, 8, [2] + [1] * 7)],
)
def test_many_context_warmup_keeps_each_chunk_inside_sequence_limit(
    context_only, tokens, batch_size, expected
):
    requests, _, resources = _warmup_fixture(context_only)
    config = requests._config
    config.max_batch_size = batch_size
    config.max_seq_len = 4
    batch = requests.create_warmup_request(
        resources, tokens, 1 if context_only else 0, least_requests=False
    )
    assert batch is not None
    assert not batch.generation_requests
    assert [request.context_chunk_size for request in batch.context_requests] == expected
    assert sum(expected) == tokens
    assert max(expected) <= config.max_seq_len - 1
    assert len(expected) == min(tokens, config.max_batch_size)
