# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import random
import unittest
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Literal, Optional, Union
from unittest import mock

import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import (decoder_config,
                                                            input_preparer)
from _torch.pyexecutor.engine_test_utils import DummyModel
from parameterized import parameterized
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.attention.backends import (FlashInferAttention,
                                                    FlashInferAttentionMetadata)
from tensorrt_llm._torch.attention.backends import \
    flashinfer as flashinfer_backend
from tensorrt_llm._torch.attention.backends.flashinfer import (
    FlashInferWrappers, PlanParams)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs, PredefinedAttentionMask)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor import resource_manager
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import \
    DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import \
    DecoderInputBuffers
from tensorrt_llm._torch.pyexecutor.executor_request_queue import \
    RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import \
    KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.request_utils import merge_helix_requests
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.scheduler import (BindCapacityScheduler,
                                                      BindMicroBatchScheduler,
                                                      ScheduledRequests,
                                                      SimpleScheduler)
from tensorrt_llm._torch.utils import helix_local_len, model_extra_attrs
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.executor import SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX


@dataclass(kw_only=True, frozen=True)
class _HelixCase:
    rank: int
    position: int
    capture_graph: bool = False
    cached_tokens: Optional[int] = None
    query_length: int = 1
    kv_layout: Literal["HND", "NHD"] = "HND"
    empty_cache: bool = False
    speculative: bool = False
    batch_size: int = 1


@skip_pre_blackwell
@pytest.mark.parametrize("scenario", [
    pytest.param(_HelixCase(rank=0, position=64), id="active"),
    pytest.param(_HelixCase(rank=1, position=64), id="inactive"),
    pytest.param(_HelixCase(rank=1, position=0), id="empty"),
    pytest.param(_HelixCase(rank=0, position=64, capture_graph=True),
                 id="replay-active"),
    pytest.param(_HelixCase(
        rank=1, position=127, capture_graph=True, cached_tokens=31),
                 id="replay-owner-transition"),
    pytest.param(_HelixCase(rank=0, position=64, kv_layout="NHD", batch_size=2),
                 id="nhd-strided"),
    pytest.param(_HelixCase(rank=1, position=0, empty_cache=True),
                 id="no-pages"),
    pytest.param(_HelixCase(rank=0, position=31, query_length=3),
                 id="grouped-query-unsupported"),
    pytest.param(_HelixCase(rank=0, position=64, speculative=True),
                 id="speculative-single-query-unsupported"),
])
@torch.inference_mode()
def test_helix_generation_statistics(scenario: _HelixCase) -> None:
    rank, position = scenario.rank, scenario.position
    capture_graph, cached_tokens = scenario.capture_graph, scenario.cached_tokens
    query_length, kv_layout = scenario.query_length, scenario.kv_layout
    batch_size = scenario.batch_size
    num_rows = batch_size * query_length
    request_ids = list(range(batch_size))
    generator = torch.Generator(device="cuda").manual_seed(42)
    num_heads, num_kv_heads, head_dim, page_size = 4, 2, 128, 32
    mapping = Mapping(world_size=2,
                      cp_size=2,
                      rank=rank,
                      cp_config={"cp_type": CpType.HELIX})
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=256,
                      enable_block_reuse=False,
                      host_cache_size=0),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=256,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.BF16,
        spec_config=(MTPDecodingConfig(max_draft_len=max(query_length - 1, 1))
                     if query_length > 1 or scenario.speculative else None),
    )
    try:
        cache_tokens = (0 if scenario.empty_cache else
                        (1 if position == 0 else 128))
        requests = manager.add_dummy_requests(request_ids,
                                              [cache_tokens] * batch_size,
                                              is_gen=not scenario.empty_cache)
        assert requests is not None
        cache = manager.get_buffers(0, kv_layout=kv_layout)
        if kv_layout == "NHD":
            cache = cache.transpose(2, 3)
        cache.normal_(std=0.25, generator=generator)
        page_ids = manager.get_batch_cache_indices(request_ids)
        qkv = torch.randn((num_rows, (num_heads + 2 * num_kv_heads) * head_dim),
                          device="cuda",
                          dtype=torch.bfloat16,
                          generator=generator)
        query, key, value = qkv.split([
            num_heads * head_dim, num_kv_heads * head_dim,
            num_kv_heads * head_dim
        ],
                                      dim=-1)
        if batch_size > 1:
            assert not key.is_contiguous()
            assert not value.is_contiguous()
            assert not torch.equal(key[0], key[1])
            assert not torch.equal(value[0], value[1])
        expected_cache = cache.clone()
        local_before = helix_local_len(position, page_size, 2, rank)
        if cached_tokens is not None:
            local_before = cached_tokens
        positions = list(range(position, position + query_length))
        ownership = [(token_position // page_size) % 2 == rank
                     for token_position in positions]
        visible_lengths = []
        for request_index, request_pages in enumerate(page_ids):
            count = local_before
            for token_index, owns_token in enumerate(ownership):
                row = request_index * query_length + token_index
                if owns_token:
                    page = request_pages[count // page_size]
                    slot = count % page_size
                    expected_cache[page, 0, :, slot].copy_(key[row].view(
                        num_kv_heads, head_dim))
                    expected_cache[page, 1, :, slot].copy_(value[row].view(
                        num_kv_heads, head_dim))
                    count += 1
                visible_lengths.append(count)

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.full((batch_size, ), query_length,
                                dtype=torch.int32),
            num_contexts=0,
            request_ids=request_ids,
            max_num_requests=batch_size,
            max_num_tokens=num_rows,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[local_before] * batch_size),
            mapping=mapping,
            kv_layout=kv_layout,
        )
        if capture_graph:
            metadata = metadata.create_cuda_graph_metadata(batch_size)
        metadata.update_helix_param(
            [position] * batch_size, [not any(ownership)] * batch_size,
            helix_owned_new_tokens=([0] *
                                    batch_size if query_length > 1 else None))
        with model_extra_attrs({}):
            if query_length > 1 or scenario.speculative:
                with pytest.raises(NotImplementedError,
                                   match="single-token queries"):
                    metadata.prepare()
                return
            metadata.prepare()
        if scenario.empty_cache:
            assert torch.all(metadata.positions < 0)
        attention = FlashInferAttention(layer_idx=0,
                                        num_heads=num_heads,
                                        head_dim=head_dim,
                                        num_kv_heads=num_kv_heads)
        statistics = torch.full((num_rows, num_heads, 2),
                                float("nan"),
                                device="cuda")
        forward_args = AttentionForwardArgs(softmax_stats_tensor=statistics)
        output = attention.forward(query,
                                   key,
                                   value,
                                   metadata,
                                   forward_args=forward_args)
        if capture_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = attention.forward(query,
                                           key,
                                           value,
                                           metadata,
                                           forward_args=forward_args)
            query.add_(0.125)
            next_position = position + query_length
            owns_next = (next_position // page_size) % 2 == rank
            count = local_before + sum(ownership)
            metadata.kv_cache_params.num_cached_tokens_per_seq = [count
                                                                  ] * batch_size
            metadata.update_helix_param([next_position] * batch_size,
                                        [not owns_next] * batch_size)
            with model_extra_attrs({}):
                metadata.prepare()
            if owns_next:
                for request_index, request_pages in enumerate(page_ids):
                    page = request_pages[count // page_size]
                    slot = count % page_size
                    expected_cache[page, 0, :,
                                   slot].copy_(key[request_index].view(
                                       num_kv_heads, head_dim))
                    expected_cache[page, 1, :,
                                   slot].copy_(value[request_index].view(
                                       num_kv_heads, head_dim))
                count += 1
            visible_lengths = [count] * batch_size
            graph.replay()

        torch.testing.assert_close(cache.view(torch.uint8),
                                   expected_cache.view(torch.uint8),
                                   atol=0,
                                   rtol=0)
        expected_outputs = []
        expected_normalizers = []
        for token_index, visible in enumerate(visible_lengths):
            request_pages = page_ids[token_index // query_length]
            if visible:
                keys = torch.cat([
                    expected_cache[page, 0].transpose(0, 1)
                    for page in request_pages
                ])[:visible].float()
                values = torch.cat([
                    expected_cache[page, 1].transpose(0, 1)
                    for page in request_pages
                ])[:visible].float()
                keys = keys.repeat_interleave(num_heads // num_kv_heads, dim=1)
                values = values.repeat_interleave(num_heads // num_kv_heads,
                                                  dim=1)
                token_query = query[token_index].view(1, num_heads,
                                                      head_dim).float()
                scores = torch.einsum("qhd,khd->qhk", token_query,
                                      keys) * head_dim**-0.5
                expected_outputs.append(
                    torch.einsum("qhk,khd->qhd", scores.softmax(-1),
                                 values).reshape(1, -1))
                expected_normalizers.append(scores.logsumexp(-1))
            else:
                expected_outputs.append(
                    torch.zeros((1, num_heads * head_dim), device="cuda"))
                expected_normalizers.append(
                    torch.full((1, num_heads), float("-inf"), device="cuda"))
        expected_output = torch.cat(expected_outputs)
        expected_lse = torch.cat(expected_normalizers)
        torch.testing.assert_close(output.float(),
                                   expected_output,
                                   atol=1e-2,
                                   rtol=1e-2)
        actual_lse = statistics[..., 0] + statistics[..., 1].log()
        torch.testing.assert_close(actual_lse,
                                   expected_lse,
                                   atol=1e-4,
                                   rtol=1e-4)
        assert {
            wrappers.decode_wrapper._backend
            for wrappers in metadata._plan_params_to_wrappers.values()
        } == {"fa2"}
    finally:
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize(
    "window_size,capture_graph,manager_version,page_size", [
        pytest.param(32, False, 2, 32, id="v2-eager"),
        pytest.param(47, True, 2, 32, id="v2-replay"),
        pytest.param(32, True, 1, 32, id="v1-prompt-decode-replay"),
        pytest.param(4, True, 2, 1, id="short-row-mask-replay"),
    ])
@torch.inference_mode()
def test_helix_global_window(window_size: int, capture_graph: bool,
                             manager_version: int, page_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    num_heads, num_kv_heads, head_dim = 4, 2, 128
    query_shift = 0.0 if page_size == 1 else 0.125
    positions = ([32, 63, 78] if manager_version == 1 else [7] *
                 32 if page_size == 1 else [63, 96])
    prompt_lens = [32, 32, 47] if manager_version == 1 else [0] * len(positions)
    batch_size = len(positions)
    query = torch.randn((batch_size, num_heads * head_dim),
                        dtype=torch.bfloat16,
                        device="cuda",
                        generator=generator)
    if not capture_graph or page_size == 1:
        query.zero_()
    keys = torch.randn((batch_size, 128, num_kv_heads, head_dim),
                       dtype=torch.bfloat16,
                       device="cuda",
                       generator=generator) * 0.25
    values = torch.randn_like(keys)
    partials = []
    normalizers = []
    full_partials = []
    full_normalizers = []
    for rank in range(2):
        mapping = Mapping(world_size=2,
                          cp_size=2,
                          rank=rank,
                          cp_config={"cp_type": CpType.HELIX})
        manager_class = KVCacheManager if manager_version == 1 else KVCacheManagerV2
        manager = manager_class(
            KvCacheConfig(max_tokens=max(1024, 128 * batch_size),
                          enable_block_reuse=False,
                          host_cache_size=0),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_seq_len=256,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        try:

            def owns_token(request_index: int, token_position: int) -> bool:
                if manager_version == 1 and token_position >= prompt_lens[
                        request_index]:
                    token_position -= prompt_lens[request_index]
                return (token_position // page_size) % 2 == rank

            generation_requests = manager.add_dummy_requests(list(
                range(batch_size)), [128] * batch_size,
                                                             is_gen=True)
            assert generation_requests is not None
            cache = manager.get_buffers(0, kv_layout="HND")
            cache.zero_()
            page_ids = manager.get_batch_cache_indices(list(range(batch_size)))
            for request_index, position in enumerate(positions):
                local_index = 0
                for global_index in range(position):
                    if not owns_token(request_index, global_index):
                        continue
                    page = page_ids[request_index][local_index // page_size]
                    slot = local_index % page_size
                    cache[page, 0, :, slot].copy_(keys[request_index,
                                                       global_index])
                    cache[page, 1, :, slot].copy_(values[request_index,
                                                         global_index])
                    local_index += 1
            local_before = [
                sum(
                    owns_token(request_index, token_position)
                    for token_position in range(position))
                for request_index, position in enumerate(positions)
            ]
            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.ones(batch_size, dtype=torch.int32),
                num_contexts=0,
                request_ids=list(range(batch_size)),
                max_num_requests=batch_size,
                max_num_tokens=batch_size,
                kv_cache_manager=manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=local_before),
                mapping=mapping,
            )
            if capture_graph:
                metadata = metadata.create_cuda_graph_metadata(batch_size)
            scheduled_requests = ScheduledRequests()
            scheduled_requests.context_requests_last_chunk = []
            scheduled_requests.generation_requests = generation_requests
            preparer = None
            if manager_version == 1:
                config = decoder_config(max_batch_size=batch_size,
                                        max_num_tokens=128,
                                        max_seq_len=256,
                                        dtype=torch.bfloat16,
                                        attention_backend=FlashInferAttention,
                                        disable_overlap_scheduler=True)
                preparer = input_preparer(
                    config,
                    model=DummyModel(torch.bfloat16),
                    mapping=mapping,
                    buffers=DecoderInputBuffers.for_config(config, False))
                for request_index, request in enumerate(generation_requests):
                    request.total_input_len_cp = prompt_lens[request_index]
                    request.prompt_len = sum(
                        owns_token(request_index, token_position)
                        for token_position in range(prompt_lens[request_index]))
                    request.py_prompt_len = request.prompt_len
                    request.py_seq_slot = request_index
                    request.py_batch_idx = None

            def prepare_metadata(step_positions: list[int]) -> None:
                local_before = [
                    sum(
                        owns_token(request_index, token_position)
                        for token_position in range(position))
                    for request_index, position in enumerate(step_positions)
                ]
                inactive = [
                    not owns_token(request_index, position)
                    for request_index, position in enumerate(step_positions)
                ]
                with model_extra_attrs({}):
                    if preparer is not None:
                        for request_index, request in enumerate(
                                generation_requests):
                            request.py_decoding_iter = (
                                step_positions[request_index] -
                                prompt_lens[request_index] + 1)
                            request.py_helix_is_inactive_rank = inactive[
                                request_index]
                            request.seqlen_this_rank_cp = (
                                local_before[request_index] +
                                (not inactive[request_index]))
                        preparer.prepare_inputs(
                            scheduled_requests=scheduled_requests,
                            kv_cache_manager=manager,
                            attn_metadata=metadata,
                            enable_spec_decode=False,
                            runtime_draft_len=0,
                            is_dummy=False,
                            maybe_graph=capture_graph)
                    else:
                        metadata.kv_cache_params.num_cached_tokens_per_seq = local_before
                        metadata.update_helix_param(step_positions, inactive)
                        metadata.prepare()

            prepare_metadata(positions)
            attention = FlashInferAttention(layer_idx=0,
                                            num_heads=num_heads,
                                            num_kv_heads=num_kv_heads,
                                            head_dim=head_dim)
            rank_query = query.clone()
            key = torch.stack([
                keys[index, position]
                for index, position in enumerate(positions)
            ])
            value = torch.stack([
                values[index, position]
                for index, position in enumerate(positions)
            ])
            statistics = torch.empty((batch_size, num_heads, 2), device="cuda")
            forward_args = AttentionForwardArgs(
                attention_window_size=window_size,
                softmax_stats_tensor=statistics)
            output = attention.forward(rank_query,
                                       key,
                                       value,
                                       metadata,
                                       forward_args=forward_args)
            final_positions = positions
            if capture_graph:
                full_attention = FlashInferAttention(layer_idx=0,
                                                     num_heads=num_heads,
                                                     num_kv_heads=num_kv_heads,
                                                     head_dim=head_dim)
                full_statistics = torch.empty_like(statistics)
                full_args = AttentionForwardArgs(
                    softmax_stats_tensor=full_statistics)
                full_output = full_attention.forward(rank_query,
                                                     key,
                                                     value,
                                                     metadata,
                                                     forward_args=full_args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = attention.forward(rank_query,
                                               key,
                                               value,
                                               metadata,
                                               forward_args=forward_args)
                    full_output = full_attention.forward(rank_query,
                                                         key,
                                                         value,
                                                         metadata,
                                                         forward_args=full_args)
                final_positions = [position + 1 for position in positions]
                rank_query.add_(query_shift)
                key.copy_(
                    torch.stack([
                        keys[index, position]
                        for index, position in enumerate(final_positions)
                    ]))
                value.copy_(
                    torch.stack([
                        values[index, position]
                        for index, position in enumerate(final_positions)
                    ]))
                prepare_metadata(final_positions)
                graph.replay()
                full_partials.append(
                    full_output.view(batch_size, num_heads,
                                     head_dim).float().clone())
                full_normalizers.append((full_statistics[..., 0] +
                                         full_statistics[..., 1].log()).clone())
            partials.append(
                output.view(batch_size, num_heads, head_dim).float().clone())
            normalizers.append(
                (statistics[..., 0] + statistics[..., 1].log()).clone())
        finally:
            manager.shutdown()
    weights = torch.stack(normalizers).softmax(dim=0)
    combined = (torch.stack(partials) * weights[..., None]).sum(dim=0)
    final_query = query.float().view(batch_size, num_heads, head_dim)
    if capture_graph:
        final_query = (query + query_shift).float().view(
            batch_size, num_heads, head_dim)
    for request_index, position in enumerate(final_positions):
        start = max(0, position + 1 - window_size)
        full_keys = keys[request_index,
                         start:position + 1].float().repeat_interleave(2, dim=1)
        full_values = values[request_index, start:position +
                             1].float().repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,khd->hk", final_query[request_index],
                              full_keys) * head_dim**-0.5
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), full_values)
        torch.testing.assert_close(combined[request_index],
                                   expected,
                                   atol=5e-3,
                                   rtol=5e-3)
        torch.testing.assert_close(
            torch.stack(normalizers)[:, request_index].logsumexp(0),
            scores.logsumexp(-1),
            atol=1e-4,
            rtol=1e-4)
        if capture_graph:
            all_keys = keys[request_index, :position +
                            1].float().repeat_interleave(2, dim=1)
            all_values = values[request_index, :position +
                                1].float().repeat_interleave(2, dim=1)
            all_scores = torch.einsum("hd,khd->hk", final_query[request_index],
                                      all_keys) * head_dim**-0.5
            expected_full = torch.einsum("hk,khd->hd", all_scores.softmax(-1),
                                         all_values)
            full_weights = torch.stack(full_normalizers)[:,
                                                         request_index].softmax(
                                                             dim=0)
            combined_full = (torch.stack(full_partials)[:, request_index] *
                             full_weights[..., None]).sum(dim=0)
            torch.testing.assert_close(combined_full,
                                       expected_full,
                                       atol=5e-3,
                                       rtol=5e-3)
            torch.testing.assert_close(
                torch.stack(full_normalizers)[:, request_index].logsumexp(0),
                all_scores.logsumexp(-1),
                atol=1e-4,
                rtol=1e-4)
    if window_size == 32 and manager_version == 2:
        assert torch.isneginf(normalizers[0][0]).all()
        assert torch.count_nonzero(partials[0][0]) == 0


@skip_pre_blackwell
@torch.inference_mode()
def test_helix_multibeam_generation_rejected() -> None:
    beam_width, num_heads, head_dim, page_size = 2, 2, 128, 32
    mapping = Mapping(world_size=2,
                      cp_size=2,
                      rank=0,
                      cp_config={"cp_type": CpType.HELIX})
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=1024,
                      enable_block_reuse=False,
                      host_cache_size=0),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=256,
        max_batch_size=1,
        max_beam_width=beam_width,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.BF16)
    try:
        requests = merge_helix_requests([
            RequestQueueItem(
                0,
                tensorrt_llm.bindings.executor.Request(
                    input_token_ids=[1] * 64,
                    max_tokens=2,
                    sampling_config=SamplingConfig(beam_width=beam_width)))
        ],
                                        cp_rank=0,
                                        cp_size=2,
                                        tokens_per_block=page_size,
                                        exclude_last_generation_logits=False)
        assert len(requests) == 1
        generation_request = requests[0]
        assert generation_request.py_beam_width == beam_width
        batch = ScheduledRequests()
        batch.reset_context_requests(requests)
        manager.prepare_resources(batch)
        generation_request.context_current_position = generation_request.prompt_len
        generation_request.state = LlmRequestState.GENERATION_IN_PROGRESS
        for beam_index in range(beam_width):
            generation_request.add_new_token(beam_index + 1, beam_index)
        generation_request.py_decoding_iter += 1
        generation_request.py_seq_slot = 0
        batch.reset_context_requests([])
        batch.generation_requests = requests
        manager.prepare_resources(batch)
        cache = manager.get_buffers(0, kv_layout="HND")
        cache.fill_(0.25)
        expected_cache = cache.clone()
        config = decoder_config(max_batch_size=1,
                                max_beam_width=beam_width,
                                max_num_tokens=128,
                                max_seq_len=256,
                                dtype=torch.bfloat16,
                                attention_backend=FlashInferAttention)
        preparer = input_preparer(config,
                                  model=DummyModel(torch.bfloat16),
                                  mapping=mapping,
                                  buffers=DecoderInputBuffers.for_config(
                                      config, False))
        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.ones(beam_width, dtype=torch.int32),
            num_contexts=0,
            request_ids=[0],
            max_num_requests=beam_width,
            max_num_tokens=beam_width,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True),
            mapping=mapping)
        with model_extra_attrs({}), pytest.raises(NotImplementedError,
                                                  match="beam width one"):
            preparer.prepare_inputs(scheduled_requests=batch,
                                    kv_cache_manager=manager,
                                    attn_metadata=metadata,
                                    enable_spec_decode=False,
                                    runtime_draft_len=0,
                                    is_dummy=False)
        assert metadata.request_ids == [0]
        assert metadata.seq_lens.tolist() == [1, 1]
        torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
    finally:
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
@torch.inference_mode()
def test_helix_v1_scheduled_window(overlap: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(321)
    prompt_lens = [64, 79]
    batch_size, num_steps = len(prompt_lens), 65
    num_heads, num_kv_heads, head_dim, page_size, window_size = 4, 2, 128, 32, 32
    keys = torch.randn((batch_size, 160, num_kv_heads, head_dim),
                       device="cuda",
                       dtype=torch.bfloat16,
                       generator=generator) * 0.25
    values = torch.randn(keys.shape,
                         device="cuda",
                         dtype=torch.bfloat16,
                         generator=generator)
    queries = torch.zeros((num_steps, batch_size, num_heads, head_dim),
                          device="cuda",
                          dtype=torch.bfloat16)
    checkpoints = (31, 32, 63, 64)
    partials = defaultdict(list)
    normalizers = defaultdict(list)
    for rank in range(2):
        mapping = Mapping(world_size=2,
                          cp_size=2,
                          rank=rank,
                          cp_config={"cp_type": CpType.HELIX})
        manager = KVCacheManager(
            KvCacheConfig(max_tokens=1024,
                          enable_block_reuse=False,
                          host_cache_size=0),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_seq_len=256,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16)
        try:
            requests = merge_helix_requests(
                [
                    RequestQueueItem(
                        index,
                        tensorrt_llm.bindings.executor.Request(
                            input_token_ids=[1] * prompt_len,
                            max_tokens=num_steps + 1))
                    for index, prompt_len in enumerate(prompt_lens)
                ],
                cp_rank=rank,
                cp_size=2,
                tokens_per_block=page_size,
                exclude_last_generation_logits=False)
            scheduler = SimpleScheduler(
                BindCapacityScheduler(batch_size, manager.impl, None),
                BindMicroBatchScheduler(batch_size, max_num_tokens=128))

            def schedule() -> ScheduledRequests:
                inflight = ReqIdsSet()
                result = scheduler.schedule_request(requests, inflight)
                assert not result.paused_requests
                batch = ScheduledRequests()
                batch.reset_context_requests(result.context_requests)
                batch.generation_requests = result.generation_requests
                return batch

            manager.prepare_resources(schedule())
            cache = manager.get_buffers(0, kv_layout="HND")
            cache.zero_()
            page_ids = manager.get_batch_cache_indices(list(range(batch_size)))
            for request_index, request in enumerate(requests):
                assert request.py_position_ids is not None
                for local_index, global_index in enumerate(
                        request.py_position_ids):
                    page = page_ids[request_index][local_index // page_size]
                    slot = local_index % page_size
                    cache[page, 0, :, slot].copy_(keys[request_index,
                                                       global_index])
                    cache[page, 1, :, slot].copy_(values[request_index,
                                                         global_index])
                request.context_current_position = request.prompt_len
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
                request.add_new_token(1, 0)
                request.py_decoding_iter += 1
                request.py_seq_slot = request_index
            config = decoder_config(max_batch_size=batch_size,
                                    max_num_tokens=128,
                                    max_seq_len=256,
                                    dtype=torch.bfloat16,
                                    attention_backend=FlashInferAttention,
                                    disable_overlap_scheduler=not overlap)
            preparer = input_preparer(config,
                                      model=DummyModel(torch.bfloat16),
                                      mapping=mapping,
                                      buffers=DecoderInputBuffers.for_config(
                                          config, False))
            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.ones(batch_size, dtype=torch.int32),
                num_contexts=0,
                request_ids=list(range(batch_size)),
                max_num_requests=batch_size,
                max_num_tokens=batch_size,
                kv_cache_manager=manager,
                kv_cache_params=KVCacheParams(use_cache=True),
                mapping=mapping).create_cuda_graph_metadata(batch_size)
            attention = FlashInferAttention(layer_idx=0,
                                            num_heads=num_heads,
                                            num_kv_heads=num_kv_heads,
                                            head_dim=head_dim)
            query = torch.empty((batch_size, num_heads * head_dim),
                                dtype=torch.bfloat16,
                                device="cuda")
            key = torch.empty((batch_size, num_kv_heads, head_dim),
                              dtype=torch.bfloat16,
                              device="cuda")
            value = torch.empty_like(key)
            statistics = torch.empty((batch_size, num_heads, 2), device="cuda")
            forward_args = AttentionForwardArgs(
                attention_window_size=window_size,
                softmax_stats_tensor=statistics)
            sampled_tokens = SampleStateTensors(new_tokens=torch.ones(
                (1, batch_size, 1), dtype=torch.int32, device="cuda"))
            graph = torch.cuda.CUDAGraph()
            output = None
            for step in range(num_steps):
                batch = schedule()
                assert len(batch.generation_requests) == batch_size
                manager.prepare_resources(batch)
                with model_extra_attrs({}):
                    prepared = preparer.prepare_inputs(
                        scheduled_requests=batch,
                        kv_cache_manager=manager,
                        attn_metadata=metadata,
                        new_tensors_device=sampled_tokens
                        if overlap and step > 0 else None,
                        enable_spec_decode=False,
                        runtime_draft_len=0,
                        is_dummy=False,
                        maybe_graph=True)
                assert prepared.kwargs["input_ids"].flatten().tolist(
                ) == [1] * batch_size
                assert prepared.kwargs["position_ids"].flatten().tolist() == [
                    prompt_len + step for prompt_len in prompt_lens
                ]
                query.copy_(queries[step].flatten(1))
                key.copy_(
                    torch.stack([
                        keys[index, prompt_len + step]
                        for index, prompt_len in enumerate(prompt_lens)
                    ]))
                value.copy_(
                    torch.stack([
                        values[index, prompt_len + step]
                        for index, prompt_len in enumerate(prompt_lens)
                    ]))
                if step == 0:
                    attention.forward(query,
                                      key,
                                      value,
                                      metadata,
                                      forward_args=forward_args)
                    with torch.cuda.graph(graph):
                        output = attention.forward(query,
                                                   key,
                                                   value,
                                                   metadata,
                                                   forward_args=forward_args)
                else:
                    graph.replay()
                if step in checkpoints:
                    assert output is not None
                    partials[step].append(
                        output.view(batch_size, num_heads,
                                    head_dim).float().clone())
                    normalizers[step].append(
                        (statistics[..., 0] + statistics[..., 1].log()).clone())
                if not overlap or step > 0:
                    for request in requests:
                        request.add_new_token(1, 0)
                        request.py_decoding_iter += 1
                manager.update_resources(batch)
        finally:
            manager.shutdown()
    for step in checkpoints:
        for request_index, prompt_len in enumerate(prompt_lens):
            position = prompt_len + step
            begin = position - window_size + 1
            reference_keys = keys[request_index, begin:position +
                                  1].float().repeat_interleave(2, dim=1)
            reference_values = values[request_index, begin:position +
                                      1].float().repeat_interleave(2, dim=1)
            scores = torch.einsum("hd,thd->ht", queries[step,
                                                        request_index].float(),
                                  reference_keys) / head_dim**0.5
            expected = torch.einsum("ht,thd->hd", scores.softmax(-1),
                                    reference_values)
            rank_lse = torch.stack(normalizers[step])[:, request_index]
            combined = (torch.stack(partials[step])[:, request_index] *
                        rank_lse.softmax(0)[..., None]).sum(0)
            torch.testing.assert_close(combined, expected, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(rank_lse.logsumexp(0),
                                       scores.logsumexp(-1),
                                       atol=1e-4,
                                       rtol=1e-4)


@skip_pre_blackwell
@torch.inference_mode()
def test_helix_v1_unmodified_warmup_window(
        request: pytest.FixtureRequest) -> None:
    generator = torch.Generator(device="cuda").manual_seed(456)
    batch_size, num_heads, num_kv_heads, head_dim, page_size = 2, 4, 2, 128, 32
    for rank in range(2):
        mapping = Mapping(world_size=2,
                          cp_size=2,
                          rank=rank,
                          cp_config={"cp_type": CpType.HELIX})
        manager = KVCacheManager(
            KvCacheConfig(max_tokens=1024,
                          enable_block_reuse=False,
                          host_cache_size=0),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_seq_len=256,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16)
        request.addfinalizer(manager.shutdown)
        resources = resource_manager.ResourceManager(
            {resource_manager.ResourceManagerType.KV_CACHE_MANAGER: manager})
        config = decoder_config(max_batch_size=batch_size,
                                max_num_tokens=128,
                                max_seq_len=256,
                                dtype=torch.bfloat16,
                                attention_backend=FlashInferAttention)
        model = DummyModel(torch.bfloat16)
        factory = DummyRequestFactory(
            config=config,
            model=model,
            get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
            release_request_state=None)
        with factory.release_batch_context(
                factory.create_cuda_graph_warmup_request(resources,
                                                         batch_size=batch_size,
                                                         draft_len=0),
                resources) as batch:
            try:
                assert batch is not None
                requests = batch.generation_requests
                assert [len(request.get_tokens(0))
                        for request in requests] == [255, 2]
                for dummy in requests:
                    assert dummy.py_prompt_len == helix_local_len(
                        dummy.total_input_len_cp, page_size, 2, rank)
                    assert dummy.seqlen_this_rank_cp == dummy.py_prompt_len + (
                        rank == 0)
                    assert dummy.py_helix_is_inactive_rank == (rank != 0)
                    assert dummy.seqlen_this_rank_cp <= len(
                        manager.get_cache_indices(dummy)) * page_size
                cache = manager.get_buffers(0, kv_layout="HND")
                cache.zero_()
                metadata = FlashInferAttentionMetadata(
                    seq_lens=torch.ones(batch_size, dtype=torch.int32),
                    num_contexts=0,
                    request_ids=[request.py_request_id for request in requests],
                    max_num_requests=batch_size,
                    max_num_tokens=batch_size,
                    kv_cache_manager=manager,
                    kv_cache_params=KVCacheParams(use_cache=True),
                    mapping=mapping).create_cuda_graph_metadata(batch_size)
                preparer = input_preparer(
                    config,
                    model=model,
                    mapping=mapping,
                    buffers=DecoderInputBuffers.for_config(config, False))
                with model_extra_attrs({}):
                    preparer.prepare_inputs(scheduled_requests=batch,
                                            kv_cache_manager=manager,
                                            attn_metadata=metadata,
                                            enable_spec_decode=False,
                                            runtime_draft_len=0,
                                            is_dummy=True,
                                            maybe_graph=True)
                query = torch.randn((batch_size, num_heads * head_dim),
                                    device="cuda",
                                    dtype=torch.bfloat16,
                                    generator=generator)
                key = torch.zeros((batch_size, num_kv_heads, head_dim),
                                  device="cuda",
                                  dtype=torch.bfloat16)
                value = torch.zeros_like(key)
                statistics = torch.empty((batch_size, num_heads, 2),
                                         device="cuda")
                attention = FlashInferAttention(layer_idx=0,
                                                num_heads=num_heads,
                                                num_kv_heads=num_kv_heads,
                                                head_dim=head_dim)
                args = AttentionForwardArgs(attention_window_size=16,
                                            softmax_stats_tensor=statistics)
                attention.forward(query,
                                  key,
                                  value,
                                  metadata,
                                  forward_args=args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = attention.forward(query,
                                               key,
                                               value,
                                               metadata,
                                               forward_args=args)
                graph.replay()
                assert torch.count_nonzero(output) == 0
                actual_lse = statistics[..., 0] + statistics[..., 1].log()
                expected_lengths = []
                for dummy in requests:
                    position = dummy.total_input_len_cp
                    owned = sum(
                        (token // page_size) % 2 == rank
                        for token in range(max(0, position -
                                               15), position)) + (rank == 0)
                    expected_lengths.append(owned)
                expected_lse = torch.tensor(expected_lengths,
                                            device="cuda").float().log()
                torch.testing.assert_close(actual_lse,
                                           expected_lse[:, None].expand(
                                               -1, num_heads),
                                           atol=1e-4,
                                           rtol=1e-4)
            finally:
                torch.cuda.synchronize()


@pytest.mark.cpu_only
def test_flashinfer_rejects_swa_scratch_before_buffer_allocation() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_swa_scratch_reuse = True
    metadata = object.__new__(FlashInferAttentionMetadata)
    metadata.kv_cache_manager = manager
    with pytest.raises(
            ValueError,
            match="FlashInfer attention does not support SWA scratch reuse"):
        metadata.__post_init__()


class TestingFlashInferAttentionMetadata(FlashInferAttentionMetadata):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_times_planned = defaultdict(int)

    def get_num_plans(self, plan_params) -> int:
        return self._num_times_planned[plan_params]

    def _plan_with_params(self, plan_params, flashinfer_backend: str = "fa2"):
        if self.needs_plan(plan_params):
            self._num_times_planned[plan_params] += 1
        return super()._plan_with_params(plan_params, flashinfer_backend)


@dataclass(repr=False)
class Scenario:
    num_layers: int
    num_heads: int
    num_kv_heads: Union[int, List[Optional[int]]]
    head_dim: int
    dtype: torch.dtype
    full_page_table: bool = False

    def __repr__(self) -> str:
        if isinstance(self.num_kv_heads, int):
            num_kv_heads_str = str(self.num_kv_heads)
        else:
            num_kv_heads_str = '_'.join(map(str, self.num_kv_heads))
        return f"num_layers:{self.num_layers}-num_heads:{self.num_heads}-num_kv_heads:{num_kv_heads_str}-head_dim:{self.head_dim}-dtype:{self.dtype}{'-full_page_table' if self.full_page_table else ''}"


@dataclass
class CUDAGraphTestScenario:
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype


class TestFlashInferAttention(unittest.TestCase):

    def test_swa_page_sanitization_uses_local_window_order(self) -> None:
        metadata = object.__new__(FlashInferAttentionMetadata)
        metadata.kv_cache_manager = SimpleNamespace(
            layer_offsets={
                4: 0,
                5: 1,
                6: 2
            },
            max_attention_window_vec=[128, None, 128],
        )

        sliding_page_indices = torch.tensor([BAD_PAGE_INDEX, 3],
                                            dtype=torch.int32)
        metadata._sanitize_swa_page_indices(sliding_page_indices, layer_idx=4)
        torch.testing.assert_close(sliding_page_indices,
                                   torch.tensor([0, 3], dtype=torch.int32))

        full_page_indices = torch.tensor([BAD_PAGE_INDEX, 3], dtype=torch.int32)
        metadata._sanitize_swa_page_indices(full_page_indices, layer_idx=5)
        torch.testing.assert_close(
            full_page_indices,
            torch.tensor([BAD_PAGE_INDEX, 3], dtype=torch.int32),
        )

    def test_attention_layer_indices_ignore_zero_kv_layers(self) -> None:

        class FakeHybridManager:

            def __init__(self) -> None:
                self.layer_offsets = {10: 0, 20: 1, 30: 2}
                self.num_kv_heads_per_layer = [8, 0, 8]

            def is_attention_layer(self, layer_idx: int) -> bool:
                return layer_idx != 20

        self.assertEqual(
            flashinfer_backend._get_attention_layer_indices(
                FakeHybridManager()), [10, 30])

    def test_attention_layer_indices_support_v1_manager(self) -> None:

        class FakeV1Manager:

            def __init__(self) -> None:
                self.layer_offsets = {10: 0, 20: 1}

        self.assertEqual(
            flashinfer_backend._get_attention_layer_indices(FakeV1Manager()),
            [10, 20])

    def test_hybrid_v2_metadata_uses_first_attention_layer_for_prepare(
            self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        class FakeHybridManager:
            blocks_in_primary_pool = 4
            max_blocks_per_seq = 4
            tokens_per_block = 32
            is_vswa = False

            def __init__(self) -> None:
                self.layer_offsets = {10: 0, 20: 1}
                self.layer_to_pool_mapping_dict = {0: 0, 1: 0}

            def is_attention_layer(self, layer_idx: int) -> bool:
                return layer_idx == 20

            def get_layer_page_index_scale(self, layer_idx: int) -> int:
                return 1

        manager = FakeHybridManager()
        manager.get_buffers = mock.Mock(
            side_effect=lambda layer_idx: torch.empty(4) if layer_idx == 20 else
            self.fail("recurrent layer queried for KV buffers"))
        manager.get_batch_cache_indices_flat = mock.Mock(
            return_value=torch.tensor([2], dtype=torch.int32))

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.zeros(1, dtype=torch.int32),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[1],
            ),
            max_num_requests=1,
            max_num_tokens=1,
            kv_cache_manager=manager,
            request_ids=[7],
            workspace_buffer=torch.empty(1, dtype=torch.uint8, device="cuda"),
            mamba_metadata=False,
        )

        self.assertEqual(metadata._primary_kv_layer_idx, 20)
        manager.get_buffers.assert_called_once_with(20)

        metadata.prepare()

        manager.get_batch_cache_indices_flat.assert_called_once_with(
            [7], [1], layer_idx=20)

    def test_generation_page_table_uses_reserved_block_count(self):
        manager = SimpleNamespace(get_batch_cache_indices=mock.Mock(
            return_value=[list(range(325))]))

        self.assertEqual(
            flashinfer_backend._get_page_table_num_blocks(manager, [98, 99],
                                                          [3, 324],
                                                          num_contexts=1),
            [3, 325],
        )
        manager.get_batch_cache_indices.assert_called_once_with([99])

    def test_decode_query_width_is_part_of_plan_params(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            num_contexts=0,
            kv_cache_manager=None,
            request_ids=[0, 1],
            max_num_requests=3,
            max_num_tokens=18,
        )

        def return_plan_params(plan_params, _flashinfer_backend):
            return plan_params

        with mock.patch.object(
                metadata,
                "_plan_with_params",
                side_effect=return_plan_params,
        ):
            single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata.seq_lens = torch.tensor([1, 1, 1], dtype=torch.int32)
            larger_single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata.seq_lens = torch.tensor([1, 1], dtype=torch.int32)
            metadata._uses_full_generation_page_table = True
            full_page_single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata._uses_full_generation_page_table = False
            metadata._is_shared_kv_draft_view = True
            shared_draft_single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata._is_shared_kv_draft_view = False
            metadata.seq_lens = torch.tensor([6, 6], dtype=torch.int32)
            multi_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata.seq_lens = torch.tensor([6, 6, 6], dtype=torch.int32)
            larger_batch_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )

        self.assertEqual(single_token_plan.q_len_per_req, 1)
        self.assertEqual(larger_single_token_plan.q_len_per_req, 1)
        self.assertEqual(single_token_plan, larger_single_token_plan)
        self.assertEqual(single_token_plan, full_page_single_token_plan)
        self.assertEqual(single_token_plan, shared_draft_single_token_plan)
        self.assertEqual(multi_token_plan.q_len_per_req, 6)
        self.assertNotEqual(single_token_plan, multi_token_plan)
        # The live generation batch size deliberately does not key the cache. Eager metadata
        # replans every wrapper on each prepare() and sizes their block tables by
        # max_num_requests, so keying on it would allocate a fresh wrapper per batch size seen.
        self.assertEqual(multi_token_plan, larger_batch_plan)

        metadata.seq_lens = torch.tensor([6, 5], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "uniform query length"):
            metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )

    def test_cuda_graph_metadata_owns_a_private_plan_cache(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            num_contexts=0,
            kv_cache_manager=None,
            request_ids=[0, 1],
            max_num_requests=4,
            max_num_tokens=16,
        )
        plan_params = PlanParams(
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            attention_mask_type=AttentionMaskType.causal,
        )
        eager_wrappers = FlashInferWrappers(is_planned=True)
        metadata._plan_params_to_wrappers[plan_params] = eager_wrappers

        graph_metadata = metadata.create_cuda_graph_metadata(2)

        # Wrappers belong to a metadata instance, and the runner builds one metadata per captured
        # batch size. That partitioning -- not the cache key -- is what keeps a graph wrapper from
        # being reused across batch sizes, so PlanParams need not carry the generation batch size.
        self.assertTrue(graph_metadata.is_cuda_graph)
        self.assertIsNot(graph_metadata._plan_params_to_wrappers,
                         metadata._plan_params_to_wrappers)
        self.assertEqual(graph_metadata._plan_params_to_wrappers, {})
        self.assertIs(metadata._plan_params_to_wrappers[plan_params],
                      eager_wrappers)

    def test_generation_page_table_keeps_logical_positions(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=256),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        try:
            kv_cache_manager.add_dummy_requests([0], [32],
                                                is_gen=True,
                                                max_num_draft_tokens=3)
            reserved_blocks = kv_cache_manager.get_batch_cache_indices([0])
            self.assertEqual(len(reserved_blocks[0]), 2)

            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.full((1, ), 4, dtype=torch.int32),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[28],
                    use_full_generation_page_table=True,
                ),
                max_num_requests=1,
                max_num_tokens=4,
                kv_cache_manager=kv_cache_manager,
                request_ids=[0],
            )
            metadata.prepare()

            self.assertEqual(metadata.num_blocks, [2])
            torch.testing.assert_close(
                metadata.paged_kv_indptr_decode[:2],
                torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._paged_kv_last_page_len[:1],
                torch.tensor([32], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._logical_kv_lens[:1],
                torch.tensor([32], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata.positions,
                torch.tensor([28, 29, 30, 31], dtype=torch.int32,
                             device="cuda"),
            )
        finally:
            kv_cache_manager.shutdown()

    def test_spec_decode_offsets_update_append_and_decode_lengths(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=256),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=2,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        try:
            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.full((2, ), 4, dtype=torch.int32),
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True),
                max_num_requests=2,
                max_num_tokens=8,
                kv_cache_manager=kv_cache_manager,
            )
            cached_token_lens = torch.tensor([31, 62],
                                             dtype=torch.int32,
                                             device="cuda")
            positions = torch.tensor([31, 32, 33, 34, 62, 63, 64, 65],
                                     dtype=torch.int32,
                                     device="cuda")
            metadata._cached_token_lens[:2].copy_(cached_token_lens)
            metadata._logical_kv_lens[:2].copy_(cached_token_lens + 4)
            metadata._uses_full_generation_page_table = True
            metadata._positions[:8].copy_(positions)
            kv_lens_buffer = torch.tensor([35, 66],
                                          dtype=torch.int32,
                                          device="cuda")
            metadata._plan_params_to_wrappers = {
                object():
                FlashInferWrappers(
                    is_planned=True,
                    decode_wrapper=SimpleNamespace(
                        _kv_lens_buffer=kv_lens_buffer),
                )
            }
            offsets = torch.tensor([-3, -1], dtype=torch.int32, device="cuda")

            metadata.apply_spec_decode_kv_lens_offsets(
                offsets,
                num_generations=2,
                tokens_per_generation=4,
            )

            torch.testing.assert_close(
                metadata._cached_token_lens[:2],
                torch.tensor([28, 61], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._logical_kv_lens[:2],
                torch.tensor([32, 65], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._positions[:8],
                torch.tensor([28, 29, 30, 31, 61, 62, 63, 64],
                             dtype=torch.int32,
                             device="cuda"),
            )
            torch.testing.assert_close(
                kv_lens_buffer,
                torch.tensor([32, 65], dtype=torch.int32, device="cuda"),
            )

            metadata.apply_spec_decode_kv_lens_offsets(
                offsets,
                num_generations=2,
                tokens_per_generation=4,
                restore=True,
            )
            torch.testing.assert_close(metadata._cached_token_lens[:2],
                                       cached_token_lens)
            torch.testing.assert_close(metadata._logical_kv_lens[:2],
                                       cached_token_lens + 4)
            torch.testing.assert_close(metadata._positions[:8], positions)
            torch.testing.assert_close(
                kv_lens_buffer,
                torch.tensor([35, 66], dtype=torch.int32, device="cuda"),
            )
        finally:
            kv_cache_manager.shutdown()

    # 5 tokens per request (4 drafts + 1 bonus). The FP8 cases use a cast (scale 1.0) cache.
    @parameterized.expand([(head_dim, fp8_kv) for head_dim in (256, 512)
                           for fp8_kv in (False, True)])
    def test_xqa_target_verify(self, head_dim: int, fp8_kv: bool):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("FlashInfer XQA requires SM90, SM100 or SM120")

        num_heads, num_kv_heads, tokens_per_gen = 8, 2, 5
        num_drafts = tokens_per_gen - 1
        cached = [59, 75]
        kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=512),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=32,
            max_seq_len=256,
            max_batch_size=2,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=(tensorrt_llm.bindings.DataType.FP8
                   if fp8_kv else tensorrt_llm.bindings.DataType.BF16),
            spec_config=MTPDecodingConfig(max_draft_len=num_drafts),
        )
        try:
            kv_cache_manager.add_dummy_requests(
                [0, 1], [c + tokens_per_gen + num_drafts for c in cached],
                is_gen=True,
                max_num_draft_tokens=num_drafts)
            buf = kv_cache_manager.get_buffers(0)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(0)
            randn = functools.partial(torch.randn,
                                      dtype=torch.bfloat16,
                                      device="cuda",
                                      generator=generator)
            buf.copy_(randn(buf.shape).to(buf.dtype))
            initial_kv = buf.clone()

            def make_layer(quant_algo):
                return FlashInferAttention(
                    layer_idx=0,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_kv_heads=num_kv_heads,
                    quant_config=(QuantConfig(quant_algo=quant_algo,
                                              kv_cache_quant_algo=QuantAlgo.FP8)
                                  if fp8_kv else None))

            layer = make_layer(QuantAlgo.FP8)
            num_tokens = 2 * tokens_per_gen
            q = randn(num_tokens, num_heads * head_dim)
            k = randn(num_tokens, num_kv_heads * head_dim)
            v = randn(k.shape)
            # SM90 XQA runs an FP8 cache with head_dim <= 256 on FP8 MMAs, rounding Q and the
            # softmax weights to e4m3: allow one e4m3 step (2**-4) on the O(1) outputs.
            fp8_mma = (fp8_kv and head_dim <= 256
                       and torch.cuda.get_device_capability()[0] == 9)
            tol = dict(atol=2**-4 if fp8_mma else 2e-2, rtol=2e-2)
            out_scale = torch.tensor([4.0], device="cuda")

            def run(num_cached,
                    offsets=None,
                    metadata=None,
                    backend=layer,
                    **forward_kwargs):
                # Undo the previous run's K/V append.
                buf.copy_(initial_kv)
                kv_cache_params = KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=num_cached,
                    use_full_generation_page_table=True,
                )
                if metadata is None:
                    metadata = FlashInferAttentionMetadata(
                        seq_lens=torch.full((2, ),
                                            tokens_per_gen,
                                            dtype=torch.int32),
                        num_contexts=0,
                        kv_cache_params=kv_cache_params,
                        max_num_requests=2,
                        max_num_tokens=num_tokens,
                        kv_cache_manager=kv_cache_manager,
                        request_ids=[0, 1],
                    )
                else:
                    metadata.kv_cache_params = kv_cache_params
                metadata.prepare()
                if offsets is not None:
                    metadata.apply_spec_decode_kv_lens_offsets(
                        torch.tensor(offsets, dtype=torch.int32, device="cuda"),
                        num_generations=2,
                        tokens_per_generation=tokens_per_gen,
                    )
                return backend.forward(q, k, v, metadata,
                                       **forward_kwargs), metadata

            def reference_attention(window):
                """fp32 attention over each request's cached K/V and this step's K/V."""
                buf.copy_(initial_kv)
                cache = kv_cache_manager.get_buffers(0, kv_layout="HND")
                pages = kv_cache_manager.get_batch_cache_indices([0, 1])
                out = []
                for i, (num_cached, ids) in enumerate(zip(cached, pages)):
                    rows = slice(i * tokens_per_gen, (i + 1) * tokens_per_gen)
                    # [2, tokens, num_kv_heads, head_dim]
                    cached_kv = cache[torch.tensor(ids)].permute(1, 0, 3, 2, 4)
                    cached_kv = cached_kv.flatten(1, 2)[:, :num_cached]
                    new_kv = torch.stack(
                        (k[rows], v[rows])).view(2, tokens_per_gen,
                                                 num_kv_heads, head_dim)
                    kv = torch.cat((cached_kv, new_kv.to(cache.dtype)), dim=1)
                    key, value = kv.float().repeat_interleave(num_heads //
                                                              num_kv_heads,
                                                              dim=2)
                    query = q[rows].float().view(tokens_per_gen, num_heads,
                                                 head_dim)
                    scores = torch.einsum("qhd,khd->hqk", query,
                                          key) / head_dim**0.5
                    key_pos = torch.arange(num_cached + tokens_per_gen,
                                           device="cuda")
                    query_pos = key_pos[num_cached:, None]
                    masked = key_pos > query_pos
                    if window is not None:
                        masked |= key_pos <= query_pos - window
                    probs = scores.masked_fill(masked,
                                               float("-inf")).softmax(-1)
                    out.append(
                        torch.einsum("hqk,khd->qhd", probs,
                                     value).reshape(tokens_per_gen, -1))
                return torch.cat(out).to(q.dtype)

            # Sliding-window and windowless layers must both match the reference. The windowless
            # case runs last: the overlap checks below reuse its reference and metadata.
            for window in (48, None):
                reference = reference_attention(window)
                exact, metadata = run(cached, attention_window_size=window)
                # The full page table routes this generation-only step to XQA.
                self.assertTrue(metadata._xqa_target.in_use)
                torch.testing.assert_close(exact, reference, **tol)
                if fp8_kv:
                    # With an FP8 cache XQA writes the FP8 input of an FP8 o_proj: the output
                    # times out_scale (o_proj's inverse input scale), cast to FP8.
                    fp8_out, _ = run(cached,
                                     attention_window_size=window,
                                     out_scale=out_scale)
                    self.assertEqual(fp8_out.dtype, torch.float8_e4m3fn)
                    torch.testing.assert_close(fp8_out.float() / 4.0,
                                               exact.float(),
                                               atol=1e-2,
                                               rtol=7e-2)
                    # An NVFP4 o_proj also passes out_scale but takes no FP8 input.
                    nvfp4_out, _ = run(cached,
                                       attention_window_size=window,
                                       backend=make_layer(QuantAlgo.NVFP4),
                                       out_scale=out_scale)
                    torch.testing.assert_close(nvfp4_out, exact)

            # Overlap: the host assumed all drafts of request 0 were accepted; the device
            # offset rewinds it, so the step appends and attends exactly as the reference. The
            # first run builds the XQA inputs inside forward, after the offsets; the second
            # rebuilds them in prepare(), before the offsets.
            for prepared in (None, metadata):
                corrected, overlap = run([cached[0] + num_drafts, cached[1]],
                                         [-num_drafts, 0],
                                         metadata=prepared)
                # Request 0 reserves three pages, though the corrected length fills only two.
                self.assertEqual(overlap.num_blocks[0], 3)
                torch.testing.assert_close(corrected, reference, **tol)
        finally:
            kv_cache_manager.shutdown()

    def test_separate_kv_draft_metadata_uses_draft_manager(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("FlashInfer trtllm-gen requires SM100 or SM103")

        def create_manager():
            return KVCacheManagerV2(
                KvCacheConfig(max_tokens=256),
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=1,
                num_kv_heads=1,
                head_dim=128,
                tokens_per_block=32,
                max_seq_len=64,
                max_batch_size=2,
                mapping=Mapping(world_size=1, tp_size=1, rank=0),
                dtype=tensorrt_llm.bindings.DataType.BF16,
            )

        target_manager = create_manager()
        draft_manager = create_manager()
        try:
            target_manager.add_dummy_requests([0, 1], [31, 45], is_gen=True)
            draft_manager.add_dummy_requests([0, 1], [31, 45],
                                             is_gen=True,
                                             max_num_draft_tokens=3)
            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.ones(2, dtype=torch.int32),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[30, 44],
                ),
                max_num_requests=2,
                max_num_tokens=8,
                kv_cache_manager=target_manager,
                request_ids=[0, 1],
                is_cuda_graph=True,
            )
            metadata.prepare()
            draft_metadata = metadata.get_draft_metadata(draft_manager)

            self.assertIs(draft_metadata.kv_cache_manager, draft_manager)
            self.assertFalse(hasattr(metadata, "kv_lens_cuda"))
            torch.testing.assert_close(
                draft_metadata.kv_lens_cuda[:2],
                torch.tensor([31, 45], dtype=torch.int32, device="cuda"),
            )
            draft_blocks = draft_manager.get_batch_cache_indices([0, 1])
            self.assertEqual(draft_metadata.num_blocks,
                             list(map(len, draft_blocks)))

            layer = FlashInferAttention(
                layer_idx=0,
                num_heads=1,
                num_kv_heads=1,
                head_dim=128,
                flashinfer_backend="trtllm-gen",
            )
            q = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            self.assertEqual(
                layer.forward(q, k, v, draft_metadata).shape, q.shape)
            for wrappers in draft_metadata._plan_params_to_wrappers.values():
                torch.testing.assert_close(
                    wrappers.decode_wrapper._kv_lens_buffer[:2],
                    draft_metadata.kv_lens_cuda[:2],
                )

            with mock.patch.object(
                    draft_metadata,
                    "_plan_with_params",
                    wraps=draft_metadata._plan_with_params,
            ) as replan, mock.patch.object(
                    draft_metadata,
                    "_build_decode_block_tables",
            ) as refresh_block_tables:
                metadata.prepare()
            replan.assert_not_called()
            self.assertEqual(refresh_block_tables.call_count,
                             len(draft_metadata._plan_params_to_wrappers))
        finally:
            target_manager.shutdown()
            draft_manager.shutdown()

    def test_ragged_no_kv_cuda_graph_uses_stable_indptr_aliases(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        class FakeRaggedWrapper:

            def plan(self, **kwargs):
                self.plan_kwargs = kwargs

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            num_contexts=2,
            kv_cache_manager=None,
            request_ids=[0, 1],
            max_num_requests=2,
            max_num_tokens=8,
        )
        metadata.is_cuda_graph = True
        plan_params = PlanParams(
            num_heads=2,
            num_kv_heads=2,
            head_dim=4,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            attention_mask_type=AttentionMaskType.padding,
        )
        wrapper = FakeRaggedWrapper()

        with mock.patch.object(
                flashinfer_backend.flashinfer,
                "BatchPrefillWithRaggedKVCacheWrapper",
                FakeRaggedWrapper,
        ):
            metadata._plan_ragged_no_kv(plan_params, wrapper, "cudnn")

        self.assertIn("kv_indptr", wrapper.plan_kwargs)
        self.assertNotIn("v_indptr", wrapper.plan_kwargs)
        self.assertNotIn("o_indptr", wrapper.plan_kwargs)

    @parameterized.expand([
        Scenario(num_layers=1,
                 num_heads=32,
                 num_kv_heads=8,
                 head_dim=128,
                 dtype=torch.bfloat16),
        Scenario(num_layers=2,
                 num_heads=32,
                 num_kv_heads=8,
                 head_dim=64,
                 dtype=torch.float16),
        Scenario(num_layers=2,
                 num_heads=32,
                 num_kv_heads=[8, 16],
                 head_dim=128,
                 dtype=torch.bfloat16),
        Scenario(num_layers=3,
                 num_heads=32,
                 num_kv_heads=[8, None, 16],
                 head_dim=64,
                 dtype=torch.float16),
        Scenario(num_layers=1,
                 num_heads=32,
                 num_kv_heads=8,
                 head_dim=128,
                 dtype=torch.bfloat16,
                 full_page_table=True),
        Scenario(num_layers=3,
                 num_heads=32,
                 num_kv_heads=[8, None, 16],
                 head_dim=64,
                 dtype=torch.float16,
                 full_page_table=True),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_flashinfer_attention(self, scenario: Scenario):
        if (scenario.full_page_table
                and torch.cuda.get_device_capability()[0] not in (9, 10, 12)):
            self.skipTest("FlashInfer XQA requires SM90, SM100 or SM120")
        num_layers = scenario.num_layers
        num_heads = scenario.num_heads
        num_kv_heads = scenario.num_kv_heads
        head_dim = scenario.head_dim
        dtype = scenario.dtype

        device = torch.device('cuda')

        # TODO: make these a part of the scenario?
        num_gens = 2
        context_sequence_lengths = [3, 2]
        sequence_lengths = context_sequence_lengths + [1] * num_gens
        past_seen_tokens = [30, 40, 62, 75]
        batch_size = num_gens + len(context_sequence_lengths)
        request_ids = list(range(batch_size))
        token_nums = (torch.tensor(sequence_lengths) +
                      torch.tensor(past_seen_tokens)).tolist()

        num_blocks = 16
        tokens_per_block = 128
        max_seq_len = tokens_per_block * num_blocks
        mapping = Mapping(world_size=1, tp_size=1, rank=0)

        if dtype == torch.float16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype for unit test")

        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
        kv_cache_manager = KVCacheManagerV2(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        try:
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            for i in kv_cache_manager.layer_offsets:
                buf = kv_cache_manager.get_buffers(i)
                if buf is not None:
                    torch.nn.init.normal_(buf)
                    del buf

            if isinstance(num_kv_heads, int):
                num_kv_heads = [num_kv_heads] * num_layers

            contexts_per_layer = []
            gens_per_layer = []

            for layer_idx in range(num_layers):
                kv_heads = num_kv_heads[layer_idx]
                if kv_heads is None:
                    continue

                context_qs = [
                    torch.randn(sequence_length,
                                num_heads * head_dim,
                                dtype=dtype,
                                device=device)
                    for sequence_length in context_sequence_lengths
                ]

                context_ks = [
                    torch.randn(sequence_length,
                                kv_heads * head_dim,
                                dtype=dtype,
                                device=device)
                    for sequence_length in context_sequence_lengths
                ]
                context_vs = [
                    torch.randn(sequence_length,
                                kv_heads * head_dim,
                                dtype=dtype,
                                device=device)
                    for sequence_length in context_sequence_lengths
                ]

                contexts_per_layer.append((context_qs, context_ks, context_vs))

                gen_qs = [
                    torch.randn(1,
                                num_heads * head_dim,
                                dtype=dtype,
                                device=device) for _ in range(num_gens)
                ]

                gen_ks = [
                    torch.randn(1,
                                kv_heads * head_dim,
                                dtype=dtype,
                                device=device) for _ in range(num_gens)
                ]

                gen_vs = [
                    torch.randn(1,
                                kv_heads * head_dim,
                                dtype=dtype,
                                device=device) for _ in range(num_gens)
                ]

                gens_per_layer.append((gen_qs, gen_ks, gen_vs))

            layers = [
                FlashInferAttention(
                    layer_idx=layer_idx,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_kv_heads=kv_heads,
                ) for layer_idx, kv_heads in enumerate(num_kv_heads)
                if kv_heads is not None
            ]

            # [context_1, context_2, gen_1, gen_2]
            results_1 = []

            seq_lens = torch.tensor(sequence_lengths).int()
            attn_metadata = TestingFlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=len(context_sequence_lengths),
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=past_seen_tokens,
                    use_full_generation_page_table=scenario.full_page_table),
                max_num_requests=4,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )

            attn_metadata.prepare()
            for attn_layer_idx, flashinfer_attn in enumerate(layers):
                context_qs, context_ks, context_vs = contexts_per_layer[
                    attn_layer_idx]
                gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

                q = torch.cat((*context_qs, *gen_qs))
                k = torch.cat((*context_ks, *gen_ks))
                v = torch.cat((*context_vs, *gen_vs))

                result_1 = flashinfer_attn.forward(q, k, v, attn_metadata)
                self.assertEqual(result_1.size()[0],
                                 sum(context_sequence_lengths) + num_gens)

                # validate kv cache was updated expectedly
                cache_buf = kv_cache_manager.get_buffers(
                    flashinfer_attn.layer_idx,
                    kv_layout=attn_metadata.kv_layout)
                if attn_metadata.kv_layout == "HND":
                    cache_buf = cache_buf.transpose(2, 3).contiguous()
                assert cache_buf is not None
                num_kv_heads = cache_buf.size(-2)

                # validate contexts
                block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(
                    request_ids)
                for seq_id in range(len(context_sequence_lengths)):
                    # get a contiguous copy of the cache for the sequence
                    block_ids = block_ids_per_seq[seq_id]
                    last_block_len = attn_metadata.paged_kv_last_page_len[
                        seq_id]
                    cached_kvs = torch.concat(
                        cache_buf[block_ids, :].unbind(dim=0), dim=1)
                    # only look at new tokens added
                    cached_kvs = cached_kvs[:, past_seen_tokens[seq_id]:
                                            last_block_len]

                    # compare to input kvs
                    torch.testing.assert_close(
                        cached_kvs[0].to(context_ks[seq_id].dtype),
                        context_ks[seq_id].view(-1, num_kv_heads, head_dim))
                    torch.testing.assert_close(
                        cached_kvs[1].to(context_vs[seq_id].dtype),
                        context_vs[seq_id].view(-1, num_kv_heads, head_dim))

                # validate generations (same way)
                for gen_seq_id in range(num_gens):
                    seq_id = len(context_sequence_lengths) + gen_seq_id
                    block_ids = block_ids_per_seq[seq_id]
                    last_block_len = attn_metadata.paged_kv_last_page_len[
                        seq_id]
                    cached_kvs = torch.concat(
                        cache_buf[block_ids, :].unbind(dim=0),
                        dim=1)[:, past_seen_tokens[seq_id]:last_block_len]

                    torch.testing.assert_close(
                        cached_kvs[0],
                        gen_ks[gen_seq_id].view(-1, num_kv_heads, head_dim))
                    torch.testing.assert_close(
                        cached_kvs[1],
                        gen_vs[gen_seq_id].view(-1, num_kv_heads, head_dim))

                results_1.append(result_1)
                if scenario.full_page_table:
                    self.assertTrue(attn_metadata._xqa_target.in_use)
                del cache_buf

            for plan_params in attn_metadata._plan_params_to_wrappers.keys():
                self.assertEqual(attn_metadata.get_num_plans(plan_params), 1)

            # prepare() defers re-planning to forward_impl only when multiple
            # wrappers share one workspace_buffer (hybrid attention); for the
            # single-wrapper case it re-plans eagerly so cuda-graph capture works.
            attn_metadata.prepare()
            defer_plan = len(attn_metadata._plan_params_to_wrappers) > 1
            for wrappers in attn_metadata._plan_params_to_wrappers.values():
                self.assertEqual(wrappers.is_planned, not defer_plan)

            # [context_1, gen_1]
            results_2 = []
            num_cached_tokens_per_seq = [
                j for j in [
                    past_seen_tokens[0], past_seen_tokens[len(
                        context_sequence_lengths)]
                ]
            ]

            seq_lens = torch.tensor([context_sequence_lengths[0], 1],
                                    dtype=torch.int)
            attn_metadata = TestingFlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=num_cached_tokens_per_seq,
                    use_full_generation_page_table=scenario.full_page_table),
                max_num_requests=2,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=[0, 2],
            )

            attn_metadata.prepare()

            for attn_layer_idx, flashinfer_attn in enumerate(layers):
                context_qs, context_ks, context_vs = contexts_per_layer[
                    attn_layer_idx]
                gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

                result_2 = flashinfer_attn.forward(
                    torch.cat((context_qs[0], gen_qs[0])),
                    torch.cat((context_ks[0], gen_ks[0])),
                    torch.cat((context_vs[0], gen_vs[0])), attn_metadata)
                self.assertEqual(result_2.size()[0],
                                 context_sequence_lengths[0] + 1)
                results_2.append(result_2)

            for plan_params in attn_metadata._plan_params_to_wrappers.keys():
                self.assertEqual(attn_metadata.get_num_plans(plan_params), 1)

            # prepare() defers re-planning to forward_impl only when multiple
            # wrappers share one workspace_buffer (hybrid attention); for the
            # single-wrapper case it re-plans eagerly so cuda-graph capture works.
            attn_metadata.prepare()
            defer_plan = len(attn_metadata._plan_params_to_wrappers) > 1
            for wrappers in attn_metadata._plan_params_to_wrappers.values():
                self.assertEqual(wrappers.is_planned, not defer_plan)

            # [context_2, gen_2]
            results_3 = []
            num_cached_tokens_per_seq = [
                j for j in [
                    past_seen_tokens[1], past_seen_tokens[
                        len(context_sequence_lengths) + 1]
                ]
            ]

            seq_lens = torch.tensor([context_sequence_lengths[1], 1],
                                    dtype=torch.int)
            attn_metadata = TestingFlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=num_cached_tokens_per_seq,
                    use_full_generation_page_table=scenario.full_page_table),
                max_num_requests=2,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=[1, 3],
            )

            attn_metadata.prepare()
            for attn_layer_idx, flashinfer_attn in enumerate(layers):
                context_qs, context_ks, context_vs = contexts_per_layer[
                    attn_layer_idx]
                gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

                result_3 = flashinfer_attn.forward(
                    torch.cat((context_qs[1], gen_qs[1])),
                    torch.cat((context_ks[1], gen_ks[1])),
                    torch.cat((context_vs[1], gen_vs[1])), attn_metadata)
                self.assertEqual(result_3.size()[0],
                                 context_sequence_lengths[1] + 1)
                results_3.append(result_3)

            for plan_params in attn_metadata._plan_params_to_wrappers.keys():
                self.assertEqual(attn_metadata.get_num_plans(plan_params), 1)

            # prepare() defers re-planning to forward_impl only when multiple
            # wrappers share one workspace_buffer (hybrid attention); for the
            # single-wrapper case it re-plans eagerly so cuda-graph capture works.
            attn_metadata.prepare()
            defer_plan = len(attn_metadata._plan_params_to_wrappers) > 1
            for wrappers in attn_metadata._plan_params_to_wrappers.values():
                self.assertEqual(wrappers.is_planned, not defer_plan)

            # assert value

            for result_1, result_2, result_3 in zip(results_1, results_2,
                                                    results_3):
                torch.testing.assert_close(
                    torch.cat((
                        result_1[:context_sequence_lengths[0] +
                                 context_sequence_lengths[1], :],
                        result_1[sum(context_sequence_lengths
                                     ):sum(context_sequence_lengths) + 2],
                    )),
                    torch.cat((
                        result_2[:context_sequence_lengths[0], :],
                        result_3[:context_sequence_lengths[1], :],
                        result_2[context_sequence_lengths[0]:, :],
                        result_3[context_sequence_lengths[1]:, :],
                    )))

        finally:
            kv_cache_manager.shutdown()

    @parameterized.expand([
        CUDAGraphTestScenario(
            batch_size=1,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            dtype=torch.float16,
        ),
        CUDAGraphTestScenario(
            batch_size=16,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            dtype=torch.bfloat16,
        ),
        CUDAGraphTestScenario(
            batch_size=16,
            num_heads=32,
            num_kv_heads=[32, 16],
            head_dim=128,
            dtype=torch.bfloat16,
        ),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_attention_with_cuda_graphs(
            self, test_scenario: CUDAGraphTestScenario) -> None:
        # This test exercises our CUDAGraph metadata class and makes sure
        # that the flashinfer attention layer is compatible with graph capture/replay.
        # We compare the CUDA graph results to the results without CUDA graph.
        batch_size = test_scenario.batch_size
        num_heads = test_scenario.num_heads
        num_kv_heads = test_scenario.num_kv_heads
        head_dim = test_scenario.head_dim
        dtype = test_scenario.dtype
        device = 'cuda'

        # For simplicity, just use 1 page per request in this example.
        tokens_per_block = 128
        past_seen_tokens = [
            random.randint(1, tokens_per_block - 1) for _ in range(batch_size)
        ]
        request_ids = list(range(batch_size))
        token_nums = (torch.tensor(past_seen_tokens) + 1).tolist()

        num_blocks = 16
        max_seq_len = tokens_per_block * num_blocks
        num_layers = 1 if isinstance(num_kv_heads, int) else len(num_kv_heads)
        mapping = Mapping(world_size=1, tp_size=1, rank=0)

        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
        if dtype == torch.float16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype for unit test")

        kv_cache_manager = KVCacheManagerV2(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        try:
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            gen_qs = []
            gen_ks = []
            gen_vs = []

            for i in range(num_layers):
                gen_qs.append([
                    torch.randn(1,
                                num_heads * head_dim,
                                dtype=dtype,
                                device=device) for _ in range(batch_size)
                ])

                kv_heads = num_kv_heads if isinstance(num_kv_heads,
                                                      int) else num_kv_heads[i]
                gen_ks.append([
                    torch.randn(1,
                                kv_heads * head_dim,
                                dtype=dtype,
                                device=device) for _ in range(batch_size)
                ])

                gen_vs.append([
                    torch.randn(1,
                                kv_heads * head_dim,
                                dtype=dtype,
                                device=device) for _ in range(batch_size)
                ])

            layers = []
            for i in range(num_layers):
                kv_heads = num_kv_heads if isinstance(num_kv_heads,
                                                      int) else num_kv_heads[i]
                layers.append(
                    FlashInferAttention(
                        layer_idx=i,
                        head_dim=head_dim,
                        num_heads=num_heads,
                        num_kv_heads=kv_heads,
                    ))

            seq_lens = torch.ones((batch_size, ), dtype=torch.int)
            attn_metadata_ref = TestingFlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
                max_num_requests=batch_size,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )

            attn_metadata_ref.kv_cache_manager = kv_cache_manager

            workspace = torch.empty(1024 * 1024 * 128,
                                    dtype=torch.int,
                                    device='cuda')
            attn_metadata_cuda_graph = TestingFlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=0,
                is_cuda_graph=True,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
                workspace_buffer=workspace,
                max_num_requests=batch_size,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )

            attn_metadata_ref.prepare()
            attn_metadata_cuda_graph.prepare()

            results_ref = []

            for i in range(num_layers):
                q = torch.cat(gen_qs[i])
                k = torch.cat(gen_ks[i])
                v = torch.cat(gen_vs[i])
                layer = layers[i]
                results_ref.append(layer.forward(q, k, v, attn_metadata_ref))

            graph = torch.cuda.CUDAGraph()
            for i in range(num_layers):
                layer = layers[i]
                q = torch.cat(gen_qs[i])
                k = torch.cat(gen_ks[i])
                v = torch.cat(gen_vs[i])
                # Warmup run, required by PT
                for _ in range(2):
                    layer.forward(q, k, v, attn_metadata_cuda_graph)

            results_actual = []
            with torch.cuda.graph(graph):
                for i in range(num_layers):
                    layer = layers[i]
                    q = torch.cat(gen_qs[i])
                    k = torch.cat(gen_ks[i])
                    v = torch.cat(gen_vs[i])
                    results_actual.append(
                        layer.forward(q, k, v, attn_metadata_cuda_graph))

            graph.replay()

            for result_actual, result_ref in zip(results_actual, results_ref):
                torch.testing.assert_close(result_actual,
                                           result_ref,
                                           atol=1e-2,
                                           rtol=0)

        finally:
            kv_cache_manager.shutdown()

    def test_ragged_prefill_no_kv_cache_uses_cudnn_plan(self) -> None:
        """Ragged QKV with ``kv_cache_manager=None`` plans via cuDNN (``_plan_ragged_cudnn_no_kv``)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = torch.device("cuda")
        num_heads = 8
        num_kv_heads = 8
        head_dim = 80
        hidden_size = num_heads * head_dim
        dtype = torch.bfloat16
        per_sequence_token_counts = [24, 56]
        num_context_sequences = len(per_sequence_token_counts)
        total_tokens = sum(per_sequence_token_counts)

        attn_metadata = TestingFlashInferAttentionMetadata(
            max_num_requests=max(num_context_sequences, 128),
            max_num_tokens=max(total_tokens * 2, 8192),
            kv_cache_manager=None,
        )
        attn_metadata.seq_lens = torch.tensor(
            per_sequence_token_counts,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        attn_metadata.num_contexts = num_context_sequences
        attn_metadata.request_ids = list(range(1, num_context_sequences + 1))
        attn_metadata.prompt_lens = list(per_sequence_token_counts)
        attn_metadata.prepare()

        layer = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        query_states = torch.randn(
            total_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        key_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        value_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        attention_output = layer.forward(
            query_states,
            key_states,
            value_states,
            attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )

        self.assertEqual(attention_output.shape, (total_tokens, hidden_size))
        self.assertGreaterEqual(len(attn_metadata._plan_params_to_wrappers), 1)
        # FlashInfer stores the chosen implementation on the wrapper (see
        # BatchPrefillWithRaggedKVCacheWrapper.__init__: self._backend = backend).
        # TRT-LLM no-cache ragged path passes backend="cudnn" in flashinfer.py.
        for flashinfer_wrappers in attn_metadata._plan_params_to_wrappers.values(
        ):
            ragged_wrapper = flashinfer_wrappers.ragged_prefill_wrapper
            self.assertIsNotNone(ragged_wrapper)
            self.assertEqual(
                getattr(ragged_wrapper, "_backend", None),
                "cudnn",
                msg="No-KV ragged prefill should use FlashInfer's cudnn backend",
            )

    def test_ragged_prefill_no_kv_cache_with_cuda_graphs(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = torch.device("cuda")
        num_heads = 8
        num_kv_heads = 8
        head_dim = 80
        hidden_size = num_heads * head_dim
        dtype = torch.bfloat16
        per_sequence_token_counts = [24, 56]
        num_context_sequences = len(per_sequence_token_counts)
        total_tokens = sum(per_sequence_token_counts)
        max_num_requests = 8

        seq_lens = torch.tensor(
            per_sequence_token_counts,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        request_ids = list(range(1, num_context_sequences + 1))

        attn_metadata_ref = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=num_context_sequences,
            max_num_requests=max_num_requests,
            max_num_tokens=max(total_tokens * 2, 8192),
            kv_cache_manager=None,
            request_ids=request_ids,
        )
        attn_metadata_cuda_graph = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=num_context_sequences,
            is_cuda_graph=True,
            max_num_requests=max_num_requests,
            max_num_tokens=max(total_tokens * 2, 8192),
            kv_cache_manager=None,
            request_ids=request_ids,
        )
        attn_metadata_ref.prompt_lens = list(per_sequence_token_counts)
        attn_metadata_cuda_graph.prompt_lens = list(per_sequence_token_counts)
        attn_metadata_ref.prepare()
        attn_metadata_cuda_graph.prepare()

        layer = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        query_states = torch.randn(
            total_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        key_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        value_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        result_ref = layer.forward(
            query_states,
            key_states,
            value_states,
            attn_metadata_ref,
            attention_mask=PredefinedAttentionMask.FULL,
        )

        graph = torch.cuda.CUDAGraph()
        for _ in range(2):
            layer.forward(
                query_states,
                key_states,
                value_states,
                attn_metadata_cuda_graph,
                attention_mask=PredefinedAttentionMask.FULL,
            )

        with torch.cuda.graph(graph):
            result_actual = layer.forward(
                query_states,
                key_states,
                value_states,
                attn_metadata_cuda_graph,
                attention_mask=PredefinedAttentionMask.FULL,
            )

        graph.replay()
        torch.testing.assert_close(result_actual, result_ref, atol=1e-2, rtol=0)


# --------------------------------------------------------------------------
# FlashInfer decode plan refresh under CUDA graphs
#
# Graph replay never re-enters ``forward_impl``: whatever split-KV schedule
# ``plan()`` computed at capture time is what every replay runs. A model with
# more than one decode wrapper (per-layer sliding window plus global attention)
# defers its plan to ``forward_impl``, so the recorded page tuple and
# ``_refresh_fa2_cuda_graph_plans()`` are the only things that keep a replayed
# plan current. This is plan bookkeeping rather than kernel maths, so it is
# testable without a GPU. (Relocated from test_flashinfer_plan_refresh.py.)
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _FakePlanParams:
    """Only the two fields ``_refresh_fa2_cuda_graph_plans`` looks at.

    ``eq=False`` keeps identity semantics, so two default instances are two
    distinct cache keys -- as two real per-layer plan params are.
    """

    attention_mask_data: Optional[object] = None
    multi_item_params: Optional[object] = None


class _FakeMetadata:
    """The three attributes the refresh reads, plus a recording re-plan.

    Constructing real metadata needs a KV cache manager and a device, and the
    method under test touches none of that: it reads ``num_blocks``,
    ``num_contexts`` and the plan cache, and calls ``_plan_with_params``.
    """

    def __init__(self, num_blocks, num_contexts, wrappers_by_params):
        self.num_blocks = num_blocks
        self.num_contexts = num_contexts
        self._plan_params_to_wrappers = wrappers_by_params
        self.replanned = []

    def _plan_with_params(self, plan_params):
        self.replanned.append(plan_params)
        wrappers = self._plan_params_to_wrappers[plan_params]
        wrappers.fa2_plan_num_blocks = tuple(
            self.num_blocks[self.num_contexts:])
        wrappers.is_planned = True

    refresh = FlashInferAttentionMetadata._refresh_fa2_cuda_graph_plans


def _refresh_wrappers(page_tuple):
    return FlashInferWrappers(is_planned=True, fa2_plan_num_blocks=page_tuple)


@pytest.mark.cpu_only
def test_page_table_change_replans_every_recorded_wrapper():
    """The multi-wrapper (variable sliding window) case the fix was written for.

    Every decode wrapper that was graph-captured now carries a page tuple, so a
    change to the generation page counts must re-plan all of them -- not just
    the first, and not just the tensor-core one.
    """
    windowed, global_ = _FakePlanParams(), _FakePlanParams()
    cache = {
        windowed: _refresh_wrappers((3, 3)),
        global_: _refresh_wrappers((3, 3))
    }
    # num_contexts=1: the first entry is a context request and is not part of
    # the generation page tuple.
    meta = _FakeMetadata([9, 4, 4], 1, cache)

    meta.refresh()

    assert meta.replanned == [windowed, global_]
    assert cache[windowed].fa2_plan_num_blocks == (4, 4)
    assert all(w.is_planned for w in cache.values())


@pytest.mark.cpu_only
def test_an_unchanged_page_table_does_not_replan():
    """The plan and its stream sync run only on a page-count change."""
    params = _FakePlanParams()
    cache = {params: _refresh_wrappers((4, 4))}
    meta = _FakeMetadata([4, 4], 0, cache)

    meta.refresh()

    assert meta.replanned == []
    assert cache[params].is_planned is True


@pytest.mark.cpu_only
def test_no_generations_clears_the_recorded_page_tuple():
    """A context-only step has no decode plan to keep current."""
    params = _FakePlanParams()
    cache = {params: _refresh_wrappers((4, 4))}
    meta = _FakeMetadata([7], 1, cache)

    meta.refresh()

    assert meta.replanned == []
    assert cache[params].fa2_plan_num_blocks is None
