# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local offload admission and cached-prefill lifecycle, without decode attention."""

import pickle
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)


def _run_offload_rank(layout: str, reuse: bool, overlap: bool) -> int:
    from _torch.attention.sparse.deepseek_v4.test_deepseek_v4_cache_manager import (
        TestDeepseekV4CacheManager,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=2,
        rank=rank,
        tp_size=1 if layout == "pp" else 2,
        pp_size=2 if layout == "pp" else 1,
        enable_attention_dp=layout == "adp",
    )
    fixture = TestDeepseekV4CacheManager()
    with torch.cuda.stream(torch.cuda.Stream()):
        manager, config = fixture._create_deepseek_v4_cache_manager(
            tokens_per_block=128,
            max_batch_size=2,
            max_seq_len=512,
            compress_ratios=[4, 4, 128, 1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            indexer_k_dtype="fp8",
            host_cache_size=32 << 20,
            enable_kv_cache_offload=True,
            enable_block_reuse=reuse,
            enable_swa_scratch_reuse=False,
            disable_overlap_scheduler=not overlap,
            mapping=mapping,
        )
    executor = object.__new__(PyExecutor)
    executor.kv_cache_manager = manager
    executor.dist = Distributed.get(mapping)
    executor.enable_attention_dp = mapping.enable_attention_dp
    requests = []
    batch = ScheduledRequests()
    # ADP ranks may have no local requests while their peers perform prefill.
    has_request = layout != "adp" or rank == 0
    try:
        for iteration in range(2 if reuse else 1):
            request = fixture._create_request(20 + iteration, 129) if has_request else None
            if request is not None:
                requests.append(request)
                assert manager.prepare_context(request)
                reused = request.context_current_position
                assert reused == (128 if iteration else 0)
                assert manager.resize_context(request, request.context_chunk_size)
                batch.context_requests_last_chunk = [request]
            else:
                batch.context_requests_last_chunk = []
            batch.generation_requests = []
            manager.prepare_resources(batch)
            executor._validate_kv_cache_admission(batch)
            if request is not None:
                metadata = DeepseekV4TrtllmAttentionMetadata(
                    seq_lens=torch.tensor([request.context_chunk_size], dtype=torch.int32),
                    num_contexts=1,
                    max_num_requests=2,
                    kv_cache_params=KVCacheParams(
                        use_cache=True, num_cached_tokens_per_seq=[reused]
                    ),
                    kv_cache_manager=manager,
                    request_ids=[request.py_request_id],
                    prompt_lens=[request.context_chunk_size],
                    max_num_tokens=512,
                    mapping=mapping,
                    sparse_attention_config=config,
                )
                metadata.prepare()
                assert metadata.sparse_offload_state is None
                if iteration:
                    for layer in manager.pp_layers:
                        if config.compress_ratios[layer] != 4:
                            continue
                        indices = fixture._get_page_indices(
                            request, manager, layer, DeepseekV4AttentionType.COMPRESS
                        )
                        values = manager.get_buffers(layer, DeepseekV4AttentionType.COMPRESS)
                        assert (values[indices[0]].view(torch.uint8) == rank + 17).all()
                else:
                    producer = torch.cuda.Stream()
                    producer.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(producer):
                        if rank == 1:
                            torch.cuda._sleep(2_000_000)
                        for layer in manager.pp_layers:
                            if config.compress_ratios[layer] == 4:
                                manager.get_buffers(layer, DeepseekV4AttentionType.COMPRESS).view(
                                    torch.uint8
                                ).fill_(rank + 17)
                    # This is the executor's common forward-completion dependency.
                    torch.cuda.current_stream().wait_stream(producer)
                request.context_current_position = request.prompt_len
                request.add_new_token(1, 0)
            manager.update_context_resources(batch)
            if request is not None:
                cache = manager.kv_cache_map[request.py_request_id]
                assert cache.history_length == 129
                for layer in manager.pp_layers:
                    if config.compress_ratios[layer] == 4:
                        group = manager.impl.get_layer_group_id(
                            manager._layer_attn_to_layer_id[layer, DeepseekV4AttentionType.COMPRESS]
                        )
                        assert cache.get_page_storage_snapshot(group).eligible_history_blocks == 0
                assert manager.try_allocate_generation(request)
                batch.context_requests_last_chunk = []
                batch.generation_requests = [request]
            executor._validate_kv_cache_admission(batch)
            manager._publish_sparse_metadata()
            if request is not None:
                for layer in manager.pp_layers:
                    if config.compress_ratios[layer] == 4:
                        group = manager.impl.get_layer_group_id(
                            manager._layer_attn_to_layer_id[layer, DeepseekV4AttentionType.COMPRESS]
                        )
                        snapshot = cache.get_page_storage_snapshot(group)
                        assert list(snapshot.cache_levels) == [1, 0]
                        assert snapshot.eligible_history_blocks == 1

        # PP peers must reject a propagated schedule if any rank failed local
        # allocation. All ranks take the same failure path before model work.
        local_output = SimpleNamespace(
            context_requests=[], generation_requests=batch.generation_requests if rank == 0 else []
        )
        if layout != "adp":
            with pytest.raises(RuntimeError, match="admission differs across ranks"):
                executor._validate_kv_cache_admission(batch, local_output)
        return rank
    finally:
        for request in reversed(requests):
            manager.free_resources(request)
        manager.shutdown()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize("layout", ["tp", "pp", "adp"])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_deepseek_v4_runtime_offload_ranks(layout: str, reuse: bool, overlap: bool) -> None:
    with MPIPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_offload_rank, layout, reuse, overlap) for _ in range(2)]
        assert sorted(future.result(timeout=180) for future in futures) == [0, 1]
