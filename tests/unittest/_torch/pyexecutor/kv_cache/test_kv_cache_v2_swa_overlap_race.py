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

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType, LayerType
from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

_BATCH = 4
_TOKENS_PER_BLOCK = 16
_TOKENS_PER_REQUEST = 5 * _TOKENS_PER_BLOCK
_SWA_WINDOW = 2 * _TOKENS_PER_BLOCK
_STAGING_SOURCES = ("copy_idx", "page_table", "scratch_metadata")


def _build_manager() -> KVCacheManagerV2:
    model_config = ModelConfigCpp(
        vocab_size=32000,
        num_layers=2,
        num_attention_layers=2,
        num_rnn_layers=0,
        num_heads=2,
        hidden_size=32,
        data_type=DataType.HALF,
    )
    model_config.layer_types = [LayerType.ATTENTION] * 2
    model_config.set_num_kv_heads(2)

    return KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            max_attention_window=[_SWA_WINDOW],
            free_gpu_memory_fraction=0.1,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=True,
            tokens_per_block=_TOKENS_PER_BLOCK,
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=128,
        max_batch_size=2 * _BATCH,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        model_config=model_config,
        max_beam_width=1,
    )


def _empty_offsets(manager: KVCacheManagerV2) -> torch.Tensor:
    return torch.full(
        (
            manager.num_attention_op_pools,
            _BATCH,
            2,
            manager.max_blocks_per_seq,
        ),
        -12345,
        dtype=torch.int32,
        device="cuda",
    )


def _copy_offsets(
    manager: KVCacheManagerV2,
    dst_tensor: torch.Tensor,
    request_ids: list[int],
    copy_idx: torch.Tensor,
) -> None:
    manager._copy_batch_block_offsets_per_layer(
        dst_tensor,
        request_ids,
        copy_idx,
        len(request_ids),
        len(request_ids),
    )


def _overwrite_source(manager: KVCacheManagerV2, source: str, copy_idx: torch.Tensor) -> None:
    if source == "copy_idx":
        copy_idx.fill_(manager.max_batch_size - 1)
    elif source == "scratch_metadata":
        manager._host_scratch_begs_staging.fill_(777777)
        manager._host_scratch_ends_staging.fill_(777777)
        manager._host_scratch_slots_staging.fill_(777777)
    elif source == "page_table":
        manager.host_kv_cache_block_offsets.fill_(777777)
    else:
        raise AssertionError(f"Unknown staging source: {source}")


def _queue_copy_behind_gate(
    manager: KVCacheManagerV2,
    dst_tensor: torch.Tensor,
    request_ids: list[int],
    copy_idx: torch.Tensor,
) -> torch.cuda.Stream:
    gate_stream = torch.cuda.Stream()
    gate_event = torch.cuda.Event()
    gate_lhs = torch.ones((8192, 8192), dtype=torch.float16, device="cuda")
    gate_rhs = torch.ones_like(gate_lhs)

    with torch.cuda.stream(gate_stream):
        torch.mm(gate_lhs, gate_rhs)
        gate_event.record(gate_stream)

    with torch.cuda.stream(manager._stream):
        manager._stream.wait_event(gate_event)
        _copy_offsets(manager, dst_tensor, request_ids, copy_idx)

    assert not gate_event.query()
    return gate_stream


def _assert_scratch_offset(
    manager: KVCacheManagerV2,
    offsets: torch.Tensor,
    request_ids: list[int],
) -> None:
    for op_pool_idx in range(manager.num_attention_op_pools):
        for role_idx in range(2):
            pool_id = int(manager._device_attention_op_pool_ids[op_pool_idx, role_idx].item())
            scale = int(manager._device_attention_op_scales[op_pool_idx, role_idx].item())
            layer_offset = int(
                manager._device_attention_op_layer_offsets[op_pool_idx, role_idx].item()
            )
            scratch_pages = int(
                manager._device_attention_op_scratch_pages[op_pool_idx, role_idx].item()
            )
            for context_idx, request_id in enumerate(request_ids):
                desc = manager.kv_cache_map[request_id].get_scratch_desc(pool_id)
                if (
                    desc is None
                    or not desc.slot_ids
                    or desc.range.beg >= manager.max_blocks_per_seq
                ):
                    continue
                block_position = desc.range.beg
                total_offset = (block_position - desc.range.beg) * scratch_pages
                slot_idx = total_offset // scale
                if slot_idx >= len(desc.slot_ids):
                    continue
                expected = desc.slot_ids[slot_idx] * scale + (total_offset + layer_offset) % scale
                assert (
                    offsets[op_pool_idx, context_idx, role_idx, block_position].item() == expected
                )
                return
    raise AssertionError("No nonempty SWA scratch descriptor matched an output offset")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("source", _STAGING_SOURCES)
def test_swa_staging_snapshot_survives_host_overwrite(source: str) -> None:
    manager = _build_manager()
    request_ids = list(range(1, 1 + _BATCH))

    try:
        manager.add_dummy_requests(
            request_ids=request_ids,
            token_nums=[_TOKENS_PER_REQUEST] * _BATCH,
        )
        scratch_descs = [
            manager.kv_cache_map[request_id].get_scratch_desc(pool_id)
            for request_id in request_ids
            for pool_id in range(manager.num_pools)
        ]
        assert any(desc is not None and desc.slot_ids for desc in scratch_descs)
        copy_idx = manager.index_mapper.get_copy_index(request_ids, _BATCH, 1)

        baseline = _empty_offsets(manager)
        with torch.cuda.stream(manager._stream):
            _copy_offsets(manager, baseline, request_ids, copy_idx)
        manager._stream.synchronize()
        baseline = baseline.clone()

        raced = _empty_offsets(manager)
        gate_stream = _queue_copy_behind_gate(manager, raced, request_ids, copy_idx)
        second = None
        second_request_ids = list(reversed(request_ids))
        if source == "scratch_metadata":
            second_copy_idx = manager.index_mapper.get_copy_index(second_request_ids, _BATCH, 1)
            second = _empty_offsets(manager)
            with torch.cuda.stream(manager._stream):
                _copy_offsets(manager, second, second_request_ids, second_copy_idx)
        _overwrite_source(manager, source, copy_idx)
        gate_stream.synchronize()
        manager._stream.synchronize()

        assert torch.any(baseline != -12345)
        if source == "scratch_metadata":
            assert second is not None and torch.any(second != -12345)
            _assert_scratch_offset(manager, baseline, request_ids)
            _assert_scratch_offset(manager, second, second_request_ids)
        torch.testing.assert_close(raced, baseline)
    finally:
        manager.shutdown()
