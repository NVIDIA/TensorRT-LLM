# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUDA graph coverage for GLM draft cache slot-table replay."""

from unittest.mock import Mock, call

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GLM draft slot-table replay requires CUDA graph support",
)


def test_draft_slot_tables_refresh_before_cuda_graph_replay() -> None:
    """Captured draft attention reads each batch's independently allocated slots."""
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
        Glm5NextCacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.metadata import (
        Glm5NextMamba2Metadata,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.speculative.interface import (
        prepare_attn_metadata_for_draft_replay,
        restore_attn_metadata_after_draft_replay,
    )

    batch_size = 3
    table_width = 4
    target_tables = torch.tensor(
        [[80, 81, 82, 83], [70, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.long,
        device="cuda",
    )
    target_tables_before = target_tables.clone()

    glm_metadata = object.__new__(Glm5NextMamba2Metadata)
    glm_metadata.glm_block_tables = target_tables
    glm_metadata._glm_draft_block_tables = None
    glm_metadata._glm_draft_block_tables_cpu = None

    target_manager = Mock(spec=Glm5NextCacheManager)
    draft_manager = Mock(spec=Glm5NextCacheManager)
    target_host_offsets = torch.zeros((1,), dtype=torch.int32)
    draft_host_offsets = torch.ones((1,), dtype=torch.int32)
    target_manager.host_kv_cache_block_offsets = target_host_offsets
    draft_manager.host_kv_cache_block_offsets = draft_host_offsets

    rows_by_request = {
        (1, 2, 3): [[90], [91, 92], []],
        (11, 12, 13): [[4, 5, 6, 7], [8], []],
        (21, 22, 23): [[9], [10, 11, 12, 13], []],
    }

    def lookup_draft_slots(request_ids: list[int]) -> list[list[int]]:
        """Return each batch's slots and reject host lookups during capture."""
        assert not torch.cuda.is_current_stream_capturing()
        return rows_by_request[tuple(request_ids)]

    draft_manager.get_batch_slot_tables.side_effect = lookup_draft_slots

    attn_metadata = object.__new__(TrtllmAttentionMetadata)
    attn_metadata.kv_cache_manager = target_manager
    attn_metadata.draft_kv_cache_manager = draft_manager
    target_offsets = torch.zeros((1, batch_size, 2, table_width), dtype=torch.int32, device="cuda")
    draft_offsets = torch.ones_like(target_offsets)
    attn_metadata.kv_cache_block_offsets = target_offsets
    attn_metadata.draft_kv_cache_block_offsets = draft_offsets
    attn_metadata.host_kv_cache_block_offsets = target_host_offsets
    attn_metadata.enable_flash_mla = False
    attn_metadata.draft_replay_swapped_attrs = {}
    attn_metadata.mamba_metadata = glm_metadata
    attn_metadata.request_ids = [1, 2, 3]
    attn_metadata.seq_lens = torch.tensor([1, 1, 0], dtype=torch.int32)

    assert glm_metadata._glm_draft_block_tables is None
    assert glm_metadata._glm_draft_block_tables_cpu is None
    warmup_state = prepare_attn_metadata_for_draft_replay(attn_metadata, draft_manager)
    assert warmup_state is not None
    draft_tables = glm_metadata._glm_draft_block_tables
    draft_tables_cpu = glm_metadata._glm_draft_block_tables_cpu
    assert draft_tables is not None
    assert draft_tables_cpu is not None
    torch.cuda.synchronize()
    torch.testing.assert_close(
        draft_tables,
        torch.tensor(
            [[90, 0, 0, 0], [91, 92, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
            device="cuda",
        ),
    )
    restore_attn_metadata_after_draft_replay(attn_metadata, warmup_state)
    assert glm_metadata.glm_block_tables is target_tables
    assert attn_metadata.kv_cache_manager is target_manager
    assert attn_metadata.kv_cache_block_offsets is target_offsets
    assert attn_metadata.host_kv_cache_block_offsets is target_host_offsets
    torch.testing.assert_close(target_tables, target_tables_before)
    lookup_count_before_capture = draft_manager.get_batch_slot_tables.call_count

    captured_tables = torch.empty_like(target_tables)
    graph = torch.cuda.CUDAGraph()
    capture_state: dict | None = None
    captured_draft_address = 0

    with torch.cuda.graph(graph):
        capture_state = prepare_attn_metadata_for_draft_replay(attn_metadata, draft_manager)
        assert capture_state is not None
        assert glm_metadata.glm_block_tables is draft_tables
        assert attn_metadata.kv_cache_manager is draft_manager
        assert attn_metadata.kv_cache_block_offsets is draft_offsets
        assert attn_metadata.host_kv_cache_block_offsets is draft_host_offsets
        captured_draft_address = glm_metadata.glm_block_tables.data_ptr()
        captured_tables.copy_(glm_metadata.glm_block_tables)
        restore_attn_metadata_after_draft_replay(attn_metadata, capture_state)

    assert draft_manager.get_batch_slot_tables.call_count == lookup_count_before_capture
    assert captured_draft_address == draft_tables.data_ptr()
    assert captured_draft_address != target_tables.data_ptr()
    assert glm_metadata.glm_block_tables is target_tables
    assert attn_metadata.kv_cache_manager is target_manager
    assert attn_metadata.kv_cache_block_offsets is target_offsets
    assert attn_metadata.host_kv_cache_block_offsets is target_host_offsets
    torch.testing.assert_close(
        draft_tables,
        torch.tensor(
            [[90, 0, 0, 0], [91, 92, 0, 0], [0, 0, 0, 0]],
            dtype=torch.long,
            device="cuda",
        ),
    )
    torch.testing.assert_close(target_tables, target_tables_before)

    replay_cases = [
        ([11, 12, 13], [[4, 5, 6, 7], [8, 0, 0, 0], [0, 0, 0, 0]]),
        ([21, 22, 23], [[9, 0, 0, 0], [10, 11, 12, 13], [0, 0, 0, 0]]),
    ]
    for request_ids, expected_rows in replay_cases:
        attn_metadata.request_ids = request_ids
        replay_state = prepare_attn_metadata_for_draft_replay(attn_metadata, draft_manager)
        assert replay_state is not None
        assert glm_metadata.glm_block_tables is draft_tables
        assert attn_metadata.kv_cache_manager is draft_manager
        assert attn_metadata.kv_cache_block_offsets is draft_offsets
        assert attn_metadata.host_kv_cache_block_offsets is draft_host_offsets
        assert glm_metadata.glm_block_tables.data_ptr() == captured_draft_address

        graph.replay()
        torch.cuda.synchronize()
        expected_tables = torch.tensor(expected_rows, dtype=torch.long, device="cuda")
        torch.testing.assert_close(captured_tables, expected_tables)

        restore_attn_metadata_after_draft_replay(attn_metadata, replay_state)
        assert glm_metadata.glm_block_tables is target_tables
        assert attn_metadata.kv_cache_manager is target_manager
        assert attn_metadata.kv_cache_block_offsets is target_offsets
        assert attn_metadata.host_kv_cache_block_offsets is target_host_offsets
        torch.testing.assert_close(target_tables, target_tables_before)

    assert draft_manager.get_batch_slot_tables.call_args_list == [
        call([1, 2, 3]),
        call([11, 12, 13]),
        call([21, 22, 23]),
    ]
