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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.interface import (
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)

pytestmark = pytest.mark.cpu_only


def _metadata() -> TrtllmAttentionMetadata:
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata.enable_flash_mla = False
    metadata.kv_cache_manager = Mock(name="target_manager")
    metadata.kv_cache_block_offsets = torch.tensor([10, 20], dtype=torch.int32)
    metadata.host_kv_cache_block_offsets = metadata.kv_cache_block_offsets.clone()
    metadata.draft_kv_cache_block_offsets = torch.tensor([30, 40], dtype=torch.int32)
    metadata.draft_replay_swapped_attrs = {}
    metadata.mamba_metadata = SimpleNamespace(
        prepare_for_draft_forward=Mock(), restore_after_draft_forward=Mock()
    )
    return metadata


def test_backend_preparation_failure_restores_metadata_and_allows_retry() -> None:
    metadata = _metadata()
    draft_manager = Mock(host_kv_cache_block_offsets=torch.tensor([30, 40]))
    target_manager = metadata.kv_cache_manager
    target_offsets = metadata.kv_cache_block_offsets
    target_host_offsets = metadata.host_kv_cache_block_offsets
    target_table = torch.tensor([1, 2])
    draft_table = torch.tensor([3, 4])
    metadata.backend_table = target_table

    def fail_backend_preparation(attn_metadata: TrtllmAttentionMetadata) -> None:
        assert attn_metadata.kv_cache_manager is draft_manager
        attn_metadata.swap_for_draft("backend_table", draft_table)
        raise LookupError("draft backend lookup failed")

    hook = metadata.mamba_metadata.prepare_for_draft_forward
    hook.side_effect = fail_backend_preparation

    with pytest.raises(LookupError, match="draft backend lookup failed"):
        prepare_attn_metadata_for_draft_replay(metadata, draft_manager)

    assert metadata.kv_cache_manager is target_manager
    assert metadata.kv_cache_block_offsets is target_offsets
    assert metadata.host_kv_cache_block_offsets is target_host_offsets
    assert metadata.backend_table is target_table
    assert metadata.draft_replay_swapped_attrs == {}
    metadata.mamba_metadata.restore_after_draft_forward.assert_not_called()

    hook.side_effect = None
    hook.return_value = {"backend_table": target_table}
    saved_state = prepare_attn_metadata_for_draft_replay(metadata, draft_manager)

    assert metadata.kv_cache_manager is draft_manager
    assert metadata.kv_cache_block_offsets is metadata.draft_kv_cache_block_offsets
    assert metadata.host_kv_cache_block_offsets is draft_manager.host_kv_cache_block_offsets
    restore_attn_metadata_after_draft_replay(metadata, saved_state)

    assert metadata.kv_cache_manager is target_manager
    assert metadata.kv_cache_block_offsets is target_offsets
    assert metadata.host_kv_cache_block_offsets is target_host_offsets
    assert metadata.draft_replay_swapped_attrs == {}
    metadata.mamba_metadata.restore_after_draft_forward.assert_called_once_with(
        metadata, hook.return_value
    )


def test_flash_mla_preparation_failure_restores_target_buffers() -> None:
    metadata = _metadata()
    metadata.enable_flash_mla = True
    metadata._flash_mla_metadata_valid = True
    metadata.request_ids = [17]
    metadata.block_ids_per_seq = torch.tensor([[1, 2]], dtype=torch.int32)
    metadata.kv_block_ids_per_seq = torch.tensor([[3, 4]], dtype=torch.int32)
    metadata.draft_block_ids_per_seq = torch.zeros_like(metadata.block_ids_per_seq)
    metadata.draft_kv_block_ids_per_seq = torch.zeros_like(metadata.kv_block_ids_per_seq)
    draft_manager = Mock(host_kv_cache_block_offsets=torch.tensor([30, 40]))
    draft_manager.get_block_ids_per_seq.side_effect = LookupError("draft block lookup failed")
    target_fields = {
        name: getattr(metadata, name)
        for name in (
            "kv_cache_manager",
            "kv_cache_block_offsets",
            "host_kv_cache_block_offsets",
            "block_ids_per_seq",
            "kv_block_ids_per_seq",
        )
    }

    with (
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        pytest.raises(LookupError, match="draft block lookup failed"),
    ):
        prepare_attn_metadata_for_draft_replay(metadata, draft_manager)

    draft_manager.get_block_ids_per_seq.assert_called_once_with([17])
    for name, original in target_fields.items():
        assert getattr(metadata, name) is original
    torch.testing.assert_close(
        metadata.block_ids_per_seq, torch.tensor([[1, 2]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.kv_block_ids_per_seq, torch.tensor([[3, 4]], dtype=torch.int32)
    )
    assert not metadata._flash_mla_metadata_valid
    assert metadata.draft_replay_swapped_attrs == {}
    metadata.mamba_metadata.prepare_for_draft_forward.assert_not_called()
    metadata.mamba_metadata.restore_after_draft_forward.assert_not_called()
