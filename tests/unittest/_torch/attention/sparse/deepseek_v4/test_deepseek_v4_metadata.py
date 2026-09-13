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

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

pytestmark = pytest.mark.cpu_only


class _RecordingKvCacheManager:
    max_seq_len = 256
    tokens_per_block = 128

    def __init__(self) -> None:
        self.compute_calls = []
        self.copy_calls = []

    def compute_sliding_block_tables(self, request_ids: list[int], num_contexts: int) -> None:
        self.compute_calls.append((list(request_ids), num_contexts))

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: int | None = None,
    ) -> None:
        self.copy_calls.append(
            (dst_tensor, list(request_ids), beam_width, num_contexts, num_seqs, max_blocks)
        )


class _RecordingDeepseekV4CacheManager(DeepseekV4CacheManager):
    max_seq_len = 256
    tokens_per_block = 128

    def __init__(self) -> None:
        self.compute_calls = []
        self.copy_calls = []

    def compute_sliding_block_tables(self, request_ids: list[int], num_contexts: int) -> None:
        self.compute_calls.append((list(request_ids), num_contexts))

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: list[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: int | None = None,
    ) -> None:
        if not self.compute_calls:
            raise AssertionError("draft sliding block tables must be computed before copy")
        self.copy_calls.append(
            (dst_tensor, list(request_ids), beam_width, num_contexts, num_seqs, max_blocks)
        )
        raise RuntimeError("draft copy reached")


def _minimal_metadata_for_base_prepare() -> DeepseekV4TrtllmAttentionMetadata:
    metadata = object.__new__(DeepseekV4TrtllmAttentionMetadata)
    metadata.kv_cache_manager = _RecordingKvCacheManager()
    metadata.draft_kv_cache_manager = _RecordingDeepseekV4CacheManager()
    metadata.request_ids = [11, 12]
    metadata._seq_lens = torch.tensor([3, 2], dtype=torch.int)
    metadata._seq_lens_kv = torch.tensor([3, 2], dtype=torch.int)
    metadata._seq_lens_cuda = metadata._seq_lens
    metadata._seq_lens_kv_cuda = metadata._seq_lens_kv
    metadata._num_contexts = 1
    metadata._num_generations = 1
    metadata._num_tokens = 5
    metadata._num_ctx_tokens = 3
    metadata.prompt_lens = [3, 1]
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[0, 1],
        num_extra_kv_tokens=0,
    )
    metadata.beam_width = 1
    metadata.enable_flash_mla = False
    metadata.enable_helix = False
    metadata.enable_context_mla_with_cached_kv = False
    metadata.is_spec_decoding_enabled = False
    metadata.runtime_features = None
    metadata.kv_cache_block_offsets = torch.empty(2, 1, 1, 1, dtype=torch.int32)
    metadata.draft_kv_cache_block_offsets = torch.empty(2, 1, 1, 1, dtype=torch.int32)
    metadata.prompt_lens_cpu = torch.empty(2, dtype=torch.int)
    metadata.prompt_lens_cuda = torch.empty(2, dtype=torch.int)
    metadata.kv_lens = torch.empty(2, dtype=torch.int)
    metadata.kv_lens_cuda = torch.empty(2, dtype=torch.int)
    metadata.host_total_kv_lens = torch.empty(2, dtype=torch.int)
    metadata.host_request_types = torch.empty(2, dtype=torch.int)
    return metadata


def test_prepare_computes_draft_sliding_block_tables_before_draft_copy() -> None:
    """DeepSeek-V4 MTP draft KV managers need their sliding tables prepared.

    The base TRT-LLM metadata prepare path copies block offsets from both the
    target and draft managers. DeepSeek-V4's copy path consumes precomputed
    sliding-window tables, so the draft manager must compute them before the
    base prepare reaches copy_batch_block_offsets().
    """
    metadata = _minimal_metadata_for_base_prepare()

    with pytest.raises(RuntimeError, match="draft copy reached"):
        DeepseekV4TrtllmAttentionMetadata.prepare(metadata)

    assert metadata.kv_cache_manager.compute_calls == [([11, 12], 1)]
    assert metadata.draft_kv_cache_manager.compute_calls == [([11, 12], 1)]
    assert len(metadata.kv_cache_manager.copy_calls) == 1
    assert len(metadata.draft_kv_cache_manager.copy_calls) == 1


def test_prepare_skips_non_deepseek_draft_sliding_table_lookalike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = object.__new__(DeepseekV4TrtllmAttentionMetadata)
    metadata.kv_cache_manager = MagicMock()
    metadata.draft_kv_cache_manager = MagicMock()
    metadata.request_ids = [11]
    metadata.num_contexts = 1

    def _stop_at_base_prepare(self: TrtllmAttentionMetadata) -> None:
        raise RuntimeError("base prepare reached")

    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", _stop_at_base_prepare)

    with pytest.raises(RuntimeError, match="base prepare reached"):
        DeepseekV4TrtllmAttentionMetadata.prepare(metadata)

    metadata.draft_kv_cache_manager.compute_sliding_block_tables.assert_not_called()


def test_prepare_allows_plain_draft_manager_without_sliding_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlainDraftManager:
        pass

    metadata = object.__new__(DeepseekV4TrtllmAttentionMetadata)
    metadata.kv_cache_manager = MagicMock()
    metadata.draft_kv_cache_manager = PlainDraftManager()
    metadata.request_ids = [11]
    metadata.num_contexts = 1

    def _stop_at_base_prepare(self: TrtllmAttentionMetadata) -> None:
        raise RuntimeError("base prepare reached")

    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", _stop_at_base_prepare)

    with pytest.raises(RuntimeError, match="base prepare reached"):
        DeepseekV4TrtllmAttentionMetadata.prepare(metadata)
