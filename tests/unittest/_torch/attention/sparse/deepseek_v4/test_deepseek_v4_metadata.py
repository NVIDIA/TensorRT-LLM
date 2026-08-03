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

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata


def test_prepare_computes_draft_sliding_block_tables_before_base_prepare(monkeypatch):
    """DeepSeek-V4 MTP draft KV managers need their sliding tables prepared.

    The base TRT-LLM metadata prepare path copies block offsets from both the
    target and draft managers. DeepSeek-V4's copy path consumes precomputed
    sliding-window tables, so the draft manager must compute them before the
    base prepare reaches copy_batch_block_offsets().
    """
    metadata = object.__new__(DeepseekV4TrtllmAttentionMetadata)
    metadata.kv_cache_manager = MagicMock()
    metadata.draft_kv_cache_manager = MagicMock()
    metadata.request_ids = [11, 12, 13]
    metadata.num_contexts = 2

    def stop_at_base_prepare(self):
        raise RuntimeError("base prepare reached")

    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", stop_at_base_prepare)

    with pytest.raises(RuntimeError, match="base prepare reached"):
        DeepseekV4TrtllmAttentionMetadata.prepare(metadata)

    metadata.kv_cache_manager.compute_sliding_block_tables.assert_called_once_with(
        metadata.request_ids,
        metadata.num_contexts,
    )
    metadata.draft_kv_cache_manager.compute_sliding_block_tables.assert_called_once_with(
        metadata.request_ids,
        metadata.num_contexts,
    )
