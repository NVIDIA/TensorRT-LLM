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
"""Contract tests for the base ``AttentionMetadata.update_helix_param`` hook.

The base hook is a no-op for the plain-helix parameters, but it must reject
``helix_owned_new_tokens``: a backend that inherits the base hook has no
storage for the owned counts, so silently accepting them would corrupt
helix + speculative-decode runs. Only backends that override the hook (the
TRTLLM backend) may consume them. These checks run on CPU.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends import VanillaAttentionMetadata


def _make_metadata() -> VanillaAttentionMetadata:
    """Minimal metadata using the base-class update_helix_param hook."""
    return VanillaAttentionMetadata(
        seq_lens=torch.tensor([1], dtype=torch.int),
        num_contexts=0,
        max_num_requests=1,
        max_num_tokens=8,
        kv_cache_manager=None,
        request_ids=[0],
    )


def test_base_update_helix_param_accepts_plain_helix():
    """Omitted or None owned counts keep the base hook a no-op."""
    metadata = _make_metadata()
    metadata.update_helix_param(
        helix_position_offsets=[0],
        helix_is_inactive_rank=[False],
    )
    metadata.update_helix_param(
        helix_position_offsets=[0],
        helix_is_inactive_rank=[False],
        helix_owned_new_tokens=None,
    )


def test_base_update_helix_param_rejects_owned_new_tokens():
    """Non-None owned counts must raise instead of being silently dropped."""
    metadata = _make_metadata()
    with pytest.raises(NotImplementedError, match="helix_owned_new_tokens"):
        metadata.update_helix_param(
            helix_position_offsets=[0],
            helix_is_inactive_rank=[False],
            helix_owned_new_tokens=[1],
        )
