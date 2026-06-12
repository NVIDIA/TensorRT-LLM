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

from tensorrt_llm._torch.speculative.suffix_automaton import SAConfig, SuffixAutomatonManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_ensure_workspace_reserves_gpu_workspace():
    """_ensure_workspace() should make SA workspace visible in device free-memory accounting."""
    max_seq_len = 64
    max_num_requests = 2
    max_draft_len = 3
    manager = SuffixAutomatonManager(
        SAConfig(max_seq_len=max_seq_len, max_slots=max_num_requests),
        max_num_requests=max_num_requests,
    )

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info()

        manager._ensure_workspace(max_draft_len)
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()

        assert free_after < free_before
    finally:
        manager.shutdown()


def test_warmup_dummy_request_ids_do_not_create_request_state():
    """Synthetic resize/cudagraph request IDs should not populate real SA request slots."""
    max_seq_len = 64
    max_num_requests = 2
    max_draft_len = 3
    manager = SuffixAutomatonManager(
        SAConfig(max_seq_len=max_seq_len, max_slots=max_num_requests),
        max_num_requests=max_num_requests,
    )

    try:
        warmup_request_ids = [100, 101]
        manager.prepare(warmup_request_ids, max_draft_len)
        accepted_tokens = torch.ones(
            (len(warmup_request_ids), max_draft_len + 1), dtype=torch.int32, device="cuda"
        )
        num_accepted_tokens = torch.ones(
            (len(warmup_request_ids),), dtype=torch.int32, device="cuda"
        )

        manager.extend(
            warmup_request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
        )

        assert manager._request_to_slot == {}
        assert manager._active_slots == set()
    finally:
        manager.shutdown()
