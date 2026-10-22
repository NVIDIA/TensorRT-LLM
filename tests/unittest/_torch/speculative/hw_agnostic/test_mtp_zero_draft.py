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
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm._torch.speculative.mtp import MTPWorker
from tensorrt_llm.llmapi import MTPDecodingConfig


@pytest.mark.cpu_only
@pytest.mark.parametrize("worker_cls", [MTPWorker, MTPEagleWorker], ids=["mtp", "mtp-eagle"])
@pytest.mark.parametrize(
    "use_gdn_replay,num_contexts",
    [(True, 1), (True, 2), (False, 0), (None, 0)],
    ids=["gdn-mixed", "context-only", "other-hybrid", "attention"],
)
def test_mtp_zero_draft_commits_gdn_replay(
    worker_cls: type[MTPWorker] | type[MTPEagleWorker],
    use_gdn_replay: bool | None,
    num_contexts: int,
) -> None:
    worker = worker_cls(MTPDecodingConfig(max_draft_len=3))
    worker._execute_guided_decoder_if_present = mock.Mock()
    worker._sample_tokens_for_batch = mock.Mock(return_value=torch.tensor([3, 6]))
    handler = mock.Mock()
    worker.register_auxiliary_state_handler(handler)
    manager = SimpleNamespace(update_mamba_states=mock.Mock())
    if use_gdn_replay is not None:
        manager.use_gdn_mtp_replay = use_gdn_replay
    state_indices = torch.tensor([3, 1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_seqs=2,
        num_contexts=num_contexts,
        kv_cache_manager=manager,
        mamba_metadata=SimpleNamespace(state_indices=state_indices),
    )
    result = worker._forward_impl(
        input_ids=mock.sentinel.input_ids,
        position_ids=mock.sentinel.position_ids,
        hidden_states=mock.sentinel.hidden_states,
        logits=torch.zeros(2, 8),
        attn_metadata=attn_metadata,
        spec_metadata=SimpleNamespace(runtime_draft_len=0),
        draft_model=mock.sentinel.draft_model,
    )

    assert result["next_draft_tokens"].shape == (2, 0)
    torch.testing.assert_close(result["new_tokens"], torch.tensor([[3], [6]], dtype=torch.int))
    torch.testing.assert_close(result["new_tokens_lens"], torch.ones(2, dtype=torch.int))
    if use_gdn_replay and num_contexts < 2:
        manager.update_mamba_states.assert_called_once_with(
            attn_metadata=attn_metadata,
            num_accepted_tokens=result["new_tokens_lens"],
            state_indices=state_indices,
        )
        handler.commit_speculative_states.assert_called_once()
        accepted, indices, contexts = handler.commit_speculative_states.call_args.args
        assert accepted is result["new_tokens_lens"]
        torch.testing.assert_close(indices, state_indices)
        assert contexts == num_contexts
    else:
        manager.update_mamba_states.assert_not_called()
        handler.commit_speculative_states.assert_not_called()
