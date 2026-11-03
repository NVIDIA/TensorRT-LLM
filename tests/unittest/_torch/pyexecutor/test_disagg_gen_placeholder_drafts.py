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
"""Placeholder drafts at the disaggregated generation handoff.

A generation request that arrives from the context server gets zero placeholder drafts when the
context server sent none. They have no draft-probability row on the generation server, so with
rejection sampling the request must ask for the one-hot row.
"""

import types
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType


def _disagg_generation_handoff(ctx_draft_tokens, rejection: bool, spec_decode: bool = True):
    request = types.SimpleNamespace(
        is_disagg_generation_transmission_complete=True,
        py_request_id=41,
        py_beam_width=1,
        py_draft_tokens=[],
        py_needs_onehot_draft_probs=False,
        context_phase_params=types.SimpleNamespace(
            first_gen_tokens=[17], draft_tokens=ctx_draft_tokens
        ),
        add_new_token=lambda token, beam: None,
    )
    executor = types.SimpleNamespace(
        model_engine=types.SimpleNamespace(
            max_total_draft_tokens=7,
            spec_config=types.SimpleNamespace(use_rejection_sampling=rejection)
            if spec_decode
            else None,
        ),
        enable_spec_decode=spec_decode,
        draft_model_engine=None,
        resource_manager=types.SimpleNamespace(
            resource_managers={ResourceManagerType.SEQ_SLOT_MANAGER: Mock()}
        ),
        disagg=types.SimpleNamespace(
            completed_gen_receives=lambda batch: batch.generation_requests,
            try_finish_gen_receive=lambda req: True,
        ),
        _setup_sampler_step=Mock(),
        _update_sampler_state_for_disagg_gen_request=Mock(return_value=True),
        _maybe_prepend_logprobs_and_logits=Mock(),
    )
    PyExecutor._prepare_disagg_gen_transmission_complete(
        executor, types.SimpleNamespace(generation_requests=[request])
    )
    return request


@pytest.mark.parametrize("ctx_draft_tokens", [None, [], [0] * 7], ids=["none", "empty", "zeros"])
@pytest.mark.parametrize("rejection", [True, False], ids=["rejection", "strict"])
def test_disagg_generation_placeholder_drafts_request_onehot_row(
    ctx_draft_tokens, rejection: bool
) -> None:
    # Placeholder drafts have no draft-probability row on the generation server.
    request = _disagg_generation_handoff(ctx_draft_tokens, rejection)
    assert request.py_draft_tokens == [0] * 7
    assert request.py_needs_onehot_draft_probs is rejection


@pytest.mark.parametrize(
    "ctx_draft_tokens", [[5, 0, 9, 9, 1, 2, 3], [5, 9]], ids=["full", "padded"]
)
def test_disagg_generation_real_drafts_keep_their_rows(ctx_draft_tokens) -> None:
    request = _disagg_generation_handoff(ctx_draft_tokens, rejection=True)
    assert request.py_draft_tokens == ctx_draft_tokens + [0] * (7 - len(ctx_draft_tokens))
    assert request.py_needs_onehot_draft_probs is False


def test_disagg_generation_without_spec_decode_has_no_drafts() -> None:
    request = _disagg_generation_handoff(None, rejection=False, spec_decode=False)
    assert request.py_draft_tokens == []
    assert request.py_needs_onehot_draft_probs is False
