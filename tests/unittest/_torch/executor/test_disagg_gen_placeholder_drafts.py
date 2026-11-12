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

A generation request that arrives from the context server carries the context server's draft
tokens, or none. The generation server fills missing drafts with zeros so the first step has the
draft-length input shape. These placeholders have no draft-probability row on the generation
server, so with rejection sampling the request must ask for the one-hot row; otherwise the first
verify step reads an unwritten row and can accept token 0.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

K = 7


def _executor(rejection, spec_decode=True):
    return SimpleNamespace(
        resource_manager=SimpleNamespace(
            resource_managers={
                ResourceManagerType.SEQ_SLOT_MANAGER: SimpleNamespace(
                    prepare_resources=lambda requests: None
                ),
            }
        ),
        _setup_sampler_step=lambda requests: None,
        model_engine=SimpleNamespace(
            enable_spec_decode=spec_decode,
            max_total_draft_tokens=K,
            spec_config=SimpleNamespace(use_rejection_sampling=rejection) if spec_decode else None,
        ),
        kv_cache_transceiver=None,
        _update_sampler_state_for_disagg_gen_request=lambda req, beam_width, tokens: True,
        _maybe_prepend_logprobs_and_logits=lambda req, beam_width: None,
    )


def _arrived(ctx_draft_tokens):
    req = SimpleNamespace(
        is_disagg_generation_transmission_complete=True,
        py_request_id=1,
        prompt_len=16,
        py_beam_width=1,
        py_draft_tokens=[],
        py_needs_onehot_draft_probs=False,
        context_phase_params=SimpleNamespace(first_gen_tokens=[42], draft_tokens=ctx_draft_tokens),
        new_tokens=[],
    )
    req.add_new_token = lambda token, beam: req.new_tokens.append(token)
    return req


def _handoff(executor, req):
    PyExecutor._prepare_disagg_gen_transmission_complete(
        executor, SimpleNamespace(generation_requests=[req])
    )
    return req


@pytest.mark.parametrize("ctx_draft_tokens", [None, [], [0] * K], ids=["none", "empty", "zeros"])
@pytest.mark.parametrize("rejection", [True, False], ids=["rejection", "strict"])
def test_placeholder_drafts_request_onehot_row(ctx_draft_tokens, rejection):
    req = _handoff(_executor(rejection), _arrived(ctx_draft_tokens))
    assert req.py_draft_tokens == [0] * K
    assert req.py_needs_onehot_draft_probs is rejection
    assert req.new_tokens == [42]


@pytest.mark.parametrize("rejection", [True, False], ids=["rejection", "strict"])
def test_real_drafts_keep_their_rows(rejection):
    drafts = [5, 0, 9, 9, 1, 2, 3]
    req = _handoff(_executor(rejection), _arrived(list(drafts)))
    assert req.py_draft_tokens == drafts
    assert req.py_needs_onehot_draft_probs is False


def test_flag_already_set_is_kept():
    req = _arrived([3] * K)
    req.py_needs_onehot_draft_probs = True
    _handoff(_executor(rejection=True), req)
    assert req.py_needs_onehot_draft_probs is True


def test_no_spec_decode_has_no_drafts():
    req = _handoff(_executor(rejection=False, spec_decode=False), _arrived(None))
    assert req.py_draft_tokens == []
    assert req.py_needs_onehot_draft_probs is False
