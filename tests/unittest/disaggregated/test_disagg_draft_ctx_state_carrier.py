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
"""F0a context-first: the drafter window decision reaches the generation request in serving.

It rode the response's opaque_state, which becomes the C++ ContextPhaseParams' serialized
DataTransceiverState: the bytes were parsed and re-serialized, so in serving every generation
request saw no state (no_ctx_state) and asked for no draft pages. The decision now rides its own
DisaggregatedParams field, draft_ctx_state, set on the context worker's LlmResult (like
remote_tail_start) and carried through the serving protocol.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import _response_draft_ctx_state
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "aux, context_only, expected",
    [
        ((9000, 3, True), True, [9000, 3, 1]),
        ((0, 0, False), True, [0, 0, 0]),  # the window did not travel: the receiver asks none
        ((64, 0, True), False, None),  # a generation request's own aux (generation-first) stays off
        (None, True, None),  # no window decision (F0a off or not engaged)
    ],
)
def test_the_context_response_carries_the_decision(aux, context_only, expected):
    request = SimpleNamespace(py_draft_ctx_aux=aux, is_context_only_request=context_only)
    assert _response_draft_ctx_state(request) == expected


def _context_result(draft_ctx_state=None):
    result = SimpleNamespace(
        is_final=True,
        decoding_iter=1,
        avg_decoded_tokens_per_iter=1.0,
        context_phase_params=SimpleNamespace(
            first_gen_tokens=[7],
            req_id=1,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=0,
            disagg_info_endpoint=None,
        ),
        finish_reasons=[tllm.FinishReason.LENGTH],
        output_token_ids=[[7]],
        sequence_index=0,
        cum_log_probs=None,
        log_probs=None,
        generation_logits=None,
        context_logits=None,
        request_perf_metrics=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
    )
    if draft_ctx_state is not None:
        result.draft_ctx_state = draft_ctx_state
    return result


@pytest.mark.parametrize("state", [[9000, 3, 1], [0, 0, 0], None])
def test_the_disaggregated_params_carry_draft_ctx_state(state):
    result = GenerationResultBase(id=1, sampling_params=SamplingParams(max_tokens=4))
    result._handle_response(SimpleNamespace(result=_context_result(state), has_error=lambda: False))
    params = result.disaggregated_params
    assert params.request_type == "context_only"
    assert params.draft_ctx_state == state
    assert params.opaque_state is None


def test_draft_ctx_state_never_enters_the_cpp_context_phase_params():
    """The generation request builds its C++ ContextPhaseParams from opaque_state only."""
    params = DisaggregatedParams(
        request_type="generation_only",
        ctx_request_id=1,
        draft_ctx_state=[9000, 3, 1],
    )
    cpp = params.get_context_phase_params()
    assert cpp.opaque_state is None
    assert params.draft_ctx_state == [9000, 3, 1]
