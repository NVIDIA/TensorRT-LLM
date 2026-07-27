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
from uuid import uuid4

import pytest

from tensorrt_llm._torch.disaggregation.handoff import HandoffEventState, HandoffLifecycleEvent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.executor.base_worker import _send_rsp
from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.executor.utils import is_control_only_llm_response
from tensorrt_llm.sampling_params import SamplingParams


def _params():
    return SimpleNamespace(
        logical_request_id=17,
        prefill_artifact_id=str(uuid4()),
        artifact_version=3,
        handoff_attempt_uuid=str(uuid4()),
        consumer_grant_id=str(uuid4()),
        transfer_session_id=str(uuid4()),
    )


def test_handoff_event_from_params_preserves_exact_identity():
    params = _params()

    event = HandoffLifecycleEvent.from_params(
        params,
        HandoffEventState.HANDOFF_COMMITTED,
    )

    assert event.committed
    assert event.session.logical_request_id == params.logical_request_id
    assert str(event.consumer_grant_id) == params.consumer_grant_id


def test_handoff_event_rejects_protocol_downgrade():
    event = HandoffLifecycleEvent.from_params(
        _params(),
        HandoffEventState.HANDOFF_FAILED,
        reason="writer failed",
    )

    with pytest.raises(ValueError, match="protocol version 1"):
        HandoffLifecycleEvent(
            session=event.session,
            consumer_grant_id=event.consumer_grant_id,
            state=event.state,
            lifecycle_protocol_version=0,
        )


@pytest.mark.asyncio
async def test_control_only_handoff_event_does_not_complete_generation():
    event = HandoffLifecycleEvent.from_params(
        _params(),
        HandoffEventState.HANDOFF_COMMITTED,
    )
    response = LlmResponse(
        request_id=17,
        client_id=3,
        disagg_handoff_event=event,
    )
    result = GenerationResultBase(17, SamplingParams())

    assert is_control_only_llm_response(response)
    assert result._handle_response(response)
    assert not result._done
    assert await result._wait_disagg_handoff_event() == event


def test_control_only_response_does_not_consume_postproc_first_response_params():
    event = HandoffLifecycleEvent.from_params(
        _params(),
        HandoffEventState.HANDOFF_COMMITTED,
    )
    sampling_params = SamplingParams()
    disaggregated_params = _params()
    result = SimpleNamespace(
        _params_transmitted=False,
        sampling_params=sampling_params,
        postproc_params=None,
        disaggregated_params=disaggregated_params,
        _streaming=True,
    )
    queued = []
    worker = SimpleNamespace(
        frontend_result_queues=None,
        result_queue=None,
        _results={
            3: result,
        },
        postproc_config=SimpleNamespace(num_postprocess_workers=1),
        postproc_queues=[SimpleNamespace(put=queued.append)],
    )

    _send_rsp(
        worker,
        LlmResponse(
            request_id=17,
            client_id=3,
            disagg_handoff_event=event,
        ),
    )

    assert not result._params_transmitted
    assert queued[0].sampling_params is None
    assert queued[0].disaggregated_params is None

    _send_rsp(
        worker,
        LlmResponse(
            request_id=17,
            client_id=3,
            result=SimpleNamespace(is_final=False),
        ),
    )

    assert result._params_transmitted
    assert queued[1].sampling_params is sampling_params
    assert queued[1].disaggregated_params is disaggregated_params


def test_conflicting_handoff_event_is_rejected():
    params = _params()
    committed = HandoffLifecycleEvent.from_params(
        params,
        HandoffEventState.HANDOFF_COMMITTED,
    )
    failed = HandoffLifecycleEvent.from_params(
        params,
        HandoffEventState.HANDOFF_FAILED,
        reason="late conflicting failure",
    )
    result = GenerationResultBase(17, SamplingParams())

    assert result._handle_response(LlmResponse(request_id=17, disagg_handoff_event=committed))
    with pytest.raises(RuntimeError, match="conflicting"):
        result._handle_response(LlmResponse(request_id=17, disagg_handoff_event=failed))
