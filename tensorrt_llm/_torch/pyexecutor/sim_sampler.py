# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SimSampler: dummy sampler for simulation mode."""

from __future__ import annotations

from typing import Optional

from tensorrt_llm.bindings.executor import FinishReason

from .llm_request import LlmRequestState
from .sampler import SampleState, Sampler
from .scheduler import ScheduledRequests


class SimSampler(Sampler):
    """Sampler that generates dummy tokens and advances request state.

    Used in simulation mode. Each call to update_requests adds one dummy
    token per request and checks whether max_new_tokens has been reached.
    Records per-request token timestamps on the SimClock.
    """

    DUMMY_TOKEN_ID = 0

    def __init__(self, clock=None):
        self._clock = clock

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs: dict, num_context_logits_prefix_sum: list,
                     resource_manager=None):
        all_requests = (scheduled_requests.context_requests +
                        scheduled_requests.generation_requests)
        return SampleState(requests=all_requests)

    def update_requests(self, state: SampleState, resource_manager=None):
        for request in state.requests:
            if request.is_generation_complete_state:
                continue

            # Register request on first encounter
            if (self._clock is not None
                    and request.request_id not in self._clock.request_stats):
                self._clock.register_request(
                    request.request_id,
                    input_length=request.orig_prompt_len)

            # add_new_token is a C++ binding that appends the token and
            # updates internal sequence length tracking.
            request.add_new_token(self.DUMMY_TOKEN_ID, 0)
            request.py_decoding_iter += 1

            # Record token timestamp on the simulated clock
            if self._clock is not None:
                self._clock.record_token(request.request_id)

            num_generated = request.get_num_tokens(0) - request.orig_prompt_len
            if num_generated >= request.max_new_tokens:
                request.state = LlmRequestState.GENERATION_COMPLETE
                request.set_finished_reason(FinishReason.LENGTH, 0)

    def is_generation_model(self) -> bool:
        return True
