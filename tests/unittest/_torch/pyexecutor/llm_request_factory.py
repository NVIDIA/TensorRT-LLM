# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared factory for real ``LlmRequest`` objects built through the executor bindings.

Used by tests that need the C++-backed request rather than a stub: the base
class exposes read-only properties (``is_generation_only_request``,
``prepopulated_prompt_len``) that a ``SimpleNamespace`` cannot pin down.
"""

from typing import Sequence

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, executor_request_to_llm_request
from tensorrt_llm.bindings import executor as trtllm


def make_llm_request(
    request_id: int,
    request_type: trtllm.RequestType,
    *,
    input_token_ids: Sequence[int] = (1, 2, 3, 4),
    max_tokens: int = 8,
) -> LlmRequest:
    """Build a real ``LlmRequest`` of ``request_type`` via ``executor_request_to_llm_request``.

    A generation-only request always arrives from a context server, so it
    carries the first generated token in its context phase params.
    """
    context_phase_params = (
        trtllm.ContextPhaseParams([100], request_id, None, None, None, None)
        if request_type == trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY
        else None
    )
    executor_request = trtllm.Request(
        input_token_ids=list(input_token_ids),
        max_tokens=max_tokens,
        type=request_type,
        context_phase_params=context_phase_params,
    )
    return executor_request_to_llm_request(
        request_id,
        executor_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )
