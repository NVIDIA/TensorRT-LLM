# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response preparation shared by execution and transfer-completion paths."""

from copy import copy, deepcopy

from .llm_request import LlmRequest, LlmResponse


def publish_execution_metadata(request: LlmRequest) -> None:
    """Publish sampler progress at a caller-selected native consumption boundary.

    Overlap scheduling can read the previous native iteration while Python is
    already sampling the next one. Callers choose when to publish rather than
    keeping these fields synchronized on every write.
    """
    request.draft_tokens = request.py_draft_tokens or []
    request.decoding_iter = request.py_decoding_iter


def prepare_response(
    request: LlmRequest,
    *,
    publish_metadata: bool = False,
    include_latest_logits: bool = False,
) -> LlmResponse | None:
    """Create one native result and snapshot response metadata and optional outputs.

    Callers retain emission, performance-timing and teardown decisions. Transfer
    completion keeps its native state until after this call; early-token paths
    can expose the latest logits before overlap exclusion would hide them.
    """
    if publish_metadata:
        publish_execution_metadata(request)
    logits = request.py_result.get_latest_logits_unexcluded() if include_latest_logits else None
    response = request.create_response()
    if response is None:
        return None
    result = response.result
    result.cached_tokens = request.cached_tokens
    result.ctx_computed_tokens = getattr(request, "py_ctx_computed_tokens", 0)
    result.ctx_first_begin = getattr(request, "py_ctx_first_begin", -1)
    result.ctx_num_chunks = getattr(request, "py_ctx_num_chunks", 0)
    result.per_pos_drafted = copy(request.py_per_pos_drafted)
    result.per_pos_accepted = copy(request.py_per_pos_accepted)
    if request.py_total_draft_tokens > 0:
        result.spec_dec_totals = (
            request.py_total_accepted_draft_tokens,
            request.py_total_draft_tokens,
        )
    params = request.py_disaggregated_params
    if params is not None and params.ctx_usage is not None:
        result.ctx_usage = deepcopy(params.ctx_usage)
    if logits is not None:
        result.generation_logits = logits
    return response
