# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The connector's generation-only guard, driven by a real ``LlmRequest``.

``KvCacheConnectorManager.get_num_new_matched_tokens`` is reached from two
directions. The C++ trampoline in ``cpp/tensorrt_llm/nanobind/batch_manager/
kvCacheConnector.cpp`` passes ``LlmRequest const&``, which nanobind casts by
copy, so that caller hands in a ``bindings.internal.batch_manager.LlmRequest``
-- the base class, where ``is_generation_only_request`` is a ``def_prop_ro``.
Python callers hold the ``_torch`` subclass. Both must read the flag the same
way.

Every other connector test stubs the request, and a stub satisfies whichever
spelling the guard happens to use. Only a real request pins the two together.
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, executor_request_to_llm_request
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only


def _make_llm_request(request_id: int, request_type: trtllm.RequestType) -> LlmRequest:
    # A generation-only request always arrives from a context server, so it
    # carries the first generated token in its context phase params.
    context_phase_params = (
        trtllm.ContextPhaseParams([100], request_id, None, None, None, None)
        if request_type == trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY
        else None
    )

    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3, 4],
        max_tokens=8,
        type=request_type,
        context_phase_params=context_phase_params,
    )

    return executor_request_to_llm_request(
        request_id,
        executor_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )


def test_generation_only_guard_reads_the_request_type():
    """The guard must fire on the request type, not on attribute truthiness.

    Spelling ``is_generation_only_request`` as a method on the subclass makes
    ``request.is_generation_only_request`` a bound method, which is always
    truthy. The guard then rejects every request, context ones included, and
    nothing raises or logs at the attribute access itself.
    """
    worker = MagicMock()
    scheduler = MagicMock()
    scheduler.get_num_new_matched_tokens.return_value = (0, False)

    manager = KvCacheConnectorManager(worker, scheduler=scheduler)

    ctx_req = _make_llm_request(1, trtllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION)
    assert not ctx_req.is_generation_only_request
    assert manager.get_num_new_matched_tokens(ctx_req, 0) == 0
    assert scheduler.get_num_new_matched_tokens.call_count == 1

    gen_req = _make_llm_request(2, trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY)
    assert gen_req.is_generation_only_request
    with pytest.raises(RuntimeError, match="generation-only"):
        manager.get_num_new_matched_tokens(gen_req, 0)

    # The connector was never consulted about the generation-only request.
    assert scheduler.get_num_new_matched_tokens.call_count == 1
