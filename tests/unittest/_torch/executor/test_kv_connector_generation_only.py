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
from _torch.executor.llm_request_factory import make_llm_request

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm.bindings import executor as trtllm

pytestmark = pytest.mark.cpu_only


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

    ctx_req = make_llm_request(1, trtllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION)
    assert not ctx_req.is_generation_only_request
    assert manager.get_num_new_matched_tokens(ctx_req, 0) == 0
    assert scheduler.get_num_new_matched_tokens.call_count == 1

    gen_req = make_llm_request(2, trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY)
    assert gen_req.is_generation_only_request
    with pytest.raises(RuntimeError, match="generation-only"):
        manager.get_num_new_matched_tokens(gen_req, 0)

    # The connector was never consulted about the generation-only request.
    assert scheduler.get_num_new_matched_tokens.call_count == 1
