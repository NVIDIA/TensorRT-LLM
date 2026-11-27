# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``context_only`` KV connector rejects, at admission, requests that would take a generation step."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings.internal.batch_manager import LlmRequestType

pytestmark = pytest.mark.cpu_only

CTX_ONLY = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
GEN_ONLY = LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
BOTH = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION


def _executor(context_only):
    ex = object.__new__(PyExecutor)
    ex.kv_connector_manager = SimpleNamespace(context_only=context_only)
    return ex


def _request(request_type, max_new_tokens=1):
    return SimpleNamespace(
        py_llm_request_type=request_type, max_new_tokens=max_new_tokens, py_beam_width=1
    )


@pytest.mark.parametrize(
    "request_type, max_new_tokens",
    [(GEN_ONLY, 1), (BOTH, 4)],
    ids=["generation_only", "context_and_generation"],
)
def test_requests_that_would_decode_are_rejected(request_type, max_new_tokens):
    with pytest.raises(ValueError, match="context_only"):
        _executor(True)._validate_context_only_connector(_request(request_type, max_new_tokens))


@pytest.mark.parametrize(
    "request_type, max_new_tokens",
    [(CTX_ONLY, 4), (BOTH, 1)],
    ids=["context_only", "one_token"],
)
def test_context_only_and_one_token_requests_pass(request_type, max_new_tokens):
    _executor(True)._validate_context_only_connector(_request(request_type, max_new_tokens))


@pytest.mark.parametrize("manager", [None, SimpleNamespace(context_only=False)])
def test_without_a_context_only_connector_nothing_is_checked(manager):
    ex = object.__new__(PyExecutor)
    ex.kv_connector_manager = manager
    ex._validate_context_only_connector(_request(GEN_ONLY, 8))
