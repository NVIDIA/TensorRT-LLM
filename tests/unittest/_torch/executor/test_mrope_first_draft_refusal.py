# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A first-draft request in a speculative batch of an mRoPE model is refused.

The first-draft packing adds neither an mRoPE position span nor a delta read slot, and the
attention kernel indexes the deltas by generation batch index, so every generation request
packed after it would read another request's delta. The engine builds such a request nowhere
today; the runner says so loudly instead of misaligning the slots if one ever appears.
"""

from unittest.mock import Mock

import pytest
import test_pytorch_model_engine as engine_tests  # module import: its TestCase is not collected

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi import MTPDecodingConfig


def _prepare(first_draft: bool) -> None:
    case = engine_tests.PyTorchModelEngineTestCase()
    model_engine, kv_cache_manager, attn_metadata = case._setup_mrope_engine(
        spec_config=MTPDecodingConfig(max_draft_len=1)
    )
    try:
        # A generation request with an mRoPE delta, then a plain one after it.
        first = case._make_mrope_gen_request(4, 1, 0, 10)
        first.py_is_first_draft = first_draft
        plain = case._make_mrope_gen_request(6, 2, 1, 20)
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [first, plain]
        model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=Mock(
                _force_non_greedy_for_capture=False, context_prompt_lookahead_tokens=None
            ),
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False,
        )
    finally:
        kv_cache_manager.shutdown()


def test_first_draft_request_on_an_mrope_model_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="First-draft requests on an mRoPE model"):
        _prepare(first_draft=True)


def test_the_same_batch_without_a_first_draft_request_is_prepared() -> None:
    _prepare(first_draft=False)
