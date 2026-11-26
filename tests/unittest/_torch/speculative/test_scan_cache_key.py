# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-model sampling scan must not reuse the result of another request that shares id, state and slot."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import _sampling_scan_cache_key


@pytest.mark.cpu_only
def test_scan_cache_key_tells_requests_with_the_same_id_apart():
    metadata = SimpleNamespace(
        is_ragged_verify=False,
        runtime_draft_len=3,
        group_all_greedy_sample=None,
        dummy_slot_row=0,
    )

    def request():
        return SimpleNamespace(
            py_request_id=8,
            py_seq_slot=0,
            state=LlmRequestState.CONTEXT_INIT,
            is_dummy_request=False,
        )

    warmup_dummy, real = request(), request()
    assert _sampling_scan_cache_key(metadata, [warmup_dummy]) != _sampling_scan_cache_key(
        metadata, [real]
    )
    assert _sampling_scan_cache_key(metadata, [real]) == _sampling_scan_cache_key(metadata, [real])
