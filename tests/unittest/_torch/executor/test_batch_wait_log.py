# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Smoke test for PyExecutor._maybe_log_batch_wait_decision rank gating.

Per ``py_executor.py:3060-3063`` the method early-returns when ``self.dist.rank
!= 0``. The docstring claims env-var gating via ``TLLM_LOG_BATCH_WAIT=1``
which does NOT exist in the implementation; the only gate is rank.

This test is a regression net — if a refactor removes the rank gate, the
function would crash at line 3107 (``self.batch_wait_iters_count``) since
we never set the wait counters on the fake executor.
"""
from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def test_log_decision_early_returns_on_nonzero_rank():
    """Non-rank-0 path should early-return without touching wait counters."""
    fake = PyExecutor.__new__(PyExecutor)
    fake.dist = Mock()
    fake.dist.rank = 1  # non-zero rank → early return at py_executor.py:3062

    # If the gate is broken, the function would proceed to read
    # self.batch_wait_iters_count (line 3107) which we never set →
    # AttributeError. The successful no-crash path verifies the gate works.
    PyExecutor._maybe_log_batch_wait_decision(
        fake,
        context_requests=[],
        generation_requests=[],
        num_scheduled_tokens=0,
        wait_threshold=0.5,
        should_waiting=False,
    )
