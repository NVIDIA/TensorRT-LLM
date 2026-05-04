# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Smoke test for ``PyExecutor._maybe_log_batch_wait_decision`` rank gating.

The method early-returns when ``self.dist.rank != 0`` and otherwise reads
``self.batch_wait_iters_count`` and ``self.batch_wait_timeout_iters`` to
format diagnostic output. If a refactor removes the rank gate, the function
would crash on the wait counters since this test never sets them on the fake.
"""

from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def test_log_decision_early_returns_on_nonzero_rank():
    """Non-rank-0 path early-returns without touching wait counters."""
    fake = PyExecutor.__new__(PyExecutor)
    fake.dist = Mock()
    fake.dist.rank = 1  # non-zero rank → early return

    # No exception expected. If the gate is broken, the function would proceed
    # to read attributes (``batch_wait_iters_count``, ``batch_wait_timeout_iters``)
    # that we never set on the fake, raising AttributeError.
    PyExecutor._maybe_log_batch_wait_decision(
        fake,
        context_requests=[],
        generation_requests=[],
        num_scheduled_tokens=0,
        wait_threshold=0.5,
        should_waiting=False,
    )
