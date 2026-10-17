# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""bench.py warms up at the largest measured point, so a fresh session's first window is not cold."""

from agent_flow.workflows.trtllm_modeling_v2.tune.bench import warmup_shape


def test_the_warmup_defaults_to_the_largest_point():
    assert warmup_shape([1, 128]) == (128, 256)


def test_explicit_warmup_values_win():
    assert warmup_shape([1, 128], concurrency=16, prompts=16) == (16, 16)
    assert warmup_shape([1, 128], concurrency=32) == (32, 64)
