# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape-scoped slot-indexed capacity; the gathered sampler retains its measured cap."""

import os
import subprocess
import sys

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.sampling import (  # noqa: E501
    m31_chain_spec_mc as entry,
)


@pytest.mark.parametrize(
    "k,cap", [(0, 48), (1, 48), (2, 48), (3, 112), (4, 48), (6, 48), (7, 56), (8, 48)]
)
def test_slot_capacity_is_scoped(k, cap):
    assert entry.SLOTS_WIDE
    assert entry.max_slot_batch(k, 200064) == cap
    assert entry.max_batch() == 48
    assert entry.WORKSPACE_MAX_BATCH == 112


@pytest.mark.parametrize("vocab", [0, 16384, 200060, 200068, 262144])
@pytest.mark.parametrize("k", [3, 7])
def test_other_vocab_keeps_previous_capacity(k, vocab):
    assert entry.max_slot_batch(k, vocab) == 48


@pytest.mark.parametrize("disabled", [False, True])
def test_slot_wide_kill_in_fresh_process(disabled):
    env = {**os.environ, "TRTLLM_M31_DISABLE_REJ_SLOTS_WIDE": "1" if disabled else "0"}
    code = (
        f"import {entry.__name__} as e; "
        "print(e.max_slot_batch(3,200064),e.max_slot_batch(7,200064),e.max_batch())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines()[-1] == ("48 48 48" if disabled else "112 56 48")
