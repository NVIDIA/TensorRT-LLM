# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Forced Mamba snapshot chunk points with save_last_snapshot.

The GDN running state is only written into the block that a context chunk
ENDS on. save_last_snapshot materializes the last prompt block a DUPLICATE
request can reach (a duplicate matches at most prompt_len - 1 tokens, so for
a block-aligned prompt that is one block before the prompt end) as a
reusable snapshot, and a chunk must end exactly on that block's boundary —
otherwise the block enters the reuse tree holding a stale state and later
reusers silently generate wrong output. These tests pin the chunk-point
computation for every boundary case.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import CppMambaHybridCacheManager


@pytest.fixture(autouse=True)
def _two_chunk_save_last_schedule(monkeypatch):
    # These tests describe the two-chunk save-last schedule; the folded schedule
    # (the default) has its own tests (test_fold_save_last_*).
    monkeypatch.setenv("TLLM_MAMBA_FOLD_SAVE_LAST", "0")


def _make_manager(interval, tokens_per_block=32, enable_block_reuse=True, save_last=True):
    manager = object.__new__(CppMambaHybridCacheManager)
    manager.enable_block_reuse = enable_block_reuse
    manager.kv_cache_config = SimpleNamespace(
        enable_block_reuse=enable_block_reuse,
        mamba_state_config=SimpleNamespace(
            periodic_snapshot_interval=interval,
            additional_snapshot_offsets_from_start=[],
            additional_snapshot_offsets_from_end=[],
        ),
    )
    manager.linear_attention_metadata = SimpleNamespace(
        states_snapshot_interval=interval if enable_block_reuse else 0,
        save_last_snapshot=save_last,
    )
    manager.tokens_per_block = tokens_per_block
    return manager


def _points(manager, prompt_len):
    request = SimpleNamespace(prompt_len=prompt_len, expect_snapshot_points=None)
    manager.prepare_expect_snapshot_points([request])
    return request.expect_snapshot_points


@pytest.mark.parametrize(
    "prompt_len,interval,expected",
    [
        # Production shape: 7639-token prompt, interval 4096 — interval
        # points plus the reachable terminal block boundary.
        (7639, 4096, [4096, 7616]),
        # Block-aligned prompt: the last full block ends AT prompt_len and a
        # duplicate can never reach it (it matches at most prompt_len - 1
        # tokens), so the terminal point lands one block earlier. This is the
        # aligned-prompt reuse-floor trap: without this point every duplicate
        # of a 7648-token prompt falls back to the 4096 interval snapshot.
        (7648, 4096, [4096, 7616]),
        # Reachable block coincides with an interval multiple: no duplicate.
        (8193, 4096, [4096, 8192]),
        # Prompt shorter than one interval: only the terminal point.
        (100, 4096, [96]),
        # Prompt shorter than one block: nothing to snapshot.
        (31, 4096, []),
        # Aligned prompt shorter than one interval: terminal point one block
        # back from the prompt end.
        (64, 4096, [32]),
        # Dense interval (current default 256): extra point still inserted
        # between the last interval multiple and the prompt end.
        (7639, 256, list(range(256, 7639, 256)) + [7616]),
        # Aligned prompt that is ALSO an interval multiple: the reachable
        # point (7648) precedes the final interval point (7680), so it must
        # be inserted in sorted position, not appended.
        (7680, 256, sorted(list(range(256, 7681, 256)) + [7648])),
    ],
)
def test_snapshot_points_include_last_full_block(prompt_len, interval, expected):
    assert _points(_make_manager(interval), prompt_len) == expected


def test_points_are_strictly_increasing_and_below_prompt_end():
    for prompt_len in range(1, 700):
        points = _points(_make_manager(interval=256, tokens_per_block=32), prompt_len)
        assert points == sorted(set(points))
        assert all(p <= prompt_len for p in points)
        # Every point must be a chunkable boundary (multiple of the block).
        assert all(p % 32 == 0 for p in points)


def test_save_last_disabled_keeps_interval_points_only():
    manager = _make_manager(interval=4096, save_last=False)
    assert _points(manager, 7639) == [4096]


def test_reuse_disabled_yields_no_points():
    manager = _make_manager(interval=4096, enable_block_reuse=False)
    assert _points(manager, 7639) == []


def test_zero_interval_yields_no_points():
    manager = _make_manager(interval=0)
    assert _points(manager, 7639) == []
