# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the fp8 context-MLA attention-workspace reservation and its admission cap.

The estimator reserves KV-cache headroom for the total_kv_len-scaled context-MLA workspace and carries the
exact token cap that reserve covers onto the KV manager; the scheduler reads that cap directly (rather than
re-deriving it from pool layout) and trims context requests whose summed attended KV would exceed it. These
tests cover the pure-Python pieces: the reservation gate (reuse-on / chunked-off), the workspace reserve/cap
math, the L_cap derivation, the per-request attended-KV computation, the admission trim, the carried-cap
read (V1/V2 layout-independent), and the non-MLA no-op gate.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    get_mla_context_workspace_bytes_per_token,
    get_mla_context_workspace_kv_len_cap,
    get_mla_context_workspace_reserve,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _ctx_req(
    context_current_position,
    is_first_context_chunk,
    estimated_reusable_tokens,
    context_chunk_size,
    orig_prompt_len,
):
    return SimpleNamespace(
        context_current_position=context_current_position,
        is_first_context_chunk=is_first_context_chunk,
        estimated_reusable_tokens=estimated_reusable_tokens,
        context_chunk_size=context_chunk_size,
        orig_prompt_len=orig_prompt_len,
    )


@pytest.mark.parametrize(
    "req,expected",
    [
        # V1: reuse not yet applied at schedule time; credit is in estimated_reusable_tokens.
        (_ctx_req(0, True, 100, 50, 150), 150),
        # V2: context_current_position already advanced past the reused prefix.
        (_ctx_req(100, True, 100, 50, 150), 150),
        # Fresh prefill, no reuse.
        (_ctx_req(0, True, 0, 80, 80), 80),
        # Middle (non-first) chunk: reuse credit does not apply, position already advanced.
        (_ctx_req(200, False, 0, 100, 500), 300),
        # Reuse estimate above the prompt length is clamped to the prompt length.
        (_ctx_req(0, True, 999, 50, 150), 150),
    ],
)
def test_context_attended_kv_len(req, expected):
    assert PyExecutor._context_attended_kv_len(req) == expected


def _make_executor(cap):
    # Bypass __init__; the trim only reads the cached cap and the two helper methods.
    exe = object.__new__(PyExecutor)
    exe._ctx_mla_kv_len_cap = cap  # pre-set so the cap getter skips the C++ binding
    return exe


def test_cap_trims_tail_by_total_kv_len():
    exe = _make_executor(cap=120)
    reqs = [_ctx_req(0, True, 0, 50, 50) for _ in range(3)]  # cumulative 50, 100, 150
    kept = exe._cap_context_by_total_kv_len(reqs)
    assert len(kept) == 2  # the third would push cumulative to 150 > 120


def test_cap_keeps_all_when_within_budget():
    exe = _make_executor(cap=200)
    reqs = [_ctx_req(0, True, 0, 50, 50) for _ in range(3)]
    assert len(exe._cap_context_by_total_kv_len(reqs)) == 3


def test_cap_always_keeps_first_even_if_it_alone_exceeds():
    exe = _make_executor(cap=120)
    reqs = [_ctx_req(0, True, 0, 200, 200), _ctx_req(0, True, 0, 50, 50)]
    kept = exe._cap_context_by_total_kv_len(reqs)
    assert len(kept) == 1  # first kept despite exceeding cap (forward-progress guard)


def test_cap_single_request_never_trimmed():
    exe = _make_executor(cap=1)
    reqs = [_ctx_req(0, True, 0, 500, 500)]
    assert exe._cap_context_by_total_kv_len(reqs) == reqs


def test_no_cap_returns_untouched():
    exe = _make_executor(cap=None)  # non-fp8-MLA model: no reservation, no trimming
    reqs = [_ctx_req(0, True, 0, 50, 50) for _ in range(5)]
    assert exe._cap_context_by_total_kv_len(reqs) is reqs


@pytest.mark.parametrize(
    "override,expected",
    [
        # Default (no override): never-stall worst case min(bs, num_tokens) * max_seq_len
        # = min(64, 8192) * 4096.
        (None, 64 * 4096),
        # In-range override passes through.
        (100_000, 100_000),
        # Below the max_seq_len floor is raised to max_seq_len (one request must always fit).
        (1_000, 4096),
        # Above the worst case is capped at the worst case (a larger value only over-reserves).
        (10**12, 64 * 4096),
    ],
)
def test_kv_len_cap_default_floor_and_ceiling(override, expected):
    # Reuse on + chunked off: a cap is reserved (default worst case, or the clamped override).
    cfg = SimpleNamespace(enable_block_reuse=True, fp8_context_mla_kv_len_cap=override)
    assert (
        get_mla_context_workspace_kv_len_cap(
            cfg,
            max_batch_size=64,
            max_num_tokens=8192,
            max_seq_len=4096,
            enable_chunked_prefill=False,
        )
        == expected
    )


# ---- reservation gate: only when reuse can grow the workspace past the profiled floor ----


@pytest.mark.parametrize(
    "enable_block_reuse,enable_chunked_prefill",
    [
        # Reuse off: summed attended KV bounded by max_num_tokens (already profiled) -> no reservation.
        (False, False),
        # Chunked prefill: each attention launch bounded by its own chunk buffer -> no reservation.
        (True, True),
        (False, True),
    ],
)
def test_kv_len_cap_none_when_reuse_cannot_grow_workspace(
    enable_block_reuse, enable_chunked_prefill
):
    # None cap -> configure_kv_cache_capacity reserves nothing and applies no admission cap for these
    # unaffected configs, even with an explicit override present.
    cfg = SimpleNamespace(enable_block_reuse=enable_block_reuse, fp8_context_mla_kv_len_cap=999_999)
    assert (
        get_mla_context_workspace_kv_len_cap(
            cfg,
            max_batch_size=64,
            max_num_tokens=8192,
            max_seq_len=4096,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        is None
    )


def test_workspace_bytes_zero_for_non_mla_model():
    # No kv_lora_rank on the config -> not MLA -> 0 (early return, no binding call needed).
    model_config = SimpleNamespace(pretrained_config=SimpleNamespace(), quant_config=None)
    assert get_mla_context_workspace_bytes_per_token(model_config, Mock()) == 0


# ---- workspace reserve / admission-cap math (estimator side) ----


@pytest.mark.parametrize(
    "budget,k,w,kv_len_cap,expected_cap",
    [
        # Worst case fits the budget: reserve = w * kv_len_cap, so cap = kv_len_cap.
        (10**12, 1000, 100, 50_000, 50_000),
        # Memory-constrained: w * kv_len_cap exceeds the per-token split, so cap = budget / (k + w).
        # budget/(k+w) = 1_100_000 / 1100 = 1000.
        (1_100_000, 1000, 100, 10**9, 1000),
    ],
)
def test_workspace_reserve_cap(budget, k, w, kv_len_cap, expected_cap):
    reserve, cap = get_mla_context_workspace_reserve(budget, k, w, kv_len_cap)
    assert cap == expected_cap
    assert cap == int(reserve / w)  # the cap is exactly what the reserve covers
    assert reserve <= w * kv_len_cap  # never reserves beyond the worst case
    assert cap <= kv_len_cap  # never admits more than L_cap


@pytest.mark.parametrize(
    "budget,k,w,kv_len_cap",
    [
        (0, 1000, 100, 50_000),  # no budget
        (10**12, 0, 100, 50_000),  # k <= 0
        (10**12, 1000, 0, 50_000),  # w <= 0
        (10**12, 1000, 100, None),  # no L_cap
        (10**12, 1000, 100, 0),  # L_cap == 0
    ],
)
def test_workspace_reserve_zero_for_bad_inputs(budget, k, w, kv_len_cap):
    # Any non-positive input -> reserve nothing and carry no cap (no admission cap applied).
    assert get_mla_context_workspace_reserve(budget, k, w, kv_len_cap) == (0, None)


# ---- carried admission-cap read (executor side, V1/V2) ----


def _executor_with_manager(manager, is_warmup=False):
    # Bypass __init__; the cap getter only reads is_warmup and the cap carried on kv_cache_manager. The
    # estimator is the single decision point -- the scheduler never re-derives w or the pool layout.
    # Set the _is_warmup backing field directly -- the is_warmup property setter also propagates into
    # model_engine, which this bare (no-__init__) executor doesn't have.
    exe = object.__new__(PyExecutor)
    exe._is_warmup = is_warmup
    exe.kv_cache_manager = manager
    return exe


@pytest.mark.parametrize(
    "manager",
    [
        # V1-style manager: unified-pool block count present but must NOT be consulted anymore.
        SimpleNamespace(
            fp8_ctx_mla_kv_len_cap=262144, blocks_in_primary_pool=4096, tokens_per_block=64
        ),
        # V2-style manager: page-index upper bound overstates capacity; must NOT be consulted.
        SimpleNamespace(
            fp8_ctx_mla_kv_len_cap=262144, blocks_in_primary_pool=10**9, tokens_per_block=64
        ),
    ],
)
def test_ctx_cap_reads_carried_value_ignoring_pool_layout(manager):
    # Both managers carry the same estimator cap; the (very different) pool layout is irrelevant.
    exe = _executor_with_manager(manager)
    assert exe._get_ctx_mla_kv_len_cap() == 262144


@pytest.mark.parametrize(
    "manager",
    [
        # Estimator made no reservation -- non-fp8-MLA, reuse off / chunked prefill, or estimation skipped.
        SimpleNamespace(fp8_ctx_mla_kv_len_cap=None),
        # Attribute absent (manager not built by the estimator): treated the same as None, no admission cap.
        SimpleNamespace(),
    ],
)
def test_ctx_cap_none_when_no_reservation(manager):
    # A carried None (or absent) cap means no headroom to enforce -> admission disabled, no crash.
    exe = _executor_with_manager(manager)
    assert exe._get_ctx_mla_kv_len_cap() is None


def test_ctx_cap_no_cap_during_warmup():
    # Estimation/warmup runs fresh-prefill dummies against a throwaway manager that reserved nothing: don't
    # cap even if a cap were carried, and short-circuit before reading the manager.
    exe = _executor_with_manager(SimpleNamespace(fp8_ctx_mla_kv_len_cap=262144), is_warmup=True)
    assert exe._get_ctx_mla_kv_len_cap() is None
