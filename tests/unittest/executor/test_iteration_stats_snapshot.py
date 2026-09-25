# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observation ownership, report parity, and producer buffer bounds."""

import json
import pickle
from collections import deque
from dataclasses import fields
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm.executor.iteration_stats import (
    DisServingRequestSnapshot,
    InflightBatchingSnapshot,
    IterationStatsFrame,
    IterationStatsSnapshot,
    KVCacheStatsSnapshot,
    RankStatsSnapshot,
    RequestStatsSnapshot,
    SpecDecodingSnapshot,
    capture_legacy_kv_iteration_stats,
    materialize_stats_batch,
    prepare_stats_batch,
)

pytestmark = pytest.mark.cpu_only


def test_frame_pickle_preserves_owned_rank_and_request_observations():
    payload = SimpleNamespace(
        num_context_requests=1,
        num_ctx_tokens=128,
        num_ctx_kv_tokens=64,
        num_gen_requests=2,
        num_gen_kv_tokens=500,
        num_paused_requests=0,
        num_paused_kv_tokens=0,
    )
    frame = IterationStatsFrame(
        stats=IterationStatsSnapshot(iter=9, timestamp="09-25-2026 12:30:00.000000"),
        req_stats=[
            RequestStatsSnapshot(id=17, stage="GENERATION_COMPLETE", num_generated_tokens=32)
        ],
        rank_payloads=(
            RankStatsSnapshot.capture(0, payload),
            RankStatsSnapshot.capture(1, payload),
        ),
        sequence=42,
    )
    with patch(
        "tensorrt_llm.executor.iteration_stats._materialize_frame", side_effect=AssertionError
    ):
        prepared = prepare_stats_batch([frame])
        received = pickle.loads(pickle.dumps(prepared))
    payload.num_ctx_tokens = 999
    frame.stats.iter = 99
    frame.req_stats[0].num_generated_tokens = 999
    reports = materialize_stats_batch(received)
    assert [report["attentionDpRank"] for report in reports] == [0, 1]
    assert [report["statsSequence"] for report in reports] == [42, 42]
    assert all(report["iter"] == 9 for report in reports)
    assert all(report["inflightBatchingStats"]["numCtxTokens"] == 128 for report in reports)
    assert reports[0]["requestStats"][0]["numGeneratedTokens"] == 32
    assert "requestStats" not in reports[1]


def test_public_report_matches_native_json_contract():
    from tensorrt_llm.bindings import executor as native

    stats = IterationStatsSnapshot(
        iter=19,
        timestamp="09-25-2026 12:30:00.123456",
        iter_latency_ms=8.0,
        new_active_requests_queue_latency_ms=4.5,
        num_new_active_requests=3,
        num_active_requests=5,
        num_queued_requests=7,
        num_completed_requests=2,
        max_num_active_requests=16,
        gpu_mem_usage=1024,
        kv_cache_stats=KVCacheStatsSnapshot(32, 24, 8, 16, 17, 13, 4, 12, 0.25),
        specdec_stats=SpecDecodingSnapshot(8, 4, 2, 2.0),
    )
    for index, member in enumerate(fields(InflightBatchingSnapshot), 1):
        setattr(stats.inflight_batching_stats, member.name, index)
    native_stats = native.IterationStats()
    for member in fields(stats):
        if member.name not in ("kv_cache_stats", "inflight_batching_stats", "specdec_stats"):
            setattr(native_stats, member.name, getattr(stats, member.name))
    native_stats.static_batching_stats = native.StaticBatchingStats()
    native_kv = native.KvCacheStats()
    for member in fields(stats.kv_cache_stats):
        setattr(native_kv, member.name, getattr(stats.kv_cache_stats, member.name))
    native_stats.kv_cache_stats = native_kv
    native_ifb = native.InflightBatchingStats()
    for member in fields(stats.inflight_batching_stats):
        setattr(native_ifb, member.name, getattr(stats.inflight_batching_stats, member.name))
    native_stats.inflight_batching_stats = native_ifb
    native_spec = native.SpecDecodingStats()
    native_spec.num_draft_tokens = 8
    native_spec.num_accepted_tokens = 4
    native_spec.num_requests_with_draft_tokens = 2
    native_spec.iter_latency_ms = 2.0
    native_spec.acceptance_length = 3.0
    native_spec.draft_overhead = 0.25
    native_stats.specdec_stats = native_spec
    actual = materialize_stats_batch([IterationStatsFrame(stats)])[0]
    for key in ("attentionDpRank", "schedulerMode", "statsSequence", "statsDroppedFrames"):
        actual.pop(key)
    assert actual == json.loads(native_stats.to_json_str())


def test_request_report_matches_native_json_contract():
    from tensorrt_llm.bindings import executor as native

    request = RequestStatsSnapshot(
        id=25,
        stage="GENERATION_COMPLETE",
        context_prefill_position=128,
        num_generated_tokens=32,
        avg_num_decoded_tokens_per_iter=1.5,
        scheduled=True,
        dis_serving_stats=DisServingRequestSnapshot(3.5, 4096),
        alloc_total_blocks_per_request=16,
        alloc_new_blocks_per_request=12,
        reused_blocks_per_request=4,
        missed_blocks_per_request=12,
        kv_cache_hit_rate_per_request=0.25,
    )
    expected = native.RequestStats()
    for member in fields(request):
        if member.name not in ("stage", "dis_serving_stats"):
            setattr(expected, member.name, getattr(request, member.name))
    expected.stage = native.RequestStage.GENERATION_COMPLETE
    disagg = native.DisServingRequestStats()
    disagg.kv_cache_transfer_ms = 3.5
    disagg.kv_cache_size = 4096
    expected.dis_serving_stats = disagg
    actual = materialize_stats_batch(
        [
            IterationStatsFrame(
                IterationStatsSnapshot(),
                req_stats=[request],
            )
        ]
    )[0]["requestStats"][0]
    assert actual == json.loads(expected.to_json_str())


def test_legacy_kv_capture_does_not_keep_native_rows():
    from tensorrt_llm.executor.iteration_stats import _LEGACY_KV_FIELDS

    row = SimpleNamespace(**dict.fromkeys(_LEGACY_KV_FIELDS, 0))
    row.iter_reused_blocks = 7
    captured = capture_legacy_kv_iteration_stats({16: row})
    row.iter_reused_blocks = 99
    received = pickle.loads(pickle.dumps(captured))
    assert received[16].iter_reused_blocks == 7
    assert received[16] is not row


def test_buffer_eviction_is_bounded_and_visible_after_drain():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = SimpleNamespace(
        stats_lock=Lock(),
        stats=deque(),
        max_stats_len=2,
        _stats_sequence=0,
        _stats_dropped_frames=0,
        enable_iter_perf_stats=True,
    )
    for iteration in range(5):
        PyExecutor._append_stats_frame(
            executor, IterationStatsFrame(IterationStatsSnapshot(iter=iteration))
        )
    assert len(executor.stats) == 2
    received = PyExecutor.get_latest_iteration_stats(executor)
    assert not executor.stats
    reports = materialize_stats_batch(received)
    assert [report["iter"] for report in reports] == [3, 4]
    assert [report["statsSequence"] for report in reports] == [4, 5]
    assert reports[-1]["statsDroppedFrames"] == 3
    PyExecutor._append_stats_frame(executor, IterationStatsFrame(IterationStatsSnapshot(iter=9)))
    assert executor.stats[0].sequence == 6
    assert executor.stats[0].dropped_frames == 3
