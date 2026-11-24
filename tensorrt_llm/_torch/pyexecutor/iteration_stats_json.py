# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``/metrics`` JSON row of one IterationStats record."""

import json
from typing import Optional

from .kv_cache_stats import append_kv_cache_iteration_stats


def iteration_stats_row(
    stats,
    req_stats=None,
    kv_iter_stats=None,
    attention_dp_rank: Optional[int] = None,
    host_step_time_ms: Optional[float] = None,
    prev_device_step_time_ms: Optional[float] = None,
    scheduler_mode: Optional[str] = None,
    gpu_forward_time_ms: Optional[float] = None,
    mm_encoder_stats: Optional[dict] = None,
    is_queue_snapshot: bool = False,
) -> dict:
    """``stats`` as a dict, tagged with its attention-DP rank (0 without attention DP), with its
    request and KV-cache iteration stats and the profiler's per-loop timings when present.

    ``is_queue_snapshot`` adds ``"isQueueSnapshot": true``: the record was taken without a
    forward pass, so it reports the queues and its duration and per-loop timings are not
    measurements. Measured records do not have the key."""
    row = json.loads(stats.to_json_str())
    row["attentionDpRank"] = 0 if attention_dp_rank is None else attention_dp_rank
    if req_stats:
        row["requestStats"] = [json.loads(req_stat.to_json_str()) for req_stat in req_stats]
    append_kv_cache_iteration_stats(row, kv_iter_stats)
    # hostStepTimeMS is one clean loop; prevDeviceStepTimeMS lags it by a loop (the profiler's
    # ping-pong events); gpuForwardTimeMS is this record's forward; under the "overlap" scheduler
    # iterLatencyMS spans about two loops.
    if host_step_time_ms is not None:
        row["hostStepTimeMS"] = host_step_time_ms
    if prev_device_step_time_ms is not None:
        row["prevDeviceStepTimeMS"] = prev_device_step_time_ms
    if gpu_forward_time_ms is not None:
        row["gpuForwardTimeMS"] = gpu_forward_time_ms
    if mm_encoder_stats is not None:
        row["multimodalEncoderStats"] = mm_encoder_stats
    if scheduler_mode is not None:
        row["schedulerMode"] = scheduler_mode
    if is_queue_snapshot:
        row["isQueueSnapshot"] = True
    return row
