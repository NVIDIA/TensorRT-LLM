# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stream FP4 MLA decode: the 4x kernel on the first requests and the 2x packed kernel on the last ones, concurrently.

The four-CTA-cluster kernel can only occupy the 92 TPC pairs that host 4-CTA clusters (184 of the 212 SMs); the
28 SMs on the unpaired TPCs idle for its whole duration.  ``fp4_mla_twostream.plan_two_stream`` moves exactly as
many of the batch's last requests to the two-CTA packed kernel (one whole-KV task per request: no plan kernel, no
reducer) as drop one whole 4x wave, and this module runs the two arms on two streams forked from the caller's
stream and joined back, so the pair captures into CUDA graphs as two branches.

Placement is what makes the gain real: the 4x grid must take the pairable TPCs before the 2x pairs land on them.
Measured at B128/q4/30K: 2x launched first 478.7 us (gain gone), 4x first 453.6, 2x first but 4x on the
high-priority stream under CUDA graphs 456.9 -- torch instantiates graphs with
``cudaGraphInstantiateFlagUseNodePriority``, so the captured node priority decides the dispatch order at replay.
Hence the 4x arm runs on the highest-priority stream (``torch.cuda.Stream.priority_range()[1]``, -3) and is
enqueued first; the 2x arm runs on a priority -1 stream (a pool nothing else in the process uses: plain
``torch.cuda.Stream()`` objects come from a 32-entry round-robin pool per priority level and a priority-0 side
stream could alias the caller's capture or executor stream and silently serialize the arms).

The sub-batches are plain views.  Request ``i`` owns packed Q rows ``[i * rows, (i + 1) * rows)`` and their
swizzled scales (40 B per row in 128-row blocks, so a split at a 128-row group is a block boundary), output tokens
``[i * q, (i + 1) * q)``, ``kv_lens[i]`` and the fixed-stride generation page table entries
``[i * stride, (i + 1) * stride)`` with ``paged_kv_indptr_decode[i] = i * stride`` (absolute offsets): the 2x arm
keeps the whole table with the suffix of the indptr, the 4x arm gets the table prefix (its envelope caps the
table at ``num_sequences * MAX_PAGES_PER_SEQ``).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .config import FP4_MLA_Q_LOGICAL_DIM, FP4_MLA_SCALE_ROW_GROUP, _ceil_div
from .layout import _get_fp4_mla_swizzled_scale_size
from .packed_plan import _PackedPlanGeometry, _packed_wave_tasks

# (high-priority stream for the 4x arm, low-priority stream for the 2x arm, (fork, join_high, join_low) events) per
# device.  The events are re-recorded every call (one host thread issues the op), like multi_stream_utils does.
_twostream_state: dict[int, tuple[torch.cuda.Stream, torch.cuda.Stream, tuple[torch.cuda.Event, ...]]] = {}


def _get_twostream_state(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream, tuple[torch.cuda.Event, ...]]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    state = _twostream_state.get(index)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            # the warm-up forward of every graph bucket runs this op eagerly first; creating streams, events and the
            # SM-count / compile caches inside a (global-mode) capture is not allowed
            raise RuntimeError(
                "FP4 MLA two-stream decode: streams must be created by an eager (warm-up) call before capture."
            )
        least, greatest = torch.cuda.Stream.priority_range()
        high_priority = min(least, greatest)
        low_priority = max(min(-1, max(least, greatest) - 1), high_priority)
        state = (
            torch.cuda.Stream(device=device, priority=high_priority),
            torch.cuda.Stream(device=device, priority=low_priority),
            (torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()),
        )
        _twostream_state[index] = state
    return state


def run_cutedsl_twostream_decode(
    module: Any,
    metadata: Any,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor,
    v_sf: torch.Tensor,
    kv_global_scale: torch.Tensor,
    q_global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    page_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    num_requests_2x: int,
    rows_per_seq: int,
    max_kv_len: int,
    sm_scale: float,
    v_page_offset: int,
    run_4x: Callable[..., None],
    run_2x: Callable[..., None],
) -> None:
    """Run ``run_4x`` (``decode_4x._run_cutedsl_4x_decode``) on the first ``num_requests - num_requests_2x``
    requests and ``run_2x`` (``decode._run_cutedsl_packed_decode``, one split, no stats) on the last
    ``num_requests_2x`` on two streams; returns once both are joined back to the current stream."""
    r2 = int(num_requests_2x)
    r4 = int(num_requests) - r2
    if not 0 < r2 < num_requests:
        raise RuntimeError(f"FP4 MLA two-stream decode needs 0 < 2x requests {r2} < {num_requests}.")
    if rows_per_seq % FP4_MLA_SCALE_ROW_GROUP != 0:
        raise RuntimeError("FP4 MLA two-stream decode needs whole 128-row scale groups per request.")
    if page_indptr.numel() != num_requests + 1 or kv_lens.numel() < num_requests:
        raise RuntimeError("FP4 MLA two-stream decode: page table / KV length views do not cover the batch.")
    if src_page_ids.numel() % num_requests != 0:
        raise RuntimeError("FP4 MLA two-stream decode needs the fixed-stride generation page table.")
    page_stride = src_page_ids.numel() // num_requests
    rows4 = r4 * rows_per_seq
    sf_bytes4 = _get_fp4_mla_swizzled_scale_size(rows4, FP4_MLA_Q_LOGICAL_DIM)
    tokens4 = r4 * query_len
    q_fp4_4x, q_fp4_2x = q_fp4[:rows4], q_fp4[rows4:]
    q_sf_4x, q_sf_2x = q_sf[:sf_bytes4], q_sf[sf_bytes4:]
    output_4x, output_2x = output[:tokens4], output[tokens4:]
    kv_lens_4x, kv_lens_2x = kv_lens[:r4], kv_lens[r4:num_requests]
    indptr_4x, indptr_2x = page_indptr[: r4 + 1], page_indptr[r4 : num_requests + 1]
    src_page_ids_4x = src_page_ids[: r4 * page_stride]
    # 2x arm geometry: one whole-KV task per request, compiled for the bucket's KV capacity.
    m_tiles = rows_per_seq // FP4_MLA_SCALE_ROW_GROUP
    kv_tiles = _ceil_div(max(int(max_kv_len), 1), 256)
    split_kv_len = max(kv_tiles * 256, 256)
    geometry_2x = _PackedPlanGeometry(
        r2,
        query_len,
        1,
        _packed_wave_tasks(output.device, m_tiles),
        module._select_runtime_kv_profile(split_kv_len) // 256,
        r2,
        split_kv_len,
        None,
        None,
        None,
    )
    high, low, (fork, join_high, join_low) = _get_twostream_state(output.device)
    current = torch.cuda.current_stream(output.device)
    if current == high or current == low:
        raise RuntimeError("FP4 MLA two-stream decode: the caller's stream must not be one of the arm streams.")
    fork.record(current)
    high.wait_event(fork)
    low.wait_event(fork)
    with torch.cuda.stream(high):
        run_4x(
            q_fp4_4x,
            q_sf_4x,
            kv_cache,
            sf_cache,
            v_packed,
            v_sf,
            kv_global_scale,
            q_global_scale,
            src_page_ids_4x,
            indptr_4x,
            kv_lens_4x,
            output_4x,
            num_heads=num_heads,
            query_len=query_len,
            num_requests=r4,
            max_kv_len=max_kv_len,
            sm_scale=sm_scale,
            v_page_offset=v_page_offset,
            metadata=metadata,
        )
    with torch.cuda.stream(low):
        run_2x(
            module,
            metadata,
            q_fp4_2x,
            q_sf_2x,
            kv_cache,
            sf_cache,
            v_packed,
            v_sf,
            kv_global_scale,
            q_global_scale,
            src_page_ids,
            indptr_2x,
            kv_lens_2x,
            output_2x,
            num_heads=num_heads,
            query_len=query_len,
            num_requests=r2,
            rows_per_seq=rows_per_seq,
            sm_scale=sm_scale,
            v_page_offset=v_page_offset,
            helix_kv_bounds=None,
            softmax_stats_tensor=None,
            plan_geometry=geometry_2x,
        )
    join_high.record(high)
    join_low.record(low)
    current.wait_event(join_high)
    current.wait_event(join_low)
