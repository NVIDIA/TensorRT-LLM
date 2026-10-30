# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-CTA-cluster (4x) decode dispatch: envelope decision, optional dispatch recorder, split-mode workspaces and the runner.

The kernel (``fp4_mla_cutedsl_4x``) is the four-CTA-cluster variant of the FP4 MLA decode (256 packed rows of one
request per cluster, two CTA pairs splitting the KV tiles and exchanging P over DSMEM; same FP4 operands as the
two-CTA packed kernel). ``run_fp4_mla_attention_decode`` hands it the packed-Q decode calls inside its static envelope when ``TRTLLM_FP4_MLA_CUTEDSL_4X=1`` and keeps the
packed 2x path for everything else (``fp4_mla_cutedsl_4x_host.envelope_reason_static``)."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from .config import FP4_MLA_CUTEDSL_4X_ENV as FP4_MLA_CUTEDSL_4X_ENV
from .config import _fp4_mla_cutedsl_4x_enabled as _fp4_mla_cutedsl_4x_enabled
from .layout import _ensure_workspace_tensor
from .packed_plan import _get_sm_count

# Dispatch recorder for end-to-end validation (``TRTLLM_FP4_MLA_4X_STATS_DIR=<dir>``): every 4x dispatch
# decision (taken, or the static reason the call stayed on 2x) is counted per (requests, query_len, max_kv_len,
# graph-capture vs eager) and written as JSON per process, rewritten whenever a new bucket appears and at exit.
# Default off; it never changes the decision itself.
FP4_MLA_4X_STATS_DIR_ENV = "TRTLLM_FP4_MLA_4X_STATS_DIR"
_fp4_mla_4x_stats: dict[str, int] = {}
_fp4_mla_4x_stats_path: Optional[str] = None


def _fp4_mla_4x_record(
    reason: Optional[str], num_requests: int, query_len: int, max_kv_len: int
) -> None:
    global _fp4_mla_4x_stats_path
    stats_dir = os.getenv(FP4_MLA_4X_STATS_DIR_ENV)
    if not stats_dir:
        return
    try:
        import atexit
        import json
        import socket

        capturing = torch.cuda.is_current_stream_capturing()
        key = (
            f"{'4x' if reason is None else '2x'}|B{num_requests}|q{query_len}|kv{max_kv_len}|"
            f"{'graph' if capturing else 'eager'}|{reason or ''}"
        )
        new_key = key not in _fp4_mla_4x_stats
        _fp4_mla_4x_stats[key] = _fp4_mla_4x_stats.get(key, 0) + 1

        def _dump() -> None:
            try:
                os.makedirs(stats_dir, exist_ok=True)
                with open(_fp4_mla_4x_stats_path, "w") as f:
                    json.dump(_fp4_mla_4x_stats, f, indent=1, sort_keys=True)
            except Exception:
                pass

        if _fp4_mla_4x_stats_path is None:
            _fp4_mla_4x_stats_path = os.path.join(
                stats_dir, f"dispatch_{socket.gethostname()}_{os.getpid()}.json"
            )
            atexit.register(_dump)
        if new_key:
            _dump()
    except Exception:
        pass


def _fp4_mla_cutedsl_4x_reason(
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    max_kv_len: int,
    device: torch.device,
    helix_kv_bounds: Optional[torch.Tensor],
    softmax_stats_tensor: Optional[torch.Tensor],
    v_packed: Optional[torch.Tensor],
) -> Optional[str]:
    """None when the 4x kernel takes this decode call, else why it stays on the 2x path (static shapes only)."""
    from . import fp4_mla_cutedsl_4x_host as host_4x

    return host_4x.envelope_reason_static(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=max_kv_len,
        sm_count=_get_sm_count(device),
        helix=helix_kv_bounds is not None,
        softmax_stats=softmax_stats_tensor is not None,
        v_packed=v_packed is not None,
    )


def _fp4_mla_4x_split_buffers(
    metadata: Any,
    kv_lens: torch.Tensor,
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    device: torch.device,
) -> dict:
    """Plan + FP32 partial workspace of the 4x split mode for this step (fp4_mla_cutedsl_4x_plan).

    The plan is written once per step (keyed on the generation-length serial): a single-CTA kernel on the
    corrected KV lengths, captured like any other node under CUDA graphs. Buffers live on the metadata like the
    2x packed path's (``_ensure_workspace_tensor``: grow-only, never during capture).
    """
    from .fp4_mla_cutedsl_4x_plan import launch_4x_plan, plan_4x_length, plan_4x_task_capacity

    rows = num_heads * query_len
    pairs_per_request = rows // 256
    task_capacity = plan_4x_task_capacity(num_requests, pairs_per_request)
    plan = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_4x_plan",
        (plan_4x_length(num_requests, task_capacity),),
        dtype=torch.int32,
        device=device,
    )
    # flat, grow-only workspaces viewed per call (a leading slice of a flat buffer is always contiguous)
    partial = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_4x_split_partial",
        (task_capacity * rows * 512,),
        dtype=torch.float32,
        device=device,
    ).view(task_capacity, rows, 512)
    maxima = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_4x_split_maxima",
        (task_capacity * rows,),
        dtype=torch.float32,
        device=device,
    ).view(task_capacity, rows)
    sums = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_4x_split_sums",
        (task_capacity * rows,),
        dtype=torch.float32,
        device=device,
    ).view(task_capacity, rows)
    key = (
        num_requests,
        pairs_per_request,
        rows,
        task_capacity,
        kv_lens.data_ptr(),
        plan.data_ptr(),
        metadata.fp4_mla_state.generation_lengths_serial,
    )
    if metadata.fp4_mla_state.plan_4x_key != key:
        launch_4x_plan(
            kv_lens,
            plan,
            num_requests=num_requests,
            task_capacity=task_capacity,
            pairs_per_request=pairs_per_request,
            rows_per_request=rows,
        )
        metadata.fp4_mla_state.plan_4x_key = key
    return dict(
        plan=plan,
        partial=partial,
        maxima=maxima,
        sums=sums,
        task_capacity=task_capacity,
        # static grid: capacity x pairs clusters; the slots past the plan's used tasks exit at kernel entry
        used_tasks=task_capacity,
    )


def _run_cutedsl_4x_decode(
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
    max_kv_len: int,
    sm_scale: float,
    v_page_offset: int,
    metadata: Any = None,
) -> None:
    """Four-CTA-cluster (4x) decode (fp4_mla_cutedsl_4x_host): same tensors as the packed path, whose packed Q layout
    (row = token * heads + head, heads * query_len % 256 == 0) is the layout the 4x entry takes.

    One wave of pair tasks at short KV runs the FULL-only launch (output written directly, no stats). Larger
    batches (and under-filled waves at long KV) run the split mode: a device plan gives every request its own
    KV split (whole requests fill the full waves, the tail is cut to fill the leftover cluster slots), the kernel
    writes FP32 partials for the split requests and ``reduce_packed_splits`` merges them.
    """
    from . import fp4_mla_cutedsl_4x_host as host_4x

    rows = num_heads * query_len
    if not output.is_contiguous() or output.numel() != num_requests * rows * 512:
        raise RuntimeError("FP4 MLA 4x decode requires a contiguous [tokens, heads, 512] output.")
    split = None
    if host_4x.use_split_mode(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=max_kv_len,
    ):
        if metadata is None:
            raise RuntimeError("FP4 MLA 4x split mode requires the attention metadata workspace.")
        split = _fp4_mla_4x_split_buffers(
            metadata,
            kv_lens,
            num_heads=num_heads,
            query_len=query_len,
            num_requests=num_requests,
            device=output.device,
        )
    host_4x.run_fp4_mla_decode_4x_full(
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        kv_global_scale,
        src_page_ids,
        page_indptr,
        kv_lens,
        output,
        max_kv_len=max_kv_len,
        sm_scale=sm_scale,
        num_heads=num_heads,
        query_len_per_seq=query_len,
        q_global_scale=q_global_scale,
        v_page_offset=v_page_offset,
        # the packed planner has already validated the CSR tables; never synchronize here
        validate_contents=False,
        split=split,
    )
    if split is None:
        return
    from .fp4_mla_packed_reduce import reduce_packed_splits

    # Single-task requests were written by the kernel; the reducer merges the split ones (and only pays its
    # launch when the plan kept every request whole). The kernel's denominators carry kv_global_scale, so the
    # output scale is the constant 1 (no device read).
    reduce_packed_splits(
        split["partial"],
        split["maxima"],
        split["sums"],
        output.view(num_requests * rows, 512),
        q_global_scale,
        kv_global_scale,
        num_requests=num_requests,
        num_splits=1,
        valid_m=rows,
        physical_m=rows,
        sm_scale=sm_scale,
        plan=split["plan"],
        output_scale=1.0,
        stats=None,
        sm_count=_get_sm_count(output.device),
        task_capacity=split["task_capacity"],
    )
