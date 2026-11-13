# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch geometry and per-step KV split plan of the packed FP4 MLA decode."""

import os
from typing import Any, NamedTuple, Sequence

import torch

from .config import (
    _FP4_MLA_PACKED_MAX_SPLITS,
    _FP4_MLA_PACKED_MIN_TILES_PER_SPLIT,
    _FP4_MLA_SPLIT_KV_ENV,
    FP4_MLA_SCALE_ROW_GROUP,
    _ceil_div,
    _fp4_mla_cutedsl_kernel_module,
    _fp4_mla_packed_rows_per_seq,
    _fp4_mla_packed_wave_plan_enabled,
    _fp4_mla_split_on_demand_enabled,
)
from .layout import _ensure_workspace_tensor, _max_sequences

_SM_COUNT_CACHE: dict[int, int] = {}


def _get_sm_count(device: torch.device) -> int:
    """Return the SM (multiprocessor) count for ``device``, cached per index."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNT_CACHE.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = count
    return count


def _wave_tasks(sm_count: int, m_tiles: int) -> int:
    """Tasks that fill one wave of 2-CTA clusters on ``sm_count`` SMs."""
    return max(1, max(1, sm_count // 2) // max(1, m_tiles))


def _packed_wave_tasks(device: torch.device, m_tiles: int) -> int:
    """Tasks that fill one wave of 2-CTA clusters."""
    return _wave_tasks(_get_sm_count(device), m_tiles)


def _graph_split_budget(num_requests: int, wave_tasks: int) -> int:
    """Slots per request of a CUDA-graph decode: two waves of tasks (the plan's finest target) plus one rounding
    task per request, so the target share fits whenever the batch is skewed; never fewer than two."""
    splits = _ceil_div(2 * wave_tasks, max(1, num_requests)) + 1
    return max(2, min(_FP4_MLA_PACKED_MAX_SPLITS, splits))


def _select_packed_split_count(
    num_requests: int,
    m_tiles: int,
    device: torch.device,
    *,
    max_kv_len: int,
    is_cuda_graph: bool,
) -> int:
    """Task slots per request (grid.z = requests * splits): enough for two
    waves of 2-CTA clusters plus one task per request, and never fewer than
    two so a long request in a large batch can still be split (the kernel
    shares the slots by KV length and leaves them unused when the batch is
    even).

    The count depends only on the request count and tile count, so a CUDA
    graph bucket always replays the same geometry. Eager calls know the real
    longest KV and keep at least _FP4_MLA_PACKED_MIN_TILES_PER_SPLIT KV tiles
    per split. ``TRTLLM_FP4_MLA_SPLIT_KV`` overrides the count for testing.
    """
    override = os.getenv(_FP4_MLA_SPLIT_KV_ENV, "auto")
    if override != "auto":
        return max(1, min(int(override), _FP4_MLA_PACKED_MAX_SPLITS))
    splits = _graph_split_budget(num_requests, _packed_wave_tasks(device, m_tiles))
    if not is_cuda_graph:
        kv_tiles = _ceil_div(max(max_kv_len, 1), 256)
        splits = max(1, min(splits, _ceil_div(kv_tiles, _FP4_MLA_PACKED_MIN_TILES_PER_SPLIT)))
    return splits


# On-demand split budget (TRTLLM_FP4_MLA_SPLIT_ON_DEMAND, default on). The budget of _select_packed_split_count
# costs a call one cluster launch per slot, the plan and the reduction launch even when the step's balanced plan
# keeps every request whole, as it does for an even batch that fills the machine: each wave of empty 2-CTA
# clusters takes ~2 us (DeepSeek-R1 B128/q4 on Rubin: +10.8 us per call, independent of the KV length). A step
# whose plan splits keeps the budget, so a long request still spreads over the other requests' slots; any other
# step launches one task per request. The decision is the host mirror of the kernels' plan (plan_packed_tasks);
# a graph runner keys its decode graphs by it (packed_split_variants) and hands it to the graph's metadata
# (Fp4MlaState.packed_split).


def packed_split_needed(
    kv_lens: Sequence[int],
    *,
    query_len: int,
    m_tiles: int,
    sm_count: int,
    splits: int | None = None,
) -> bool:
    """Whether the balanced plan of a decode step with these per-request KV lengths splits a request.

    The host mirror of the kernels' plan (``plan_packed_tasks``) over ``splits`` slots per request (default: the
    CUDA-graph budget). False when every request keeps one task, so the step needs no split budget.
    """
    num_requests = len(kv_lens)
    if num_requests == 0:
        return False
    wave_tasks = _wave_tasks(sm_count, m_tiles)
    if splits is None:
        splits = _graph_split_budget(num_requests, wave_tasks)
    if splits <= 1:
        return False
    from .fp4_mla_packed_reduce import plan_packed_tasks, request_kv_tiles

    _, begins = plan_packed_tasks(
        request_kv_tiles(list(kv_lens), query_len), splits * num_requests, wave_tasks
    )
    return begins[-1] > num_requests


def packed_split_variants(num_requests: int, *, m_tiles: int, sm_count: int) -> tuple[bool, ...]:
    """Split decisions a decode graph of this batch shape can replay, the values a graph runner keys it by.

    An even batch keeps every request whole once it fills two thirds of a wave (the plan splits a request only
    when it outlasts 1.5x the balanced share), so such a shape gets an unsplit and a split graph. A smaller batch
    splits at any KV worth splitting and keeps the split graph alone; a step it would keep whole (a very short
    one) replays the split graph, as without the on-demand rule.
    """
    wave_tasks = _wave_tasks(sm_count, m_tiles)
    return (False, True) if 3 * num_requests >= 2 * wave_tasks else (True,)


def _host_generation_kv_lens(metadata: Any, num_requests: int) -> list[int] | None:
    """KV lengths (past + new tokens) of the step's generation requests, the last ``num_requests`` of
    ``kv_lens_runtime`` on the host, or None. ``prompt_lens_cpu_runtime`` is no substitute: in generation it
    holds the new tokens per request."""
    lens = getattr(metadata, "kv_lens_runtime", None)
    if (
        isinstance(lens, torch.Tensor)
        and lens.device.type == "cpu"
        and lens.numel() >= num_requests
    ):
        return [int(x) for x in lens[lens.numel() - num_requests :].tolist()]
    return None


def _on_demand_split_count(
    splits: int,
    metadata: Any,
    *,
    num_requests: int,
    m_tiles: int,
    query_len: int,
    device: torch.device,
    helix: bool,
) -> int:
    """Slots per request after the on-demand rule: ``splits`` when the step's plan splits a request, else 1.

    A decode graph keyed by the decision (``Fp4MlaState.packed_split``) replays it; a graph without one keeps the
    budget, since its replays may split; an eager step decides on its host KV lengths. Helix (skewed per-rank KV
    slices) and a ``TRTLLM_FP4_MLA_SPLIT_KV`` override keep the budget.
    """
    if (
        splits <= 1
        or helix
        or not _fp4_mla_split_on_demand_enabled()
        or os.getenv(_FP4_MLA_SPLIT_KV_ENV, "auto") != "auto"
    ):
        return splits
    decided = getattr(metadata.fp4_mla_state, "packed_split", None)
    if decided is not None:
        return splits if decided else 1
    if bool(getattr(metadata, "is_cuda_graph", False)):
        return splits
    kv_lens = _host_generation_kv_lens(metadata, num_requests)
    if kv_lens is None:
        return splits
    needed = packed_split_needed(
        kv_lens,
        query_len=query_len,
        m_tiles=m_tiles,
        sm_count=_get_sm_count(device),
        splits=splits,
    )
    return splits if needed else 1


_SPLIT_TASK_BOUND_CACHE: dict[tuple[int, int], int] = {}


def packed_split_task_bound(max_requests: int, m_tiles: int, device: torch.device) -> int:
    """Largest ``splits * requests`` any batch up to ``max_requests`` can ask for: the capacity of the shared
    split-KV scratch (``layout._ensure_workspace_tensor``). The graph rule bounds the eager rule (eager only caps
    the splits further by the KV length) and the eager floor of 32 slots is kept."""
    override = os.getenv(_FP4_MLA_SPLIT_KV_ENV, "auto")
    if override != "auto":
        splits = max(1, min(int(override), _FP4_MLA_PACKED_MAX_SPLITS))
        return max(32, splits * max(1, int(max_requests)))
    wave_tasks = _packed_wave_tasks(device, m_tiles)
    key = (wave_tasks, max(1, int(max_requests)))
    bound = _SPLIT_TASK_BOUND_CACHE.get(key)
    if bound is None:
        bound = 32
        for requests in range(1, key[1] + 1):
            splits = max(
                2, min(_FP4_MLA_PACKED_MAX_SPLITS, _ceil_div(2 * wave_tasks, requests) + 1)
            )
            bound = max(bound, splits * requests)
        _SPLIT_TASK_BOUND_CACHE[key] = bound
    return bound


def packed_split_scratch_bound(
    metadata: Any, num_heads: int, query_len: int, device: torch.device, *, max_requests: int
) -> tuple[int, int]:
    """(slot bound, slot x row bound) of the shared packed split scratch over every query width the engine can
    decode (1 .. 1 + max_total_draft_tokens, ``query_len`` included). Narrow rows mean more wave tasks and more
    splits, so neither bound is monotonic in the width: both are maxima over all of them."""
    width = max(int(query_len), 1 + int(getattr(metadata, "max_total_draft_tokens", None) or 0))
    slots_bound, slot_rows_bound = 32, 0
    for q in range(1, width + 1):
        rows = _fp4_mla_packed_rows_per_seq(num_heads, q)
        if rows <= 0:
            continue
        slots = packed_split_task_bound(max_requests, rows // FP4_MLA_SCALE_ROW_GROUP, device)
        slots_bound = max(slots_bound, slots)
        slot_rows_bound = max(slot_rows_bound, slots * rows)
    return slots_bound, slot_rows_bound


class _PackedPlanGeometry(NamedTuple):
    """Per-step launch geometry of the packed split-KV decode."""

    num_requests: int
    query_len: int
    splits: int
    wave_tasks: int
    profile_tiles: int
    task_capacity: int
    split_kv_len: int
    helix_kv_bounds: torch.Tensor | None
    plan: torch.Tensor | None
    key: tuple | None
    # capacity of the shared split-KV scratch: task slots and slot x row products (0 = single task per request)
    task_bound: int = 0
    slot_rows_bound: int = 0


def _fp4_mla_packed_plan_geometry(
    metadata: Any,
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    device: torch.device,
    helix_kv_bounds: torch.Tensor | None,
) -> _PackedPlanGeometry | None:
    """Slots, task-length cap and plan buffer of this step's packed decode.

    None when the packed path is off. The geometry depends only on the batch
    shape and the page-table capacity, so the generation cache update (the
    first FP4 MLA kernel of a step) and the attention decode derive the same
    one, and whichever runs first writes the step's plan fused with the
    generation lengths (``_fp4_mla_uniform_generation_lengths``). ``key`` is
    None when every request keeps a single task and no plan is needed.
    """
    rows_per_seq = _fp4_mla_packed_rows_per_seq(num_heads, query_len)
    if rows_per_seq == 0 or num_requests <= 0:
        return None
    if not metadata.fp4_mla_state.device_page_table:
        return None
    # The fixed-stride generation page-table capacity (see _max_generation_pages).
    max_pages = int(metadata.fp4_mla_state.page_table_stride)
    if max_pages <= 0:
        return None
    max_kv_len = max_pages * metadata.page_size
    is_cuda_graph = bool(getattr(metadata, "is_cuda_graph", False))
    m_tiles = rows_per_seq // FP4_MLA_SCALE_ROW_GROUP
    splits = _select_packed_split_count(
        num_requests,
        m_tiles,
        device,
        max_kv_len=max_kv_len,
        is_cuda_graph=is_cuda_graph,
    )
    splits = _on_demand_split_count(
        splits,
        metadata,
        num_requests=num_requests,
        m_tiles=m_tiles,
        query_len=query_len,
        device=device,
        helix=helix_kv_bounds is not None or bool(getattr(metadata, "enable_helix", False)),
    )
    # Tasks that fill one wave of 2-CTA clusters: the plan's task-length reference.
    wave_tasks = _packed_wave_tasks(device, m_tiles)
    # The runtime KV profile only has to cover one task: the kernel clamps the
    # plan's task length to it, and ceil(longest / (splits - 1)) tiles per task
    # always fits the splits * requests slots. A batch that fills the machine
    # with one task per request keeps the full profile so even batches never
    # have to split.
    kv_tiles = _ceil_div(max(max_kv_len, 1), 256)
    if splits == 1 or wave_tasks <= 2 * num_requests:
        task_tiles = kv_tiles
    else:
        task_tiles = min(kv_tiles, max(_ceil_div(kv_tiles, splits - 1), 2))
    split_kv_len = max(task_tiles * 256, 256)
    # The kernels clamp the plan to the compiled profile (a bucket >= split_kv_len).
    module = _fp4_mla_cutedsl_kernel_module()
    profile_tiles = module._select_runtime_kv_profile(split_kv_len) // 256
    if splits == 1:
        return _PackedPlanGeometry(
            num_requests,
            query_len,
            1,
            wave_tasks,
            profile_tiles,
            num_requests,
            split_kv_len,
            helix_kv_bounds,
            None,
            None,
        )
    from .fp4_mla_packed_reduce import packed_plan_length

    tasks = splits * num_requests
    # Graph buckets hold exactly their tasks; eager metadata grows on demand. The buffers themselves are
    # process-wide (layout._ensure_workspace_tensor), sized once for the largest batch any bucket can bring.
    task_capacity = tasks if is_cuda_graph else max(tasks, 32)
    max_requests = max(_max_sequences(metadata), num_requests)
    slots_bound, slot_rows_bound = packed_split_scratch_bound(
        metadata, num_heads, query_len, device, max_requests=max_requests
    )
    task_bound = max(slots_bound, task_capacity)
    slot_rows_bound = max(slot_rows_bound, task_capacity * rows_per_seq)
    plan = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_packed_plan",
        (packed_plan_length(num_requests, task_capacity),),
        dtype=torch.int32,
        device=device,
        capacity_numel=packed_plan_length(max_requests, task_bound),
    )
    kv_lens = metadata.fp4_mla_state.generation_kv_lens
    key = (
        num_requests,
        splits,
        wave_tasks,
        profile_tiles,
        query_len,
        task_capacity,
        kv_lens.data_ptr() if isinstance(kv_lens, torch.Tensor) else 0,
        helix_kv_bounds.data_ptr() if helix_kv_bounds is not None else 0,
        plan.data_ptr(),
    )
    return _PackedPlanGeometry(
        num_requests,
        query_len,
        splits,
        wave_tasks,
        profile_tiles,
        task_capacity,
        split_kv_len,
        helix_kv_bounds,
        plan,
        key,
        task_bound,
        slot_rows_bound,
    )


def _fp4_mla_helix_plan_bounds(
    metadata: Any, num_tokens: int, *, use_helix: bool
) -> torch.Tensor | None:
    """Per-token Helix KV bounds the decode plans on, or None (validated at decode)."""
    if not use_helix or not bool(getattr(metadata, "_helix_spec_tokens_valid", False)):
        return None
    bounds = getattr(metadata, "helix_kv_bounds", None)
    if (
        not isinstance(bounds, torch.Tensor)
        or bounds.dtype != torch.int32
        or bounds.ndim != 1
        or bounds.numel() < num_tokens
        or not bounds.is_contiguous()
    ):
        return None
    return bounds[:num_tokens]


def _launch_fp4_mla_packed_plan(
    geometry: _PackedPlanGeometry,
    kv_lens: torch.Tensor,
    metadata: Any,
    *,
    generation_lengths: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
) -> None:
    """Write the step's packed split plan, fused with the generation lengths when given."""
    from .fp4_mla_packed_reduce import launch_packed_plan

    launch_packed_plan(
        kv_lens,
        geometry.plan,
        num_requests=geometry.num_requests,
        num_splits=geometry.splits,
        wave_tasks=geometry.wave_tasks,
        max_tiles_per_task=geometry.profile_tiles,
        query_len=geometry.query_len,
        helix_kv_bounds=geometry.helix_kv_bounds,
        generation_lengths=generation_lengths,
        wave_plan=_fp4_mla_packed_wave_plan_enabled(),
    )
    metadata.fp4_mla_state.packed_plan_key = geometry.key
