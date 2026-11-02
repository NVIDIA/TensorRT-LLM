# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch geometry and per-step KV split plan of the packed FP4 MLA decode."""

import os
from typing import Any, NamedTuple

import torch

from .config import (
    _FP4_MLA_PACKED_MAX_SPLITS,
    _FP4_MLA_PACKED_MIN_TILES_PER_SPLIT,
    _FP4_MLA_SPLIT_KV_ENV,
    FP4_MLA_PACKED_WAVE_PLAN_ENV,
    FP4_MLA_SCALE_ROW_GROUP,
    _ceil_div,
    _env_enabled_default,
    _fp4_mla_cutedsl_kernel_module,
    _fp4_mla_packed_rows_per_seq,
)
from .layout import _ensure_workspace_tensor

_SM_COUNT_CACHE: dict[int, int] = {}


def _get_sm_count(device: torch.device) -> int:
    """Return the SM (multiprocessor) count for ``device``, cached per index."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNT_CACHE.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = count
    return count


def _packed_wave_tasks(device: torch.device, m_tiles: int) -> int:
    """Tasks that fill one wave of 2-CTA clusters."""
    clusters_per_wave = max(1, _get_sm_count(device) // 2)
    return max(1, clusters_per_wave // max(1, m_tiles))


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
    # Slots for two waves of tasks (the plan's finest target) plus one rounding
    # task per request, so the target share fits whenever the batch is skewed.
    wave_tasks = _packed_wave_tasks(device, m_tiles)
    splits = _ceil_div(2 * wave_tasks, max(1, num_requests)) + 1
    splits = max(2, min(_FP4_MLA_PACKED_MAX_SPLITS, splits))
    if not is_cuda_graph:
        kv_tiles = _ceil_div(max(max_kv_len, 1), 256)
        splits = max(1, min(splits, _ceil_div(kv_tiles, _FP4_MLA_PACKED_MIN_TILES_PER_SPLIT)))
    return splits


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
    # Graph buckets hold exactly their tasks; eager metadata grows on demand.
    task_capacity = tasks if is_cuda_graph else max(tasks, 32)
    plan = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_packed_plan",
        (packed_plan_length(num_requests, task_capacity),),
        dtype=torch.int32,
        device=device,
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
        wave_plan=_env_enabled_default(FP4_MLA_PACKED_WAVE_PLAN_ENV, True),
    )
    metadata.fp4_mla_state.packed_plan_key = geometry.key
