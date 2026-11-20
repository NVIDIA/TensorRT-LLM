# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3's packed FP4 MLA generation decode: step metadata, task plan, decode and split reduction.

The host side of the served FP4 MLA generation decode (6904cd9672,
``attention/backends/fp4_mla/{packed_plan,metadata,decode}.py``) for the one configuration the
deployment runs: the CuTe DSL backend on SM107 with K3's packed decode kernel module
(``kernels/fp4_mla_mufu16``, served ``set_fp4_mla_cutedsl_kernel_module``), the packed per-request
query layout, 8 task slots per request (``TRTLLM_FP4_MLA_SPLIT_KV=8``), the wave plan (no makespan
plan) and the page plan. The task plan and the split reduction are ``kernels/fp4_mla_packed_reduce``
(the served file); the KV cache layout, the length / page-table kernels and the workspaces are the
FP4 MLA backend's (``metadata.fp4_mla_state``), whose helpers this module calls where #21996 kept
them byte-identical.

Per step, the first call that needs it (``prelaunch_generation_metadata`` on the model's plan stream,
else the step's first FP4 MLA cache update) writes the generation lengths fused with the plan, then
the device page table; the state's ``packed_plan_key`` marks the plan current until the backend
invalidates the step (``prepare`` / ``on_update_kv_lens``) or :func:`begin_step` forgets it.

FP4 MLA cache update (``_fp4_mla_cache_update``) and decode share :func:`packed_plan_geometry` and
:func:`uniform_generation_lengths`, as served: the cache update never reaches the backend's own
plan of the same name.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple, Optional

import torch

from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    _FP4_MLA_PACKED_MAX_M,
    _FP4_MLA_PACKED_MAX_SPLITS,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_Q_LOGICAL_DIM,
    FP4_MLA_Q_PACKED_DIM,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_SCALE_ROW_GROUP,
    FP4_MLA_TOKENS_PER_BLOCK,
    _cutedsl_backend_available,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _ensure_workspace_tensor,
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _get_fp4_mla_q_global_scale,
    _get_fp4_mla_swizzled_scale_size,
    _validate_fp4_mla_attention_q_shape,
    _validate_fp4_mla_cache_shape,
    _validate_fp4_mla_kv_storage_shape,
    get_fp4_mla_v_scale_pool_view,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _fp4_mla_generation_page_ids,
    _get_linear_mtp_query_len_per_seq,
    _max_generation_pages,
    materialize_fp4_mla_device_page_table,
    populate_fp4_mla_generation_lengths,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..common.k3_ull_features import active as _features
from ._k3_fp4_mla_cache import (
    _get_cutedsl_persistent_v_packed_cache,
    _get_fp4_mla_v_packed_pool_base,
    _get_fp4_mla_v_scale_pool_base,
)


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


# The served backend read these from the environment; 1007804's values:
# Task slots per request of the packed decode (grid.z = requests * splits): TRTLLM_FP4_MLA_SPLIT_KV=8.
_SPLITS = 8
assert 1 < _SPLITS <= _FP4_MLA_PACKED_MAX_SPLITS
# The makespan plan (TRTLLM_FP4_MLA_LPT_PLAN=1): on; it takes precedence over the wave plan
# (TRTLLM_FP4_MLA_PACKED_WAVE_PLAN, unset; K3's module has SUPPORTS_ORDERED_PLAN).
_LPT_PLAN = True
_WAVE_PLAN = True
# Where the decode releases a late-released fold consumer (TRTLLM_K3_ULL3_MLA_TAIL_TRIGGER=epilogue):
# once its producer warps have finished the last KV tile (the kernel module's
# LAUNCH_DEPENDENTS_EPILOGUE); False: only the decode's completion.
_ULL3_MLA_TAIL_TRIGGER_EPILOGUE = True
# The packed per-request query layout (TRTLLM_FP4_MLA_PACKED_Q, unset): on; the CuTe DSL backend
# without the fused V transpose (TRTLLM_FP4_MLA_ATTENTION_BACKEND,
# TRTLLM_FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE, unset).
_PACKED_Q = True
#: The backend name the FP4 MLA layout helpers take (served ``_FP4_MLA_CUTEDSL_BACKEND``).
CUTEDSL_BACKEND = "cutedsl"


@functools.cache
def cutedsl_backend() -> bool:
    """Whether the FP4 MLA layers run the CuTe DSL backend: SM107 with a CuTe DSL that builds
    the FP4 MLA kernels (the served backend selection with TRTLLM_FP4_MLA_ATTENTION_BACKEND
    unset). Elsewhere the deployment does not run."""
    return get_sm_version() == 107 and _cutedsl_backend_available()


def packed_rows_per_seq(num_heads: int, query_len: int) -> int:
    """Rows one request owns in the packed Q layout (``_fp4_mla_packed_rows_per_seq`` with the
    deployment's settings), or 0 when the batch does not pack."""
    if not (_PACKED_Q and cutedsl_backend()) or num_heads <= 0 or query_len <= 0:
        return 0
    rows = _ceil_div(num_heads * query_len, FP4_MLA_SCALE_ROW_GROUP) * FP4_MLA_SCALE_ROW_GROUP
    return rows if rows <= _FP4_MLA_PACKED_MAX_M else 0


# Workspace names (``fp4_mla_state.workspaces``); the backend's own buffers have other names.
_PLAN_WS = "_k3_fp4_mla_packed_plan"
_PARTIAL_WS = "_k3_fp4_mla_split_partial"
_PARTIAL_F16_WS = "_k3_fp4_mla_split_partial_f16"
_MAXIMA_WS = "_k3_fp4_mla_split_maxima"
_SUMS_WS = "_k3_fp4_mla_split_sums"


def _kernel_module() -> Any:
    from ...kernels.k3r_fp4_mla import fp4_mla_mufu16

    return fp4_mla_mufu16


def _reduce_module() -> Any:
    from ...kernels.k3r_fp4_mla import fp4_mla_packed_reduce

    return fp4_mla_packed_reduce


def applies(attn: Any) -> bool:
    """Whether ``attn`` (a generation ``K3MlaAttnState``) runs its FP4 MLA generation here: an
    FP4 MLA layer on the CuTe DSL backend."""
    return bool(getattr(attn, "uses_fp4_mla_attention", False) and cutedsl_backend())


# ---------------------------------------------------------------------------
# Per-step state.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Step:
    """The served ``Fp4MlaState`` fields #21996's state does not carry (kept on the metadata's
    ``fp4_mla_state``): whether another kernel writes this step's query rows, the count of
    decode-metadata kernels this module launched and that count when the query kernel was
    launched (:func:`mark_query_launch`)."""

    q_rows_external: bool = False
    metadata_launches: int = 0
    query_launch_mark: Optional[int] = None


def _step(metadata: Any) -> _Step:
    state = metadata.fp4_mla_state
    step = getattr(state, "_k3_fp4_mla_decode", None)
    if step is None:
        step = _Step()
        state._k3_fp4_mla_decode = step
    return step


def begin_step(metadata: Any) -> None:
    """Forget the previous step's plan (the K3 model calls this at the start of every forward; the
    backend's ``prepare`` already did for a new step)."""
    state = getattr(metadata, "fp4_mla_state", None)
    if state is not None:
        state.packed_plan_key = None


def _count_metadata_launch(metadata: Any) -> None:
    """Count a kernel that writes decode metadata on the current stream; a decode launched right
    after one must not read that metadata before its wait."""
    _step(metadata).metadata_launches += 1


def mark_query_launch(metadata: Any) -> None:
    """Record that the kernel writing this step's query rows was just launched (the fused
    query): a decode launched as its programmatic dependent (``features.ull2_mla_decode_pdl``)
    requires that no metadata kernel was launched since."""
    step = _step(metadata)
    step.query_launch_mark = step.metadata_launches


# ---------------------------------------------------------------------------
# Plan geometry and step metadata.
# ---------------------------------------------------------------------------

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


class PackedPlanGeometry(NamedTuple):
    """Per-step launch geometry of the packed split-KV decode."""

    num_requests: int
    query_len: int
    splits: int
    wave_tasks: int
    profile_tiles: int
    task_capacity: int
    split_kv_len: int
    helix_kv_bounds: torch.Tensor | None
    plan: torch.Tensor
    key: tuple
    # Decode clusters per task and the wave's clusters when the makespan plan is on, else 0.
    lpt_m_tiles: int
    lpt_wave_clusters: int
    # The wave plan (task length by simulated waves, full tasks first).
    wave_plan: bool


def packed_plan_geometry(
    metadata: Any,
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    device: torch.device,
    helix_kv_bounds: torch.Tensor | None,
) -> PackedPlanGeometry | None:
    """Slots, task-length cap and plan buffer of this step's packed decode.

    None when the packed Q layout is off or there are no generation pages. The geometry depends
    only on the batch shape and the page-table capacity, so the prelaunch and the decode derive
    the same one.
    """
    rows_per_seq = packed_rows_per_seq(num_heads, query_len)
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
    splits = _SPLITS
    # Tasks that fill one wave of 2-CTA clusters: the plan's task-length reference.
    wave_tasks = _packed_wave_tasks(device, m_tiles)
    # The runtime KV profile only has to cover one task: the kernel clamps the
    # plan's task length to it, and ceil(longest / (splits - 1)) tiles per task
    # always fits the splits * requests slots. A batch that fills the machine
    # with one task per request keeps the full profile so even batches never
    # have to split.
    kv_tiles = _ceil_div(max(max_kv_len, 1), 256)
    if wave_tasks <= 2 * num_requests:
        task_tiles = kv_tiles
    else:
        task_tiles = min(kv_tiles, max(_ceil_div(kv_tiles, splits - 1), 2))
    split_kv_len = max(task_tiles * 256, 256)
    # The kernels clamp the plan to the compiled profile (a bucket >= split_kv_len).
    module = _kernel_module()
    if not _features().fp4_mla_page_plan:
        split_kv_len = max(split_kv_len, module.SMEM_P4_PAGE_PLAN_PROFILE_KV + 1)
    profile_tiles = module._select_runtime_kv_profile(split_kv_len) // 256
    reduce = _reduce_module()
    tasks = splits * num_requests
    # Graph buckets hold exactly their tasks; eager metadata grows on demand.
    task_capacity = tasks if is_cuda_graph else max(tasks, 32)
    plan = _ensure_workspace_tensor(
        metadata,
        _PLAN_WS,
        (reduce.packed_plan_length(num_requests, task_capacity),),
        dtype=torch.int32,
        device=device,
    )
    kv_lens = metadata.fp4_mla_state.generation_kv_lens
    lpt_m_tiles = m_tiles if _LPT_PLAN else 0
    lpt_wave_clusters = max(1, _get_sm_count(device) // 2) if lpt_m_tiles else 0
    wave_plan = _WAVE_PLAN
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
        lpt_m_tiles,
        wave_plan,
    )
    return PackedPlanGeometry(
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
        lpt_m_tiles,
        lpt_wave_clusters,
        wave_plan,
    )


def helix_plan_bounds(metadata: Any, num_tokens: int, *, use_helix: bool) -> torch.Tensor | None:
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


def _launch_packed_plan(
    geometry: PackedPlanGeometry,
    kv_lens: torch.Tensor,
    metadata: Any,
    *,
    generation_lengths: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
) -> None:
    """Write the step's packed split plan, fused with the generation lengths when given."""
    _count_metadata_launch(metadata)
    _reduce_module().launch_packed_plan(
        kv_lens,
        geometry.plan,
        num_requests=geometry.num_requests,
        num_splits=geometry.splits,
        wave_tasks=geometry.wave_tasks,
        max_tiles_per_task=geometry.profile_tiles,
        query_len=geometry.query_len,
        helix_kv_bounds=geometry.helix_kv_bounds,
        generation_lengths=generation_lengths,
        lpt_m_tiles=geometry.lpt_m_tiles,
        lpt_wave_clusters=geometry.lpt_wave_clusters,
        wave_plan=geometry.wave_plan,
    )
    metadata.fp4_mla_state.packed_plan_key = geometry.key


def uniform_generation_lengths(
    metadata: Any,
    num_gen_tokens: int,
    num_gen: int,
    plan_geometry: PackedPlanGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the preallocated CUDA generation KV and append lengths of the backend's state.

    With ``plan_geometry`` the launch that writes the lengths also writes this module's plan of
    the step (one CuTe DSL kernel instead of two launches), and a stale plan is rewritten even
    when the lengths are already current.
    """
    if num_gen <= 0 or num_gen_tokens % num_gen != 0:
        raise RuntimeError("FP4 MLA generation requires a non-empty uniform request batch.")
    if plan_geometry is not None and plan_geometry.num_requests != num_gen:
        raise RuntimeError("FP4 MLA packed plan geometry does not match the generation batch.")

    state = metadata.fp4_mla_state
    num_contexts = metadata.num_contexts
    num_seqs = metadata.num_seqs
    kv_lens_gen = metadata.kv_lens_cuda_runtime[num_contexts:num_seqs]
    prompt_lens_gen = metadata.prompt_lens_cuda_runtime[num_contexts:num_seqs]
    corrected_kv_lens = getattr(state, "generation_kv_lens", None)
    generation_lens = getattr(state, "generation_append_lens", None)
    tensors = (
        kv_lens_gen,
        prompt_lens_gen,
        corrected_kv_lens,
        generation_lens,
    )
    if (
        not all(isinstance(tensor, torch.Tensor) for tensor in tensors)
        or corrected_kv_lens.numel() < num_gen
        or generation_lens.numel() < num_gen
        or not all(tensor.is_cuda for tensor in tensors)
    ):
        raise RuntimeError("FP4 MLA generation lengths require preallocated CUDA buffers.")

    record_for_capture = bool(
        getattr(metadata, "is_cuda_graph", False)
        and torch.cuda.is_current_stream_capturing()
        and not getattr(state, "generation_lengths_capture_recorded", False)
    )
    precomputed = (
        not record_for_capture
        and state.generation_lengths_num_tokens == num_gen_tokens
        and state.generation_lengths_num_seqs == num_gen
        and state.generation_lengths_num_contexts == num_contexts
    )
    plan_pending = plan_geometry is not None and state.packed_plan_key != plan_geometry.key
    if not precomputed:
        if plan_pending:
            _launch_packed_plan(
                plan_geometry,
                kv_lens_gen,
                metadata,
                generation_lengths=(
                    prompt_lens_gen,
                    corrected_kv_lens[:num_gen],
                    generation_lens[:num_gen],
                    num_gen_tokens // num_gen,
                ),
            )
        else:
            _count_metadata_launch(metadata)
            populate_fp4_mla_generation_lengths(
                kv_lens_gen,
                prompt_lens_gen,
                corrected_kv_lens[:num_gen],
                generation_lens[:num_gen],
                num_gen_tokens=num_gen_tokens,
                num_gen=num_gen,
            )
        state.generation_lengths_num_tokens = num_gen_tokens
        state.generation_lengths_num_seqs = num_gen
        state.generation_lengths_num_contexts = num_contexts
        if record_for_capture:
            state.generation_lengths_capture_recorded = True
    elif plan_pending:
        _launch_packed_plan(plan_geometry, corrected_kv_lens[:num_gen], metadata)
    return corrected_kv_lens[:num_gen], generation_lens[:num_gen]


def materialize_page_table(
    metadata: Any, generation_kv_lens: Optional[torch.Tensor] = None
) -> None:
    """The backend's fixed-stride device page table of this forward, from the final device
    lengths (once per forward; counted as a metadata launch when it launches)."""
    state = metadata.fp4_mla_state
    if not bool(getattr(state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA cache update requires fixed-stride device page metadata.")
    if bool(getattr(state, "device_page_table_valid", False)):
        return
    num_contexts = int(metadata.num_contexts)
    num_sequences = int(metadata.num_seqs)
    if generation_kv_lens is None:
        if num_sequences > num_contexts:
            num_generation_tokens = int(metadata.num_tokens) - int(metadata.num_ctx_tokens)
            generation_kv_lens, _ = uniform_generation_lengths(
                metadata, num_generation_tokens, num_sequences - num_contexts
            )
        else:
            generation_kv_lens = metadata.kv_lens_cuda_runtime[num_contexts:num_sequences]
    _count_metadata_launch(metadata)
    materialize_fp4_mla_device_page_table(
        metadata,
        metadata.kv_lens_cuda_runtime[:num_sequences],
        generation_kv_lens,
    )


def prelaunch_generation_metadata(
    metadata: Any, *, num_heads: int, use_helix: bool, device: torch.device
) -> bool:
    """``features.ull4_mla_plan_early``: launch on the current stream what the first FP4 MLA
    generation layer of a step launches for the step's metadata: the generation lengths fused
    with this module's plan of ``num_heads`` query heads, then the device page table. Later calls
    of the step find them current and launch nothing. Returns whether it launched (a
    generation-only step with the device page table); the caller orders the step's first FP4 MLA
    kernel after this stream."""
    state = getattr(metadata, "fp4_mla_state", None)
    if state is None or not state.device_page_table:
        return False
    num_contexts = int(metadata.num_contexts)
    num_gen = int(metadata.num_seqs) - num_contexts
    num_tokens = int(metadata.num_tokens)
    if num_contexts != 0 or num_gen <= 0 or num_tokens % num_gen != 0 or num_heads <= 0:
        return False
    # The plan workspace is reallocated on a device mismatch, and tensors carry an index.
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    plan_geometry = packed_plan_geometry(
        metadata,
        num_heads=num_heads,
        query_len=num_tokens // num_gen,
        num_requests=num_gen,
        device=device,
        helix_kv_bounds=helix_plan_bounds(metadata, num_tokens, use_helix=use_helix),
    )
    kv_lens_gen, _ = uniform_generation_lengths(
        metadata, num_tokens, num_gen, plan_geometry=plan_geometry
    )
    materialize_page_table(metadata, kv_lens_gen)
    return True


# ---------------------------------------------------------------------------
# Decode.
# ---------------------------------------------------------------------------


def _run_packed_decode(
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
    rows_per_seq: int,
    sm_scale: float,
    v_page_offset: int,
    helix_kv_bounds: torch.Tensor | None,
    softmax_stats_tensor: torch.Tensor | None,
    plan_geometry: PackedPlanGeometry,
    metadata_launches: int,
) -> None:
    """Packed CuTe DSL decode: one kernel over (split, request) tasks.

    The splits write partials (FP32, or normalized FP16 with ``features.fp4_mla_f16_partials``)
    that one reduction kernel folds into the output and stats (or a ``DeferredPackedReduce``
    consumer folds). Nothing pads, replicates, or copies Q or the output.

    ``metadata_launches`` is the metadata-launch count when the decode call began; with
    ``features.ull2_mla_decode_pdl`` the decode is launched programmatically only if no metadata
    kernel was launched since and the query rows come from another kernel (the fused query).
    """
    valid_m, physical_m = module.packed_query_rows(num_heads, query_len)
    if physical_m != rows_per_seq:
        raise RuntimeError(
            f"FP4 MLA packed Q rows {rows_per_seq} do not match the kernel geometry {physical_m}."
        )
    if v_packed is None:
        raise RuntimeError("FP4 MLA packed decode requires the persistent packed-V cache.")
    if not output.is_contiguous() or output.numel() != num_requests * valid_m * 512:
        raise RuntimeError(
            "FP4 MLA packed decode requires a contiguous [tokens, heads, 512] output."
        )
    device = output.device
    splits = plan_geometry.splits
    step = _step(metadata)
    use_pdl_wait = (
        _features().ull2_mla_decode_pdl
        and step.q_rows_external
        and (step.query_launch_mark is None or step.query_launch_mark == metadata_launches)
    )
    output_rows = output.view(num_requests * valid_m, 512)
    stats_rows = None
    if softmax_stats_tensor is not None:
        stats_rows = softmax_stats_tensor.view(num_requests * valid_m, 2)
    common = dict(
        max_kv_len=plan_geometry.split_kv_len,
        sm_scale=sm_scale,
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        num_splits=splits,
        q_global_scale=q_global_scale,
        v_page_offset=v_page_offset,
        helix_kv_bounds=helix_kv_bounds,
    )
    reduce = _reduce_module()
    # TRTLLM_FP4_MLA_REDUCE_PDL / TRTLLM_FP4_MLA_F16_PARTIALS in the served backend.
    reduce_pdl = _features().fp4_mla_reduce_pdl
    deferred = reduce.UllDeferredPackedReduce.active()
    launch_dependents = reduce_pdl and (deferred is None or deferred.release_early)
    if deferred is not None and not deferred.release_early:
        launch_dependents = (
            module.LAUNCH_DEPENDENTS_EPILOGUE if _ULL3_MLA_TAIL_TRIGGER_EPILOGUE else False
        )
    f16_partials = _features().fp4_mla_f16_partials and torch.float16 in tuple(
        getattr(module, "SPLIT_PARTIAL_DTYPES", ())
    )
    tasks = splits * num_requests
    task_capacity = plan_geometry.task_capacity
    plan = plan_geometry.plan
    # The first call of the step wrote the plan fused with the generation lengths; a stale key
    # means this call plans on other bounds.
    if metadata.fp4_mla_state.packed_plan_key != plan_geometry.key:
        _launch_packed_plan(plan_geometry, kv_lens, metadata)
    partial = _ensure_workspace_tensor(
        metadata,
        _PARTIAL_F16_WS if f16_partials else _PARTIAL_WS,
        (task_capacity, physical_m, 512),
        dtype=torch.float16 if f16_partials else torch.float32,
        device=device,
    )[:tasks]
    maxima = _ensure_workspace_tensor(
        metadata,
        _MAXIMA_WS,
        (task_capacity, physical_m),
        dtype=torch.float32,
        device=device,
    )[:tasks]
    sums = _ensure_workspace_tensor(
        metadata,
        _SUMS_WS,
        (task_capacity, physical_m),
        dtype=torch.float32,
        device=device,
    )[:tasks]
    module.run_trtllm_fp4_mla_decode_packed(
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
        output_rows,
        split_kv_buffers=(partial, maxima, sums),
        plan=plan,
        **common,
        launch_dependents=launch_dependents,
        pdl_wait=use_pdl_wait and step.metadata_launches == metadata_launches,
        kv_ready_operand=True,
    )
    reduce.reduce_packed_splits(
        partial,
        maxima,
        sums,
        output_rows,
        q_global_scale,
        kv_global_scale,
        num_requests=num_requests,
        num_splits=splits,
        valid_m=valid_m,
        physical_m=physical_m,
        sm_scale=sm_scale,
        plan=plan,
        # FP16 partials are already in output units.
        output_scale=1.0 if f16_partials else 1.0 / module.FP4_MLA_P_GLOBAL_SCALE,
        stats=stats_rows,
        sm_count=_get_sm_count(device),
        use_pdl=reduce_pdl,
        defer_with=reduce.UllDeferredPackedReduce,
    )


def run_decode(
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q: torch.Tensor,
    output: torch.Tensor,
    *,
    sm_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    prequantized_q: torch.Tensor,
    prequantized_q_sf: torch.Tensor,
    q_batch_capacity: int,
    softmax_stats_tensor: torch.Tensor | None = None,
) -> None:
    """MLA decode with FP4 QK and FP4 PV tensor-core matmuls on the packed query rows the
    generation cache update prepared (and wrote, or left to the fused query, which wrote them;
    served ``run_fp4_mla_attention_decode``, packed CuTe DSL branch).

    ``q`` (``[tokens, heads, latent + RoPE]``) provides shapes; ``output`` is ``[tokens, heads,
    512]`` and ``softmax_stats_tensor`` the Helix ``[tokens, heads, 2]`` (max, sum) statistics.
    """
    head_dim = kv_lora_rank + qk_rope_head_dim
    metadata_launches = _step(metadata).metadata_launches
    if qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "FP4 MLA K residual attention requires "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {qk_rope_head_dim}."
        )
    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)
    if metadata.page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            f"FP4 MLA attention decode requires page_size={FP4_MLA_TOKENS_PER_BLOCK}, "
            f"got {metadata.page_size}."
        )
    if q.ndim != 3 or q.shape[-1] != head_dim:
        raise ValueError(
            "FP4 MLA attention Q must have shape "
            f"[tokens, heads, {head_dim}], got {tuple(q.shape)}."
        )
    if not q.is_contiguous():
        raise ValueError("FP4 MLA attention Q must be contiguous.")

    num_queries = q.shape[0]
    if num_queries == 0:
        raise ValueError("FP4 MLA attention decode requires at least one query token.")
    num_gen_seqs = metadata.num_seqs - metadata.num_contexts
    query_len_per_seq = _get_linear_mtp_query_len_per_seq(
        metadata,
        num_queries=num_queries,
        num_gen_seqs=num_gen_seqs,
    )
    num_heads = q.shape[1]
    if output.shape[:2] != (num_queries, num_heads):
        raise ValueError("FP4 MLA attention output batch dimensions do not match.")

    if get_sm_version() != 107:
        raise RuntimeError(
            "K3's FP4 MLA decode requires Rubin SM107; "
            f"current architecture is SM{get_sm_version()}."
        )
    if not cutedsl_backend():
        raise RuntimeError(
            "K3's FP4 MLA decode requires a CuTe DSL build that provides the tcgen05 "
            "descriptor K-segment offset, and cuda-python."
        )
    if not 0 < num_heads <= 128 or kv_lora_rank != 512:
        raise ValueError(
            "K3's FP4 MLA decode requires 1-128 local heads and kv_lora_rank=512, got "
            f"num_heads={num_heads}, kv_lora_rank={kv_lora_rank}."
        )

    helix_spec_tokens_valid = bool(getattr(metadata, "_helix_spec_tokens_valid", False))
    helix_kv_bounds = None
    if softmax_stats_tensor is not None:
        if query_len_per_seq != 1 and not helix_spec_tokens_valid:
            raise NotImplementedError(
                "FP4 MLA multi-token Helix requires speculative per-token metadata."
            )
        expected_stats_shape = (num_queries, num_heads, 2)
        if (
            softmax_stats_tensor.shape != expected_stats_shape
            or softmax_stats_tensor.dtype != torch.float32
            or softmax_stats_tensor.device != q.device
            or not softmax_stats_tensor.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix requires contiguous same-device float32 softmax "
                f"stats with shape {expected_stats_shape}."
            )
        if helix_spec_tokens_valid:
            helix_kv_bounds = getattr(metadata, "helix_kv_bounds", None)
            if (
                not isinstance(helix_kv_bounds, torch.Tensor)
                or helix_kv_bounds.dtype != torch.int32
                or helix_kv_bounds.device != q.device
                or helix_kv_bounds.ndim != 1
                or helix_kv_bounds.numel() < num_queries
                or not helix_kv_bounds.is_contiguous()
            ):
                raise ValueError(
                    "FP4 MLA speculative Helix KV bounds must be a contiguous "
                    "same-device int32 tensor covering every query token."
                )
            helix_kv_bounds = helix_kv_bounds[:num_queries]
    state = metadata.fp4_mla_state
    v_sf_pool = getattr(state, "v_scale_pool", None)
    if v_sf_pool is None:
        raise RuntimeError(
            "FP4 MLA attention decode requires the auxiliary V scale pool to be allocated."
        )

    global_scale = _get_fp4_mla_global_scale(metadata, q.device)
    _validate_fp4_mla_attention_q_shape(head_dim, FP4_MLA_Q_RESIDUAL_DIM)
    if prequantized_q is None or prequantized_q_sf is None or q_batch_capacity is None:
        raise RuntimeError(
            "FP4 MLA decode requires Q prequantized by the fused generation cache update."
        )
    capacity = int(q_batch_capacity)
    # The generation cache update records the Q layout it produced.
    rows_per_seq = int(getattr(state, "prequantized_q_rows_per_seq", None) or 0)
    if rows_per_seq == 0:
        raise NotImplementedError(
            "K3's FP4 MLA decode reads the packed per-request Q layout; the generation cache "
            "update prepared the unpacked one (TRTLLM_FP4_MLA_PACKED_Q=0, or more than "
            "1024 rows per request)."
        )
    if rows_per_seq != packed_rows_per_seq(num_heads, query_len_per_seq):
        raise RuntimeError(
            "FP4 MLA prequantized Q layout does not match the packed decode "
            f"configuration (rows per request {rows_per_seq})."
        )
    active_q_rows = num_gen_seqs * rows_per_seq
    active_q_sf_bytes = _get_fp4_mla_swizzled_scale_size(active_q_rows, FP4_MLA_Q_LOGICAL_DIM)
    if (
        q.dtype != torch.bfloat16
        or capacity <= 0
        or num_queries > capacity
        or prequantized_q.ndim != 2
        or prequantized_q.shape[0] < active_q_rows
        or prequantized_q.shape[1] != FP4_MLA_Q_PACKED_DIM
        or prequantized_q_sf.numel() < active_q_sf_bytes
        or prequantized_q.dtype != torch.uint8
        or not prequantized_q.is_contiguous()
        or prequantized_q_sf.dtype != torch.float8_e4m3fn
        or not prequantized_q_sf.is_contiguous()
    ):
        raise ValueError("FP4 MLA prequantized Q does not satisfy the fused-Q contract.")
    q_fp4 = prequantized_q[:active_q_rows]
    q_sf = prequantized_q_sf[:active_q_sf_bytes]
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, q.device)

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=CUTEDSL_BACKEND,
    )
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=kv_lora_rank)[local_layer].view(
        torch.float8_e4m3fn
    )
    # The kv_lens runtime alias can lag at the decode anchor (seq_lens == 1) under CUDA graph /
    # one-engine MTP; the generation lengths recover the true total per sequence so the
    # per-query causal masking sees the full 1 + draft_len window.
    plan_geometry = packed_plan_geometry(
        metadata,
        num_heads=num_heads,
        query_len=query_len_per_seq,
        num_requests=num_gen_seqs,
        device=q.device,
        helix_kv_bounds=helix_kv_bounds,
    )
    if plan_geometry is None:
        raise RuntimeError("FP4 MLA packed decode requires generation cache pages.")
    kv_lens, _ = uniform_generation_lengths(
        metadata, num_queries, num_gen_seqs, plan_geometry=plan_geometry
    )
    materialize_page_table(metadata, kv_lens)
    src_page_ids = _fp4_mla_generation_page_ids(metadata, num_gen_seqs)
    if _max_generation_pages(metadata) == 0:
        raise RuntimeError("FP4 MLA attention decode requires generation cache pages.")

    module = _kernel_module()
    v_packed = _get_cutedsl_persistent_v_packed_cache(
        metadata,
        local_layer,
        kv_cache,
        v_head_dim=kv_lora_rank,
        page_size=metadata.page_size,
        block_v=module.SMEM_P4_V_N_PER_CTA,
    )
    core_v_packed = _get_fp4_mla_v_packed_pool_base(metadata)
    if core_v_packed is None:
        raise RuntimeError("Persistent FP4 MLA V packing requires a stable full-pool base.")
    get_v_page_offset = getattr(metadata.kv_cache_manager, "get_mla_v_packed_page_offset", None)
    v_page_offset = (
        int(get_v_page_offset(local_layer))
        if callable(get_v_page_offset)
        else local_layer * kv_cache.shape[0]
    )
    page_bytes = kv_lora_rank * (metadata.page_size // 2)
    if core_v_packed.data_ptr() + v_page_offset * page_bytes != v_packed.data_ptr():
        raise RuntimeError(
            "Persistent FP4 MLA V-packed layer view does not match its full-pool base and page "
            "offset."
        )
    v_sf_pool_base = _get_fp4_mla_v_scale_pool_base(metadata)
    if v_sf_pool_base is None:
        v_sf_pool_base = v_sf_pool.flatten(0, 1).view(torch.uint8)
        if v_sf_pool_base.data_ptr() != v_sf_pool.data_ptr():
            raise RuntimeError("Persistent FP4 MLA V-scale pool must flatten without a copy.")
    if (
        not isinstance(v_sf_pool_base, torch.Tensor)
        or v_sf_pool_base.dtype != torch.uint8
        or v_sf_pool_base.device != v_sf.device
        or v_sf_pool_base.ndim != 2
        or v_sf_pool_base.shape[1] != v_sf_pool.shape[-1]
        or not v_sf_pool_base.is_contiguous()
    ):
        raise RuntimeError(
            "Persistent FP4 MLA V-scale pool base must be a contiguous two-dimensional uint8 "
            "tensor with the configured page stride."
        )
    core_v_sf = v_sf_pool_base.view(torch.float8_e4m3fn)
    get_v_sf_page_offset = getattr(metadata.kv_cache_manager, "get_mla_v_scale_page_offset", None)
    v_sf_page_offset = (
        int(get_v_sf_page_offset(local_layer))
        if callable(get_v_sf_page_offset)
        else local_layer * kv_cache.shape[0]
    )
    if v_sf_page_offset != v_page_offset:
        raise RuntimeError(
            "Persistent FP4 MLA V-packed and V-scale pools require matching encoded layer offsets."
        )
    expected_v_sf_ptr = (
        core_v_sf.data_ptr() + v_sf_page_offset * v_sf_pool.stride(1) * v_sf_pool.element_size()
    )
    if expected_v_sf_ptr != v_sf.data_ptr():
        raise RuntimeError(
            "Persistent FP4 MLA V-scale layer view does not match its full-pool base and page "
            "offset."
        )
    _run_packed_decode(
        module,
        metadata,
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        core_v_packed,
        core_v_sf,
        global_scale,
        q_global_scale,
        src_page_ids,
        state.paged_kv_indptr_decode[: num_gen_seqs + 1],
        kv_lens,
        output,
        num_heads=num_heads,
        query_len=query_len_per_seq,
        num_requests=num_gen_seqs,
        rows_per_seq=rows_per_seq,
        sm_scale=float(sm_scale),
        v_page_offset=v_page_offset,
        helix_kv_bounds=helix_kv_bounds,
        softmax_stats_tensor=softmax_stats_tensor,
        plan_geometry=plan_geometry,
        metadata_launches=metadata_launches,
    )


def run_generation(
    attn: Any,
    q: torch.Tensor,
    metadata: Any,
    *,
    softmax_stats_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The absorbed generation attention of the FP4 MLA layer ``attn`` (its generation
    ``K3MlaAttnState``, :func:`applies`): what the served backend's ``forward`` returns for a
    generation-only call with the fused query ``q`` (``[tokens, heads * (latent + RoPE)]``), after
    the generation cache update.
    Returns ``[tokens, heads * kv_lora_rank]`` in ``q``'s dtype; ``softmax_stats_tensor`` receives
    the Helix statistics. Served: ``TrtllmAttention.forward`` -> ``Fp4MlaFmha.run_mla_generation``
    -> ``run_fp4_mla_attention_decode`` (CPU work and the BF16 output allocation besides the
    decode)."""
    kv_lora_rank = attn.kv_lora_rank or 0
    qk_rope_head_dim = attn.qk_rope_head_dim or 0
    num_heads = attn.num_heads
    num_tokens = q.shape[0]
    if metadata.num_generations <= 0:
        raise RuntimeError("FP4 MLA generation requires generation requests.")
    kv_cache_manager = metadata.kv_cache_manager
    if kv_cache_manager is None or kv_cache_manager.kv_factor != 1:
        raise RuntimeError("FP4 MLA requires a SELF-K-only KV cache manager.")
    if metadata.beam_width != 1:
        raise NotImplementedError("FP4 MLA does not support beam search.")
    state = metadata.fp4_mla_state
    if state is None or state.hp_pool is None:
        raise RuntimeError("FP4 MLA requires prepared high-precision KV state.")
    if (
        getattr(metadata, "helix_position_offsets", None) is not None
        and softmax_stats_tensor is None
    ):
        # Helix combines the per-rank partial outputs with these stats.
        raise RuntimeError("FP4 MLA Helix requires softmax_stats_tensor.")
    if not bool(getattr(state, "generation_cache_scattered", False)):
        raise RuntimeError(
            "FP4 MLA generation requires fused RoPE/Q quantization and "
            "cache/HP-pool update before attention."
        )
    state.generation_cache_scattered = False
    features = _features()
    logger.info_once(
        f"Kimi K3 FP4 MLA generation decode: k3_rubin packed kernel, {_SPLITS} KV splits, "
        f"wave_plan={_WAVE_PLAN}, lpt_plan={_LPT_PLAN}, page_plan={features.fp4_mla_page_plan}, "
        f"reduce_pdl={features.fp4_mla_reduce_pdl}, f16_partials={features.fp4_mla_f16_partials}, "
        f"decode_pdl={features.ull2_mla_decode_pdl}, helix={softmax_stats_tensor is not None}",
        key="k3_fp4_mla_generation_decode",
    )
    output = q.new_empty((num_tokens, num_heads * kv_lora_rank))
    sm_scale = 1.0 / (attn.q_scaling * ((attn.qk_nope_head_dim or 0) + qk_rope_head_dim) ** 0.5)
    try:
        run_decode(
            metadata,
            attn.layer_idx,
            attn.get_fp4_mla_local_layer_idx(metadata),
            q.view(num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim),
            output.view(num_tokens, num_heads, kv_lora_rank),
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            prequantized_q=state.prequantized_q,
            prequantized_q_sf=state.prequantized_q_sf,
            q_batch_capacity=state.q_batch_capacity,
            softmax_stats_tensor=softmax_stats_tensor,
        )
    finally:
        state.prequantized_q = None
        state.prequantized_q_sf = None
        state.q_batch_capacity = None
        state.prequantized_q_rows_per_seq = None
    return output
