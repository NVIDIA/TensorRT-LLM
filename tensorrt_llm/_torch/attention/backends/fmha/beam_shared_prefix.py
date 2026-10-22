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

"""Beam-search generation attention that reads the shared prompt once per request.

Every beam of a beam-search request attends to the same prompt KV, which the
KV-cache manager stores once (beam 0's blocks) and shares across beams. The
fallback decode kernel still walks that prompt once per (beam, head) pair; at
wide beams the shared prefix dominates the step.

This library splits a generation step into two parts and merges them with the
standard log-sum-exp recombination:

* **prefix**: one FlashInfer paged-prefill call per step where the request's
  ``beam_width`` queries form the Q tile and the KV is the shared prompt
  ``[0, prompt_len)`` read through beam 0's page table (no causal mask: every
  beam sees the whole prompt);
* **tail**: the few generated positions ``[prompt_len, kv_len - 1)``, gathered
  per beam through ``cache_indirection`` (a generated token lives in the block
  of the beam that produced it), plus the current token taken straight from
  the layer input. This part is tiny, so it runs as plain tensor math.

The current token's K/V is written into the beam's own block at
``kv_len - 1``, exactly as the fallback kernel does, so the two paths are
interchangeable step by step.

Scope: generation-only batches of a dense GQA/MHA model with an unquantized
bf16/fp16 paged KV cache (manager V1), RoPE already applied by the module
(``pos_embd_params`` not handed to the backend), no sliding window, sinks,
custom masks, or speculative decoding. Mixed context+generation batches fall
back to the regular library. The per-step planning (page table of the prompt,
tail length) is host work driven from ``TrtllmAttentionMetadata.prepare``,
which keeps the captured CUDA graph free of host synchronization.

The same prefix/tail split serves ``n``-sample requests (``beam_width == 1``),
whose children are independent requests that map the same leading prompt pages
when KV-cache block reuse lets them share the parent's blocks. Rows whose block
tables start with identical pages form a group; the group's common full pages
are read once for all of its queries and every row attends to its own remaining
pages (partial prompt block and generated tokens) with a paged decode call. The
two states are merged as above. This path needs a non-CUDA-graph batch and
declines (the regular kernel runs) when no group shares a prefix.

Opt in with ``TLLM_FMHA_LIBS=+beam_shared_prefix``.
``TLLM_BEAM_SHARED_PREFIX_MAX_TAIL`` (default 8) bounds how many generated
positions a beam may carry under CUDA graphs; longer generations are rejected
loudly at planning time rather than truncated.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

LIB_NAME = "beam_shared_prefix"
"""Registry name; also the key of the per-metadata plan state in
``TrtllmAttentionMetadata.fmha_plan_caches``."""

SAMPLING_KEY = LIB_NAME + "_sampling"
"""Key of the per-metadata plan state of the n-sample path in
``TrtllmAttentionMetadata.fmha_plan_caches``."""

MAX_TAIL_ENV = "TLLM_BEAM_SHARED_PREFIX_MAX_TAIL"
DEFAULT_MAX_TAIL = 8
BACKEND_ENV = "TLLM_BEAM_SHARED_PREFIX_BACKEND"
"""FlashInfer prefill backend for the prefix call. Default ``fa2``, the kernel
family the TRTLLM FlashInfer backend pins for paged prefill; ``fa3`` or
``auto`` select the Hopper kernels."""
DEFAULT_BACKEND = "fa2"
CHECK_ENV = "TLLM_BEAM_SHARED_PREFIX_CHECK"
"""Set to 1 to synchronize after every stage of the step and validate the
page table and the prefix attention against dense references. Debugging aid:
it serializes the GPU and must stay off for measurements."""

_LOG2E = 1.4426950408889634
# Split-KV partial results of the prefix call. FlashInfer recommends 128 MB
# and raises (rather than overflows) if a plan needs more.
_FLOAT_WORKSPACE_BYTES = 128 * 1024 * 1024
_float_workspace: Dict[torch.device, torch.Tensor] = {}


def _partial_out_dtype(dtype: torch.dtype) -> torch.dtype:
    """Output dtype of the FlashInfer calls whose results are merged afterwards.

    FlashInfer cannot write fp32 outputs. A bf16 partial output adds a second
    bf16 rounding on top of the one of the merged result, which leaves the
    step measurably less accurate than a single decode kernel (about 1.3x its
    RMS error against exact attention). fp16 keeps three more mantissa bits,
    so the merged result carries the error of one bf16 rounding. A normalized
    attention output is a convex combination of V rows, so it fits in fp16
    whenever every V entry does (|v| < 65504).
    """
    return torch.float16 if dtype == torch.bfloat16 else dtype


def _check_enabled() -> bool:
    return os.environ.get(CHECK_ENV, "0") == "1"


def _backend_from_env() -> str:
    value = os.environ.get(BACKEND_ENV, "").strip()
    return value or DEFAULT_BACKEND


def _get_float_workspace(device: torch.device) -> torch.Tensor:
    """Shared FlashInfer float workspace. Every wrapper runs on the same stream
    sequentially, so one buffer per device serves all plan states."""
    workspace = _float_workspace.get(device)
    if workspace is None:
        workspace = torch.empty(_FLOAT_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _float_workspace[device] = workspace
    return workspace


def _max_tail_from_env() -> int:
    value = os.environ.get(MAX_TAIL_ENV)
    if value is None or not value.strip():
        return DEFAULT_MAX_TAIL
    tail = int(value)
    if tail < 0:
        raise ValueError(f"{MAX_TAIL_ENV} must be >= 0, got {tail}.")
    return tail


class _PlanKey:
    """Per-layer shape family a FlashInfer wrapper is planned for."""

    __slots__ = ("num_heads", "num_kv_heads", "head_dim", "dtype", "sm_scale")

    def __init__(
        self, num_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, sm_scale: float
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.sm_scale = sm_scale

    def _tuple(self):
        return (self.num_heads, self.num_kv_heads, self.head_dim, self.dtype, self.sm_scale)

    def __eq__(self, other) -> bool:
        return isinstance(other, _PlanKey) and self._tuple() == other._tuple()

    def __hash__(self) -> int:
        return hash(self._tuple())


@dataclass
class _HostPlan:
    """Host-side description of one generation step, shared by all layers."""

    step: int
    num_requests: int
    beam_width: int
    qo_indptr: torch.Tensor
    """[num_requests + 1] int32 host, beam_width queries per request."""
    paged_kv_indptr: torch.Tensor
    """[num_requests + 1] int32 host."""
    paged_kv_indices: torch.Tensor
    """[paged_kv_indptr[-1]] int32 host, prompt pages of beam 0 per request."""
    paged_kv_last_page_len: torch.Tensor
    """[num_requests] int32 host."""
    tail_len: int
    """Generated positions (excluding the current token) the tail covers.
    Static under CUDA graphs, the batch maximum otherwise."""


@dataclass
class _StepIndices:
    """Device index tensors of one step, computed by the first layer that runs
    and reused by the others (they depend on the batch, not the layer)."""

    step: int
    write_page: torch.Tensor
    """[rows] int64: pool page holding the current token of each beam row."""
    write_slot: torch.Tensor
    """[rows] int64: slot of the current token inside that page."""
    tail_page: Optional[torch.Tensor]
    """[rows, tail_len] int64: pool page of each past generated position."""
    tail_slot: Optional[torch.Tensor]
    """[rows, tail_len] int64."""
    tail_valid: Optional[torch.Tensor]
    """[rows, tail_len] bool: position exists (< current position)."""
    prefix_empty: torch.Tensor
    """[rows] bool: the row has no cached token before the current one (only
    warmup/dummy rows). Its prefix call runs over a one-token placeholder and
    is masked out of the merge."""
    captured: bool
    """Computed while a CUDA graph was being captured. A graph must record the
    index computation itself: tensors computed by an eager warmup iteration
    live outside the graph's memory pool and are freed by the next plan, so a
    replay reading them would read recycled memory."""


@dataclass
class _BeamStepState:
    """Per-metadata plan state kept in ``metadata.fmha_plan_caches[LIB_NAME]``.

    A metadata object belongs to exactly one execution mode (eager, or one
    CUDA graph batch size), so the FlashInfer wrappers it owns are created with
    the matching ``use_cuda_graph`` setting and, under graphs, a fixed batch.
    """

    is_cuda_graph: bool
    pool_index: int
    page_stride: int
    """Divisor turning a K entry of ``kv_cache_block_offsets`` into a pool page
    index: ``layers_in_pool * kv_factor`` for the standard pool layout."""
    max_tail: int
    num_pages: int
    """Pages in the primary pool; bounds every page index handed to a kernel."""
    plan: Optional[_HostPlan] = None
    wrappers: Dict[_PlanKey, Any] = field(default_factory=dict)
    planned_step: Dict[_PlanKey, int] = field(default_factory=dict)
    indices: Optional[_StepIndices] = None
    step_counter: int = 0


def _pool_layout(metadata: "TrtllmAttentionMetadata", local_layer_idx: int) -> tuple[int, int]:
    """(pool index, page stride) of a layer's primary pool.

    Mirrors the offset encoding of ``WindowBlockManager::setOffsets`` for the
    standard (block-major) pool layout: the K entry of a block is
    ``page * layers_in_pool * kv_factor``.
    """
    pool_mapping = metadata.host_kv_cache_pool_mapping
    if pool_mapping is None or pool_mapping.ndim != 2 or pool_mapping.shape[1] < 2:
        raise RuntimeError("KV-cache pool mapping must have shape [num_layers, >=2].")
    pool_index = int(pool_mapping[local_layer_idx, 0])
    layers_in_pool = int((pool_mapping[:, 0] == pool_index).sum())
    kv_factor = int(metadata.kv_cache_manager.kv_factor)
    return pool_index, layers_in_pool * kv_factor


def _is_v2_manager(kv_cache_manager) -> bool:
    return hasattr(kv_cache_manager, "index_mapper")


def _page_layout(
    metadata: "TrtllmAttentionMetadata", layer_idx: int, local_layer_idx: int
) -> tuple[int, int, int]:
    """(pool index, scale, divisor) of a layer: the page of the layer's pool view
    (``get_buffers``) that a host block-table entry names is ``entry * scale // divisor``.

    V1 entries are K offsets (``page * layers_in_pool * kv_factor``). V2 entries
    are base page indices that the device-side copy multiplies by the pool's
    index scale, and ``get_buffers`` folds K and V into one page.
    """
    manager = metadata.kv_cache_manager
    if _is_v2_manager(manager):
        pool_index = int(manager.layer_to_pool_mapping_dict[manager.layer_offsets[layer_idx]])
        return pool_index, manager.get_layer_page_index_scale(layer_idx), int(manager.kv_factor)
    pool_index, page_stride = _pool_layout(metadata, local_layer_idx)
    return pool_index, 1, page_stride


def _table_rows(metadata: "TrtllmAttentionMetadata", num_rows: int) -> torch.Tensor:
    """Row of the host block table of each generation row (context-free batch).

    V1 fills the table in batch order. V2 keeps it indexed by the request's
    slot in its index mapper.
    """
    manager = metadata.kv_cache_manager
    if not _is_v2_manager(manager):
        return torch.arange(num_rows, dtype=torch.int64)
    request_ids = metadata.request_ids
    if request_ids is None or len(request_ids) < num_rows:
        raise RuntimeError(f"{LIB_NAME}: the batch's request ids are required with KV manager V2.")
    beam_stride = int(manager.max_beam_width)
    return torch.tensor(
        [manager.index_mapper.get_index(request_ids[i]) * beam_stride for i in range(num_rows)],
        dtype=torch.int64,
    )


def _kv_manager_supported(kv_cache_manager) -> bool:
    """Only the V1 paged manager exposes the per-layer pool view and the
    host block-offset table this library reads."""
    if kv_cache_manager is None:
        return False
    for attr in ("get_buffers", "host_kv_cache_block_offsets", "kv_factor", "tokens_per_block"):
        if not hasattr(kv_cache_manager, attr):
            return False
    if getattr(kv_cache_manager, "kv_factor", 0) != 2:
        return False
    # The layer-first pool layout (recurrent-state caches) encodes offsets
    # differently; those managers never run dense beam search anyway.
    if getattr(kv_cache_manager, "is_linear_attention", False):
        return False
    return True


def metadata_eligible(metadata: "TrtllmAttentionMetadata") -> bool:
    """Batch-level preconditions shared by planning and library selection."""
    return (
        IS_FLASHINFER_AVAILABLE
        and metadata.beam_width > 1
        and metadata.num_generations > 0
        and metadata.num_contexts == 0
        and not metadata.is_cross
        and not metadata.enable_helix
        and not metadata.is_spec_decoding_enabled
        and not metadata.use_spec_decoding
        and metadata.cache_indirection is not None
        and metadata.kv_cache_block_offsets is not None
        and metadata.kv_cache_params is not None
        and metadata.kv_cache_params.use_cache
        and metadata.kv_cache_params.num_extra_kv_tokens == 0
        and _kv_manager_supported(metadata.kv_cache_manager)
        and not _is_v2_manager(metadata.kv_cache_manager)
        and metadata.num_generations % metadata.beam_width == 0
    )


def has_active_plan(metadata: "TrtllmAttentionMetadata") -> bool:
    """Whether ``prepare`` planned the current batch for this library.

    The FMHA selection cache keys on batch shape alone, so this state is part
    of its key: without it, a selection made on a step that had no plan would
    be reused on steps that do.
    """
    for key in (LIB_NAME, SAMPLING_KEY):
        state = metadata.fmha_plan_caches.get(key)
        if state is not None and state.plan is not None:
            return True
    return False


def plan_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Per-step host planning, called from ``TrtllmAttentionMetadata.prepare``.

    Builds the shared-prompt page table of every request from the host block
    offsets the manager just filled, decides the tail length, and re-plans the
    FlashInfer wrappers the layers registered on earlier steps. Runs before any
    forward of the step, so under CUDA graphs it precedes capture and replay.
    """
    sampling_state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    if sampling_state is not None:
        sampling_state.plan = None
    if sampling_metadata_eligible(metadata):
        _plan_sampling_step(metadata)
        return
    if not metadata_eligible(metadata):
        return
    state = metadata.fmha_plan_caches.get(LIB_NAME)
    if state is None:
        # The layer-independent layout facts come from layer 0's pool; a
        # layer of a different pool creates its own wrapper key below.
        pool_index, page_stride = _pool_layout(metadata, 0)
        kv_cache_manager = metadata.kv_cache_manager
        first_layer = min(kv_cache_manager.layer_offsets)
        pool = kv_cache_manager.get_buffers(first_layer, kv_layout="HND")
        state = _BeamStepState(
            is_cuda_graph=metadata.is_cuda_graph,
            pool_index=pool_index,
            page_stride=page_stride,
            max_tail=_max_tail_from_env(),
            num_pages=int(pool.size(0)),
        )
        metadata.fmha_plan_caches[LIB_NAME] = state
    state.step_counter += 1

    beam_width = metadata.beam_width
    rows = metadata.num_generations
    num_requests = rows // beam_width
    num_ctx = metadata.num_contexts
    tokens_per_block = int(metadata.tokens_per_block)

    kv_lens = metadata.kv_lens_runtime[num_ctx : num_ctx + rows].to(torch.int64)
    prompt_lens = metadata.prompt_lens_cpu_runtime[num_ctx : num_ctx + rows].to(torch.int64)
    # The prefix ends at the current token at the latest: warmup/dummy
    # generation rows can carry no generated token yet, in which case the
    # current position is the prompt's last token and the tail is empty.
    prompt_lens = torch.minimum(prompt_lens, kv_lens - 1)
    # Generated positions before the current token; uniform within a request.
    tail_lens = kv_lens - prompt_lens - 1
    max_tail = int(tail_lens.max().item()) if rows > 0 else 0
    # A row with nothing cached before the current token (kv_len == 1) would
    # give FlashInfer a zero-length KV segment. Plan a one-token placeholder
    # instead; _step_indices flags the row and the merge drops its prefix.
    prompt_lens = prompt_lens.clamp(min=1)
    if state.is_cuda_graph:
        if max_tail > state.max_tail:
            raise RuntimeError(
                f"beam_shared_prefix: a beam carries {max_tail} generated positions but "
                f"CUDA graphs were captured for at most {state.max_tail}. Raise "
                f"{MAX_TAIL_ENV} to the maximum number of generated tokens, or remove "
                f"{LIB_NAME} from TLLM_FMHA_LIBS."
            )
        tail_len = state.max_tail
    else:
        tail_len = max_tail

    request_prompt_lens = prompt_lens.view(num_requests, beam_width)[:, 0]
    ctx_blocks = (request_prompt_lens + tokens_per_block - 1) // tokens_per_block
    last_page_len = request_prompt_lens - (ctx_blocks - 1) * tokens_per_block
    host_offsets = metadata.host_kv_cache_block_offsets
    assert host_offsets is not None
    offsets_k = host_offsets[state.pool_index, :, 0, :]
    pages = []
    for request_idx in range(num_requests):
        row = num_ctx + request_idx * beam_width
        n_blocks = int(ctx_blocks[request_idx])
        pages.append(offsets_k[row, :n_blocks].to(torch.int64) // state.page_stride)
    paged_kv_indices = (
        torch.cat(pages).clamp(0, state.num_pages - 1).to(torch.int32)
        if pages
        else torch.zeros((0,), dtype=torch.int32)
    )
    paged_kv_indptr = torch.zeros((num_requests + 1,), dtype=torch.int32)
    paged_kv_indptr[1:] = torch.cumsum(ctx_blocks, dim=0).to(torch.int32)
    qo_indptr = (torch.arange(num_requests + 1, dtype=torch.int32) * beam_width).to(torch.int32)

    state.plan = _HostPlan(
        step=state.step_counter,
        num_requests=num_requests,
        beam_width=beam_width,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=last_page_len.to(torch.int32),
        tail_len=tail_len,
    )
    state.indices = None
    for key, wrapper in state.wrappers.items():
        _plan_wrapper(state, key, wrapper, metadata)


def _plan_wrapper(
    state: _BeamStepState, key: _PlanKey, wrapper: Any, metadata: "TrtllmAttentionMetadata"
) -> None:
    plan = state.plan
    assert plan is not None
    wrapper.plan(
        plan.qo_indptr,
        plan.paged_kv_indptr,
        plan.paged_kv_indices,
        plan.paged_kv_last_page_len,
        key.num_heads,
        key.num_kv_heads,
        key.head_dim,
        int(metadata.tokens_per_block),
        causal=False,
        sm_scale=key.sm_scale,
        q_data_type=key.dtype,
        kv_data_type=key.dtype,
        o_data_type=key.dtype,
    )
    state.planned_step[key] = plan.step


def _create_wrapper(
    state: _BeamStepState, key: _PlanKey, metadata: "TrtllmAttentionMetadata"
) -> Any:
    device = metadata.kv_cache_block_offsets.device
    kwargs: Dict[str, Any] = {}
    if state.is_cuda_graph:
        plan = state.plan
        assert plan is not None
        max_blocks = metadata.kv_cache_manager.max_blocks_per_seq
        kwargs = dict(
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros((plan.num_requests + 1,), dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.zeros(
                (plan.num_requests + 1,), dtype=torch.int32, device=device
            ),
            paged_kv_indices_buf=torch.zeros(
                (max(1, plan.num_requests * max_blocks),), dtype=torch.int32, device=device
            ),
            paged_kv_last_page_len_buf=torch.zeros(
                (plan.num_requests,), dtype=torch.int32, device=device
            ),
        )
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _get_float_workspace(device), "HND", backend=_backend_from_env(), **kwargs
    )


def _step_indices(
    state: _BeamStepState, metadata: "TrtllmAttentionMetadata", num_pages: int
) -> _StepIndices:
    """Device-side page/slot indices of the current token and the tail.

    Everything derives from device tensors ``prepare`` refreshes each step
    (``kv_lens_cuda_runtime``, ``prompt_lens_cuda_runtime``,
    ``kv_cache_block_offsets``, ``cache_indirection``), so a CUDA graph that
    captured this computation recomputes it correctly on replay.
    """
    plan = state.plan
    assert plan is not None
    capturing = torch.cuda.is_current_stream_capturing()
    cached = state.indices
    if cached is not None and cached.step == plan.step and (cached.captured or not capturing):
        return cached

    beam_width = plan.beam_width
    rows = plan.num_requests * beam_width
    num_ctx = metadata.num_contexts
    tokens_per_block = int(metadata.tokens_per_block)
    offsets_k = metadata.kv_cache_block_offsets[state.pool_index, :, 0, :]
    max_blocks = offsets_k.size(-1)
    device = offsets_k.device

    kv_len = metadata.kv_lens_cuda_runtime[num_ctx : num_ctx + rows].to(torch.int64)
    prompt_len = metadata.prompt_lens_cuda_runtime[num_ctx : num_ctx + rows].to(torch.int64)
    row_ids = torch.arange(rows, device=device, dtype=torch.int64)
    cache_rows = row_ids + num_ctx
    current = (kv_len - 1).clamp(min=0)
    # Same clamp as the host plan: the prefix never extends past the current token.
    prompt_len = torch.minimum(prompt_len, current)
    prefix_empty = prompt_len < 1
    write_page = (
        offsets_k[cache_rows, (current // tokens_per_block).clamp(max=max_blocks - 1)].to(
            torch.int64
        )
        // state.page_stride
    ).clamp(0, num_pages - 1)
    write_slot = current % tokens_per_block

    tail_page = tail_slot = tail_valid = None
    if plan.tail_len > 0:
        cache_indirection = metadata.cache_indirection
        assert cache_indirection is not None
        max_positions = cache_indirection.size(-1)
        request_ids = row_ids // beam_width
        beam_ids = row_ids % beam_width
        positions = prompt_len.view(-1, 1) + torch.arange(
            plan.tail_len, device=device, dtype=torch.int64
        ).view(1, -1)
        tail_valid = positions < current.view(-1, 1)
        positions = positions.clamp(min=0, max=max_positions - 1)
        # The beam whose block holds this position, per the sampler's
        # indirection table (rows are the generation requests of the batch).
        src_beam = cache_indirection[request_ids.view(-1, 1), beam_ids.view(-1, 1), positions]
        src_beam = src_beam.to(torch.int64).clamp(0, beam_width - 1)
        src_rows = num_ctx + request_ids.view(-1, 1) * beam_width + src_beam
        blocks = (positions // tokens_per_block).clamp(max=max_blocks - 1)
        tail_page = (offsets_k[src_rows, blocks].to(torch.int64) // state.page_stride).clamp(
            0, num_pages - 1
        )
        tail_slot = positions % tokens_per_block

    state.indices = _StepIndices(
        step=plan.step,
        write_page=write_page,
        write_slot=write_slot,
        tail_page=tail_page,
        tail_slot=tail_slot,
        tail_valid=tail_valid,
        prefix_empty=prefix_empty,
        captured=capturing,
    )
    return state.indices


def merge_prefix_and_tail(
    q: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    tail_valid: Optional[torch.Tensor],
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    *,
    num_kv_heads: int,
    sm_scale: float,
    prefix_empty: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attend ``q`` to the tail K/V and merge with the prefix partial state.

    Args:
        q: [rows, num_heads, head_dim] queries (RoPE applied).
        k_tail, v_tail: [rows, tail, num_kv_heads, head_dim] tail keys/values,
            the current token included (last position).
        tail_valid: [rows, tail] bool, or None when every position is valid.
        prefix_out: [rows, num_heads, head_dim] normalized prefix attention.
        prefix_lse: [rows, num_heads] fp32 log-sum-exp of the prefix scores
            **in base 2**, FlashInfer's convention
            (``lse = log2(sum_j 2^(s_j * log2(e)))`` with ``s_j`` the scaled
            scores).
        prefix_empty: [rows] bool, or None. Rows whose prefix state is a
            placeholder and must not contribute.

    Returns:
        [rows, num_heads, head_dim] attention over prefix + tail, in fp32.
    """
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    if prefix_empty is not None:
        prefix_lse = prefix_lse.masked_fill(prefix_empty.view(rows, 1), float("-inf"))
        prefix_out = prefix_out.masked_fill(prefix_empty.view(rows, 1, 1), 0.0)
    q_grouped = q.view(rows, num_kv_heads, group, head_dim).float()
    # Scores in the log2 domain so they combine directly with the prefix LSE.
    scores = torch.einsum("rkgd,rtkd->rkgt", q_grouped, k_tail.float()) * (sm_scale * _LOG2E)
    if tail_valid is not None:
        scores = scores.masked_fill(~tail_valid.view(rows, 1, 1, -1), float("-inf"))
    tail_max = scores.amax(dim=-1, keepdim=True)
    probs = torch.exp2(scores - tail_max)
    tail_sum = probs.sum(dim=-1)
    tail_out = torch.einsum("rkgt,rtkd->rkgd", probs, v_tail.float())
    tail_max = tail_max.squeeze(-1)

    prefix_lse = prefix_lse.view(rows, num_kv_heads, group).float()
    merged_max = torch.maximum(prefix_lse, tail_max)
    w_prefix = torch.exp2(prefix_lse - merged_max)
    w_tail = torch.exp2(tail_max - merged_max)
    numerator = prefix_out.view(rows, num_kv_heads, group, head_dim).float() * w_prefix.unsqueeze(
        -1
    ) + tail_out * w_tail.unsqueeze(-1)
    denominator = (w_prefix + tail_sum * w_tail).unsqueeze(-1)
    return (numerator / denominator).view(rows, num_heads, head_dim)


# --------------------------------------------------------------------------- #
# n-sample requests (beam_width == 1)
# --------------------------------------------------------------------------- #
@dataclass
class _SamplingPlan:
    """Host-side description of one n-sample generation step, shared by all layers.

    Rows are generation rows in batch order. Prefix tensors describe groups in
    ``perm`` order (rows of a group are contiguous there); tail tensors and the
    KV write indices are in batch order.
    """

    step: int
    num_rows: int
    num_groups: int
    shared_tokens: int
    """Prompt tokens read once per group instead of once per row, summed over rows."""
    perm: Optional[torch.Tensor]
    """[rows] int64: batch row of each grouped position; None when the batch is already grouped."""
    qo_indptr: torch.Tensor
    prefix_indptr: torch.Tensor
    prefix_indices: torch.Tensor
    prefix_last_page_len: torch.Tensor
    prefix_empty: torch.Tensor
    """[rows] bool, grouped order: the row's group has no shared page and its
    prefix call runs over a one-token placeholder that the merge drops."""
    tail_indptr: torch.Tensor
    tail_indices: torch.Tensor
    tail_last_page_len: torch.Tensor
    write_page: torch.Tensor
    """[rows] int64, batch order: pool page of the current token."""
    write_slot: torch.Tensor
    check_pages: Optional[torch.Tensor] = None
    """[rows, max_blocks] int64 full page tables; only kept in check mode."""
    check_kv_lens: Optional[torch.Tensor] = None


@dataclass
class _SamplingDevice:
    """Device copies of the step's index tensors, made by the first layer."""

    step: int
    perm: Optional[torch.Tensor]
    inv_perm: Optional[torch.Tensor]
    prefix_empty: torch.Tensor
    write_page: torch.Tensor
    write_slot: torch.Tensor


@dataclass
class _SamplingState:
    pool_index: int
    page_scale: int
    page_div: int
    num_pages: int
    plan: Optional[_SamplingPlan] = None
    prefix_wrappers: Dict[_PlanKey, Any] = field(default_factory=dict)
    tail_wrappers: Dict[_PlanKey, Any] = field(default_factory=dict)
    planned_step: Dict[_PlanKey, int] = field(default_factory=dict)
    device: Optional[_SamplingDevice] = None
    step_counter: int = 0
    logged: bool = False
    ran: bool = False


def sampling_metadata_eligible(metadata: "TrtllmAttentionMetadata") -> bool:
    """Batch-level preconditions of the n-sample path."""
    return (
        IS_FLASHINFER_AVAILABLE
        and metadata.beam_width == 1
        and metadata.num_generations > 1
        and metadata.num_contexts == 0
        and not metadata.is_cuda_graph
        and not metadata.is_cross
        and not metadata.enable_helix
        and not metadata.is_spec_decoding_enabled
        and not metadata.use_spec_decoding
        and metadata.kv_cache_block_offsets is not None
        and metadata.kv_cache_params is not None
        and metadata.kv_cache_params.use_cache
        and metadata.kv_cache_params.num_extra_kv_tokens == 0
        and _kv_manager_supported(metadata.kv_cache_manager)
    )


def _plan_sampling_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Group the batch's rows by shared leading KV pages and build the page tables.

    Two rows share a prefix when their block tables start with the same pool
    pages. A group's shared pages are the longest common run of full pages that
    lie entirely before every member's current token. Everything after that
    run, including the current token's block, stays private to the row.
    """
    state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    if state is None:
        kv_cache_manager = metadata.kv_cache_manager
        first_layer = min(kv_cache_manager.layer_offsets)
        pool_index, page_scale, page_div = _page_layout(metadata, first_layer, 0)
        pool = kv_cache_manager.get_buffers(first_layer, kv_layout="HND")
        state = _SamplingState(
            pool_index=pool_index,
            page_scale=page_scale,
            page_div=page_div,
            num_pages=int(pool.size(0)),
        )
        metadata.fmha_plan_caches[SAMPLING_KEY] = state
    state.step_counter += 1

    rows = metadata.num_generations
    tokens_per_block = int(metadata.tokens_per_block)
    host_offsets = metadata.host_kv_cache_block_offsets
    assert host_offsets is not None
    table_rows = _table_rows(metadata, rows)
    pages_all = (
        host_offsets[state.pool_index, table_rows, 0, :].to(torch.int64)
        * state.page_scale
        // state.page_div
    ).clamp(0, state.num_pages - 1)
    kv_lens = metadata.kv_lens_runtime[:rows].to(torch.int64)
    num_blocks = (kv_lens + tokens_per_block - 1) // tokens_per_block
    # Full pages strictly before the current token (position kv_len - 1).
    private_from = (kv_lens - 1) // tokens_per_block

    by_first_page: Dict[int, list] = {}
    for row, page in enumerate(pages_all[:, 0].tolist()):
        by_first_page.setdefault(page, []).append(row)

    groups: list = []
    shared_blocks: list = []
    for members in by_first_page.values():
        shared = 0
        if len(members) > 1:
            member_idx = torch.tensor(members, dtype=torch.int64)
            cap = int(private_from[member_idx].min())
            if cap > 0:
                same = (pages_all[member_idx, :cap] == pages_all[members[0], :cap]).all(dim=0)
                shared = int(torch.cumprod(same.to(torch.int64), dim=0).sum())
        groups.append(members)
        shared_blocks.append(shared)
    if not any(blocks > 0 for blocks in shared_blocks):
        return

    order = [row for members in groups for row in members]
    perm = None if order == list(range(rows)) else torch.tensor(order, dtype=torch.int64)
    qo_indptr = torch.zeros(len(groups) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(
        torch.tensor([len(members) for members in groups], dtype=torch.int64), dim=0
    ).to(torch.int32)

    prefix_pages = []
    prefix_lens = []
    prefix_last = []
    prefix_empty = []
    row_shared = torch.zeros(rows, dtype=torch.int64)
    shared_tokens = 0
    for members, shared in zip(groups, shared_blocks):
        if shared > 0:
            prefix_pages.append(pages_all[members[0], :shared])
            prefix_last.append(tokens_per_block)
            row_shared[torch.tensor(members, dtype=torch.int64)] = shared
            shared_tokens += shared * tokens_per_block * len(members)
        else:
            prefix_pages.append(pages_all[members[0], :1])
            prefix_last.append(1)
        prefix_lens.append(int(prefix_pages[-1].numel()))
        prefix_empty.extend([shared == 0] * len(members))
    prefix_indptr = torch.zeros(len(groups) + 1, dtype=torch.int32)
    prefix_indptr[1:] = torch.cumsum(torch.tensor(prefix_lens, dtype=torch.int64), dim=0).to(
        torch.int32
    )

    block_ids = torch.arange(pages_all.size(1), dtype=torch.int64).view(1, -1)
    in_tail = (block_ids >= row_shared.view(-1, 1)) & (block_ids < num_blocks.view(-1, 1))
    tail_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    tail_indptr[1:] = torch.cumsum((num_blocks - row_shared), dim=0).to(torch.int32)
    current = kv_lens - 1
    write_page = pages_all[torch.arange(rows), current // tokens_per_block]

    state.plan = _SamplingPlan(
        step=state.step_counter,
        num_rows=rows,
        num_groups=len(groups),
        shared_tokens=shared_tokens,
        perm=perm,
        qo_indptr=qo_indptr,
        prefix_indptr=prefix_indptr,
        prefix_indices=torch.cat(prefix_pages).to(torch.int32),
        prefix_last_page_len=torch.tensor(prefix_last, dtype=torch.int32),
        prefix_empty=torch.tensor(prefix_empty, dtype=torch.bool),
        tail_indptr=tail_indptr,
        tail_indices=pages_all[in_tail].to(torch.int32),
        tail_last_page_len=(kv_lens - (num_blocks - 1) * tokens_per_block).to(torch.int32),
        write_page=write_page,
        write_slot=current % tokens_per_block,
        check_pages=pages_all if _check_enabled() else None,
        check_kv_lens=kv_lens if _check_enabled() else None,
    )
    state.device = None
    if not state.logged:
        state.logged = True
        logger.info(
            f"{LIB_NAME}: n-sample step with {rows} rows in {len(groups)} groups, "
            f"{shared_tokens} prompt tokens read once per group"
        )


def _sampling_device(state: _SamplingState, device: torch.device) -> _SamplingDevice:
    plan = state.plan
    assert plan is not None
    cached = state.device
    if cached is not None and cached.step == plan.step:
        return cached
    perm = inv_perm = None
    if plan.perm is not None:
        perm = plan.perm.to(device)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=device, dtype=perm.dtype)
    state.device = _SamplingDevice(
        step=plan.step,
        perm=perm,
        inv_perm=inv_perm,
        prefix_empty=plan.prefix_empty.to(device),
        write_page=plan.write_page.to(device),
        write_slot=plan.write_slot.to(device),
    )
    return state.device


def _sampling_wrappers(
    state: _SamplingState, key: _PlanKey, metadata: "TrtllmAttentionMetadata", device: torch.device
) -> tuple[Any, Any]:
    """FlashInfer prefix (prefill) and tail (decode) wrappers of a layer shape,
    planned once per step."""
    plan = state.plan
    assert plan is not None
    prefix = state.prefix_wrappers.get(key)
    tail = state.tail_wrappers.get(key)
    if prefix is None:
        workspace = _get_float_workspace(device)
        prefix = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "HND", backend=_backend_from_env()
        )
        tail = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "HND", use_tensor_cores=key.num_heads != key.num_kv_heads
        )
        state.prefix_wrappers[key] = prefix
        state.tail_wrappers[key] = tail
    if state.planned_step.get(key) != plan.step:
        tokens_per_block = int(metadata.tokens_per_block)
        prefix.plan(
            plan.qo_indptr,
            plan.prefix_indptr,
            plan.prefix_indices,
            plan.prefix_last_page_len,
            key.num_heads,
            key.num_kv_heads,
            key.head_dim,
            tokens_per_block,
            causal=False,
            sm_scale=key.sm_scale,
            q_data_type=key.dtype,
            kv_data_type=key.dtype,
            o_data_type=_partial_out_dtype(key.dtype),
        )
        tail.plan(
            plan.tail_indptr,
            plan.tail_indices,
            plan.tail_last_page_len,
            key.num_heads,
            key.num_kv_heads,
            key.head_dim,
            tokens_per_block,
            q_data_type=key.dtype,
            kv_data_type=key.dtype,
            o_data_type=_partial_out_dtype(key.dtype),
            sm_scale=key.sm_scale,
        )
        state.planned_step[key] = plan.step
    return prefix, tail


def merge_states(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    a_empty: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine two normalized attention states over disjoint key sets.

    ``out_*`` is [rows, heads, head_dim], ``lse_*`` is [rows, heads] in base 2.
    Rows flagged in ``a_empty`` ([rows] bool) take state ``b`` alone. Returns fp32.
    """
    if a_empty is not None:
        lse_a = lse_a.masked_fill(a_empty.view(-1, 1), float("-inf"))
        out_a = out_a.masked_fill(a_empty.view(-1, 1, 1), 0.0)
    lse_a = lse_a.float()
    lse_b = lse_b.float()
    peak = torch.maximum(lse_a, lse_b)
    w_a = torch.exp2(lse_a - peak)
    w_b = torch.exp2(lse_b - peak)
    numerator = out_a.float() * w_a.unsqueeze(-1) + out_b.float() * w_b.unsqueeze(-1)
    return numerator / (w_a + w_b).unsqueeze(-1)


_CHECK_STATS = {"calls": 0, "worst": 0.0, "floor": 0.0}
_CHECK_ROWS = 4
_CHECK_TOL = 5e-2
"""Largest per-head error, relative to that head's largest output magnitude. bf16
rounding of the two partial outputs stays near 1e-2; a wrong page or position
gives tens of percent."""


def _check_sampling_output(
    plan: _SamplingPlan,
    pool: torch.Tensor,
    q: torch.Tensor,
    merged: torch.Tensor,
    *,
    num_kv_heads: int,
    sm_scale: float,
    tokens_per_block: int,
    layer_idx: int,
) -> None:
    """Compare a few rows of the merged output with dense attention over the
    row's whole page table (shared and private pages alike)."""
    assert plan.check_pages is not None and plan.check_kv_lens is not None
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    picks = sorted({0, rows // 3, (2 * rows) // 3, rows - 1})[:_CHECK_ROWS]
    worst = 0.0
    for row in picks:
        kv_len = int(plan.check_kv_lens[row])
        num_blocks = (kv_len + tokens_per_block - 1) // tokens_per_block
        pages = plan.check_pages[row, :num_blocks].to(pool.device)
        kv = pool[pages].permute(0, 3, 1, 2, 4).reshape(-1, 2, num_kv_heads, head_dim)[:kv_len]
        q_row = q[row].float().view(num_kv_heads, group, head_dim)
        scores = torch.einsum("kgd,tkd->kgt", q_row, kv[:, 0].float()) * sm_scale
        expected = torch.einsum(
            "kgt,tkd->kgd", torch.softmax(scores, dim=-1), kv[:, 1].float()
        ).reshape(num_heads, head_dim)
        scale = expected.abs().amax(dim=-1).clamp(min=1e-3)
        err = float(((merged[row] - expected).abs().amax(dim=-1) / scale).max())
        worst = max(worst, err)
        # Noise floor of a plain bf16 attention (torch SDPA) against the same reference.
        k_rep = kv[:, 0].transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
        v_rep = kv[:, 1].transpose(0, 1).repeat_interleave(group, dim=0).unsqueeze(0)
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            q[row].view(1, num_heads, 1, head_dim).to(kv.dtype), k_rep, v_rep, scale=sm_scale
        ).view(num_heads, head_dim)
        floor = float(((sdpa.float() - expected).abs().amax(dim=-1) / scale).max())
        _CHECK_STATS["floor"] = max(_CHECK_STATS["floor"], floor)
    _CHECK_STATS["calls"] += 1
    _CHECK_STATS["worst"] = max(_CHECK_STATS["worst"], worst)
    if worst > _CHECK_TOL:
        raise RuntimeError(
            f"{LIB_NAME} check: n-sample attention disagrees with the dense reference at layer "
            f"{layer_idx}: max relative diff {worst:.4f} (rows {picks} of {rows})"
        )
    if _CHECK_STATS["calls"] % 500 == 1:
        logger.info(
            f"{LIB_NAME} check: {_CHECK_STATS['calls']} layer calls checked, worst relative diff "
            f"so far {_CHECK_STATS['worst']:.2e} (plain bf16 SDPA vs the same reference: "
            f"{_CHECK_STATS['floor']:.2e})"
        )


class BeamSharedPrefixFmha(PhasedFmha):
    """Generation-phase library for wide-beam decoding; see the module docstring."""

    supports_workspace_reclamation = False

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._pool_views: Dict[int, torch.Tensor] = {}
        # (pool index, page stride) of this layer; fixed for the manager lifetime.
        self._layout: Optional[tuple[int, int]] = None

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if not IS_FLASHINFER_AVAILABLE:
            return False
        if (
            getattr(attn, "is_mla_enable", False)
            or getattr(attn, "sparse_params", None) is not None
        ):
            return False
        if getattr(attn, "predicted_tokens_per_seq", 1) != 1:
            return False
        if getattr(attn, "attention_chunk_size", None):
            return False
        head_dim = getattr(attn, "head_dim", 0)
        num_heads = getattr(attn, "num_heads", 0)
        num_kv_heads = getattr(attn, "num_kv_heads", 0) or 0
        if head_dim not in (64, 128, 256):
            return False
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return False
        if QuantMode(getattr(attn, "quant_mode", 0)).has_kv_cache_quant():
            return False
        # RoPE must already be applied by the module: this library never
        # rotates Q/K itself.
        if PositionEmbeddingType(getattr(attn, "position_embedding_type", 0)).is_rope():
            return False
        return True

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase == FmhaPhase.CONTEXT:
            return False
        sampling = sampling_metadata_eligible(metadata)
        if not (sampling or metadata_eligible(metadata)):
            return False
        if forward_args.attention_input_type == AttentionInputType.context_only:
            return False
        if not (forward_args.is_fused_qkv and k is None and v is None):
            return False
        if not forward_args.update_kv_cache:
            return False
        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False
        if forward_args.attention_mask_data is not None or forward_args.attention_sinks is not None:
            return False
        if forward_args.relative_attention_bias is not None:
            return False
        if forward_args.out_scale is not None or forward_args.output_sf is not None:
            return False
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        window = forward_args.attention_window_size
        if (
            window is not None
            and metadata.max_seq_len is not None
            and window < metadata.max_seq_len
        ):
            return False
        # One query token per beam row.
        if q.shape[0] - metadata.num_ctx_tokens != metadata.num_generations:
            return False
        if sampling:
            sampling_state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
            # prepare() declines batches without a shared prefix.
            return sampling_state is not None and sampling_state.plan is not None
        if metadata.fmha_plan_caches.get(LIB_NAME) is None:
            # prepare() declined to plan this batch (see metadata_eligible).
            return False
        return True

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        layer_idx = self.attn.layer_idx
        pool = self._pool_views.get(layer_idx)
        if pool is None:
            pool = metadata.kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"{type(self).__name__}: layer {layer_idx} has no KV pool.")
            self._pool_views[layer_idx] = pool
        return pool

    def run_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        metadata = params.meta
        qkv = params.qkv_input
        if qkv is None:
            raise RuntimeError(f"{type(self).__name__} requires packed QKV input.")
        if metadata.beam_width == 1:
            self._run_sampling_generation(params, qkv)
            return
        state: Optional[_BeamStepState] = metadata.fmha_plan_caches.get(LIB_NAME)
        if state is None or state.plan is None:
            raise RuntimeError(
                f"{type(self).__name__}: no step plan; TrtllmAttentionMetadata.prepare() "
                "must run before the forward."
            )
        plan = state.plan
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        rows = params.num_tokens
        if rows != plan.num_requests * plan.beam_width:
            raise RuntimeError(
                f"{type(self).__name__}: {rows} generation rows do not match the planned "
                f"{plan.num_requests} x {plan.beam_width} beams."
            )
        if self._layout is None:
            self._layout = _pool_layout(metadata, attn.local_layer_idx)
        pool_index, page_stride = self._layout
        if pool_index != state.pool_index or page_stride != state.page_stride:
            raise RuntimeError(
                f"{type(self).__name__} supports a single KV pool layout per model; layer "
                f"{attn.layer_idx} uses pool {pool_index} (stride {page_stride}) but the plan "
                f"was built for pool {state.pool_index} (stride {state.page_stride})."
            )

        sm_scale = 1.0 / (math.sqrt(head_dim) * attn.q_scaling)
        key = _PlanKey(num_heads, num_kv_heads, head_dim, qkv.dtype, sm_scale)
        wrapper = state.wrappers.get(key)
        if wrapper is None or state.planned_step.get(key) != plan.step:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{type(self).__name__}: FlashInfer plan requested during CUDA graph "
                    "capture; the capture warmup must run this shape first."
                )
            if wrapper is None:
                wrapper = _create_wrapper(state, key, metadata)
                state.wrappers[key] = wrapper
            _plan_wrapper(state, key, wrapper, metadata)

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        q = qkv[:, :q_size].view(rows, num_heads, head_dim)
        k = qkv[:, q_size : q_size + kv_size].view(rows, num_kv_heads, head_dim)
        v = qkv[:, q_size + kv_size : q_size + 2 * kv_size].view(rows, num_kv_heads, head_dim)

        pool = self._pool_view(metadata)
        # Checks synchronize and read device values, neither of which is
        # permitted inside CUDA graph capture; the capture's warmup runs the
        # same shapes eagerly first, so coverage is not lost.
        check = _check_enabled() and not torch.cuda.is_current_stream_capturing()
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: before step (error from an earlier op)")
            _check_plan(state, metadata, pool)
        indices = _step_indices(state, metadata, pool.size(0))
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: step indices")
            _check_indices(indices, pool, int(metadata.tokens_per_block))

        # 1. Persist the current token's K/V in the beam's own block.
        pool[indices.write_page, 0, :, indices.write_slot, :] = k
        pool[indices.write_page, 1, :, indices.write_slot, :] = v
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: KV write")

        # 2. Shared prefix: all beams of a request against the prompt KV.
        q_contig = q.contiguous()
        prefix_out, prefix_lse = wrapper.run(q_contig, pool, return_lse=True)
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: FlashInfer prefix attention")
            _check_prefix_against_reference(
                state,
                metadata,
                pool,
                q_contig,
                prefix_out,
                prefix_lse,
                indices,
                num_kv_heads=num_kv_heads,
                sm_scale=sm_scale,
                layer_idx=attn.layer_idx,
            )

        output = params.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output tensor.")
        output = output.view(rows, num_heads, head_dim)

        # 3+4. Tail attention (past generated positions via indirection, plus
        # the current token) merged with the prefix state.
        merged = self._unfused_tail_and_merge(
            pool,
            q,
            k,
            v,
            indices,
            plan,
            prefix_out,
            prefix_lse,
            num_kv_heads=num_kv_heads,
            sm_scale=sm_scale,
        )
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: tail + merge")
            if not torch.isfinite(merged).all():
                raise RuntimeError(
                    f"{LIB_NAME} check: non-finite attention output at layer {attn.layer_idx}"
                )
        output.copy_(merged.to(output.dtype))

    def _run_sampling_generation(self, params: FmhaParams, qkv: torch.Tensor) -> None:
        attn = params.attn
        metadata = params.meta
        state: Optional[_SamplingState] = metadata.fmha_plan_caches.get(SAMPLING_KEY)
        if state is None or state.plan is None:
            raise RuntimeError(
                f"{type(self).__name__}: no n-sample step plan; "
                "TrtllmAttentionMetadata.prepare() must run before the forward."
            )
        plan = state.plan
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        rows = params.num_tokens
        if rows != plan.num_rows:
            raise RuntimeError(
                f"{type(self).__name__}: {rows} generation rows do not match the planned "
                f"{plan.num_rows}."
            )
        if self._layout is None:
            self._layout = _page_layout(metadata, attn.layer_idx, attn.local_layer_idx)
        if self._layout != (state.pool_index, state.page_scale, state.page_div):
            raise RuntimeError(
                f"{type(self).__name__} supports a single KV pool layout per model; layer "
                f"{attn.layer_idx} uses (pool, scale, divisor) {self._layout} but the plan was "
                f"built for {(state.pool_index, state.page_scale, state.page_div)}."
            )

        sm_scale = 1.0 / (math.sqrt(head_dim) * attn.q_scaling)
        key = _PlanKey(num_heads, num_kv_heads, head_dim, qkv.dtype, sm_scale)
        pool = self._pool_view(metadata)
        indices = _sampling_device(state, qkv.device)
        prefix_wrapper, tail_wrapper = _sampling_wrappers(state, key, metadata, qkv.device)

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        q = qkv[:, :q_size].view(rows, num_heads, head_dim)
        k = qkv[:, q_size : q_size + kv_size].view(rows, num_kv_heads, head_dim)
        v = qkv[:, q_size + kv_size : q_size + 2 * kv_size].view(rows, num_kv_heads, head_dim)

        # 1. Persist the current token's K/V in the row's own block.
        pool[indices.write_page, 0, :, indices.write_slot, :] = k
        pool[indices.write_page, 1, :, indices.write_slot, :] = v

        # 2. Shared prefix: the queries of each group against its common pages.
        q_grouped = q if indices.perm is None else q.index_select(0, indices.perm)
        prefix_out, prefix_lse = prefix_wrapper.run(q_grouped.contiguous(), pool, return_lse=True)
        if indices.inv_perm is not None:
            prefix_out = prefix_out.index_select(0, indices.inv_perm)
            prefix_lse = prefix_lse.index_select(0, indices.inv_perm)
            prefix_empty = indices.prefix_empty.index_select(0, indices.inv_perm)
        else:
            prefix_empty = indices.prefix_empty

        # 3. Private remainder of every row (current token included).
        tail_out, tail_lse = tail_wrapper.run(q.contiguous(), pool, return_lse=True)

        if not state.ran:
            state.ran = True
            logger.info(f"{LIB_NAME}: first n-sample attention call (layer {attn.layer_idx})")
        output = params.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output tensor.")
        merged = merge_states(prefix_out, prefix_lse, tail_out, tail_lse, prefix_empty)
        if plan.check_pages is not None:
            _check_sampling_output(
                plan,
                pool,
                q,
                merged,
                num_kv_heads=num_kv_heads,
                sm_scale=sm_scale,
                tokens_per_block=int(metadata.tokens_per_block),
                layer_idx=attn.layer_idx,
            )
        output.view(rows, num_heads, head_dim).copy_(merged.to(output.dtype))

    def _unfused_tail_and_merge(
        self,
        pool: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: _StepIndices,
        plan: _HostPlan,
        prefix_out: torch.Tensor,
        prefix_lse: torch.Tensor,
        *,
        num_kv_heads: int,
        sm_scale: float,
    ) -> torch.Tensor:
        rows = q.shape[0]
        if plan.tail_len > 0:
            assert indices.tail_page is not None and indices.tail_slot is not None
            assert indices.tail_valid is not None
            k_tail = pool[indices.tail_page, 0, :, indices.tail_slot, :]
            v_tail = pool[indices.tail_page, 1, :, indices.tail_slot, :]
            k_all = torch.cat([k_tail, k.unsqueeze(1)], dim=1)
            v_all = torch.cat([v_tail, v.unsqueeze(1)], dim=1)
            valid = torch.cat(
                [indices.tail_valid, torch.ones((rows, 1), dtype=torch.bool, device=q.device)],
                dim=1,
            )
        else:
            k_all = k.unsqueeze(1)
            v_all = v.unsqueeze(1)
            valid = None
        return merge_prefix_and_tail(
            q,
            k_all,
            v_all,
            valid,
            prefix_out,
            prefix_lse,
            num_kv_heads=num_kv_heads,
            sm_scale=sm_scale,
            prefix_empty=indices.prefix_empty,
        )


# --------------------------------------------------------------------------- #
# Check mode (TLLM_BEAM_SHARED_PREFIX_CHECK=1)
# --------------------------------------------------------------------------- #
def _sync_checkpoint(what: str) -> None:
    """Surface asynchronous CUDA errors at the stage that caused them."""
    try:
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(f"{LIB_NAME} check: CUDA error surfaced at [{what}]: {exc}") from exc


def _check_plan(
    state: _BeamStepState, metadata: "TrtllmAttentionMetadata", pool: torch.Tensor
) -> None:
    """Validate the host plan: page ids in range and equal to the manager's
    block table for beam 0 of every request."""
    plan = state.plan
    assert plan is not None
    num_pages = pool.size(0)
    indices = plan.paged_kv_indices.to(torch.int64)
    if indices.numel() and (indices.min() < 0 or indices.max() >= num_pages):
        raise RuntimeError(
            f"{LIB_NAME} check: prefix page ids out of range [0, {num_pages}): "
            f"min {int(indices.min())} max {int(indices.max())} (page stride {state.page_stride})"
        )
    request_ids = metadata.request_ids
    if request_ids is None:
        return
    gen_request_ids = list(request_ids[metadata.num_contexts :])
    if len(gen_request_ids) != plan.num_requests:
        raise RuntimeError(
            f"{LIB_NAME} check: {len(gen_request_ids)} generation request ids vs "
            f"{plan.num_requests} planned requests"
        )
    try:
        block_ids = metadata.kv_cache_manager.get_batch_cache_indices(gen_request_ids)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.warning(f"{LIB_NAME} check: could not read the manager block table: {exc!r}")
        return
    indptr = plan.paged_kv_indptr.tolist()
    for request_idx, blocks in enumerate(block_ids):
        planned = indices[indptr[request_idx] : indptr[request_idx + 1]].tolist()
        expected = [int(b) for b in blocks[: len(planned)]]
        if planned != expected:
            raise RuntimeError(
                f"{LIB_NAME} check: prefix page table of request {request_idx} differs from "
                f"the KV manager's beam-0 block ids.\n planned : {planned[:12]}...\n"
                f" manager: {expected[:12]}... (page stride {state.page_stride})"
            )
    logger.info(
        f"{LIB_NAME} check: prefix page table matches the KV manager "
        f"({plan.num_requests} requests, {int(indices.numel())} pages)"
    )


def _check_indices(indices: _StepIndices, pool: torch.Tensor, tokens_per_block: int) -> None:
    num_pages = pool.size(0)
    for name, tensor, bound in (
        ("write_page", indices.write_page, num_pages),
        ("write_slot", indices.write_slot, tokens_per_block),
        ("tail_page", indices.tail_page, num_pages),
        ("tail_slot", indices.tail_slot, tokens_per_block),
    ):
        if tensor is None or tensor.numel() == 0:
            continue
        lo, hi = int(tensor.min()), int(tensor.max())
        if lo < 0 or hi >= bound:
            raise RuntimeError(
                f"{LIB_NAME} check: {name} out of range [0, {bound}): min {lo} max {hi}"
            )


def _check_prefix_against_reference(
    state: _BeamStepState,
    metadata: "TrtllmAttentionMetadata",
    pool: torch.Tensor,
    q: torch.Tensor,
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    indices: _StepIndices,
    *,
    num_kv_heads: int,
    sm_scale: float,
    layer_idx: int,
) -> None:
    """Dense recomputation of the prefix attention from the planned pages."""
    plan = state.plan
    assert plan is not None
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    tokens_per_block = int(metadata.tokens_per_block)
    indptr = plan.paged_kv_indptr.tolist()
    last_page_len = plan.paged_kv_last_page_len.tolist()
    pages_all = plan.paged_kv_indices.to(torch.int64)
    worst_out = 0.0
    worst_lse = 0.0
    for request_idx in range(plan.num_requests):
        pages = pages_all[indptr[request_idx] : indptr[request_idx + 1]].to(pool.device)
        n_pages = pages.numel()
        if n_pages == 0:
            continue
        prefix_len = (n_pages - 1) * tokens_per_block + int(last_page_len[request_idx])
        # [pages, 2, Hkv, tpb, D] -> [L, Hkv, D]
        kv = (
            pool[pages]
            .permute(0, 3, 1, 2, 4)
            .reshape(n_pages * tokens_per_block, 2, num_kv_heads, head_dim)[:prefix_len]
        )
        k_ref = kv[:, 0].float()
        v_ref = kv[:, 1].float()
        row0 = request_idx * plan.beam_width
        row1 = row0 + plan.beam_width
        q_req = q[row0:row1].float().view(plan.beam_width, num_kv_heads, group, head_dim)
        scores = torch.einsum("rkgd,tkd->rkgt", q_req, k_ref) * sm_scale
        ref_out = torch.einsum("rkgt,tkd->rkgd", torch.softmax(scores, dim=-1), v_ref)
        ref_lse = torch.logsumexp(scores, dim=-1) * _LOG2E
        got_out = prefix_out[row0:row1].float().view(plan.beam_width, num_kv_heads, group, head_dim)
        got_lse = prefix_lse[row0:row1].view(plan.beam_width, num_kv_heads, group)
        keep = ~indices.prefix_empty[row0:row1].view(-1, 1, 1)
        # The kernel writes bf16/fp16 output, so compare with a tolerance
        # relative to the reference magnitude (bf16 keeps ~3 significant
        # digits); the LSE is fp32 and must match tightly.
        out_err = (ref_out - got_out).abs() / (ref_out.abs() + 1.0)
        worst_out = max(worst_out, float((out_err * keep.unsqueeze(-1)).max()))
        worst_lse = max(worst_lse, float(((ref_lse - got_lse).abs() * keep).max()))
    tol_out, tol_lse = 2e-2, 2e-2
    if worst_out > tol_out or worst_lse > tol_lse:
        raise RuntimeError(
            f"{LIB_NAME} check: FlashInfer prefix attention disagrees with the dense reference "
            f"at layer {layer_idx}: max |out| diff / (1 + |ref|) {worst_out:.4f}, max |lse2| "
            f"diff {worst_lse:.4f} (backend {_backend_from_env()})"
        )
    logger.info(
        f"{LIB_NAME} check: layer {layer_idx} prefix attention matches the dense "
        f"reference (max relative |out| diff {worst_out:.2e}, "
        f"max |lse2| diff {worst_lse:.2e})"
    )


__all__ = [
    "LIB_NAME",
    "MAX_TAIL_ENV",
    "BeamSharedPrefixFmha",
    "has_active_plan",
    "merge_prefix_and_tail",
    "merge_states",
    "metadata_eligible",
    "plan_step",
    "sampling_metadata_eligible",
]
