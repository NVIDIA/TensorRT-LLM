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

A decode step is split in two parts and merged with the log-sum-exp recombination:

* **prefix**: one paged-prefill call (FlashInfer, or FA4 with ``TLLM_BEAM_SHARED_PREFIX_BACKEND=fa4``)
  where a request's ``beam_width`` queries attend the shared prompt through beam 0's page table;
* **tail**: the generated positions, gathered per beam through ``cache_indirection``, plus the
  current token from the layer input.

The current token's K/V is written into the beam's own block, as the fallback kernel does.

Scope: generation rows of a dense GQA/MHA model with an unquantized bf16/fp16 paged KV cache
(KV cache manager V2 only: the legacy manager raises at planning), RoPE applied by the module,
no sliding window, sinks, custom masks or speculative decoding. In a batch that also carries
context requests, the context phase runs on the regular library. Per-step planning is host work
driven from ``TrtllmAttentionMetadata.prepare``, which keeps captured CUDA graphs free of host
synchronization.

Opt in with ``TLLM_FMHA_LIBS=+beam_shared_prefix``. ``TLLM_BEAM_SHARED_PREFIX_MAX_TAIL``
(default 8) bounds the generated positions per beam under CUDA graphs; longer generations are
rejected at planning time.
"""

from __future__ import annotations

import functools
import math
import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .fallback import FallbackFmha
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .shared_prefix_fa4 import Fa4Prefix, PrefixGroups, fa4_supported, prefix_request_tables

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

MAX_TAIL_ENV = "TLLM_BEAM_SHARED_PREFIX_MAX_TAIL"
DEFAULT_MAX_TAIL = 8
BACKEND_ENV = "TLLM_BEAM_SHARED_PREFIX_BACKEND"
"""FlashInfer prefill backend for the prefix call. Default ``fa2``, the kernel
family the TRTLLM FlashInfer backend pins for paged prefill; ``fa3`` or
``auto`` select the Hopper kernels. ``fa4`` runs the prefix through the FlashAttention-4 kernels
shared with the n-sample library (``shared_prefix_fa4``; SM90, bf16, head size 64 or 128)."""
DEFAULT_BACKEND = "fa2"
FA4_BACKEND = "fa4"
FUSED_TAIL_ENV = "TLLM_BEAM_SHARED_PREFIX_FUSED_TAIL"
"""Set to 0 to run the tail attention and merge as separate PyTorch ops instead of
the fused Triton kernel; the unfused path is the numerical reference."""
CHECK_ENV = "TLLM_BEAM_SHARED_PREFIX_CHECK"
"""Set to 1 to synchronize after every stage of the step and validate the
page table and the prefix attention against dense references. Debugging aid:
it serializes the GPU and must stay off for measurements."""

_LOG2E = 1.4426950408889634
# Split-KV partial results of the prefix call. FlashInfer recommends 128 MB as
# the floor and raises (rather than overflows) if a plan needs more.
_FLOAT_WORKSPACE_BYTES = 128 * 1024 * 1024
# The partial results are a few output-sized copies of every query row; a plan
# of 128 requests x 96 beams needs about 3.3 of them.
_SPLIT_KV_WORKSPACE_FACTOR = 4
# One buffer per KV cache manager: executors sharing a GPU must not share partials, and the buffer is
# freed with its manager.
_float_workspace: "weakref.WeakKeyDictionary[Any, torch.Tensor]" = weakref.WeakKeyDictionary()


def _backend_from_env() -> str:
    value = os.environ.get(BACKEND_ENV, "").strip()
    return value or DEFAULT_BACKEND


def _fa4_requested() -> bool:
    """Used at layer selection only (the manager caches that decision); the forward reads
    ``_Config`` instead."""
    return _backend_from_env() == FA4_BACKEND


_fa4_supported = fa4_supported


@dataclass(frozen=True)
class _Config:
    """The environment settings, read once when a metadata object first plans, not per layer."""

    backend: str
    fused_tail: bool
    check: bool
    max_tail: int

    @classmethod
    def from_env(cls) -> "_Config":
        value = os.environ.get(MAX_TAIL_ENV)
        max_tail = DEFAULT_MAX_TAIL if value is None or not value.strip() else int(value)
        if max_tail < 0:
            raise ValueError(f"{MAX_TAIL_ENV} must be >= 0, got {max_tail}.")
        return cls(
            backend=_backend_from_env(),
            fused_tail=os.environ.get(FUSED_TAIL_ENV, "1") != "0",
            check=os.environ.get(CHECK_ENV, "0") == "1",
            max_tail=max_tail,
        )

    @property
    def fa4(self) -> bool:
        return self.backend == FA4_BACKEND


def _get_float_workspace(manager: Any, device: torch.device, min_bytes: int = 0) -> torch.Tensor:
    """The manager's FlashInfer float workspace, at least ``min_bytes``; it only grows, and
    earlier wrappers keep the buffer they were created with so captured CUDA graphs stay valid."""
    workspace = _float_workspace.get(manager)
    if workspace is None or workspace.numel() < min_bytes:
        size = max(_FLOAT_WORKSPACE_BYTES, min_bytes)
        workspace = torch.empty(size, dtype=torch.uint8, device=device)
        _float_workspace[manager] = workspace
    return workspace


def _float_workspace_bytes(
    key: "_PlanKey", metadata: "TrtllmAttentionMetadata", plan_rows: int
) -> int:
    """Workspace the prefix call can need for the engine's widest batch."""
    manager = metadata.kv_cache_manager
    max_rows = max(
        plan_rows,
        int(getattr(manager, "max_batch_size", 0)) * int(getattr(manager, "max_beam_width", 1)),
    )
    return _SPLIT_KV_WORKSPACE_FACTOR * max_rows * key.num_heads * key.head_dim * key.dtype.itemsize


@functools.lru_cache(maxsize=8)
def _enabled_for(libs_env: Optional[str]) -> bool:
    from .registry import get_enabled_fmha_lib_classes

    return BeamSharedPrefixFmha in get_enabled_fmha_lib_classes()


def _lib_enabled() -> bool:
    """Whether ``TLLM_FMHA_LIBS`` enables this library; the parse is cached per value since this
    runs every beam step."""
    return _enabled_for(os.environ.get("TLLM_FMHA_LIBS"))


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
class _Fa4Slot:
    """The FlashAttention-4 prefix call of one layer shape: its buffers and fixed-address tables.

    ``upload`` stages the step's prefix groups in page-locked buffers and copies them to the
    tables a captured graph reads; the plan issues it before the forward.
    """

    prefix: Fa4Prefix
    rows: int
    staging: tuple
    copied: torch.cuda.Event
    pending: bool = False

    @classmethod
    def create(
        cls,
        key: "_PlanKey",
        rows: int,
        num_requests: int,
        max_pages: int,
        tokens_per_block: int,
        device: torch.device,
    ) -> "_Fa4Slot":
        prefix = Fa4Prefix(
            rows,
            key.num_heads,
            key.head_dim,
            1,
            Fa4Prefix.num_requests_for(num_requests, rows, 1),
            max_pages,
            key.dtype,
            device,
            tokens_per_block,
        )
        pinned = prefer_pinned()
        staging = tuple(
            torch.zeros(t.shape, dtype=t.dtype, pin_memory=pinned) for t in prefix.table_arrays()
        )
        return cls(prefix=prefix, rows=rows, staging=staging, copied=torch.cuda.Event())

    def upload(self, groups: PrefixGroups) -> None:
        if self.pending:
            # The previous copies out of the staging buffers must have run.
            self.copied.synchronize()
        prefix = self.prefix
        host = prefix_request_tables(
            groups,
            prefix.tokens_per_block,
            prefix.chunk_tokens,
            prefix.num_requests,
            prefix.block_tables.shape[1],
        )
        for staging, device, values in zip(self.staging, prefix.table_arrays(), host):
            staging.numpy()[...] = values
            device.copy_(staging, non_blocking=True)
        self.copied.record()
        self.pending = True


@dataclass
class _BeamStepState:
    """Per-metadata plan state in ``metadata.fmha_plan_caches[LIB_NAME]``; a metadata object
    belongs to one execution mode (eager, or one CUDA graph batch size)."""

    is_cuda_graph: bool
    pool_index: int
    page_scale: int
    """Page-index scale of the layers' pool (``get_layer_page_index_scale``)."""
    page_stride: int
    """Divisor turning a K entry of ``kv_cache_block_offsets`` (a base page index times the pool's
    page-index scale) into a page of the layer's ``get_buffers`` view: ``kv_factor``."""
    config: _Config
    num_pages: int
    """Pages in the primary pool; bounds every page index handed to a kernel."""
    plan: Optional[_HostPlan] = None
    wrappers: Dict[_PlanKey, Any] = field(default_factory=dict)
    fa4_slots: Dict[_PlanKey, _Fa4Slot] = field(default_factory=dict)
    planned_step: Dict[_PlanKey, int] = field(default_factory=dict)
    indices: Optional[_StepIndices] = None
    step_counter: int = 0


def _pool_layout(metadata: "TrtllmAttentionMetadata", layer_idx: int) -> tuple[int, int, int]:
    """(pool index, page-index scale, page stride) of a layer's KV pool. The block-offset table
    holds the V2 base page index times the scale; a ``get_buffers`` page is that divided by
    ``kv_factor``."""
    manager = metadata.kv_cache_manager
    pool_index = int(manager.layer_to_pool_mapping_dict[manager.layer_offsets[layer_idx]])
    return pool_index, manager.get_layer_page_index_scale(layer_idx), int(manager.kv_factor)


def _is_kv_manager_v2(kv_cache_manager) -> bool:
    """Whether the manager is KV cache manager V2: it has the index mapper, the per-layer page-index
    scale and the pool view the library reads."""
    return all(
        hasattr(kv_cache_manager, attr)
        for attr in (
            "get_buffers",
            "get_batch_cache_indices",
            "get_layer_page_index_scale",
            "index_mapper",
            "layer_to_pool_mapping_dict",
            "kv_factor",
            "tokens_per_block",
        )
    )


def _kv_manager_supported(kv_cache_manager) -> bool:
    """KV cache manager V2 with one block-offset table per pool and separate K and V."""
    if kv_cache_manager is None or not _is_kv_manager_v2(kv_cache_manager):
        return False
    if getattr(kv_cache_manager, "kv_factor", 0) != 2:
        return False
    # The layer-first pool layout (recurrent-state caches) encodes offsets
    # differently; those managers never run dense beam search anyway.
    if getattr(kv_cache_manager, "is_linear_attention", False):
        return False
    # Per-layer page tables (SWA scratch reuse, mixed page sizes or storage layouts in one lifecycle
    # group) index kv_cache_block_offsets by local layer instead of by pool.
    if getattr(kv_cache_manager, "_use_per_layer_page_tables", False):
        return False
    return True


def metadata_eligible(metadata: "TrtllmAttentionMetadata") -> bool:
    """Batch-level preconditions shared by planning and library selection."""
    return (
        IS_FLASHINFER_AVAILABLE
        and metadata.beam_width > 1
        and metadata.num_generations > 0
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
        and metadata.num_generations % metadata.beam_width == 0
    )


def plan_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Per-step host planning, called from ``TrtllmAttentionMetadata.prepare`` before any forward
    (so before CUDA graph capture and replay): builds each request's prompt page table, decides
    the tail length and re-plans the layers' prefix wrappers."""
    stale = metadata.fmha_plan_caches.get(LIB_NAME)
    if stale is not None:
        # A step that returns early below must not leave the previous step's plan in place.
        stale.plan = None
        stale.indices = None
    if metadata.beam_width <= 1 or metadata.num_generations == 0 or not _lib_enabled():
        return
    if metadata.kv_cache_manager is not None and not _is_kv_manager_v2(metadata.kv_cache_manager):
        raise NotImplementedError(
            f"{LIB_NAME} requires KV cache manager V2. Set kv_cache_config.use_kv_cache_manager_v2=True "
            f"or remove +{LIB_NAME} from TLLM_FMHA_LIBS."
        )
    if not metadata_eligible(metadata):
        return
    state = metadata.fmha_plan_caches.get(LIB_NAME)
    kv_cache_manager = metadata.kv_cache_manager
    first_layer = min(kv_cache_manager.layer_offsets)
    if state is None:
        # The layout facts come from the first layer's pool; every layer must
        # share it (run_generation checks).
        pool_index, page_scale, page_stride = _pool_layout(metadata, first_layer)
        if page_scale % page_stride:
            raise RuntimeError(
                f"{LIB_NAME}: page-index scale {page_scale} is not a multiple of kv_factor "
                f"{page_stride}; remove {LIB_NAME} from TLLM_FMHA_LIBS."
            )
        pool = kv_cache_manager.get_buffers(first_layer, kv_layout="HND")
        state = _BeamStepState(
            is_cuda_graph=metadata.is_cuda_graph,
            pool_index=pool_index,
            page_scale=page_scale,
            page_stride=page_stride,
            config=_Config.from_env(),
            num_pages=int(pool.size(0)),
        )
        metadata.fmha_plan_caches[LIB_NAME] = state
        max_seq_len = getattr(metadata, "max_seq_len", None)
        if state.is_cuda_graph and max_seq_len and max_seq_len - 2 > state.config.max_tail:
            logger.warning_once(
                f"{LIB_NAME}: CUDA graphs are captured for beams that generate at most "
                f"{state.config.max_tail} tokens ({MAX_TAIL_ENV}); a longer generation fails "
                f"the step. Set {MAX_TAIL_ENV} to the largest max_tokens you serve (each extra "
                "position costs a little per step).",
                key=f"{LIB_NAME}_max_tail",
            )
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
        if max_tail > state.config.max_tail:
            raise RuntimeError(
                f"beam_shared_prefix: a beam carries {max_tail} generated positions but "
                f"CUDA graphs were captured for at most {state.config.max_tail}. Raise "
                f"{MAX_TAIL_ENV} to the maximum number of generated tokens, or remove "
                f"{LIB_NAME} from TLLM_FMHA_LIBS."
            )
        tail_len = state.config.max_tail
    else:
        tail_len = max_tail

    request_prompt_lens = prompt_lens.view(num_requests, beam_width)[:, 0]
    ctx_blocks = (request_prompt_lens + tokens_per_block - 1) // tokens_per_block
    last_page_len = request_prompt_lens - (ctx_blocks - 1) * tokens_per_block
    # Beam 0's pages of the prompt, as pages of the layer's ``get_buffers`` view. Every beam of a
    # request holds the same prompt tokens.
    request_ids = metadata.request_ids
    if request_ids is None or len(request_ids) < num_ctx + num_requests:
        raise RuntimeError(f"{LIB_NAME}: the batch's request ids are required.")
    pages = kv_cache_manager.get_batch_cache_indices(
        list(request_ids[num_ctx : num_ctx + num_requests]),
        layer_idx=first_layer,
        num_blocks_per_seq=ctx_blocks.tolist(),
    )
    paged_kv_indices = torch.tensor(
        [page for request_pages in pages for page in request_pages], dtype=torch.int64
    )
    if paged_kv_indices.numel() != int(ctx_blocks.sum()):
        raise RuntimeError(
            f"{LIB_NAME}: the KV cache holds fewer prompt blocks than the batch's prompt lengths."
        )
    if paged_kv_indices.numel() and (
        int(paged_kv_indices.min()) < 0 or int(paged_kv_indices.max()) >= state.num_pages
    ):
        raise RuntimeError(
            f"{LIB_NAME}: the KV cache returned an invalid page index for a prompt block "
            f"(valid range [0, {state.num_pages}))."
        )
    paged_kv_indices = paged_kv_indices.to(torch.int32)
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
    for key in list(state.fa4_slots):
        _plan_fa4_slot(state, key)


def _prefix_groups(plan: _HostPlan) -> PrefixGroups:
    """One group per request: its beams' queries over the request's prompt pages."""
    return PrefixGroups(
        num_groups=plan.num_requests,
        qo_indptr=plan.qo_indptr.numpy(),
        prefix_indptr=plan.paged_kv_indptr.numpy(),
        prefix_indices=plan.paged_kv_indices.numpy(),
        prefix_last_page_len=plan.paged_kv_last_page_len.numpy(),
    )


def _plan_fa4_slot(state: _BeamStepState, key: _PlanKey) -> None:
    plan = state.plan
    assert plan is not None
    slot = state.fa4_slots[key]
    if plan.num_requests * plan.beam_width > slot.rows:
        # Eager batches grow; a captured graph's batch is fixed. The next forward recreates it.
        del state.fa4_slots[key]
        state.planned_step.pop(key, None)
        return
    slot.upload(_prefix_groups(plan))
    state.planned_step[key] = plan.step


def _create_fa4_slot(
    state: _BeamStepState, key: _PlanKey, metadata: "TrtllmAttentionMetadata"
) -> _Fa4Slot:
    plan = state.plan
    assert plan is not None
    return _Fa4Slot.create(
        key,
        plan.num_requests * plan.beam_width,
        plan.num_requests,
        int(metadata.kv_cache_manager.max_blocks_per_seq),
        int(metadata.tokens_per_block),
        metadata.kv_cache_block_offsets.device,
    )


def _plan_wrapper(
    state: _BeamStepState, key: _PlanKey, wrapper: Any, metadata: "TrtllmAttentionMetadata"
) -> None:
    plan = state.plan
    assert plan is not None
    qo_indptr = plan.qo_indptr
    paged_kv_indptr = plan.paged_kv_indptr
    paged_kv_indices = plan.paged_kv_indices
    last_page_len = plan.paged_kv_last_page_len
    if state.is_cuda_graph:
        # A graph wrapper copies these into its static buffers on every replayed step. From
        # pageable host memory that copy synchronizes the stream, so stage them pinned and hand
        # FlashInfer the indices on the device (the rest it also reads on the host).
        device = metadata.kv_cache_block_offsets.device
        qo_indptr = maybe_pin_memory(qo_indptr)
        paged_kv_indptr = maybe_pin_memory(paged_kv_indptr)
        last_page_len = maybe_pin_memory(last_page_len)
        paged_kv_indices = maybe_pin_memory(paged_kv_indices).to(device, non_blocking=True)
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        last_page_len,
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
    plan = state.plan
    assert plan is not None
    workspace = _get_float_workspace(
        metadata.kv_cache_manager,
        device,
        _float_workspace_bytes(key, metadata, plan.num_requests * plan.beam_width),
    )
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend=state.config.backend, **kwargs
    )


def _step_indices(
    state: _BeamStepState, metadata: "TrtllmAttentionMetadata", num_pages: int
) -> _StepIndices:
    """Device-side page/slot indices of the current token and the tail, derived from tensors
    ``prepare`` refreshes each step, so a captured CUDA graph recomputes them on replay."""
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
    """Attend ``q`` to the tail K/V and merge with the prefix state; returns fp32
    [rows, num_heads, head_dim].

    ``k_tail``/``v_tail`` are [rows, tail, num_kv_heads, head_dim] with the current token last;
    ``tail_valid`` [rows, tail] marks valid positions (None: all). ``prefix_lse`` is the base-2
    LSE [rows, num_heads] (FlashInfer's convention). ``prefix_empty`` [rows] marks rows whose
    prefix is a placeholder.
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


@triton.jit
def _tail_merge_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pool_ptr,
    prefix_out_ptr,
    prefix_lse_ptr,
    out_ptr,
    write_page_ptr,
    write_slot_ptr,
    tail_page_ptr,
    tail_slot_ptr,
    tail_valid_ptr,
    prefix_empty_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    stride_pool_page,
    stride_pool_kv,
    stride_pool_head,
    stride_pool_slot,
    stride_prefix_row,
    stride_prefix_head,
    stride_lse_row,
    stride_lse_head,
    stride_out_row,
    stride_out_head,
    stride_tail_row,
    tail_len,
    sm_scale_log2,
    LSE_SCALE: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program per (row, KV head): write the current K/V, attend the tail, merge.

    Online softmax in the log2 domain, started from the prefix state: the prefix
    output is already normalized, so it enters with weight 1 at max ``lse``.
    """
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    group = tl.arange(0, GROUP_PAD)
    dim = tl.arange(0, HEAD_DIM)
    group_mask = group < GROUP
    head = kv_head * GROUP + group

    q = tl.load(
        q_ptr + row * stride_q_row + head[:, None] * stride_q_head + dim[None, :],
        mask=group_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_raw = tl.load(k_ptr + row * stride_k_row + kv_head * stride_k_head + dim)
    v_raw = tl.load(v_ptr + row * stride_v_row + kv_head * stride_v_head + dim)

    write_page = tl.load(write_page_ptr + row)
    write_slot = tl.load(write_slot_ptr + row)
    write_base = (
        pool_ptr
        + write_page * stride_pool_page
        + kv_head * stride_pool_head
        + write_slot * stride_pool_slot
    )
    tl.store(write_base + dim, k_raw)
    tl.store(write_base + stride_pool_kv + dim, v_raw)

    empty = tl.load(prefix_empty_ptr + row) != 0
    lse = (
        tl.load(
            prefix_lse_ptr + row * stride_lse_row + head * stride_lse_head,
            mask=group_mask,
            other=float("-inf"),
        )
        * LSE_SCALE
    )
    acc = tl.load(
        prefix_out_ptr
        + row * stride_prefix_row
        + head[:, None] * stride_prefix_head
        + dim[None, :],
        mask=group_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    m = tl.where(empty, float("-inf"), lse)
    denom = tl.where(group_mask & (~empty), 1.0, 0.0)
    acc = tl.where(empty, 0.0, acc)

    for t in range(0, tail_len):
        valid = tl.load(tail_valid_ptr + row * stride_tail_row + t) != 0
        page = tl.load(tail_page_ptr + row * stride_tail_row + t)
        slot = tl.load(tail_slot_ptr + row * stride_tail_row + t)
        base = (
            pool_ptr
            + page * stride_pool_page
            + kv_head * stride_pool_head
            + slot * stride_pool_slot
        )
        k = tl.load(base + dim, mask=valid, other=0.0).to(tl.float32)
        v = tl.load(base + stride_pool_kv + dim, mask=valid, other=0.0).to(tl.float32)
        score = tl.sum(q * k[None, :], axis=1) * sm_scale_log2
        score = tl.where(valid, score, float("-inf"))
        m_new = tl.maximum(m, score)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m - m_safe)
        p = tl.exp2(score - m_safe)
        acc = acc * alpha[:, None] + p[:, None] * v[None, :]
        denom = denom * alpha + p
        m = m_new

    score = tl.sum(q * k_raw.to(tl.float32)[None, :], axis=1) * sm_scale_log2
    m_new = tl.maximum(m, score)
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m - m_safe)
    p = tl.exp2(score - m_safe)
    acc = acc * alpha[:, None] + p[:, None] * v_raw.to(tl.float32)[None, :]
    denom = denom * alpha + p

    out = acc / denom[:, None]
    tl.store(
        out_ptr + row * stride_out_row + head[:, None] * stride_out_head + dim[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=group_mask[:, None],
    )


def _lse_rows_log2(lse: torch.Tensor, rows: int, lse_scale: float) -> torch.Tensor:
    """The prefix LSE as [rows, heads] in base 2 (FlashInfer's layout), from either backend."""
    if lse_scale == 1.0:
        return lse
    return lse[:, :rows].t().contiguous() * lse_scale


def fused_tail_and_merge(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pool: torch.Tensor,
    indices: "_StepIndices",
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    output: torch.Tensor,
    *,
    tail_len: int,
    num_kv_heads: int,
    sm_scale: float,
    lse_scale: float = 1.0,
    lse_heads_first: bool = False,
) -> None:
    """Write the current K/V, attend the tail and merge with the prefix, in one launch.

    Same contract as ``merge_prefix_and_tail`` plus the KV write of the current
    token: ``q`` is [rows, heads, dim], ``k`` and ``v`` are [rows, kv_heads, dim],
    ``pool`` is the layer's [pages, 2, kv_heads, tokens_per_block, dim] view, and
    ``output`` receives the merged [rows, heads, dim] result. ``prefix_lse`` is [rows, heads]
    (FlashInfer) or, with ``lse_heads_first``, [heads, capacity >= rows] (FlashAttention-4);
    ``lse_scale`` turns it into base 2 (1 for FlashInfer, log2(e) for FA4's natural log).
    """
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    for tensor in (q, k, v):
        assert tensor.stride(-1) == 1
    tail_page = indices.tail_page if tail_len > 0 else indices.write_page
    tail_slot = indices.tail_slot if tail_len > 0 else indices.write_slot
    # With no tail the kernel never reads these three; any valid tensors of the right type do.
    tail_valid = (
        indices.tail_valid.view(torch.int8)
        if tail_len > 0
        else indices.prefix_empty.view(torch.int8)
    )
    _tail_merge_kernel[(rows, num_kv_heads)](
        q,
        k,
        v,
        pool,
        prefix_out,
        prefix_lse,
        output,
        indices.write_page,
        indices.write_slot,
        tail_page,
        tail_slot,
        tail_valid,
        indices.prefix_empty.view(torch.int8),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        prefix_out.stride(0),
        prefix_out.stride(1),
        prefix_lse.stride(1 if lse_heads_first else 0),
        prefix_lse.stride(0 if lse_heads_first else 1),
        output.stride(0),
        output.stride(1),
        tail_page.stride(0) if tail_len > 0 else 0,
        tail_len,
        sm_scale * _LOG2E,
        LSE_SCALE=lse_scale,
        GROUP=group,
        GROUP_PAD=triton.next_power_of_2(group),
        HEAD_DIM=head_dim,
    )


class BeamSharedPrefixFmha(PhasedFmha):
    """Generation-phase library for wide-beam decoding; see the module docstring."""

    supports_workspace_reclamation = False

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        # Context requests that share a step with the beams run on the regular library.
        self._context_impl = FallbackFmha(attn)
        self._pool_views: Dict[int, torch.Tensor] = {}
        # (pool index, page-index scale, page stride) of this layer; fixed for the manager lifetime.
        self._layout: Optional[tuple[int, int, int]] = None
        # The KV cache manager the cached views and layout belong to.
        self._pool_manager: Optional[object] = None

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
        if _fa4_requested() and not _fa4_supported(head_dim, torch.bfloat16, None):
            logger.warning(
                f"{LIB_NAME}: {BACKEND_ENV}={FA4_BACKEND} needs SM90, bf16 and head size 64 or 128 "
                "with the FlashAttention-4 kernels; the library declines this layer."
            )
            return False
        # The module must apply any position embedding itself (it hands the backend none); this
        # library implements neither RoPE nor ALiBi/relative bias.
        if (
            PositionEmbeddingType(getattr(attn, "position_embedding_type", 0))
            != PositionEmbeddingType.learned_absolute
        ):
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
        if not metadata_eligible(metadata):
            return False
        if metadata.num_contexts > 0 and not self._context_impl.is_supported(
            q, k, v, metadata, forward_args, phase=FmhaPhase.CONTEXT
        ):
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
        if _fa4_requested() and q.dtype != torch.bfloat16:
            return False
        window = forward_args.attention_window_size
        if window is not None and (metadata.max_seq_len is None or window < metadata.max_seq_len):
            return False
        # One query token per beam row.
        if q.shape[0] - metadata.num_ctx_tokens != metadata.num_generations:
            return False
        state = metadata.fmha_plan_caches.get(LIB_NAME)
        if state is None or state.plan is None:
            # prepare() did not plan this step (see metadata_eligible).
            return False
        return True

    def _refresh_manager(self, metadata: "TrtllmAttentionMetadata") -> None:
        """Drop the cached views and layout when the KV cache manager changes: the executor
        replaces the manager after memory estimation, and the old one's pools are freed."""
        manager = metadata.kv_cache_manager
        if self._pool_manager is not manager:
            self._pool_views.clear()
            self._layout = None
            self._pool_manager = manager

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        layer_idx = self.attn.layer_idx
        pool = self._pool_views.get(layer_idx)
        if pool is None:
            pool = metadata.kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"{type(self).__name__}: layer {layer_idx} has no KV pool.")
            self._pool_views[layer_idx] = pool
        return pool

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        self._context_impl.prepare_workspace(q, k, v, metadata, forward_args, workspace)

    def run_context(self, params: FmhaParams) -> None:
        self._context_impl.run_context(params)

    def run_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        metadata = params.meta
        qkv = params.qkv_input
        if qkv is None:
            raise RuntimeError(f"{type(self).__name__} requires packed QKV input.")
        state: Optional[_BeamStepState] = metadata.fmha_plan_caches.get(LIB_NAME)
        if state is None or state.plan is None:
            # The manager caches a selection by batch shape, so an unplanned batch of the same
            # shape (single beam, ineligible) can land here: run the regular kernel.
            self._context_impl.run_generation(params)
            return
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
        self._refresh_manager(metadata)
        if self._layout is None:
            self._layout = _pool_layout(metadata, attn.layer_idx)
        if self._layout != (state.pool_index, state.page_scale, state.page_stride):
            raise RuntimeError(
                f"{type(self).__name__} supports a single KV pool layout per model; layer "
                f"{attn.layer_idx} uses (pool, page-index scale, stride) {self._layout} but the "
                f"plan was built for {(state.pool_index, state.page_scale, state.page_stride)}."
            )

        sm_scale = 1.0 / (math.sqrt(head_dim) * attn.q_scaling)
        key = _PlanKey(num_heads, num_kv_heads, head_dim, qkv.dtype, sm_scale)
        use_fa4 = state.config.fa4
        wrapper = None
        slot = None
        if use_fa4:
            slot = state.fa4_slots.get(key)
        else:
            wrapper = state.wrappers.get(key)
        if (slot if use_fa4 else wrapper) is None or state.planned_step.get(key) != plan.step:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{type(self).__name__}: the prefix plan was requested during CUDA graph "
                    "capture; the capture warmup must run this shape first."
                )
            if use_fa4:
                if slot is None:
                    state.fa4_slots[key] = _create_fa4_slot(state, key, metadata)
                _plan_fa4_slot(state, key)
                slot = state.fa4_slots[key]
            else:
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
        if pool.dtype != qkv.dtype:
            raise RuntimeError(
                f"{type(self).__name__}: the KV pool is {pool.dtype} but the layer runs in "
                f"{qkv.dtype}; the library needs an unquantized cache."
            )
        # Checks synchronize and read device values, neither of which is
        # permitted inside CUDA graph capture; the capture's warmup runs the
        # same shapes eagerly first, so coverage is not lost.
        check = state.config.check and not torch.cuda.is_current_stream_capturing()
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: before step (error from an earlier op)")
            _check_plan(state, metadata, pool)
        indices = _step_indices(state, metadata, pool.size(0))
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: step indices")
            _check_indices(indices, pool, int(metadata.tokens_per_block))

        fused = state.config.fused_tail
        if not fused:
            # Persist the current token's K/V in the beam's own block.
            pool[indices.write_page, 0, :, indices.write_slot, :] = k
            pool[indices.write_page, 1, :, indices.write_slot, :] = v
            if check:
                _sync_checkpoint(f"layer {attn.layer_idx}: KV write")

        # Shared prefix: all beams of a request against the prompt KV.
        if use_fa4:
            # FlashAttention-4: normalized output [rows, heads, dim] and natural-log LSE
            # [heads, rows] (the fused kernel reads it through strides).
            fa4 = slot.prefix
            fa4.q[:rows].copy_(q)
            q_contig = fa4.q[:rows]
            fa4.run(pool, sm_scale)
            prefix_out = fa4.out[:rows]
            prefix_lse = fa4.lse
            lse_scale = _LOG2E
        else:
            q_contig = q.contiguous()
            prefix_out, prefix_lse = wrapper.run(q_contig, pool, return_lse=True)
            lse_scale = 1.0
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: FlashInfer prefix attention")
            _check_prefix_against_reference(
                state,
                metadata,
                pool,
                q_contig,
                prefix_out,
                _lse_rows_log2(prefix_lse, rows, lse_scale),
                indices,
                num_kv_heads=num_kv_heads,
                sm_scale=sm_scale,
                layer_idx=attn.layer_idx,
            )

        output = params.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output tensor.")
        output = output.view(rows, num_heads, head_dim)

        # Tail attention (generated positions via indirection, plus the current token) merged
        # with the prefix state; the fused kernel also writes the current token's K/V.
        if fused:
            fused_tail_and_merge(
                q,
                k,
                v,
                pool,
                indices,
                prefix_out,
                prefix_lse,
                output,
                tail_len=plan.tail_len,
                num_kv_heads=num_kv_heads,
                sm_scale=sm_scale,
                lse_scale=lse_scale,
                lse_heads_first=use_fa4,
            )
            merged = output
        else:
            merged = self._unfused_tail_and_merge(
                pool,
                q,
                k,
                v,
                indices,
                plan,
                prefix_out,
                _lse_rows_log2(prefix_lse, rows, lse_scale),
                num_kv_heads=num_kv_heads,
                sm_scale=sm_scale,
            )
        if check:
            _sync_checkpoint(f"layer {attn.layer_idx}: tail + merge")
            if not torch.isfinite(merged).all():
                raise RuntimeError(
                    f"{LIB_NAME} check: non-finite attention output at layer {attn.layer_idx}"
                )
        if not fused:
            output.copy_(merged.to(output.dtype))

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
    """Validate the host plan: page ids in range and equal to beam 0's entries of the manager's
    host block-offset table (base page indices by index-mapper slot)."""
    plan = state.plan
    assert plan is not None
    num_pages = pool.size(0)
    indices = plan.paged_kv_indices.to(torch.int64)
    if indices.numel() and (indices.min() < 0 or indices.max() >= num_pages):
        raise RuntimeError(
            f"{LIB_NAME} check: prefix page ids out of range [0, {num_pages}): "
            f"min {int(indices.min())} max {int(indices.max())} (page stride {state.page_stride})"
        )
    manager = metadata.kv_cache_manager
    host_offsets = getattr(manager, "host_kv_cache_block_offsets", None)
    request_ids = metadata.request_ids
    if host_offsets is None or request_ids is None:
        return
    gen_request_ids = list(request_ids[metadata.num_contexts :])
    if len(gen_request_ids) != plan.num_requests:
        raise RuntimeError(
            f"{LIB_NAME} check: {len(gen_request_ids)} generation request ids vs "
            f"{plan.num_requests} planned requests"
        )
    indptr = plan.paged_kv_indptr.tolist()
    for request_idx, request_id in enumerate(gen_request_ids):
        planned = indices[indptr[request_idx] : indptr[request_idx + 1]].tolist()
        row = manager.index_mapper.get_index(request_id) * int(manager.max_beam_width)
        base = host_offsets[state.pool_index, row, 0, : len(planned)].to(torch.int64)
        expected = (base * state.page_scale // state.page_stride).tolist()
        if planned != expected:
            raise RuntimeError(
                f"{LIB_NAME} check: prefix page table of request {request_idx} differs from "
                f"beam 0 of the KV manager's host block table.\n planned : {planned[:12]}...\n"
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
        # bf16 output with split-KV partials: tolerance relative to the reference magnitude.
        # A wrong page or length gives an error of order one; the fp32 LSE must match tightly.
        out_err = (ref_out - got_out).abs() / (ref_out.abs() + 1.0)
        worst_out = max(worst_out, float((out_err * keep.unsqueeze(-1)).max()))
        worst_lse = max(worst_lse, float(((ref_lse - got_lse).abs() * keep).max()))
    tol_out, tol_lse = 5e-2, 2e-2
    if worst_out > tol_out or worst_lse > tol_lse:
        raise RuntimeError(
            f"{LIB_NAME} check: FlashInfer prefix attention disagrees with the dense reference "
            f"at layer {layer_idx}: max |out| diff / (1 + |ref|) {worst_out:.4f}, max |lse2| "
            f"diff {worst_lse:.4f} (backend {state.config.backend})"
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
    "fused_tail_and_merge",
    "merge_prefix_and_tail",
    "metadata_eligible",
    "plan_step",
]
