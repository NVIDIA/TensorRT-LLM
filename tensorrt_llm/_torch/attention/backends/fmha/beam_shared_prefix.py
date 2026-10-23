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
custom masks, or tree speculative decoding. Mixed context+generation batches fall
back to the regular library. The per-step planning (page table of the prompt,
tail length) is host work driven from ``TrtllmAttentionMetadata.prepare``,
which keeps the captured CUDA graph free of host synchronization.

The same prefix/tail split serves ``n``-sample requests (``beam_width == 1``),
whose children are independent requests that map the same leading prompt pages
when KV-cache block reuse lets them share the parent's blocks. Rows whose block
tables start with identical pages form a group; the group's common full pages
are read once for all of its queries and every row attends to its own remaining
pages (partial prompt block and generated tokens) with a paged decode call. The
two states are merged as above. An eager batch is declined (the regular
kernel runs) unless sharing pays: ``sharing_pays`` requires wide groups with a
long enough shared prefix (``TLLM_BEAM_SHARED_PREFIX_MIN_GROUP``, default 4;
``TLLM_BEAM_SHARED_PREFIX_MIN_SHARED_TOKENS``, default 5000), because for short
prompts the regular kernel is faster. Under CUDA graphs the same decision picks
between two graph variants per batch size (``resolve_graph_variant``, part of
the graph key).

Under CUDA graphs the n-sample path is captured for every generation batch
size of two or more rows. Each captured batch size owns fixed-address buffers:
the prefix call is planned for a fixed number of groups (up to
``TLLM_BEAM_SHARED_PREFIX_MAX_GROUPS`` sharing groups plus one placeholder
group for the rows without a shared prefix, padded with empty groups), the
tail call for one entry per row, and the per-step indices (KV write page and
slot, group order, placeholder mask, tail last-page lengths) live in fixed
device buffers. ``prepare`` refreshes them before every replay with a few
asynchronous host-to-device copies, and re-plans the FlashInfer work lists
only when the groups or a page count change. The captured kernels read
nothing else that changes between steps, so any batch of the captured size is
served: surplus sharing groups and rows without a shared prefix are attended
through the tail call alone.

Speculative decoding (``n``-sample path only): with a linear draft of K tokens
every generation row carries ``q_len = 1 + K`` tokens. The prefix call stacks the
``q_len`` queries of every row of a group against the shared pages (no mask:
every shared token precedes every query); the tail is a causal FlashInfer prefill
with ``q_len`` queries per row over the row's own pages; the K/V of the step's
tokens are written at ``kv_len - q_len + j``. Under the overlap scheduler the
host KV length can exceed the true (device) length by up to K, so the host page
tables only bound the work: the write positions and the tail's last-page
lengths come from the device KV lengths (the latter may be zero or negative,
which FlashInfer's unsigned length arithmetic handles). Rejected drafts need no
handling, their stale K/V is overwritten by the next step. The draft layers run
on the same metadata with the draft KV manager swapped in and are declined.
The check mode (``TLLM_BEAM_SHARED_PREFIX_CHECK``) covers single-token steps only.

Opt in with ``TLLM_FMHA_LIBS=+beam_shared_prefix``.
``TLLM_BEAM_SHARED_PREFIX_MAX_TAIL`` (default 8) bounds how many generated
positions a beam may carry under CUDA graphs; longer generations are rejected
loudly at planning time rather than truncated.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

    from .beam_shared_prefix_kernels import (
        merge_scatter,
        step_last_len,
        step_slots,
        write_kv_gather_q,
        write_kv_gather_q_spec,
    )

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
MAX_GROUPS_ENV = "TLLM_BEAM_SHARED_PREFIX_MAX_GROUPS"
DEFAULT_MAX_GROUPS = 64
"""Sharing groups the prefix call of a captured n-sample batch can hold (capped
at half the rows). The FlashInfer work list and its float workspace grow with
it; the rows of further groups are attended through the tail call alone."""
MIN_SHARED_ENV = "TLLM_BEAM_SHARED_PREFIX_MIN_SHARED_TOKENS"
DEFAULT_MIN_SHARED_TOKENS = 5000
"""Smallest shared prompt prefix (tokens per sharing row) for which the n-sample
library is used. Below it the two attention calls plus the merge cost more GPU
time than the single regular kernel saves (measured ~0.92-0.98x at 600-1600
tokens, ~1.02-1.19x at 4000, 1.5-1.6x at 8000 on GB200, Qwen3-1.7B)."""
SPEC_MIN_SHARED_ENV = "TLLM_BEAM_SHARED_PREFIX_SPEC_MIN_SHARED_TOKENS"
DEFAULT_SPEC_MIN_SHARED_TOKENS = 1 << 30
"""Smallest shared prefix for the n-sample library with speculative decoding
(``q_len`` > 1). The default keeps it off: with EAGLE-3 (K=3) on GB200 and
Qwen3-1.7B the library was not faster than the regular multi-token generation
kernel end to end in 16 of 17 measured cells (0.77-1.00x; 1.04x at 16000 prompt
tokens, n=20), although its GPU time per step was up to 7% lower at 8000 tokens
in a short decode window: the regular kernel already reads the shared pages
from L2, the prefix call carries ``q_len`` times more queries, and the tail
(generated tokens) is read through the slower FlashInfer prefill kernel. With
the library the mean accepted length was also 4-6% lower in single-prompt
n=20 runs (open item, the attention output itself matches a dense reference to
0.5%). Set the variable to opt in."""
MIN_GROUP_ENV = "TLLM_BEAM_SHARED_PREFIX_MIN_GROUP"
DEFAULT_MIN_GROUP = 4
"""Smallest number of rows sharing a prefix for the n-sample library to be used;
reading the prompt once instead of ``n`` times only pays for wide groups."""
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

    __slots__ = ("num_heads", "num_kv_heads", "head_dim", "dtype", "sm_scale", "q_len")

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        sm_scale: float,
        q_len: int = 1,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.sm_scale = sm_scale
        self.q_len = q_len
        """Query tokens per generation row (1 + draft tokens under speculative decoding)."""

    def _tuple(self):
        return (
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.dtype,
            self.sm_scale,
            self.q_len,
        )

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
    if metadata.is_cuda_graph and not getattr(metadata, "shared_prefix_graph", True):
        # The graph variant captured with the regular kernel (see resolve_graph_variant).
        return
    sampling_state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    if sampling_state is not None:
        sampling_state.plan = None
    if sampling_metadata_eligible(metadata):
        _plan_sampling_step(metadata)
        return
    if sampling_state is not None and sampling_state.captured:
        # A graph captured with the n-sample path would replay stale buffers.
        raise RuntimeError(
            f"{LIB_NAME}: CUDA graph metadata captured with the n-sample path no longer "
            "satisfies its preconditions."
        )
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
class _Grouping:
    """Row grouping of one batch composition and the page tables derived from it.

    Rows whose block tables start with the same full pages form a sharing group
    that reads those pages once (``prefix``); every other row joins a single
    placeholder group whose prefix call runs over one token and is masked out
    of the merge. Prefix and tail tables are in ``perm`` order (the rows of a
    group are contiguous there); the KV write page is in batch order.

    A grouping stays valid while the batch keeps the same rows, each with the
    same first page and the same number of full pages before its current
    token: block tables only grow, so nothing it holds can change otherwise.
    Between two page boundaries only the last-page lengths and the write slots
    move, and those are derived per step from the KV lengths. When rows cross a
    page boundary, the groups and the prefix tables stay valid unless a
    group's shared run was cut by its members' full pages (``capped_keys``)
    and that limit moved; otherwise only the tail tables are rebuilt
    (``_refresh_tail``) and ``prefix_version`` stays.

    Host arrays are numpy (the planner runs every step and numpy's per-call
    cost is a fraction of torch's); the FlashInfer plan inputs are zero-copy
    torch views of them.
    """

    version: int
    """Changes whenever any table changes."""
    prefix_version: int
    """Changes whenever the groups or the prefix tables change."""
    num_rows: int
    table_rows: np.ndarray
    """[rows] int64: host block-table row of each batch row."""
    first_pages: np.ndarray
    """[rows] int64: first pool page of each row (cache validation)."""
    private_from: np.ndarray
    """[rows] int64: index of the page holding the row's first token of this step
    (the current token when ``q_len`` is 1). Pages before it may be shared."""
    q_len: int
    """Query tokens per row: 1, or 1 + draft tokens under speculative decoding."""
    last_pages: np.ndarray
    """[rows] int64: index of the page holding the row's last token of this step."""
    num_groups: int
    """Groups handed to the prefix call, placeholder and padding groups included."""
    num_sharing: int
    """Groups that share at least one page."""
    num_placeholder_rows: int
    shared_tokens: int
    """Prompt tokens read once per group instead of once per row, summed over rows."""
    perm: np.ndarray
    """[rows] int64: batch row of each grouped position."""
    is_identity: bool
    key_of_row: np.ndarray
    """[rows] int64: first-page key of each row."""
    capped_keys: np.ndarray
    """Keys of multi-row first pages whose members also agree on the page after
    their shared run, so the run may grow with their smallest full-page count."""
    key_caps: np.ndarray
    """[keys] int64: smallest full-page count of each key's rows."""
    shared_g: np.ndarray
    """[rows] int64, grouped order: shared pages of the row's group."""
    max_group_rows: int
    qo_indptr: torch.Tensor
    """[num_groups + 1] int32 host."""
    prefix_indptr: torch.Tensor
    prefix_indices: torch.Tensor
    prefix_last_page_len: torch.Tensor
    prefix_empty: np.ndarray
    """[rows] bool, grouped order: the row sits in the placeholder group."""
    blocks_g: np.ndarray
    """[rows] int64, grouped order: pages up to and including the current token."""
    tail_indptr: torch.Tensor
    tail_indices: torch.Tensor
    write_page: np.ndarray
    """[rows] int64, batch order: pool page of the current token."""
    base_len: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    """[rows] int32, grouped order: first position of the row's last host page,
    ``(blocks - 1) * tokens_per_block``. The device KV length minus it is the
    last page length the tail call reads (zero or negative when the host
    overestimated the length by a whole page)."""
    check_pages: Optional[torch.Tensor] = None
    """[rows, pages] int64 full page tables; only kept in check mode."""


def _indptr(counts: np.ndarray) -> torch.Tensor:
    indptr = np.zeros(counts.size + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    return torch.from_numpy(indptr)


def _build_grouping(
    version: int,
    table_rows: np.ndarray,
    pages_all: np.ndarray,
    kv_lens: np.ndarray,
    tokens_per_block: int,
    *,
    max_sharing_groups: Optional[int],
    padded_groups: Optional[int],
    keep_check_pages: bool,
    q_len: int = 1,
) -> _Grouping:
    """Group rows by common leading pages and build the prefix and tail tables.

    Two rows share a prefix when their block tables start with the same pool
    page. A group's shared pages are the longest common run of full pages that
    lie entirely before every member's current token; everything after that
    run, the current token's page included, stays private to the row.

    Args:
        pages_all: [rows, pages] int64 page tables covering every row's pages.
        kv_lens: [rows] int64, >= q_len; the KV length including this step's
            ``q_len`` tokens (the host value; under speculative decoding it may
            exceed the true length by up to the draft length, so every page the
            true length can need must be in ``pages_all``).
        q_len: query tokens per row.
        max_sharing_groups: keep at most this many sharing groups (those that
            save the most page reads; ties go to the lower first page); the
            rows of the others join the placeholder group. None keeps all.
        padded_groups: with a value, the prefix call always gets exactly this
            many groups: the sharing groups, the placeholder group (possibly
            without rows) and empty padding groups. Without one, the
            placeholder group exists only when it has rows.
    """
    i64 = np.int64
    pages_all = np.asarray(pages_all, dtype=i64)
    kv_lens = np.asarray(kv_lens, dtype=i64)
    table_rows = np.asarray(table_rows, dtype=i64)
    rows, width = pages_all.shape
    row_ids = np.arange(rows, dtype=i64)
    num_blocks = (kv_lens + tokens_per_block - 1) // tokens_per_block
    last_pages = num_blocks - 1
    private_from = np.maximum(kv_lens - q_len, 0) // tokens_per_block

    first_pages = pages_all[:, 0].copy()
    _, leader, key_of_row, members = np.unique(
        first_pages, return_index=True, return_inverse=True, return_counts=True
    )
    key_of_row = key_of_row.reshape(-1)
    num_keys = leader.size
    cap = np.full(num_keys, width, dtype=i64)
    np.minimum.at(cap, key_of_row, private_from)
    multi = members > 1
    shared = np.zeros(num_keys, dtype=i64)
    capped_keys = np.zeros(0, dtype=i64)
    if multi.any():
        # Compare one page past the cap as well (the current token's page,
        # which every row has): a run that also covers it may grow with the cap.
        span = int(cap[multi].max()) + 1
        cols = pages_all[:, :span]
        same = cols == cols[leader[key_of_row]]
        same &= np.arange(span, dtype=i64) <= cap[key_of_row][:, None]
        # Length of each row's leading run of pages equal to its group leader's.
        run = np.where(same.all(axis=1), span, np.argmin(same, axis=1))
        common = np.full(num_keys, span, dtype=i64)
        np.minimum.at(common, key_of_row, run)
        shared = np.where(multi, np.minimum(common, cap), 0)
        capped_keys = np.flatnonzero(multi & (common > cap))

    sharing = np.flatnonzero(shared > 0)
    if max_sharing_groups is not None and sharing.size > max_sharing_groups:
        saved = shared[sharing] * (members[sharing] - 1)
        sharing = sharing[np.argsort(-saved, kind="stable")[:max_sharing_groups]]
    # Sharing groups in order of their first row; the placeholder group last.
    sharing = sharing[np.argsort(leader[sharing], kind="stable")]
    num_sharing = int(sharing.size)
    rank = np.full(num_keys, num_sharing, dtype=i64)
    rank[sharing] = np.arange(num_sharing, dtype=i64)
    row_group = rank[key_of_row]
    in_sharing = row_group < num_sharing
    row_shared = np.where(in_sharing, shared[key_of_row], 0)
    perm = np.argsort(row_group, kind="stable")
    is_identity = bool((perm == row_ids).all())

    group_sizes = members[sharing]
    group_shared = shared[sharing]
    num_placeholder_rows = rows - int(group_sizes.sum())
    if num_sharing:
        lead_pages = pages_all[leader[sharing], : int(group_shared.max())]
        sharing_pages = lead_pages[np.arange(lead_pages.shape[1]) < group_shared[:, None]]
    else:
        sharing_pages = np.zeros(0, dtype=i64)
    # Any valid page serves the placeholder: its output is masked out.
    placeholder_page = (
        int(first_pages[perm[rows - num_placeholder_rows]])
        if num_placeholder_rows
        else int(first_pages[0])
    )
    if padded_groups is None:
        extra = 1 if num_placeholder_rows else 0
        sizes = np.concatenate([group_sizes, np.full(extra, num_placeholder_rows, dtype=i64)])
    else:
        extra = padded_groups - num_sharing
        if extra < 1:
            raise RuntimeError(
                f"{LIB_NAME}: {num_sharing} sharing groups leave no room for the placeholder "
                f"group in {padded_groups} padded groups."
            )
        sizes = np.concatenate(
            [group_sizes, [num_placeholder_rows], np.zeros(extra - 1, dtype=i64)]
        )
    num_groups = num_sharing + extra
    prefix_indices = np.concatenate([sharing_pages, np.full(extra, placeholder_page, dtype=i64)])
    prefix_last_page_len = np.concatenate(
        [np.full(num_sharing, tokens_per_block, dtype=np.int32), np.ones(extra, dtype=np.int32)]
    )

    blocks_g = num_blocks[perm]
    shared_g = row_shared[perm]
    pages_g = pages_all[perm]
    block_ids = np.arange(width, dtype=i64)
    in_tail = (block_ids >= shared_g[:, None]) & (block_ids < blocks_g[:, None])

    return _Grouping(
        version=version,
        prefix_version=version,
        num_rows=rows,
        table_rows=table_rows,
        first_pages=first_pages,
        private_from=private_from,
        num_groups=num_groups,
        num_sharing=num_sharing,
        num_placeholder_rows=num_placeholder_rows,
        shared_tokens=int((group_shared * group_sizes).sum()) * tokens_per_block,
        perm=perm,
        is_identity=is_identity,
        key_of_row=key_of_row,
        capped_keys=capped_keys,
        key_caps=cap,
        shared_g=shared_g,
        max_group_rows=int(sizes.max()) if sizes.size else 0,
        qo_indptr=_indptr(sizes * q_len),
        prefix_indptr=_indptr(np.concatenate([group_shared, np.ones(extra, dtype=i64)])),
        prefix_indices=torch.from_numpy(prefix_indices.astype(np.int32)),
        prefix_last_page_len=torch.from_numpy(prefix_last_page_len),
        prefix_empty=~in_sharing[perm],
        blocks_g=blocks_g,
        tail_indptr=_indptr(blocks_g - shared_g),
        tail_indices=torch.from_numpy(pages_g[in_tail].astype(np.int32)),
        write_page=pages_all[row_ids, private_from],
        q_len=q_len,
        last_pages=last_pages,
        base_len=((blocks_g - 1) * tokens_per_block).astype(np.int32),
        check_pages=torch.from_numpy(pages_all) if keep_check_pages else None,
    )


def _refresh_tail(
    grouping: _Grouping,
    version: int,
    pages_all: np.ndarray,
    kv_lens: np.ndarray,
    tokens_per_block: int,
    *,
    keep_check_pages: bool,
) -> Optional[_Grouping]:
    """Rebuild only the tail tables of ``grouping`` after rows crossed a page
    boundary, or return None when the groups themselves may have changed."""
    q_len = grouping.q_len
    private_from = np.maximum(kv_lens - q_len, 0) // tokens_per_block
    last_pages = (kv_lens - 1) // tokens_per_block
    if grouping.capped_keys.size:
        caps = grouping.key_caps.copy()
        caps[grouping.capped_keys] = pages_all.shape[1]
        np.minimum.at(caps, grouping.key_of_row, private_from)
        if not np.array_equal(caps[grouping.capped_keys], grouping.key_caps[grouping.capped_keys]):
            return None
    perm = grouping.perm
    blocks_g = last_pages[perm] + 1
    shared_g = grouping.shared_g
    in_tail = (np.arange(pages_all.shape[1]) >= shared_g[:, None]) & (
        np.arange(pages_all.shape[1]) < blocks_g[:, None]
    )
    return dataclasses.replace(
        grouping,
        version=version,
        private_from=private_from,
        last_pages=last_pages,
        blocks_g=blocks_g,
        tail_indptr=_indptr(blocks_g - shared_g),
        tail_indices=torch.from_numpy(pages_all[perm][in_tail].astype(np.int32)),
        write_page=pages_all[np.arange(pages_all.shape[0]), private_from],
        base_len=((blocks_g - 1) * tokens_per_block).astype(np.int32),
        check_pages=torch.from_numpy(pages_all) if keep_check_pages else None,
    )


@dataclass
class _SamplingPlan:
    """Host-side description of one n-sample generation step, shared by all layers.

    Rows are generation rows. Prefix and tail tensors are in ``perm`` order (the
    rows of a group are contiguous there), so one permuted copy of the queries
    serves both attention calls; the KV write indices are in batch order.
    """

    step: int
    grouping: _Grouping
    kv_lens: np.ndarray
    """[rows] int64 (>= 1), batch order."""
    tokens_per_block: int
    check_kv_lens: Optional[torch.Tensor] = None

    @property
    def tail_last_page_len_np(self) -> np.ndarray:
        """[rows] int32, grouped order: filled length of each row's last page."""
        last = (self.kv_lens[self.grouping.perm] - 1) % self.tokens_per_block + 1
        return last.astype(np.int32)

    @property
    def write_slot_np(self) -> np.ndarray:
        """[rows] int64, batch order: slot of the current token inside its page."""
        return (self.kv_lens - 1) % self.tokens_per_block

    @property
    def tail_last_page_len(self) -> torch.Tensor:
        return torch.from_numpy(self.tail_last_page_len_np)

    @property
    def write_slot(self) -> torch.Tensor:
        return torch.from_numpy(self.write_slot_np)

    @property
    def num_rows(self) -> int:
        return self.grouping.num_rows

    @property
    def q_len(self) -> int:
        return self.grouping.q_len

    @property
    def num_groups(self) -> int:
        return self.grouping.num_groups

    @property
    def shared_tokens(self) -> int:
        return self.grouping.shared_tokens

    @property
    def perm(self) -> Optional[torch.Tensor]:
        """None when the batch is already in grouped order."""
        return None if self.grouping.is_identity else torch.from_numpy(self.grouping.perm)

    @property
    def qo_indptr(self) -> torch.Tensor:
        return self.grouping.qo_indptr

    @property
    def prefix_indptr(self) -> torch.Tensor:
        return self.grouping.prefix_indptr

    @property
    def prefix_indices(self) -> torch.Tensor:
        return self.grouping.prefix_indices

    @property
    def prefix_last_page_len(self) -> torch.Tensor:
        return self.grouping.prefix_last_page_len

    @property
    def prefix_empty(self) -> torch.Tensor:
        return torch.from_numpy(self.grouping.prefix_empty)

    @property
    def has_empty(self) -> bool:
        return self.grouping.num_placeholder_rows > 0

    @property
    def tail_indptr(self) -> torch.Tensor:
        return self.grouping.tail_indptr

    @property
    def tail_indices(self) -> torch.Tensor:
        return self.grouping.tail_indices

    @property
    def write_page(self) -> torch.Tensor:
        return torch.from_numpy(self.grouping.write_page)

    @property
    def check_pages(self) -> Optional[torch.Tensor]:
        return self.grouping.check_pages


@dataclass
class _SamplingDevice:
    """Device copies of an eager step's index tensors, made by the first layer."""

    step: int
    perm: Optional[torch.Tensor]
    prefix_empty: Optional[torch.Tensor]
    """[rows] bool in grouped order, None when every group has a shared prefix."""
    write_page: Optional[torch.Tensor]
    write_slot: Optional[torch.Tensor]
    """Host-derived write indices of single-token steps (None with draft tokens,
    whose positions come from the device KV lengths)."""
    shared: Optional[torch.Tensor] = None
    """[rows] int64, grouped order: shared pages of each row (draft-token steps)."""
    last_len: Optional[torch.Tensor] = None
    """[rows] int32, grouped order: last-page length of each row's tail, from the
    device KV lengths (draft-token steps)."""


class _PlanWorkspace:
    """FlashInfer integer workspace shared by every wrapper of one role and shape.

    ``plan`` fills the page-locked buffer on the host and copies it to the
    device buffer asynchronously, and ``run`` reads the device buffer. All
    wrappers of a role run on one stream, one step at a time, so they can share
    both buffers as long as (a) a wrapper re-plans whenever another one planned
    into the buffer after it (``owner``) and (b) the host does not overwrite the
    page-locked buffer before the previous copy out of it executed (``copied``).
    """

    _INT_WORKSPACE_BYTES = 8 * 1024 * 1024

    def __init__(self, device: torch.device):
        self.int_buffer = torch.empty(self._INT_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.pinned_buffer = torch.empty(
            self._INT_WORKSPACE_BYTES, dtype=torch.uint8, pin_memory=prefer_pinned()
        )
        self.copied = torch.cuda.Event()
        self.pending = False
        self.owner = 0

    def bind(self, wrapper: Any) -> None:
        # FlashInfer 0.6 keeps the integer workspace in these two attributes;
        # reset_workspace_buffer() would allocate a fresh page-locked buffer.
        for attr in ("_int_workspace_buffer", "_pin_memory_int_workspace_buffer"):
            if not hasattr(wrapper, attr):
                raise RuntimeError(f"{LIB_NAME}: unexpected FlashInfer wrapper without {attr}.")
        wrapper._int_workspace_buffer = self.int_buffer
        wrapper._pin_memory_int_workspace_buffer = self.pinned_buffer

    def before_plan(self) -> None:
        if self.pending:
            self.copied.synchronize()

    def after_plan(self) -> int:
        self.copied.record()
        self.pending = True
        self.owner = next(_PLAN_TOKENS)
        return self.owner


_PLAN_TOKENS = itertools.count(1)
_plan_workspaces: Dict[tuple, _PlanWorkspace] = {}


def _plan_workspace(device: torch.device, role: str, key: _PlanKey) -> _PlanWorkspace:
    workspace = _plan_workspaces.get((device, role, key))
    if workspace is None:
        workspace = _PlanWorkspace(device)
        _plan_workspaces[(device, role, key)] = workspace
    return workspace


_graph_float_workspace: Dict[torch.device, torch.Tensor] = {}


def _get_graph_float_workspace(device: torch.device, nbytes: int) -> torch.Tensor:
    """Float workspace of the CUDA graph wrappers, at least ``nbytes`` large.

    Under CUDA graphs FlashInfer always splits the KV and reserves partial
    results for its padded work list, so the requirement grows with the
    captured batch. A larger request allocates a new buffer for the wrappers
    created from then on; earlier wrappers (and the graphs captured with them)
    keep the buffer they were created with.
    """
    workspace = _graph_float_workspace.get(device)
    if workspace is None or workspace.numel() < nbytes:
        workspace = torch.empty(
            max(nbytes, _FLOAT_WORKSPACE_BYTES), dtype=torch.uint8, device=device
        )
        _graph_float_workspace[device] = workspace
    return workspace


def _prefill_split_bytes(key: _PlanKey, total_rows: int, batch_size: int, num_sms: int) -> int:
    """Upper bound of the float workspace a CUDA-graph FA2 prefill plan reserves.

    Mirrors FlashInfer's ``PrefillSplitQOKVIndptr`` under CUDA graphs: the query
    tile follows from the longest possible request (``total_rows - batch_size
    + 1`` rows), the work list is padded to ``max(2 * SMs / kv_heads, tiles)``,
    and every padded entry reserves a ``tile x head_dim`` fp32 partial output
    plus its LSE per query head.
    """
    gqa = key.num_heads // key.num_kv_heads
    max_qo = (total_rows - batch_size + 1) * gqa
    tile = 128 if max_qo > 64 else (64 if max_qo > 16 else 16)
    tiles = -(-total_rows * gqa // tile) + batch_size - 1
    padded = max((2 * num_sms) // key.num_kv_heads, tiles)
    return key.num_heads * padded * tile * (key.head_dim + 1) * 4


def _graph_float_workspace_bytes(
    key: _PlanKey, rows: int, groups: int, device: torch.device
) -> int:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    prefix = _prefill_split_bytes(key, rows * key.q_len, groups, num_sms)
    if key.q_len > 1:
        # Speculative decoding: the tail is a causal prefill with q_len queries per row.
        tail = _prefill_split_bytes(key, rows * key.q_len, rows, num_sms)
    elif key.num_heads != key.num_kv_heads:
        tail = _prefill_split_bytes(key, rows, rows, num_sms)
    else:
        # Decode kernel: one fp32 partial row per query head and padded work
        # item, the work list bounded by the resident CTAs of the device.
        tail = key.num_heads * max(rows, 32 * num_sms) * (key.head_dim + 1) * 4
    return max(prefix, tail) + (1 << 20)


_AUX_INT32 = 3
"""Per-row int32 vectors at the head of the page section of ``_GraphBuffers``:
tail last-page lengths, ``base_len`` and shared page counts."""


@dataclass
class _GraphBuffers:
    """Fixed-address step inputs of one captured batch size.

    A page-locked staging buffer feeds the device buffer that the captured
    graph reads with one asynchronous copy per step; ``prepare`` issues it, so
    it precedes the replay on the stream. The buffer is int64: the write page,
    write slot and group order (``rows`` each, the index dtype of the
    gather/scatter kernels), followed by an int32 view (FlashInfer's index
    dtype) holding the tail last-page lengths and the prefix and tail page
    indices of a regrouping step, which the wrappers' plans copy into their
    own buffers.
    """

    rows: int
    num_groups: int
    prefix_capacity: int
    tail_capacity: int
    staging: torch.Tensor
    device: torch.Tensor
    rows_staging: torch.Tensor
    rows_device: torch.Tensor
    pages_staging: torch.Tensor
    pages_device: torch.Tensor
    empty_staging: torch.Tensor
    empty: torch.Tensor
    """[rows] bool, grouped order: the row's prefix state is a placeholder."""
    copied: torch.cuda.Event
    pending: bool = False
    uploaded_version: int = -1
    uploaded_prefix_version: int = -1
    prefix_slice: tuple = (0, 0)
    tail_slice: tuple = (0, 0)

    @classmethod
    def create(
        cls, rows: int, num_groups: int, max_pages: int, device: torch.device
    ) -> "_GraphBuffers":
        prefix_capacity = (num_groups - 1) * max_pages + num_groups
        tail_capacity = rows * max_pages
        pinned = prefer_pinned()
        size = 3 * rows + (_AUX_INT32 * rows + prefix_capacity + tail_capacity + 1) // 2
        staging = torch.zeros((size,), dtype=torch.int64, pin_memory=pinned)
        device_buffer = torch.zeros((size,), dtype=torch.int64, device=device)
        return cls(
            rows=rows,
            num_groups=num_groups,
            prefix_capacity=prefix_capacity,
            tail_capacity=tail_capacity,
            staging=staging,
            device=device_buffer,
            rows_staging=staging[: 3 * rows],
            rows_device=device_buffer[: 3 * rows],
            pages_staging=staging[3 * rows :].view(torch.int32),
            pages_device=device_buffer[3 * rows :].view(torch.int32),
            empty_staging=torch.zeros((rows,), dtype=torch.bool, pin_memory=pinned),
            empty=torch.zeros((rows,), dtype=torch.bool, device=device),
            copied=torch.cuda.Event(),
        )

    @property
    def write_page(self) -> torch.Tensor:
        return self.rows_device[: self.rows]

    @property
    def write_slot(self) -> torch.Tensor:
        return self.rows_device[self.rows : 2 * self.rows]

    @property
    def perm(self) -> torch.Tensor:
        return self.rows_device[2 * self.rows :]

    @property
    def tail_last_page_len(self) -> torch.Tensor:
        return self.pages_device[: self.rows]

    @property
    def base_len(self) -> torch.Tensor:
        return self.pages_device[self.rows : 2 * self.rows]

    @property
    def shared(self) -> torch.Tensor:
        return self.pages_device[2 * self.rows : 3 * self.rows]

    @property
    def prefix_indices(self) -> torch.Tensor:
        return self.pages_device[self.prefix_slice[0] : self.prefix_slice[1]]

    @property
    def tail_indices(self) -> torch.Tensor:
        return self.pages_device[self.tail_slice[0] : self.tail_slice[1]]

    def upload(self, plan: _SamplingPlan) -> None:
        """Stage the step's indices and copy them to the device buffers."""
        grouping = plan.grouping
        rows = self.rows
        if self.pending:
            # The previous step's copies out of the staging buffers must have run.
            self.copied.synchronize()
        row_staging = self.rows_staging.numpy()
        page_staging = self.pages_staging.numpy()
        row_staging[:rows] = grouping.write_page
        row_staging[rows : 2 * rows] = plan.write_slot_np
        page_staging[:rows] = plan.tail_last_page_len_np
        count = _AUX_INT32 * rows
        if grouping.version != self.uploaded_version:
            page_staging[rows : 2 * rows] = grouping.base_len
            page_staging[2 * rows : 3 * rows] = grouping.shared_g
            num_prefix = grouping.prefix_indices.numel()
            num_tail = grouping.tail_indices.numel()
            if num_prefix > self.prefix_capacity or num_tail > self.tail_capacity:
                raise RuntimeError(
                    f"{LIB_NAME}: step needs {num_prefix} prefix / {num_tail} tail pages but the "
                    f"CUDA graph buffers hold {self.prefix_capacity} / {self.tail_capacity}."
                )
            page_staging[count : count + num_prefix] = grouping.prefix_indices.numpy()
            self.prefix_slice = (count, count + num_prefix)
            count += num_prefix
            page_staging[count : count + num_tail] = grouping.tail_indices.numpy()
            self.tail_slice = (count, count + num_tail)
            count += num_tail
        if grouping.prefix_version != self.uploaded_prefix_version:
            row_staging[2 * rows :] = grouping.perm
            self.empty_staging.numpy()[:] = grouping.prefix_empty
            self.empty.copy_(self.empty_staging, non_blocking=True)
        used = 3 * rows + (count + 1) // 2
        self.device[:used].copy_(self.staging[:used], non_blocking=True)
        self.copied.record()
        self.pending = True
        self.uploaded_version = grouping.version
        self.uploaded_prefix_version = grouping.prefix_version


@dataclass
class _SamplingState:
    pool_index: int
    page_scale: int
    page_div: int
    num_pages: int
    manager_ref: Optional[weakref.ref] = None
    """The KV-cache manager the plan describes. The draft layers of one-model
    speculative decoding run on this metadata with the draft manager swapped in;
    they must not use the target's plan."""
    graph: Optional[_GraphBuffers] = None
    """Set for the metadata of a captured batch size."""
    plan: Optional[_SamplingPlan] = None
    grouping: Optional[_Grouping] = None
    table_rows_key: Optional[tuple] = None
    table_rows: Optional[np.ndarray] = None
    prefix_wrappers: Dict[_PlanKey, Any] = field(default_factory=dict)
    tail_wrappers: Dict[_PlanKey, Any] = field(default_factory=dict)
    planned_step: Dict[_PlanKey, int] = field(default_factory=dict)
    planned_tokens: Dict[_PlanKey, tuple] = field(default_factory=dict)
    """Graph wrappers: (prefix version, prefix owner token, grouping version,
    tail owner token) of their last plans, to skip re-planning while nothing
    changed."""
    device: Optional[_SamplingDevice] = None
    step_counter: int = 0
    grouping_counter: int = 0
    captured: bool = False
    slots_step: int = -1
    """Step whose slot indices ``step_slots`` last wrote (CUDA graph batches)."""
    slots_captured: bool = False
    logged: bool = False
    ran: bool = False


def _max_groups_from_env() -> int:
    value = os.environ.get(MAX_GROUPS_ENV)
    if value is None or not value.strip():
        return DEFAULT_MAX_GROUPS
    groups = int(value)
    if groups < 1:
        raise ValueError(f"{MAX_GROUPS_ENV} must be >= 1, got {groups}.")
    return groups


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def sharing_pays(
    num_rows: int,
    sharing_rows: int,
    num_sharing_groups: int,
    shared_tokens_per_row: float,
    q_len: int = 1,
) -> bool:
    """Size gate of the n-sample path: whether reading the shared prompt once pays.

    ``sharing_rows`` rows form ``num_sharing_groups`` groups whose common prefix
    is ``shared_tokens_per_row`` tokens long on average. The library is used
    when at least half the rows share, the groups are wide enough, and the
    prefix is long enough (see ``MIN_SHARED_ENV`` / ``MIN_GROUP_ENV``); otherwise
    the regular generation kernel is faster.
    """
    if num_sharing_groups <= 0 or sharing_rows <= 0:
        return False
    return (
        2 * sharing_rows >= num_rows
        and sharing_rows >= _int_env(MIN_GROUP_ENV, DEFAULT_MIN_GROUP) * num_sharing_groups
        and shared_tokens_per_row
        >= (
            _int_env(MIN_SHARED_ENV, DEFAULT_MIN_SHARED_TOKENS)
            if q_len == 1
            else _int_env(SPEC_MIN_SHARED_ENV, DEFAULT_SPEC_MIN_SHARED_TOKENS)
        )
    )


_variant_memo: list = [None, None, None, 0, 0]
"""Last decision of ``resolve_graph_variant``: first request, last request,
number of requests, gate settings, variant. Consecutive steps of one batch hit
it, which keeps the per-step host cost of the decision negligible."""


def resolve_graph_variant(batch) -> int:
    """CUDA graph variant for a generation batch: 1 uses the library, 0 the regular kernel.

    Part of the graph key, because a captured graph records one kernel sequence.
    Decided from the requests alone (children of one n-sample request share its
    prompt), before the batch is planned. Either variant is correct for any
    batch; this only picks the faster one. A batch that keeps the same first and
    last request and size as the previous call reuses its decision.
    """
    requests = batch.generation_requests
    if not requests:
        return 0
    q_len = 1 + len(getattr(requests[0], "py_draft_tokens", None) or ())
    gate = (
        _int_env(MIN_SHARED_ENV, DEFAULT_MIN_SHARED_TOKENS),
        _int_env(MIN_GROUP_ENV, DEFAULT_MIN_GROUP),
        _int_env(SPEC_MIN_SHARED_ENV, DEFAULT_SPEC_MIN_SHARED_TOKENS),
        q_len,
    )
    memo = _variant_memo
    if (
        memo[0] is requests[0]
        and memo[1] is requests[-1]
        and memo[2] == len(requests)
        and memo[3] == gate
    ):
        return memo[4]
    groups: Dict[int, list] = {}
    for request in requests:
        info = getattr(request, "_shared_prefix_info", None)
        if info is None:
            owner = request.parent_request_id if request.is_child else request.py_request_id
            info = (owner, request.py_prompt_len)
            request._shared_prefix_info = info
        entry = groups.get(info[0])
        if entry is None:
            groups[info[0]] = [1, info[1]]
        else:
            entry[0] += 1
    rows = len(requests)
    wide = [entry for entry in groups.values() if entry[0] > 1]
    sharing_rows = sum(entry[0] for entry in wide)
    tokens = sum(entry[0] * entry[1] for entry in wide)
    per_row = tokens / sharing_rows if sharing_rows else 0.0
    variant = int(sharing_pays(rows, sharing_rows, len(wide), per_row, q_len))
    _variant_memo[:] = [requests[0], requests[-1], len(requests), gate, variant]
    return variant


def sampling_metadata_eligible(metadata: "TrtllmAttentionMetadata") -> bool:
    """Batch-level preconditions of the n-sample path.

    They depend only on the batch shape and the model configuration, so they
    hold on every replay of a CUDA graph that was captured with this path.
    """
    return (
        IS_FLASHINFER_AVAILABLE
        and metadata.beam_width == 1
        and metadata.num_generations > 1
        and metadata.num_contexts == 0
        and not metadata.is_cross
        and not metadata.enable_helix
        # use_spec_decoding is not tested: the model forward leaves it set after
        # every step, so it says nothing about the next one. A tree draft is
        # declined, a linear draft is the q_len > 1 path.
        and not metadata.is_spec_decoding_enabled
        and not metadata.is_spec_dec_tree
        and not metadata.is_spec_dec_dynamic_tree
        and metadata.kv_cache_block_offsets is not None
        and metadata.kv_cache_params is not None
        and metadata.kv_cache_params.use_cache
        and _kv_manager_supported(metadata.kv_cache_manager)
    )


def _sampling_q_len(metadata: "TrtllmAttentionMetadata", rows: int) -> int:
    """Query tokens per generation row of a context-free batch, or 0 when the
    rows do not all carry the same number (the batch is declined then).

    Speculative decoding gives every generation request ``1 + draft tokens``
    tokens; the executor keeps that count uniform across a batch. Eager batches
    are checked against the per-request lengths, CUDA graph batches are uniform
    by construction.
    """
    tokens = metadata.num_tokens
    if rows <= 0 or tokens % rows:
        return 0
    q_len = tokens // rows
    if q_len > 1 and not metadata.is_cuda_graph:
        seq_lens = metadata.seq_lens
        if seq_lens is None or not bool((seq_lens[:rows] == q_len).all()):
            return 0
    return q_len


def _sampling_table_rows(
    state: _SamplingState, metadata: "TrtllmAttentionMetadata", rows: int
) -> np.ndarray:
    """``_table_rows`` cached while the batch keeps the same requests."""
    if _is_v2_manager(metadata.kv_cache_manager):
        request_ids = metadata.request_ids
        cache_key = tuple(request_ids[:rows]) if request_ids is not None else None
    else:
        cache_key = (rows,)
    if cache_key is None or cache_key != state.table_rows_key:
        state.table_rows = _table_rows(metadata, rows).numpy()
        state.table_rows_key = cache_key
    assert state.table_rows is not None
    return state.table_rows


def _plan_sampling_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Plan an n-sample generation step: group the rows and derive the page tables.

    Eager batches without any sharing group are declined (the regular kernel
    runs). A CUDA graph batch is always planned, because its graph is captured
    with this path: rows without a shared prefix take the placeholder group.
    """
    state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    rows = metadata.num_generations
    tokens_per_block = int(metadata.tokens_per_block)
    host_offsets = metadata.host_kv_cache_block_offsets
    assert host_offsets is not None
    q_len = _sampling_q_len(metadata, rows)
    if q_len == 0:
        return
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
            manager_ref=weakref.ref(kv_cache_manager),
        )
        if metadata.is_cuda_graph:
            num_groups = min(_max_groups_from_env(), rows // 2) + 1
            state.graph = _GraphBuffers.create(
                rows, num_groups, int(host_offsets.size(-1)), metadata.kv_cache_block_offsets.device
            )
        metadata.fmha_plan_caches[SAMPLING_KEY] = state
    graph = state.graph
    if graph is not None and rows != graph.rows:
        raise RuntimeError(
            f"{LIB_NAME}: CUDA graph metadata planned for {graph.rows} rows got {rows}."
        )
    state.step_counter += 1

    check = _check_enabled()
    kv_lens = np.maximum(metadata.kv_lens_runtime[:rows].numpy(), q_len).astype(np.int64)
    table_rows = _sampling_table_rows(state, metadata, rows)
    # Pages that may be shared end before the row's first token of this step; the
    # row's own pages run to the page of its last token (one page further when
    # the step's tokens cross a page boundary).
    private_from = (kv_lens - q_len) // tokens_per_block
    last_pages = (kv_lens - 1) // tokens_per_block
    offsets = host_offsets.numpy()[state.pool_index, :, 0, :]
    first_pages = np.clip(
        offsets[table_rows, 0].astype(np.int64) * state.page_scale // state.page_div,
        0,
        state.num_pages - 1,
    )
    grouping = state.grouping
    same_rows = (
        grouping is not None
        and grouping.num_rows == rows
        and grouping.q_len == q_len
        and np.array_equal(grouping.table_rows, table_rows)
        and np.array_equal(grouping.first_pages, first_pages)
    )
    if (
        not same_rows
        or not np.array_equal(grouping.private_from, private_from)
        or (q_len > 1 and not np.array_equal(grouping.last_pages, last_pages))
        or (check and grouping.check_pages is None)
    ):
        width = int(last_pages.max()) + 1
        pages_all = np.clip(
            offsets[table_rows, :width].astype(np.int64) * state.page_scale // state.page_div,
            0,
            state.num_pages - 1,
        )
        state.grouping_counter += 1
        refreshed = None
        if same_rows:
            refreshed = _refresh_tail(
                grouping,
                state.grouping_counter,
                pages_all,
                kv_lens,
                tokens_per_block,
                keep_check_pages=check,
            )
        grouping = refreshed or _build_grouping(
            state.grouping_counter,
            table_rows,
            pages_all,
            kv_lens,
            tokens_per_block,
            max_sharing_groups=None if graph is None else graph.num_groups - 1,
            padded_groups=None if graph is None else graph.num_groups,
            keep_check_pages=check,
            q_len=q_len,
        )
        state.grouping = grouping
    if graph is None:
        sharing_rows = rows - grouping.num_placeholder_rows
        if not sharing_pays(
            rows,
            sharing_rows,
            grouping.num_sharing,
            grouping.shared_tokens / max(sharing_rows, 1),
            q_len,
        ):
            return

    state.plan = _SamplingPlan(
        step=state.step_counter,
        grouping=grouping,
        kv_lens=kv_lens,
        tokens_per_block=tokens_per_block,
        check_kv_lens=torch.from_numpy(kv_lens) if check else None,
    )
    state.device = None
    if graph is not None:
        # Between page boundaries only the write slots and last-page lengths
        # move; the graph derives them from the device KV lengths (step_slots).
        if grouping.version != graph.uploaded_version:
            graph.upload(state.plan)
        for key in state.prefix_wrappers:
            _plan_graph_wrappers(state, key, metadata)
    if not state.logged:
        state.logged = True
        logger.info(
            f"{LIB_NAME}: n-sample step with {rows} rows in {grouping.num_sharing} sharing "
            f"groups ({grouping.num_placeholder_rows} rows without a shared prefix), "
            f"{grouping.shared_tokens} prompt tokens read once per group"
            + ("" if graph is None else f", CUDA graph with {graph.num_groups} prefix groups")
        )


def _sampling_device(
    state: _SamplingState, device: torch.device, metadata: "TrtllmAttentionMetadata"
) -> _SamplingDevice:
    """Device copies of an eager step's index tensors, made by the first layer with one transfer."""
    plan = state.plan
    assert plan is not None
    cached = state.device
    if cached is not None and cached.step == plan.step:
        return cached
    rows = plan.num_rows
    if plan.q_len > 1:
        grouping = plan.grouping
        parts = [
            torch.from_numpy(grouping.perm),
            torch.from_numpy(grouping.shared_g),
            torch.from_numpy(grouping.base_len.astype(np.int64)),
        ]
        if plan.has_empty:
            parts.append(plan.prefix_empty.to(torch.int64))
        packed = torch.cat(parts).to(device)
        perm = packed[:rows]
        base_len = packed[2 * rows : 3 * rows]
        last_len = torch.empty((rows,), dtype=torch.int32, device=device)
        step_last_len(metadata.kv_lens_cuda_runtime[:rows], perm, base_len, last_len)
        state.device = _SamplingDevice(
            step=plan.step,
            perm=perm,
            prefix_empty=(packed[3 * rows : 4 * rows] != 0) if plan.has_empty else None,
            write_page=None,
            write_slot=None,
            shared=packed[rows : 2 * rows],
            last_len=last_len,
        )
        return state.device
    parts = [plan.write_page, plan.write_slot]
    if plan.perm is not None:
        parts.append(plan.perm)
    if plan.has_empty:
        parts.append(plan.prefix_empty.to(torch.int64))
    packed = torch.cat(parts).to(device)
    perm = packed[2 * rows : 3 * rows] if plan.perm is not None else None
    empty = None
    if plan.has_empty:
        start = (3 if plan.perm is not None else 2) * rows
        empty = packed[start : start + rows] != 0
    state.device = _SamplingDevice(
        step=plan.step,
        perm=perm,
        prefix_empty=empty,
        write_page=packed[:rows],
        write_slot=packed[rows : 2 * rows],
    )
    return state.device


def _plan_prefix(
    wrapper: Any,
    plan: _SamplingPlan,
    key: _PlanKey,
    tokens_per_block: int,
    indices: torch.Tensor,
) -> None:
    grouping = plan.grouping
    wrapper.plan(
        grouping.qo_indptr,
        grouping.prefix_indptr,
        indices,
        grouping.prefix_last_page_len,
        key.num_heads,
        key.num_kv_heads,
        key.head_dim,
        tokens_per_block,
        causal=False,
        sm_scale=key.sm_scale,
        q_data_type=key.dtype,
        kv_data_type=key.dtype,
        o_data_type=_partial_out_dtype(key.dtype),
        max_token_per_sequence=grouping.max_group_rows * grouping.q_len,
    )


_tail_qo_indptrs: Dict[tuple, torch.Tensor] = {}


def _tail_qo_indptr(rows: int, q_len: int) -> torch.Tensor:
    """[rows + 1] int32 host: ``q_len`` queries per row of the tail call."""
    indptr = _tail_qo_indptrs.get((rows, q_len))
    if indptr is None:
        indptr = torch.arange(rows + 1, dtype=torch.int32) * q_len
        _tail_qo_indptrs[(rows, q_len)] = indptr
    return indptr


def _plan_tail(
    wrapper: Any,
    plan: _SamplingPlan,
    key: _PlanKey,
    tokens_per_block: int,
    indices: torch.Tensor,
) -> None:
    if key.q_len > 1:
        # Speculative decoding: q_len causal queries per row over the row's own
        # pages. The kernel reads the last-page lengths from device buffers that
        # carry the true KV lengths (the host lengths here only shape the plan).
        wrapper.plan(
            _tail_qo_indptr(plan.num_rows, key.q_len),
            plan.grouping.tail_indptr,
            indices,
            plan.tail_last_page_len,
            key.num_heads,
            key.num_kv_heads,
            key.head_dim,
            tokens_per_block,
            causal=True,
            sm_scale=key.sm_scale,
            q_data_type=key.dtype,
            kv_data_type=key.dtype,
            o_data_type=_partial_out_dtype(key.dtype),
            max_token_per_sequence=key.q_len,
        )
        return
    wrapper.plan(
        plan.grouping.tail_indptr,
        indices,
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


def _sampling_wrappers(
    state: _SamplingState, key: _PlanKey, metadata: "TrtllmAttentionMetadata", device: torch.device
) -> tuple[Any, Any]:
    """FlashInfer prefix (prefill) and tail (decode) wrappers of an eager layer
    shape, planned once per step."""
    plan = state.plan
    assert plan is not None
    prefix = state.prefix_wrappers.get(key)
    tail = state.tail_wrappers.get(key)
    prefix_ws = _plan_workspace(device, "prefix", key)
    tail_ws = _plan_workspace(device, "tail", key)
    if prefix is None:
        workspace = _get_float_workspace(device)
        prefix = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "HND", backend=_backend_from_env()
        )
        if key.q_len > 1:
            tail = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "HND", backend=_backend_from_env()
            )
        else:
            tail = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "HND", use_tensor_cores=key.num_heads != key.num_kv_heads
            )
        prefix_ws.bind(prefix)
        tail_ws.bind(tail)
        state.prefix_wrappers[key] = prefix
        state.tail_wrappers[key] = tail
    if state.planned_step.get(key) != plan.step:
        tokens_per_block = int(metadata.tokens_per_block)
        prefix_ws.before_plan()
        _plan_prefix(prefix, plan, key, tokens_per_block, plan.prefix_indices)
        prefix_ws.after_plan()
        tail_ws.before_plan()
        _plan_tail(tail, plan, key, tokens_per_block, plan.tail_indices)
        tail_ws.after_plan()
        state.planned_step[key] = plan.step
    if key.q_len > 1:
        # The plan copied the host last-page lengths; the kernel must read the
        # lengths derived from the device KV lengths instead.
        device_state = state.device
        assert device_state is not None and device_state.last_len is not None
        tail._paged_kv_last_page_len_buf = device_state.last_len
    return prefix, tail


def _plan_graph_wrappers(
    state: _SamplingState, key: _PlanKey, metadata: "TrtllmAttentionMetadata"
) -> None:
    """Plan the CUDA graph wrappers of a layer shape for the current step.

    ``plan`` rewrites the wrappers' fixed-address index buffers and work lists;
    the replayed kernels read them. The work lists depend only on page counts:
    the prefix call is re-planned when the groups change (``prefix_version``),
    the tail call when any row's page count changes (``version``), and either
    one when another wrapper planned into its shared integer workspace since.
    The last-page lengths reach the tail call through the uploaded step buffer.
    """
    plan = state.plan
    graph = state.graph
    assert plan is not None and graph is not None
    prefix = state.prefix_wrappers[key]
    tail = state.tail_wrappers[key]
    device = graph.rows_device.device
    prefix_ws = _plan_workspace(device, "prefix", key)
    tail_ws = _plan_workspace(device, "tail", key)
    grouping = plan.grouping
    planned = state.planned_tokens.get(key, (-1, 0, -1, 0))
    tokens_per_block = int(metadata.tokens_per_block)
    if planned[:2] != (grouping.prefix_version, prefix_ws.owner):
        prefix_ws.before_plan()
        _plan_prefix(prefix, plan, key, tokens_per_block, graph.prefix_indices)
        planned = (grouping.prefix_version, prefix_ws.after_plan()) + planned[2:]
    if planned[2:] != (grouping.version, tail_ws.owner):
        tail_ws.before_plan()
        _plan_tail(tail, plan, key, tokens_per_block, graph.tail_indices)
        planned = planned[:2] + (grouping.version, tail_ws.after_plan())
    state.planned_tokens[key] = planned
    state.planned_step[key] = plan.step


def _graph_wrappers(
    state: _SamplingState, key: _PlanKey, metadata: "TrtllmAttentionMetadata"
) -> tuple[Any, Any]:
    """CUDA graph wrappers of a layer shape, created on the eager warmup forward
    that precedes capture and planned for the current step."""
    plan = state.plan
    graph = state.graph
    assert plan is not None and graph is not None
    prefix = state.prefix_wrappers.get(key)
    if prefix is not None and state.planned_step.get(key) == plan.step:
        return prefix, state.tail_wrappers[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{LIB_NAME}: FlashInfer plan requested during CUDA graph capture; the capture "
            "warmup must run this layer shape first."
        )
    if prefix is None:
        device = graph.rows_device.device
        workspace = _get_graph_float_workspace(
            device, _graph_float_workspace_bytes(key, graph.rows, graph.num_groups, device)
        )
        groups = graph.num_groups
        prefix = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "HND",
            backend=_backend_from_env(),
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros((groups + 1,), dtype=torch.int32, device=device),
            paged_kv_indptr_buf=torch.zeros((groups + 1,), dtype=torch.int32, device=device),
            paged_kv_indices_buf=torch.zeros(
                (graph.prefix_capacity,), dtype=torch.int32, device=device
            ),
            paged_kv_last_page_len_buf=torch.zeros((groups,), dtype=torch.int32, device=device),
        )
        if key.q_len > 1:
            tail = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                "HND",
                backend=_backend_from_env(),
                use_cuda_graph=True,
                qo_indptr_buf=torch.zeros((graph.rows + 1,), dtype=torch.int32, device=device),
                paged_kv_indptr_buf=torch.zeros(
                    (graph.rows + 1,), dtype=torch.int32, device=device
                ),
                paged_kv_indices_buf=torch.zeros(
                    (graph.tail_capacity,), dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buf=graph.tail_last_page_len,
            )
        else:
            tail = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "HND",
                use_cuda_graph=True,
                use_tensor_cores=key.num_heads != key.num_kv_heads,
                paged_kv_indptr_buffer=torch.zeros(
                    (graph.rows + 1,), dtype=torch.int32, device=device
                ),
                paged_kv_indices_buffer=torch.zeros(
                    (graph.tail_capacity,), dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buffer=graph.tail_last_page_len,
            )
        _plan_workspace(device, "prefix", key).bind(prefix)
        _plan_workspace(device, "tail", key).bind(tail)
        state.prefix_wrappers[key] = prefix
        state.tail_wrappers[key] = tail
    _plan_graph_wrappers(state, key, metadata)
    return prefix, state.tail_wrappers[key]


def sampling_attention(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    prefix_wrapper: Any,
    tail_wrapper: Any,
    *,
    write_page: Optional[torch.Tensor],
    write_slot: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    prefix_empty: Optional[torch.Tensor],
    q_len: int = 1,
    kv_lens: Optional[torch.Tensor] = None,
    shared: Optional[torch.Tensor] = None,
    tokens_per_block: int = 0,
) -> None:
    """One layer of the n-sample path on planned wrappers.

    Writes the current token's K/V, runs the prefix and tail calls on the
    grouped queries, merges the two states and writes ``out`` in batch order.
    Every launch has a fixed shape and reads its indices from the given
    tensors, so the call can be captured in a CUDA graph and replayed after
    the wrappers and index tensors were refreshed in place.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer KV pool (HND).
        q: [rows, heads, head_dim]; k, v: [rows, kv_heads, head_dim], batch order.
        out: [rows, heads, head_dim] output, batch order.
        write_page, write_slot: [rows] page and slot of each row's current token.
        perm: [rows] batch row of each grouped position, or None when the batch
            is in grouped order.
        prefix_empty: [rows] bool in grouped order, or None: rows whose prefix
            state is a placeholder and must not contribute.
        q_len: query tokens per row. With speculative decoding (``q_len`` > 1)
            ``q``, ``k``, ``v`` and ``out`` hold ``rows * q_len`` tokens, row
            major; the tokens of a row sit at ``kv_lens - q_len + j``, ``kv_lens``
            being the device KV lengths in batch order, and are written into
            the pages of the row's tail table (``shared`` pages precede it). The
            tail call is then a causal prefill over the row's own pages.
    """
    rows, num_heads, head_dim = q.shape
    # 1. Persist the current token's K/V in the row's own block and gather the
    # queries into group order (one kernel).
    if q_len > 1:
        assert kv_lens is not None and shared is not None
        q_grouped = write_kv_gather_q_spec(
            pool,
            q,
            k,
            v,
            kv_lens,
            perm,
            tail_wrapper._paged_kv_indptr_buf,
            shared,
            tail_wrapper._paged_kv_indices_buf,
            q_len,
            tokens_per_block,
        )
    else:
        q_grouped = write_kv_gather_q(pool, q, k, v, write_page, write_slot, perm)

    # 2. Shared prefix: the queries of each group against its common pages.
    partial_dtype = _partial_out_dtype(q.dtype)
    prefix_out = torch.empty((rows, num_heads, head_dim), dtype=partial_dtype, device=q.device)
    prefix_lse = torch.empty((rows, num_heads), dtype=torch.float32, device=q.device)
    prefix_wrapper.run(q_grouped, pool, out=prefix_out, lse=prefix_lse, return_lse=True)

    # 3. Private remainder of every row (current token included), in the same row order.
    tail_out = torch.empty_like(prefix_out)
    tail_lse = torch.empty_like(prefix_lse)
    tail_wrapper.run(q_grouped, pool, out=tail_out, lse=tail_lse, return_lse=True)

    # 4. Merge the two states and write the result back in batch order (one
    # kernel); placeholder rows take the tail state alone.
    merge_scatter(prefix_out, prefix_lse, tail_out, tail_lse, out, perm, prefix_empty, q_len)


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
        # The KV-cache manager the cached pool views and layout belong to. The
        # executor replaces the manager after memory estimation, and CUDA graph
        # warmup runs this library with the estimation manager first.
        self._manager_ref: Optional[weakref.ref] = None

    def _bind_manager(self, metadata: "TrtllmAttentionMetadata") -> None:
        manager = metadata.kv_cache_manager
        if self._manager_ref is None or self._manager_ref() is not manager:
            self._pool_views.clear()
            self._layout = None
            self._manager_ref = weakref.ref(manager)

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
        if sampling:
            sampling_state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
            # prepare() declines batches without a shared prefix.
            if sampling_state is None or sampling_state.plan is None:
                return False
            # The plan describes the target model's KV cache. The draft layers of
            # one-model speculative decoding run on this metadata with the draft
            # manager swapped in (and with different token counts); they take the
            # regular kernel.
            manager_ref = sampling_state.manager_ref
            if manager_ref is None or manager_ref() is not metadata.kv_cache_manager:
                return False
            plan = sampling_state.plan
            return q.shape[0] == plan.num_rows * plan.q_len
        # One query token per beam row.
        if q.shape[0] - metadata.num_ctx_tokens != metadata.num_generations:
            return False
        if metadata.fmha_plan_caches.get(LIB_NAME) is None:
            # prepare() declined to plan this batch (see metadata_eligible).
            return False
        return True

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        self._bind_manager(metadata)
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
        self._bind_manager(metadata)
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
        if rows != plan.num_rows * plan.q_len:
            raise RuntimeError(
                f"{type(self).__name__}: {rows} generation tokens do not match the planned "
                f"{plan.num_rows} rows of {plan.q_len}."
            )
        self._bind_manager(metadata)
        if self._layout is None:
            self._layout = _page_layout(metadata, attn.layer_idx, attn.local_layer_idx)
        if self._layout != (state.pool_index, state.page_scale, state.page_div):
            raise RuntimeError(
                f"{type(self).__name__} supports a single KV pool layout per model; layer "
                f"{attn.layer_idx} uses (pool, scale, divisor) {self._layout} but the plan was "
                f"built for {(state.pool_index, state.page_scale, state.page_div)}."
            )

        sm_scale = 1.0 / (math.sqrt(head_dim) * attn.q_scaling)
        q_len = plan.q_len
        key = _PlanKey(num_heads, num_kv_heads, head_dim, qkv.dtype, sm_scale, q_len)
        pool = self._pool_view(metadata)
        graph = state.graph
        shared: Optional[torch.Tensor] = None
        if graph is not None:
            prefix_wrapper, tail_wrapper = _graph_wrappers(state, key, metadata)
            capturing = torch.cuda.is_current_stream_capturing()
            # Once per step, and recorded by the capture itself: an eager
            # warmup forward of the same step does not put it in the graph.
            if state.slots_step != plan.step or (capturing and not state.slots_captured):
                if q_len > 1:
                    step_last_len(
                        metadata.kv_lens_cuda_runtime[: plan.num_rows],
                        graph.perm,
                        graph.base_len,
                        graph.tail_last_page_len,
                    )
                else:
                    step_slots(
                        metadata.kv_lens_cuda_runtime[: plan.num_rows],
                        graph.perm,
                        graph.write_slot,
                        graph.tail_last_page_len,
                        int(metadata.tokens_per_block),
                    )
                state.slots_step = plan.step
                state.slots_captured = capturing
            write_page, write_slot = graph.write_page, graph.write_slot
            shared = graph.shared
            perm: Optional[torch.Tensor] = graph.perm
            prefix_empty: Optional[torch.Tensor] = graph.empty
            if capturing:
                state.captured = True
        else:
            indices = _sampling_device(state, qkv.device, metadata)
            prefix_wrapper, tail_wrapper = _sampling_wrappers(state, key, metadata, qkv.device)
            write_page, write_slot = indices.write_page, indices.write_slot
            perm, prefix_empty = indices.perm, indices.prefix_empty
            shared = indices.shared

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        q = qkv[:, :q_size].view(rows, num_heads, head_dim)
        k = qkv[:, q_size : q_size + kv_size].view(rows, num_kv_heads, head_dim)
        v = qkv[:, q_size + kv_size : q_size + 2 * kv_size].view(rows, num_kv_heads, head_dim)
        output = params.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output tensor.")
        out_view = output.view(rows, num_heads, head_dim)
        sampling_attention(
            pool,
            q,
            k,
            v,
            out_view,
            prefix_wrapper,
            tail_wrapper,
            write_page=write_page,
            write_slot=write_slot,
            perm=perm,
            prefix_empty=prefix_empty,
            q_len=q_len,
            kv_lens=metadata.kv_lens_cuda_runtime[: plan.num_rows],
            shared=shared,
            tokens_per_block=int(metadata.tokens_per_block),
        )
        if not state.ran:
            state.ran = True
            logger.info(f"{LIB_NAME}: first n-sample attention call (layer {attn.layer_idx})")
        # Check mode reads device values, which CUDA graph capture forbids; the
        # eager warmup that precedes every capture is checked instead.
        if (
            plan.check_pages is not None
            and plan.q_len == 1
            and not torch.cuda.is_current_stream_capturing()
        ):
            _check_sampling_output(
                plan,
                pool,
                q,
                out_view.float(),
                num_kv_heads=num_kv_heads,
                sm_scale=sm_scale,
                tokens_per_block=int(metadata.tokens_per_block),
                layer_idx=attn.layer_idx,
            )

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
    "sampling_attention",
    "sampling_metadata_eligible",
]
