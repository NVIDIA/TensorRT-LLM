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

"""Shared-prompt generation attention for ``n``-sample requests.

Requires KV cache manager V2 (``use_kv_cache_manager_v2=True``, the default). A
batch served by another manager, or with ``beam_width > 1``, is declined and
runs the regular kernel.

The children of an ``n``-sample request (``beam_width == 1``) are independent
requests that map the same leading prompt pages when KV-cache block reuse lets
them share the parent's blocks. The regular decode kernel reads that prompt once
per child; for a long prompt the shared prefix dominates the step.

This library splits a generation step into two parts and merges them with the
standard log-sum-exp recombination (``merge_states``). Rows whose block tables start with identical
pages form a group; the group's common full pages are read once for all of its
queries (one FlashInfer paged-prefill call, no causal mask) and every row
attends to its own remaining pages (partial prompt block and generated tokens)
with a paged decode call. The two states are merged. The current
token's K/V is written into the row's own block at ``kv_len - 1``, exactly as
the regular kernel does, so the two paths are interchangeable step by step.

Scope: generation-only batches of a dense GQA/MHA model with an unquantized
bf16/fp16 paged KV cache (manager V2), no sliding window, sinks,
custom masks, or tree speculative decoding. Mixed context+generation batches fall
back to the regular library. The per-step planning (page table of the prompt,
tail length) is host work driven from ``TrtllmAttentionMetadata.prepare``,
which keeps the captured CUDA graph free of host synchronization.

An eager batch is declined (the regular
kernel runs) unless sharing pays: ``sharing_pays`` requires wide groups with a
long enough shared prefix (``TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP``, default 4;
``TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS``, default: the cost model's break-even for
the layer shape, 5000 when it is unknown), because for short prompts the regular kernel
is faster. Under CUDA graphs the same decision picks
between two graph variants per batch size (``resolve_graph_variant``, part of
the graph key).

Under CUDA graphs the n-sample path is captured for every generation batch
size of two or more rows, and runs through the fused Triton kernel of
``sampling_shared_prefix_fused`` (prefix and tail kernels on two streams plus a reduce per
layer, no FlashInfer plan or run;
``TLLM_SAMPLING_SHARED_PREFIX_FUSED=0`` selects the FlashInfer calls) when the head
layout and dtype allow it. Each captured batch size owns fixed-address buffers:
the prefix call is planned for a fixed number of groups (up to
``TLLM_SAMPLING_SHARED_PREFIX_MAX_GROUPS`` sharing groups plus one placeholder
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
handling, their stale K/V is overwritten by the next step. Captured spec steps run
through the fused kernel too (``TLLM_SAMPLING_SHARED_PREFIX_FUSED_SPEC=0`` keeps the
FlashInfer calls): its prefix tiles stack the ``q_len`` queries of every row of a
group, its tail tile holds a row's ``q_len`` tokens causally and persists their K/V.
Whether the library pays is decided by a cost model of the layer shape
(``COST_MODEL``, ``min_shared_tokens``), not a fixed prompt length. The draft layers run
on the same metadata with the draft KV manager swapped in and are declined.
The check mode (``TLLM_SAMPLING_SHARED_PREFIX_CHECK``) covers single-token steps only.

RoPE: a module that applies RoPE itself hands no ``pos_embd_params`` to the
backend and the library sees rotated Q/K. A module that fuses RoPE into the
attention op (Qwen2, Llama, ...) hands them over, and the library then rotates the
Q and K heads of the packed QKV in place before anything else runs
(``rope_qkv_inplace``), reading the op's own cos/sin table at the device position
``kv_len - q_len + j``, which is what the op does before its K write. Supported
for ``n``-sample steps with NeoX-style rotation, no scaling or linear scaling
and a full or partial rotary dim that is a power of two; every other rotary
variant (yarn, llama3, long/dynamic scaling, mRoPE, GPT-J interleaving) is
declined and the regular kernel runs.

Opt in with ``TLLM_FMHA_LIBS=+sampling_shared_prefix``.
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
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if IS_FLASHINFER_AVAILABLE:
    import flashinfer

    from .sampling_shared_prefix_fused import (
        MAX_QUERIES_PER_TILE,
        CtxPrefix,
        FusedWorkspace,
        ctx_prefix_supported,
        default_prefix_splits,
        fused_sampling_attention,
    )
    from .sampling_shared_prefix_kernels import (
        merge_scatter,
        rope_qkv_inplace,
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

LIB_NAME = "sampling_shared_prefix"
"""Registry name; also the key of the per-metadata plan state in
``TrtllmAttentionMetadata.fmha_plan_caches``."""

SAMPLING_KEY = LIB_NAME + "_sampling"
"""Key of the per-metadata plan state of the n-sample path in
``TrtllmAttentionMetadata.fmha_plan_caches``."""

BACKEND_ENV = "TLLM_SAMPLING_SHARED_PREFIX_BACKEND"
"""FlashInfer prefill backend for the prefix call. Default ``fa2``, the kernel
family the TRTLLM FlashInfer backend pins for paged prefill; ``fa3`` or
``auto`` select the Hopper kernels."""
DEFAULT_BACKEND = "fa2"
MAX_GROUPS_ENV = "TLLM_SAMPLING_SHARED_PREFIX_MAX_GROUPS"
DEFAULT_MAX_GROUPS = 64
"""Sharing groups the prefix call of a captured n-sample batch can hold (capped
at half the rows). The FlashInfer work list and its float workspace grow with
it; the rows of further groups are attended through the tail call alone."""
MIN_SHARED_ENV = "TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS"
DEFAULT_MIN_SHARED_TOKENS = 5000
"""Smallest shared prompt prefix (tokens per sharing row) for the n-sample library when
the layer shape is not known; normally ``min_shared_tokens`` follows the cost model
(``COST_MODEL``). Below the threshold the library's extra work costs more GPU time than
re-reading the shared pages (earlier Qwen3-1.7B GQA 16/8 measurement: ~0.92-0.98x at
600-1600 tokens, ~1.02-1.19x at 4000, 1.5-1.6x at 8000)."""
SPEC_MIN_SHARED_ENV = "TLLM_SAMPLING_SHARED_PREFIX_SPEC_MIN_SHARED_TOKENS"
COST_MODEL = {
    # q_len == 1 (plain n-sample decode)
    "decode": dict(
        base_us_per_byte=9.0e-8, lib_us_per_query_dim=1.49e-8, lib_extra_us_per_row=0.61
    ),
    # q_len > 1 (speculative decoding, TRTLLM-Gen multi-token kernel); prefix on the Triton tile
    "spec": dict(base_us_per_byte=5.2e-8, lib_us_per_query_dim=1.13e-8, lib_extra_us_per_row=0.30),
    # q_len > 1; prefix on the TRTLLM-Gen context kernel (``CtxPrefix``)
    "spec_ctx": dict(
        base_us_per_byte=5.2e-8, lib_us_per_query_dim=7.0e-9, lib_extra_us_per_row=0.39
    ),
}
"""Cost model of the n-sample library, per layer, per row, per shared prompt token ``L``.
Fitted to Nsight Systems / A-B-B-A decode steps of 128 rows in 4 groups of 32, MHA 20/20
heads, head_dim 128, bf16 KV, GB200, KV cache manager V2, CUDA graphs, prompts of 600 /
1600 / 4000 / 6500 tokens, EAGLE-3 K=8 (``spec``) and without speculation (``decode``):

* the regular kernel re-reads the shared pages for every row (from L2):
  ``base_us_per_byte * kv_bytes_per_token * L`` with ``kv_bytes_per_token = 4 * kv_heads *
  head_dim`` (spec: 114 / 174 / 335 us per layer at L = 600 / 1600 / 4000; the plain
  decode kernel is ~1.8x costlier per byte at 128 rows);
* the fused kernel reads them once per group, but its tensor-core work grows with every
  query: ``lib_us_per_query_dim * q_len * heads * head_dim * L``, plus a fixed
  ``lib_extra_us_per_row`` for the partial states and the tail tile (the private pages are
  read by both and cancel).

Step time with the library minus without it, per layer, is then a straight line in
``L``. Spec (q_len 9), decode window right after the prompt (Nsight): +13 / -8 / -55 /
-113 us per layer at the four lengths (+3.4% / -1.9% / -8.5% / -12.8% of the step), i.e.
``lib_extra_us_per_row`` 0.21 and a break-even of ~1250 tokens. Over a whole generation
of 1024 new tokens the advantage shrinks, because the private tail (read by both
kernels, and slower in the fused one) grows to ~1000 tokens: the A-B-B-A tokens/s of the
library over the regular kernel is 0.97 / 0.96 / 1.03 / 1.08 at 600 / 1600 / 4000 / 6500.
The spec constants were refit for the single-purpose-launch kernel and the context-kernel prefix.
Per-layer slopes from a CUDA-graph microbenchmark (GB200, MHA 20/20, d128, q_len 9, 45 cells: 3 groups x
40, 6 x 20 and 1 x 40 rows, 150-4000 prompt tokens, 300-1800 tail tokens), with R rows, P shared and T tail
tokens: TRTLLM-Gen decode ``5.35e-4 R P + 1.43e-3 R T``, Triton prefix ``2.61e-4 R P + 1.40e-3 R T``, context
kernel prefix ``1.61e-4 R P + 1.42e-3 R T`` us (the tail terms agree; the private pages are streamed once
either way). The slopes give ``base_us_per_byte`` 5.2e-8 (5.35e-4 / 10240 B) and ``lib_us_per_query_dim``
1.13e-8 (Triton) / 7.0e-9 (context kernel, ~2.1x the tensor throughput). The microbenchmark's INTERCEPTS
are not usable: it puts +40 us on the TRTLLM-Gen decode call that the same kernel does not have inside
the model (183 us per layer at 600 + 850 tokens in Nsight, against 234 us in the microbenchmark), and it
leaves out the rotary, gather and merge kernels. ``lib_extra_us_per_row`` is therefore fitted to Nsight
Systems steps of the real model (EAGLE-3 K=8, 3 groups x 40 rows, tails ~850 tokens): the library's GPU
time per step minus the regular kernel's is +1.08 ms at 600 shared tokens (+26 us per layer) and -7.0 ms at
4000 (-171 us per layer), i.e. break-even ~1050 tokens, and the whole-generation A-B-B-A tokens/s ratios of
the library against the regular kernel agree (0.94 / 1.02-1.04 / 1.13-1.16 / 1.38 at 600 / 1600 / 4000 /
6500 tokens, default gate and forced). That gives 0.39 us per row (context kernel) and 0.30 (Triton prefix).
Below ~96 rows the gate stays off (``DEFAULT_SPEC_MIN_ROWS``: one group of 40 rows measured 0.9-1.0x of the
regular kernel up to 1600 shared tokens, 1.2x at 4000). GQA keeps no spec model (never measured faster than
the regular kernel). Plain decode (whole generation): 0.99x / 1.24x / 1.68x / 1.97x tokens/s at the
same lengths, break-even ~700 tokens. For GQA with several query heads per KV head the regular
kernel reads fewer bytes while the fused kernel's work is unchanged, so the break-even
moves out or, for q_len 9, never arrives (microbench, 16/8 heads: 1.15-1.55x slower per
layer at 600-6500 tokens). The constants are fits to those cells, not laws; the
``q_len`` scaling of the query term follows the flop count and the fixed term was not
measured across ``q_len`` within a family. ``TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS``
and ``TLLM_SAMPLING_SHARED_PREFIX_SPEC_MIN_SHARED_TOKENS`` replace the model by a fixed
threshold (0 always on, a very large value off)."""

SPEC_MIN_ROWS_ENV = "TLLM_SAMPLING_SHARED_PREFIX_SPEC_MIN_ROWS"
DEFAULT_SPEC_MIN_ROWS = 96
"""Fewest sharing rows for the library with speculative decoding. The prefix grid has
one program set per (group, KV head), so a batch of a few groups leaves SMs idle while
the regular kernel parallelises over rows: EAGLE-3 K=8, MHA 20/20, 64 rows in 2 groups
of 32 at 4000 prompt tokens measured 0.96x tokens/s (128 rows in 4 groups: 1.03x)."""
MIN_GROUP_ENV = "TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP"
DEFAULT_MIN_GROUP = 4
"""Smallest number of rows sharing a prefix for the n-sample library to be used;
reading the prompt once instead of ``n`` times only pays for wide groups."""
FUSED_ENV = "TLLM_SAMPLING_SHARED_PREFIX_FUSED"
"""Set to 0 to run captured n-sample batches through the FlashInfer prefix/tail calls instead
of the fused Triton kernel (``sampling_shared_prefix_fused``)."""
PREFIX_KERNEL_ENV = "TLLM_SAMPLING_SHARED_PREFIX_PREFIX_KERNEL"
"""``auto`` (default) serves the shared pages of captured n-sample batches through the TRTLLM-Gen
context kernel where it exists (SM100/SM103, bf16, head_dim 64/128) and through the Triton prefix
kernel elsewhere; ``triton`` forces the Triton kernel (see ``CtxPrefix``)."""
CHECK_ENV = "TLLM_SAMPLING_SHARED_PREFIX_CHECK"
"""Set to 1 to synchronize after every stage of the step and validate the
sampling attention output against a dense reference. Debugging aid:
it serializes the GPU and must stay off for measurements."""

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


def _get_float_workspace(device: torch.device, nbytes: int = 0) -> torch.Tensor:
    """Shared FlashInfer float workspace, at least ``nbytes`` large. Every wrapper
    runs on the same stream sequentially, so one buffer per device serves all plan
    states. A larger request allocates a new buffer; wrappers created earlier are
    pointed at it by ``_sampling_wrappers`` before their next plan."""
    workspace = _float_workspace.get(device)
    if workspace is None or workspace.numel() < nbytes:
        workspace = torch.empty(
            max(nbytes, _FLOAT_WORKSPACE_BYTES), dtype=torch.uint8, device=device
        )
        _float_workspace[device] = workspace
    return workspace


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


def _is_v2_manager(kv_cache_manager) -> bool:
    return hasattr(kv_cache_manager, "index_mapper")


def _page_layout(metadata: "TrtllmAttentionMetadata", layer_idx: int) -> tuple[int, int, int]:
    """(pool index, scale, divisor) of a layer of a V2 manager: the page of the
    layer's pool view (``get_buffers``) that a host block-table entry names is
    ``entry * scale // divisor``.

    V2 entries are base page indices that the device-side copy multiplies by the
    pool's index scale, and ``get_buffers`` folds K and V into one page.
    """
    manager = metadata.kv_cache_manager
    pool_index = int(manager.layer_to_pool_mapping_dict[manager.layer_offsets[layer_idx]])
    return pool_index, manager.get_layer_page_index_scale(layer_idx), int(manager.kv_factor)


def _table_rows(metadata: "TrtllmAttentionMetadata", num_rows: int) -> torch.Tensor:
    """Row of the host block table of each generation row (context-free batch).

    The V2 manager keeps the table indexed by the request's slot in its index mapper.
    """
    manager = metadata.kv_cache_manager
    request_ids = metadata.request_ids
    if request_ids is None or len(request_ids) < num_rows:
        raise RuntimeError(f"{LIB_NAME}: the batch's request ids are required with KV manager V2.")
    beam_stride = int(manager.max_beam_width)
    return torch.tensor(
        [manager.index_mapper.get_index(request_ids[i]) * beam_stride for i in range(num_rows)],
        dtype=torch.int64,
    )


def _kv_manager_supported(kv_cache_manager) -> bool:
    """Whether the manager exposes the per-layer pool view and the host
    block-offset table this library reads."""
    if kv_cache_manager is None:
        return False
    for attr in ("get_buffers", "host_kv_cache_block_offsets", "kv_factor", "tokens_per_block"):
        if not hasattr(kv_cache_manager, attr):
            return False
    if getattr(kv_cache_manager, "kv_factor", 0) != 2:
        return False
    # The layer-first pool layout (recurrent-state caches) encodes offsets
    # differently; those managers never run dense attention anyway.
    if getattr(kv_cache_manager, "is_linear_attention", False):
        return False
    return True


def has_active_plan(metadata: "TrtllmAttentionMetadata") -> bool:
    """Whether ``prepare`` planned the current batch for this library.

    The FMHA selection cache keys on batch shape alone, so this state is part
    of its key: without it, a selection made on a step that had no plan would
    be reused on steps that do.
    """
    state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    return state is not None and state.plan is not None


def plan_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Per-step host planning, called from ``TrtllmAttentionMetadata.prepare``.

    Groups the rows that share leading prompt pages and derives the page tables
    from the host block offsets the manager just filled. Runs before any forward
    of the step, so under CUDA graphs it precedes capture and replay.
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


# --------------------------------------------------------------------------- #
# n-sample requests (beam_width == 1, KV cache manager V2)
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


def _ctx_tables(
    grouping: _Grouping,
    tokens_per_block: int,
    chunk_tokens: int,
    num_requests: int,
    max_pages: int,
) -> tuple:
    """Request tables of the context-kernel prefix call (``CtxPrefix``).

    Every group with query rows becomes one request per ``chunk_tokens`` query tokens, each
    over the group's shared pages; the rest of the ``num_requests`` entries are empty.
    Returns ``(block_tables, seq_lens, cum_q, cum_kv)`` as int32 arrays.
    """
    qo = grouping.qo_indptr.numpy()
    pages_ptr = grouping.prefix_indptr.numpy()
    pages = grouping.prefix_indices.numpy()
    last = grouping.prefix_last_page_len.numpy()
    block_tables = np.zeros((num_requests, max_pages), dtype=np.int32)
    seq_lens = np.zeros((num_requests,), dtype=np.int32)
    cum_q = np.zeros((num_requests + 1,), dtype=np.int32)
    cum_kv = np.zeros((num_requests + 1,), dtype=np.int32)
    request = 0
    for group in range(grouping.num_groups):
        q0, q1 = int(qo[group]), int(qo[group + 1])
        if q1 == q0:
            continue
        first, stop = int(pages_ptr[group]), int(pages_ptr[group + 1])
        count = stop - first
        kv_len = (count - 1) * tokens_per_block + int(last[group]) if count else 0
        for start in range(q0, q1, chunk_tokens):
            if request >= num_requests:
                raise RuntimeError(
                    f"{LIB_NAME}: the step needs more context-kernel requests than the "
                    f"{num_requests} its CUDA graph buffers hold."
                )
            block_tables[request, :count] = pages[first:stop]
            seq_lens[request] = kv_len
            cum_q[request + 1] = min(start + chunk_tokens, q1)
            cum_kv[request + 1] = cum_kv[request] + kv_len
            request += 1
    cum_q[request + 1 :] = cum_q[request]
    cum_kv[request + 1 :] = cum_kv[request]
    return block_tables, seq_lens, cum_q, cum_kv


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
    meta_offset: int
    """Offset (int32 units) of the group tables inside ``pages_device``."""
    fused_prefix_indices: torch.Tensor
    fused_tail_indices: torch.Tensor
    pending: bool = False
    uploaded_version: int = -1
    uploaded_prefix_version: int = -1
    prefix_slice: tuple = (0, 0)
    tail_slice: tuple = (0, 0)
    ctx: Optional[Any] = None
    """``CtxPrefix`` serving the shared pages through the TRTLLM-Gen context kernel, if enabled."""
    ctx_staging: Optional[tuple] = None
    uploaded_ctx_version: int = -1

    @classmethod
    def create(
        cls, rows: int, num_groups: int, max_pages: int, device: torch.device
    ) -> "_GraphBuffers":
        prefix_capacity = (num_groups - 1) * max_pages + num_groups
        tail_capacity = rows * max_pages
        pinned = prefer_pinned()
        # The group tables of the fused kernel follow the page indices.
        meta_offset = _AUX_INT32 * rows + prefix_capacity + tail_capacity
        meta_offset += meta_offset % 2
        meta_len = 3 * num_groups + rows + 3
        size = 3 * rows + (meta_offset + meta_len + 1) // 2
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
            empty_staging=torch.zeros((rows + 128,), dtype=torch.bool, pin_memory=pinned),
            empty=torch.zeros((rows + 128,), dtype=torch.bool, device=device),
            copied=torch.cuda.Event(),
            meta_offset=meta_offset,
            fused_prefix_indices=torch.zeros(
                (prefix_capacity + 64,), dtype=torch.int32, device=device
            ),
            fused_tail_indices=torch.zeros((tail_capacity + 64,), dtype=torch.int32, device=device),
        )

    def _meta(self, start: int, length: int) -> torch.Tensor:
        return self.pages_device[self.meta_offset + start : self.meta_offset + start + length]

    @property
    def group_qo_indptr(self) -> torch.Tensor:
        return self._meta(0, self.num_groups + 1)

    @property
    def group_prefix_indptr(self) -> torch.Tensor:
        return self._meta(self.num_groups + 1, self.num_groups + 1)

    @property
    def group_prefix_last_page_len(self) -> torch.Tensor:
        return self._meta(2 * self.num_groups + 2, self.num_groups)

    @property
    def row_tail_indptr(self) -> torch.Tensor:
        return self._meta(3 * self.num_groups + 2, self.rows + 1)

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
            self.empty_staging.numpy()[:rows] = grouping.prefix_empty
            self.empty.copy_(self.empty_staging, non_blocking=True)
        groups = self.num_groups
        meta = page_staging[self.meta_offset :]
        meta[: groups + 1] = grouping.qo_indptr.numpy()
        meta[groups + 1 : 2 * groups + 2] = grouping.prefix_indptr.numpy()
        meta[2 * groups + 2 : 3 * groups + 2] = grouping.prefix_last_page_len.numpy()
        meta[3 * groups + 2 : 3 * groups + 3 + rows] = grouping.tail_indptr.numpy()
        used = 3 * rows + (count + 1) // 2
        self.device[:used].copy_(self.staging[:used], non_blocking=True)
        meta_start = 3 * rows + self.meta_offset // 2
        meta_stop = meta_start + (3 * groups + rows + 4) // 2
        self.device[meta_start:meta_stop].copy_(
            self.staging[meta_start:meta_stop], non_blocking=True
        )
        if grouping.version != self.uploaded_version:
            # The fused kernel reads the page lists from fixed addresses.
            lo, hi = self.prefix_slice
            self.fused_prefix_indices[: hi - lo].copy_(self.pages_device[lo:hi])
            lo, hi = self.tail_slice
            self.fused_tail_indices[: hi - lo].copy_(self.pages_device[lo:hi])
        if self.ctx is not None and grouping.prefix_version != self.uploaded_ctx_version:
            self._upload_ctx_tables(grouping)
        self.copied.record()
        self.pending = True
        self.uploaded_version = grouping.version
        self.uploaded_prefix_version = grouping.prefix_version

    def enable_ctx(self, ctx: Any) -> None:
        self.ctx = ctx
        pinned = prefer_pinned()
        self.ctx_staging = tuple(
            torch.zeros(t.shape, dtype=torch.int32, pin_memory=pinned) for t in ctx.table_arrays()
        )

    def _upload_ctx_tables(self, grouping: _Grouping) -> None:
        """Rewrite the context kernel's request tables (changes only with the groups)."""
        ctx = self.ctx
        assert ctx is not None and self.ctx_staging is not None
        tables = _ctx_tables(
            grouping,
            ctx.tokens_per_block,
            ctx.chunk_tokens,
            ctx.num_requests,
            ctx.block_tables.shape[1],
        )
        for staging, device, host in zip(self.ctx_staging, ctx.table_arrays(), tables):
            staging.numpy()[...] = host
            device.copy_(staging, non_blocking=True)
        self.uploaded_ctx_version = grouping.prefix_version


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
    fused_workspace: Optional[Any] = None
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


_model_shape: Optional[tuple] = None
"""(heads, kv_heads, head_dim) of the attention layers that can use the library, set
when a layer reports itself available; the spec cost model needs it."""


CTX_MIN_Q_LEN = 4
"""Fewest query tokens per row for the context-kernel prefix. Measured against the Triton prefix per
layer (GB200, MHA 20/20, 3 groups x 40 rows, 1000-token tails, 1600 / 4000 shared tokens): q_len 1
0.95x / 0.92x (a group is then 40 queries, too small a tile for the kernel), q_len 4 1.02x / 1.06x,
q_len 9 1.08x / 1.18x."""


def _ctx_prefix_enabled(head_dim: int, q_len: int) -> bool:
    """Whether captured batches take the shared pages through the context kernel (``CtxPrefix``)."""
    if q_len < CTX_MIN_Q_LEN:
        return False
    if os.environ.get(PREFIX_KERNEL_ENV, "auto").strip().lower() == "triton":
        return False
    if not IS_FLASHINFER_AVAILABLE or not torch.cuda.is_available():
        return False
    return ctx_prefix_supported(head_dim, torch.bfloat16)


def break_even_tokens(q_len: int, num_heads: int, num_kv_heads: int, head_dim: int) -> float:
    """Shared prompt tokens per row above which the library beats the regular kernel
    for ``q_len`` tokens per row (see ``COST_MODEL``); infinity if it never does."""
    if q_len == 1:
        cost = COST_MODEL["decode"]
    elif num_kv_heads != num_heads:
        return math.inf  # GQA with draft tokens: never measured faster than the regular kernel
    else:
        cost = COST_MODEL["spec_ctx" if _ctx_prefix_enabled(head_dim, q_len) else "spec"]
    saved_per_token = cost["base_us_per_byte"] * 4 * num_kv_heads * head_dim - (
        cost["lib_us_per_query_dim"] * q_len * num_heads * head_dim
    )
    if saved_per_token <= 0:
        return math.inf
    return cost["lib_extra_us_per_row"] / saved_per_token


def min_shared_tokens(q_len: int) -> float:
    """Smallest shared prefix (tokens per row) for the library: the env override of the
    family, else the cost model, else (layer shape unknown) the conservative default."""
    name = MIN_SHARED_ENV if q_len == 1 else SPEC_MIN_SHARED_ENV
    value = os.environ.get(name)
    if value is not None and value.strip():
        return int(value)
    if _model_shape is None:
        return DEFAULT_MIN_SHARED_TOKENS if q_len == 1 else math.inf
    return break_even_tokens(q_len, *_model_shape)


SPEC_GROUP_MODEL = dict(
    gain_us_per_token=0.0446,
    gain_us_per_token_sq=2.9e-6,
    batch_us=33.0,
    group_us=7.0,
    reference_rows=120,
)
"""Per-layer saving of the context-kernel path over the regular kernel for a speculative batch of
``reference_rows`` rows (MHA 20/20, d128, q_len 9, GB200), by group::

    saving = sum_g rows_g * (a * P_g + b * P_g^2) / reference_rows
             - batch_us * rows / reference_rows - group_us * groups

The regular kernel's cost of re-reading a prefix is convex in its length (the prefix is served from L2
until it no longer fits: 27 us per 1000 tokens between 600 and 1600, 51 between 1600 and 4000), so the
gain is quadratic in ``P_g`` and a batch is judged group by group, not by its mean prompt length.
``a``, ``b`` and ``batch_us`` solve the three Nsight points of the model (3 groups x 40 rows, tails ~850:
the library minus the regular kernel's GPU time per layer is -26 us at 600 shared tokens, +25 at 1600,
+171 at 4000, i.e. -53.8 + 0.0446 P + 2.9e-6 P^2, of which ``batch_us`` is the fixed cost of the rotary,
gather, merge and context-kernel launches after removing 3 groups of ``group_us``); ``group_us`` is the
per-group cost measured in the microbenchmark (3 groups x 40 vs 6 x 20 rows, same bytes: +20 us for 3 more
groups). Held-out ABBA checks (not used in the fit): 4x32 and 6x20 rows of mixed lengths lost 5.3% and
6.7% with the mean-length gate (model: -4 and -11 us per layer -> declined), 6x20 rows at 4000 tokens
gained 14.6% (model +150 us), 4x32 at 1600 +/-0 (+15 us)."""


def spec_batch_saving_us(group_profile, num_rows: int) -> float:
    """Per-layer saving of the context-kernel path for ``(rows, shared tokens)`` per group; see
    ``SPEC_GROUP_MODEL``."""
    m = SPEC_GROUP_MODEL
    gain = sum(
        rows * (m["gain_us_per_token"] * tokens + m["gain_us_per_token_sq"] * tokens * tokens)
        for rows, tokens in group_profile
    )
    ref = m["reference_rows"]
    return gain / ref - m["batch_us"] * num_rows / ref - m["group_us"] * len(group_profile)


def sharing_pays(
    num_rows: int,
    sharing_rows: int,
    num_sharing_groups: int,
    shared_tokens_per_row: float,
    q_len: int = 1,
    group_profile=None,
) -> bool:
    """Size gate of the n-sample path: whether reading the shared prompt once pays.

    ``sharing_rows`` rows form ``num_sharing_groups`` groups whose common prefix
    is ``shared_tokens_per_row`` tokens long on average. The library is used
    when at least half the rows share, the groups are wide enough, and the
    prefix is long enough (see ``MIN_SHARED_ENV`` / ``MIN_GROUP_ENV``); otherwise
    the regular generation kernel is faster. With draft tokens on the context-kernel
    path and ``group_profile`` (``(rows, shared tokens)`` per sharing group), the
    prefix test is the group-wise cost model ``spec_batch_saving_us`` > 0 instead
    of the mean length (unless the env fixes a threshold).
    """
    if num_sharing_groups <= 0 or sharing_rows <= 0:
        return False
    if q_len > 1 and sharing_rows < _int_env(SPEC_MIN_ROWS_ENV, DEFAULT_SPEC_MIN_ROWS):
        return False
    structure = (
        2 * sharing_rows >= num_rows
        and sharing_rows >= _int_env(MIN_GROUP_ENV, DEFAULT_MIN_GROUP) * num_sharing_groups
    )
    if not structure:
        return False
    if (
        q_len > 1
        and group_profile
        and not os.environ.get(SPEC_MIN_SHARED_ENV, "").strip()
        and _model_shape is not None
        and _model_shape[0] == _model_shape[1]
        and _ctx_prefix_enabled(_model_shape[2], q_len)
    ):
        return spec_batch_saving_us(group_profile, num_rows) > 0
    return shared_tokens_per_row >= min_shared_tokens(q_len)


FUSED_PREFIX_SPLITS = _int_env("TLLM_SAMPLING_SHARED_PREFIX_SPLITS", 0)
"""Split-K factor of each group's shared pages in the fused kernel (fixed at capture); 0 picks it
from the grid size (``default_prefix_splits``)."""
FUSED_TAIL_SPLITS = 1
CTX_TAIL_SPLITS = 1
CTX_TAIL_BLOCK = 128
CTX_TAIL_WARPS = 4
"""Tail launch of the context-kernel path, which runs alone (the prefix launch is a separate, faster
kernel): one split, 128-token tiles, 4 warps measured best for the streaming tail by itself (192 us
at 120 rows x 1000 tokens, MHA 20/20: 6.4 TB/s)."""
FUSED_GROUP_ROWS = 32
"""Rows per group served by one program per KV head; larger groups loop inside it."""


FUSED_SPEC_ENV = "TLLM_SAMPLING_SHARED_PREFIX_FUSED_SPEC"
"""Set to 0 to keep speculative steps (``q_len`` > 1) on the FlashInfer prefix/tail calls."""


def fused_supported(
    num_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, q_len: int = 1
) -> bool:
    """Whether captured n-sample batches can use the fused Triton kernel.

    The kernel holds the ``q_len`` tokens of a row and the GQA group of a KV head
    in one tile (see ``MAX_QUERIES_PER_TILE``).
    """
    return (
        _int_env(FUSED_ENV, 1) != 0
        and (q_len == 1 or _int_env(FUSED_SPEC_ENV, 1) != 0)
        and head_dim in (64, 128)
        and num_kv_heads > 0
        and num_heads % num_kv_heads == 0
        and 1 <= q_len <= 16
        and q_len * (num_heads // num_kv_heads) <= MAX_QUERIES_PER_TILE
        and dtype in (torch.bfloat16, torch.float16)
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
    if q_len > 1 and not _eager_fused(q_len):
        # Speculative steps that the fused kernel cannot serve would need FlashInfer's
        # captured tail call: its split-KV plan (the work list is padded) follows the
        # host lengths, and FlashInfer returns wrong output when the true lengths are
        # shorter (overlap scheduler). The regular kernel serves those captured steps.
        return 0
    gate = (
        _int_env(MIN_GROUP_ENV, DEFAULT_MIN_GROUP),
        min_shared_tokens(q_len),
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
    profile = [(entry[0], float(entry[1])) for entry in wide]
    variant = int(sharing_pays(rows, sharing_rows, len(wide), per_row, q_len, profile))
    _variant_memo[:] = [requests[0], requests[-1], len(requests), gate, variant]
    return variant


def sampling_metadata_eligible(metadata: "TrtllmAttentionMetadata") -> bool:
    """Batch-level preconditions of the n-sample path (KV-cache manager V2 only).

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
        and _is_v2_manager(metadata.kv_cache_manager)
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
    request_ids = metadata.request_ids
    cache_key = tuple(request_ids[:rows]) if request_ids is not None else None
    if cache_key is None or cache_key != state.table_rows_key:
        state.table_rows = _table_rows(metadata, rows).numpy()
        state.table_rows_key = cache_key
    assert state.table_rows is not None
    return state.table_rows


def _eager_fused(q_len: int) -> bool:
    """Whether eager batches use the fused kernel (the layer shape is known and supported)."""
    if _model_shape is None:
        return False
    num_heads, num_kv_heads, head_dim = _model_shape
    return fused_supported(num_heads, num_kv_heads, head_dim, torch.bfloat16, q_len)


def _group_profile(grouping: "_Grouping", tokens_per_block: int, q_len: int) -> list:
    """``(rows, shared tokens)`` of each sharing group of a grouping (see ``SPEC_GROUP_MODEL``)."""
    qo = grouping.qo_indptr.numpy()
    pages = grouping.prefix_indptr.numpy()
    return [
        (int(qo[g + 1] - qo[g]) // q_len, float(int(pages[g + 1] - pages[g]) * tokens_per_block))
        for g in range(grouping.num_sharing)
    ]


def _maybe_enable_ctx_prefix(
    graph: _GraphBuffers, metadata: "TrtllmAttentionMetadata", rows: int, q_len: int
) -> None:
    """Serve the shared pages of this captured batch size through the context kernel, if it can."""
    if _model_shape is None:
        return
    num_heads, num_kv_heads, head_dim = _model_shape
    device = metadata.kv_cache_block_offsets.device
    if not (
        fused_supported(num_heads, num_kv_heads, head_dim, torch.bfloat16, q_len)
        and _ctx_prefix_enabled(head_dim, q_len)
    ):
        return
    tokens = rows * q_len
    max_pages = int(metadata.host_kv_cache_block_offsets.size(-1))
    tokens_per_block = int(metadata.tokens_per_block)
    max_kv_len = int(metadata.max_seq_len or max_pages * tokens_per_block)
    graph.enable_ctx(
        CtxPrefix(
            tokens,
            num_heads,
            head_dim,
            q_len,
            CtxPrefix.num_requests_for(graph.num_groups, tokens, q_len),
            max_pages,
            max_kv_len,
            torch.bfloat16,
            device,
            tokens_per_block,
        )
    )


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
        pool_index, page_scale, page_div = _page_layout(metadata, first_layer)
        pool = kv_cache_manager.get_buffers(first_layer, kv_layout="HND")
        state = _SamplingState(
            pool_index=pool_index,
            page_scale=page_scale,
            page_div=page_div,
            num_pages=int(pool.size(0)),
            manager_ref=weakref.ref(kv_cache_manager),
        )
        metadata.fmha_plan_caches[SAMPLING_KEY] = state
    eager = not metadata.is_cuda_graph
    if state.graph is not None and rows != state.graph.rows:
        if not eager:
            raise RuntimeError(
                f"{LIB_NAME}: CUDA graph metadata planned for {state.graph.rows} rows got {rows}."
            )
        state.graph = None  # an eager batch changed size: new fixed-address buffers below
    if state.graph is None and (not eager or _eager_fused(q_len)):
        # Captured batches and, when the fused kernel can serve the shape, eager ones too: the
        # fused kernel reads its tables from fixed device buffers and never calls FlashInfer (whose
        # FA2 kernels return zeros for heads with extremely negative logits, e.g. Qwen2.5-7B).
        num_groups = min(_max_groups_from_env(), rows // 2) + 1
        state.graph = _GraphBuffers.create(
            rows, num_groups, int(host_offsets.size(-1)), metadata.kv_cache_block_offsets.device
        )
        _maybe_enable_ctx_prefix(state.graph, metadata, rows, q_len)
    graph = state.graph
    state.step_counter += 1

    check = _check_enabled()
    kv_lens = np.maximum(metadata.kv_lens_runtime[:rows].numpy(), q_len).astype(np.int64)
    if eager and int(kv_lens.max()) < min_shared_tokens(q_len):
        # No row has a prefix long enough for the size gate: decline before the (host
        # heavy) grouping, which otherwise runs for every eager batch of a short prompt.
        return
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
    if eager:
        sharing_rows = rows - grouping.num_placeholder_rows
        profile = _group_profile(grouping, tokens_per_block, q_len)
        if not sharing_pays(
            rows,
            sharing_rows,
            grouping.num_sharing,
            grouping.shared_tokens / max(sharing_rows, 1),
            q_len,
            profile,
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
            # A split-KV plan fixes its chunks from the host lengths, which under the
            # overlap scheduler can be a page ahead of the true device lengths; the
            # kernel then returns wrong output (visible for heads that attend to the
            # step's last few positions, e.g. Qwen2.5). An eager plan must not split.
            # CUDA graph plans cannot (see ``resolve_graph_variant``).
            **({} if getattr(wrapper, "_use_cuda_graph", False) else {"disable_split_kv": True}),
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
    # FlashInfer splits the KV of an eager prefill too and reserves its partial
    # results in the float workspace: size it from the plan instead of hoping that
    # 128 MB suffice (many query tokens x heads overflow it and abort the step).
    workspace = _get_float_workspace(
        device, _graph_float_workspace_bytes(key, plan.num_rows, plan.num_groups, device)
    )
    if prefix is not None and prefix._float_workspace_buffer is not workspace:
        for wrapper in (prefix, tail):
            wrapper._float_workspace_buffer = workspace
    if prefix is None:
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


def _rotary_dim(attn: "TrtllmAttention") -> int:
    return int(attn.rope_params.dim)


def _fused_rope_supported(attn: "TrtllmAttention") -> bool:
    """Whether the rotary embedding the attention op would apply is one that
    ``rope_qkv_inplace`` reproduces (see the module docstring)."""
    if int(attn.position_embedding_type) != int(PositionEmbeddingType.rope_gpt_neox):
        return False
    rope = attn.rope_params
    if rope.scale_type not in (RotaryScalingType.none, RotaryScalingType.linear):
        return False
    if rope.duplicate_data or getattr(attn, "rotary_cos_sin", None) is None:
        return False
    dim = _rotary_dim(attn)
    half = dim // 2
    if dim <= 0 or dim % 2 or half & (half - 1) or dim > attn.head_dim:
        return False
    # The table is [positions][dim] float32 on the device.
    table = attn.rotary_cos_sin
    return table.dtype == torch.float32 and table.numel() % dim == 0


class SamplingSharedPrefixFmha(PhasedFmha):
    """Generation-phase library for n-sample requests; see the module docstring."""

    supports_workspace_reclamation = False

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._pool_views: Dict[int, torch.Tensor] = {}
        # (pool index, scale, divisor) of this layer; fixed for the manager lifetime.
        self._layout: Optional[tuple[int, int, int]] = None
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
        position_type = PositionEmbeddingType(getattr(attn, "position_embedding_type", 0))
        if position_type.is_rope():
            if not _fused_rope_supported(attn):
                return False
        elif position_type != PositionEmbeddingType.learned_absolute:
            # ALiBi, relative, GLM and deferred embeddings are applied inside the attention op,
            # which this library bypasses.
            return False
        global _model_shape
        if _model_shape is None:  # the target layers come first; draft layers share the shape
            _model_shape = (num_heads, num_kv_heads, head_dim)
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
        if not sampling_metadata_eligible(metadata):
            return False
        if PositionEmbeddingType(getattr(self.attn, "position_embedding_type", 0)).is_rope():
            # The rotary table must cover every position the batch can reach.
            max_seq_len = metadata.max_seq_len
            if max_seq_len is None or max_seq_len > (
                self.attn.rotary_cos_sin.numel() // _rotary_dim(self.attn)
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
        window = forward_args.attention_window_size
        if (
            window is not None
            and metadata.max_seq_len is not None
            and window < metadata.max_seq_len
        ):
            return False
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
        qkv = params.qkv_input
        if qkv is None:
            raise RuntimeError(f"{type(self).__name__} requires packed QKV input.")
        self._run_sampling_generation(params, qkv)

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
            self._layout = _page_layout(metadata, attn.layer_idx)
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
        fused = graph is not None and fused_supported(
            num_heads, num_kv_heads, head_dim, qkv.dtype, q_len
        )
        if graph is not None:
            if fused:
                prefix_wrapper = tail_wrapper = None
            else:
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
        if PositionEmbeddingType(attn.position_embedding_type).is_rope():
            rope_qkv_inplace(
                qkv,
                attn.rotary_cos_sin,
                metadata.kv_lens_cuda_runtime[: plan.num_rows],
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=_rotary_dim(attn),
                q_len=q_len,
            )
        q = qkv[:, :q_size].view(rows, num_heads, head_dim)
        k = qkv[:, q_size : q_size + kv_size].view(rows, num_kv_heads, head_dim)
        v = qkv[:, q_size + kv_size : q_size + 2 * kv_size].view(rows, num_kv_heads, head_dim)
        output = params.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output tensor.")
        out_view = output.view(rows, num_heads, head_dim)
        if fused:
            assert graph is not None
            group_rows = min(FUSED_GROUP_ROWS, plan.num_rows)
            ctx = graph.ctx
            if ctx is not None and (
                qkv.dtype != ctx.q.dtype or tuple(ctx.q.shape) != (rows, num_heads, head_dim)
            ):
                ctx = None
            if ctx is not None:
                prefix_splits = 0
                tail_kwargs = dict(
                    tail_splits=CTX_TAIL_SPLITS,
                    block_nt=CTX_TAIL_BLOCK,
                    tail_num_warps=CTX_TAIL_WARPS,
                    ctx=ctx,
                )
            else:
                prefix_splits = FUSED_PREFIX_SPLITS or default_prefix_splits(
                    graph.num_groups,
                    num_kv_heads,
                    group_rows,
                    q_len,
                    num_heads // num_kv_heads,
                    torch.cuda.get_device_properties(qkv.device).multi_processor_count,
                )
                tail_kwargs = dict(tail_splits=FUSED_TAIL_SPLITS)
            slots = prefix_splits + tail_kwargs["tail_splits"]
            workspace = state.fused_workspace
            # Eager batches change their grouping, and so the split count, from step to step:
            # the arena views must cover both the slots and the query tokens of this step.
            if workspace is None or workspace.m.shape[1] < rows or workspace.m.shape[0] < slots:
                state.fused_workspace = FusedWorkspace.shared(
                    rows, num_heads, head_dim, slots, qkv.device
                )
            fused_sampling_attention(
                pool,
                q,
                k,
                v,
                out_view,
                perm=perm,
                group_qo_indptr=graph.group_qo_indptr,
                prefix_indptr=graph.group_prefix_indptr,
                prefix_indices=graph.fused_prefix_indices,
                prefix_last_page_len=graph.group_prefix_last_page_len,
                prefix_empty=prefix_empty,
                tail_indptr=graph.row_tail_indptr,
                tail_indices=graph.fused_tail_indices,
                tail_last_page_len=graph.tail_last_page_len,
                write_page=write_page,
                write_slot=write_slot,
                sm_scale=sm_scale,
                num_groups=graph.num_groups,
                q_len=q_len,
                max_group_rows=group_rows,
                workspace=state.fused_workspace,
                prefix_splits=prefix_splits,
                **tail_kwargs,
            )
        else:
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


__all__ = [
    "LIB_NAME",
    "SamplingSharedPrefixFmha",
    "has_active_plan",
    "merge_states",
    "plan_step",
    "sampling_attention",
    "sampling_metadata_eligible",
]
