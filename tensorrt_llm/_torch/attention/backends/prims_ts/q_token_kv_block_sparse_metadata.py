# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA-graph-safe compact QToken-KvBlock-Sparse-Attention metadata construction.

The public attention kernel consumes dense sparse-page indices and, for
grouped routes, a parallel packed-membership table. Q1 rows map their compact
selected logical blocks directly to encoded physical
subpage locators. Q2--Q8 rows additionally union the selected blocks of
every adjacent query and write per-page query-membership masks into a separate
dense table, packed four 8-bit masks per Int32 word. The public API names the
sparse block size explicitly: 4, 8, 16, 32, 64, or 128 tokens.
Patterns can be shared across KV heads or supplied independently for each head.

Construction uses one CUDA C++ CTA per route. Q1 maps selected and tail blocks
directly. Q2--Q8 radix-sort at most ``G * (topk + 1)`` selected/tail IDs,
segmented-OR equal-key memberships, and emit only unique pages; work and
temporary storage are independent of the model context length. On SM90 and
newer, the metadata grid releases the prepared attention grid through
programmatic dependent launch (PDL).
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from flashinfer.api_logging import flashinfer_api as flashinfer_experimental_api

from flashinfer.utils import device_support_pdl
from ._block_sparse.common import _num_sparse_pattern_heads

_Q_TOKEN_KV_BLOCK_SPARSE_MIN_BLOCK_SIZE = 4
_Q_TOKEN_KV_BLOCK_SPARSE_BLOCK_SIZES = (4, 8, 16, 32, 64, 128)
_Q_TOKEN_KV_BLOCK_SPARSE_MAX_BLOCK_TOPK = 512
# Reserve three high key bits for grouped-query membership tags and retain an
# out-of-range radix sentinel in the CUDA/CUB touched-union representation.
_Q_TOKEN_KV_BLOCK_SPARSE_MAX_SEQ_LEN_KV = (
    (1 << 29) - 1
) * _Q_TOKEN_KV_BLOCK_SPARSE_MIN_BLOCK_SIZE
_Q_TOKEN_KV_BLOCK_SPARSE_MEMBERSHIPS_PER_WORD = 4
_Q_TOKEN_KV_BLOCK_SPARSE_INT32_LOCATOR_CAPACITY = 1 << 31
_Q_TOKEN_KV_BLOCK_SPARSE_WORKSPACE_ALIGNMENT = 256


@functools.cache
def _get_prims_ts_q_token_kv_block_sparse_metadata_module() -> Any:
    """Build and cache the CUDA C++ QToken-KvBlock-Sparse-Attention metadata implementation."""

    from flashinfer.jit import gen_prims_ts_q_token_kv_block_sparse_metadata_module

    return gen_prims_ts_q_token_kv_block_sparse_metadata_module().build_and_load()


@dataclass(frozen=True)
class _PrimsTSQTokenKvBlockSparseWorkspaceViews:
    """Typed views bound to one caller-owned QToken-KvBlock-Sparse-Attention attention workspace."""

    q_token_kv_block_sparse_page_indices: torch.Tensor
    q_token_kv_block_sparse_page_memberships: torch.Tensor
    seq_lens: torch.Tensor
    attention_workspace_buffer: torch.Tensor


@dataclass(frozen=True)
class _QTokenKvBlockSparseTensorDescriptor:
    """Structural replacement-storage contract for a prepared QToken-KvBlock-Sparse-Attention plan."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype


def _describe_q_token_kv_block_sparse_tensor(
    tensor: torch.Tensor,
) -> _QTokenKvBlockSparseTensorDescriptor:
    return _QTokenKvBlockSparseTensorDescriptor(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _validate_q_token_kv_block_sparse_plan_tensor(
    tensor: torch.Tensor,
    name: str,
    descriptor: _QTokenKvBlockSparseTensorDescriptor,
) -> None:
    """Check one replacement tensor without inspecting device values."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        tuple(tensor.shape) != descriptor.shape
        or tuple(tensor.stride()) != descriptor.stride
        or tensor.device != descriptor.device
        or tensor.dtype != descriptor.dtype
    ):
        raise ValueError(
            f"{name} must preserve the shape, strides, device, and dtype "
            "validated by the QToken-KvBlock-Sparse-Attention plan"
        )


def _validate_q_token_kv_block_sparse_paged_kv_cache(
    paged_kv_cache: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return the two rank-four QToken-KvBlock-Sparse-Attention cache tensors."""

    if (
        not isinstance(paged_kv_cache, tuple)
        or len(paged_kv_cache) != 2
        or not all(isinstance(cache, torch.Tensor) for cache in paged_kv_cache)
    ):
        raise TypeError("paged_kv_cache must be a (k_cache, v_cache) tuple")
    k_cache, v_cache = paged_kv_cache
    if k_cache.ndim != 4:
        raise ValueError(
            "K and V cache tensors must have shape [pages,Hkv,storage_page_size,D]"
        )
    if (
        v_cache.shape != k_cache.shape
        or v_cache.device != k_cache.device
        or v_cache.dtype != k_cache.dtype
    ):
        raise ValueError(
            "K and V cache tensors must have matching shapes, devices, and dtypes"
        )
    return k_cache, v_cache


def _validate_q_token_kv_block_sparse_locator_capacity(
    k_cache: torch.Tensor,
    *,
    sparse_block_size: int,
) -> None:
    """Keep every valid encoded cache locator nonnegative in Int32."""

    storage_page_size = int(k_cache.shape[2])
    _validate_storage_page_size(storage_page_size)
    fragment_size = math.gcd(storage_page_size, sparse_block_size)
    num_encoded_locators = int(k_cache.shape[0]) * storage_page_size // fragment_size
    if num_encoded_locators > _Q_TOKEN_KV_BLOCK_SPARSE_INT32_LOCATOR_CAPACITY:
        raise NotImplementedError(
            "the encoded QToken-KvBlock-Sparse-Attention cache-locator extent must fit in nonnegative "
            "signed int32: got "
            f"{num_encoded_locators}, limit is {_Q_TOKEN_KV_BLOCK_SPARSE_INT32_LOCATOR_CAPACITY}"
        )


@dataclass(frozen=True)
class _PrimsTSQTokenKvBlockSparseWorkspaceLayout:
    """Layout of QToken-KvBlock-Sparse-Attention metadata outputs and disjoint kernel scratch.

    The dense page-index and packed-membership tables plus compact sequence
    lengths remain live from metadata construction through the attention
    launch. A decode split-KV plan must
    preserve its partials and self-resetting completion counters independently
    of metadata. A direct prefill plan has no split-KV storage; its small
    attention region contains only uniform-call-ABI placeholder tensors.
    Semantic inputs such as block tables and query mappings remain outside.
    """

    q_token_kv_block_sparse_page_indices_shape: tuple[int, int]
    q_token_kv_block_sparse_page_indices_bytes: int
    q_token_kv_block_sparse_page_memberships_shape: tuple[int, int]
    q_token_kv_block_sparse_page_memberships_byte_offset: int
    q_token_kv_block_sparse_page_memberships_bytes: int
    seq_lens_byte_offset: int
    seq_lens_bytes: int
    attention_workspace_byte_offset: int
    attention_scratch_bytes: int
    uses_split_kv: bool
    max_seq_len: int
    total_bytes: int

    def bind(
        self, workspace_buffer: torch.Tensor
    ) -> _PrimsTSQTokenKvBlockSparseWorkspaceViews:
        """Return zero-copy typed views over a validated byte workspace."""

        _validate_q_token_kv_block_sparse_attention_workspace(
            workspace_buffer, self.total_bytes
        )
        workspace_bytes = workspace_buffer.reshape(-1).view(torch.uint8)
        q_token_kv_block_sparse_page_indices = (
            workspace_bytes[: self.q_token_kv_block_sparse_page_indices_bytes]
            .view(torch.int32)
            .view(self.q_token_kv_block_sparse_page_indices_shape)
        )
        q_token_kv_block_sparse_page_memberships = (
            workspace_bytes[
                self.q_token_kv_block_sparse_page_memberships_byte_offset : (
                    self.q_token_kv_block_sparse_page_memberships_byte_offset
                    + self.q_token_kv_block_sparse_page_memberships_bytes
                )
            ]
            .view(torch.int32)
            .view(self.q_token_kv_block_sparse_page_memberships_shape)
        )
        seq_lens = workspace_bytes[
            self.seq_lens_byte_offset : self.seq_lens_byte_offset + self.seq_lens_bytes
        ].view(torch.int32)
        attention_workspace_buffer = workspace_bytes[
            self.attention_workspace_byte_offset : self.attention_workspace_byte_offset
            + self.attention_scratch_bytes
        ]
        return _PrimsTSQTokenKvBlockSparseWorkspaceViews(
            q_token_kv_block_sparse_page_indices=q_token_kv_block_sparse_page_indices,
            q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
            seq_lens=seq_lens,
            attention_workspace_buffer=attention_workspace_buffer,
        )


@dataclass(frozen=True)
class _PrimsTSQTokenKvBlockSparseMetadataPlan:
    """Unchecked metadata launch state with all geometry pre-resolved."""

    q_token_kv_block_sparse_page_indices: torch.Tensor
    q_token_kv_block_sparse_page_memberships: torch.Tensor
    seq_lens: torch.Tensor
    qo_indptr: Optional[torch.Tensor]
    metadata_run: Any
    group_size: int
    use_packed_q: bool
    sparse_block_size: int
    storage_page_size: int
    max_seq_len_kv: int
    release_attention_pdl: bool

    def run(
        self,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> None:
        """Launch the prepared CUDA metadata kernel without shape resolution."""

        if self.use_packed_q:
            self.metadata_run(
                block_indices,
                block_table,
                token_to_request,
                query_positions,
                self.qo_indptr,
                self.q_token_kv_block_sparse_page_indices,
                self.q_token_kv_block_sparse_page_memberships,
                self.seq_lens,
                self.group_size,
                self.storage_page_size,
                self.sparse_block_size,
                self.max_seq_len_kv,
                self.release_attention_pdl,
            )
        else:
            self.metadata_run(
                block_indices,
                block_table,
                token_to_request,
                query_positions,
                self.q_token_kv_block_sparse_page_indices,
                self.q_token_kv_block_sparse_page_memberships,
                self.seq_lens,
                self.group_size,
                self.storage_page_size,
                self.sparse_block_size,
                self.max_seq_len_kv,
                self.release_attention_pdl,
            )


@dataclass(frozen=True)
class _PrimsTSQTokenKvBlockSparsePlan:
    """Prepared compact-metadata and PrimTS-attention launch state.

    The plan binds output/workspace storage and freezes input geometry once.
    Eager calls may pass new input storage with the same shapes, strides,
    devices, and dtypes; CUDA graph replay retains its usual stable-address
    requirement. The hot path performs synchronization-free replacement-storage
    checks followed by already-prepared launches. G1 resolves logical blocks
    directly inside attention, without launching the metadata kernel; grouped
    routes first build their union. The
    original tensors are validated during preparation. Callers must keep all
    replacement tensors alive and disjoint, and must not mutate them
    concurrently with a launch or CUDA-graph replay that reads them. Call
    :meth:`run` once before capture so both metadata and attention kernels are
    compiled and their workspace state is initialized outside the graph.
    """

    _metadata_plan: _PrimsTSQTokenKvBlockSparseMetadataPlan
    _bmm1_scale: float
    _bmm2_scale: float
    _attention_plan: Any
    _query: _QTokenKvBlockSparseTensorDescriptor
    _block_indices: _QTokenKvBlockSparseTensorDescriptor
    _block_table: _QTokenKvBlockSparseTensorDescriptor
    _token_to_request: _QTokenKvBlockSparseTensorDescriptor
    _query_positions: _QTokenKvBlockSparseTensorDescriptor
    _out: _QTokenKvBlockSparseTensorDescriptor
    _fixed_query_group_size: Optional[int] = None

    def run(
        self,
        query: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        out: torch.Tensor,
        sm_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Launch prepared QToken-KvBlock-Sparse-Attention metadata and attention on the current stream."""

        _validate_q_token_kv_block_sparse_plan_tensor(query, "query", self._query)
        _validate_q_token_kv_block_sparse_plan_tensor(
            block_indices, "block_indices", self._block_indices
        )
        _validate_q_token_kv_block_sparse_plan_tensor(
            block_table, "block_table", self._block_table
        )
        _validate_q_token_kv_block_sparse_plan_tensor(
            token_to_request,
            "token_to_request",
            self._token_to_request,
        )
        _validate_q_token_kv_block_sparse_plan_tensor(
            query_positions,
            "query_positions",
            self._query_positions,
        )
        _validate_q_token_kv_block_sparse_plan_tensor(out, "out", self._out)

        from .decode import _validate_16byte_alignment

        _validate_16byte_alignment(query, "query")
        _validate_16byte_alignment(out, "out")

        direct_q1_inputs = None
        if self._attention_plan._direct_q1_inputs:
            direct_q1_inputs = (
                block_indices,
                block_table,
                token_to_request,
                query_positions,
            )
        else:
            self._metadata_plan.run(
                block_indices,
                block_table,
                token_to_request,
                query_positions,
            )
        attention_query = _flatten_fixed_q_token_kv_block_sparse_groups(
            query, self._fixed_query_group_size
        )
        attention_out = _flatten_fixed_q_token_kv_block_sparse_groups(
            out, self._fixed_query_group_size
        )
        from .decode import _validate_scale

        scale_qk = (
            self._bmm1_scale
            if sm_scale is None
            else _validate_scale(sm_scale, "sm_scale")
        )
        scale_v = (
            self._bmm2_scale if v_scale is None else _validate_scale(v_scale, "v_scale")
        )
        self._attention_plan._run_unchecked(
            attention_query,
            attention_out,
            scale_qk,
            scale_v,
            direct_q1_inputs,
        )
        return out


@dataclass(frozen=True)
class _QTokenKvBlockSparsePlanConfig:
    """Capacity and dtype contract published by the public wrapper."""

    batch_size: int
    seq_len_q: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    kv_block_size: int
    page_size: int
    block_topk: int
    max_seq_len_kv: int
    device: torch.device
    workspace_buffer: torch.Tensor
    use_packed_q: bool
    split_kv: bool
    share_pattern_across_kv_heads: bool
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    o_data_type: torch.dtype


def _q_token_tensor_abi_key(
    tensor: Optional[torch.Tensor],
    *,
    include_storage: bool = False,
) -> object:
    """Return the structural ABI, optionally including retained storage."""

    if tensor is None:
        return None
    key: tuple[object, ...] = (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device,
        tensor.dtype,
    )
    return key + (tensor.data_ptr(),) if include_storage else key


class QTokenKvBlockSparsePagedTSWrapper:
    """Plan QToken-KvBlock-Sparse-Attention capacity and run live routes.

    :meth:`plan` fixes query grouping, K/V capacity, dtypes, and the single
    caller-owned workspace. It must run outside CUDA Graph capture. The first
    eager :meth:`run` binds the live tensor ABI, compiles the selected PrimTS
    kernels, and initializes split-KV scratch. Later runs rebuild compact
    routes from the current indexer output and are CUDA-graph capturable.

    One wrapper revision owns mutable route and split-KV workspace. Runs must
    therefore be ordered on one stream or externally synchronized; unordered
    concurrent runs require distinct wrappers.

    Output and workspace must be disjoint from each other and from all live
    inputs, including the original indexer and request metadata. Storage
    overlap is not checked; this precondition also applies after rebinding
    inputs and across CUDA-graph replays.
    """

    def __init__(self) -> None:
        self._config: Optional[_QTokenKvBlockSparsePlanConfig] = None
        self._prepared_plan: Optional[_PrimsTSQTokenKvBlockSparsePlan] = None
        self._prepared_key: Optional[tuple[object, ...]] = None
        self._default_out: Optional[torch.Tensor] = None
        self._plan_lock = threading.Lock()

    def plan(
        self,
        batch_size: int,
        seq_len_q: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_block_size: int,
        page_size: int,
        block_topk: int,
        max_seq_len_kv: int,
        *,
        device: torch.device | str | int,
        workspace_buffer: torch.Tensor,
        use_packed_q: bool = False,
        split_kv: bool = True,
        share_pattern_across_kv_heads: bool = True,
        mask_type: Literal["causal"] = "causal",
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
    ) -> None:
        """Publish a capacity-only QToken-KvBlock-Sparse-Attention plan.

        ``batch_size`` is the number of query routes: ``len(qo_indptr)-1`` for
        packed Q, or ``B * Nq`` for fixed ``[B, Nq, G, Hq, D]`` Q. The planned
        ``seq_len_q`` is the maximum packed route length or fixed group size
        ``G``. ``kv_block_size`` is the indexer's sparse K/V atom and currently
        supports 4/8/16/32/64/128 tokens; ``page_size`` is the physical cache
        page. Blocks may cross cache pages. The loader uses fragments of size
        gcd(kv_block_size, page_size), resolving every fragment independently.

        ``share_pattern_across_kv_heads=True`` preserves the existing shared
        top-k ``[T,K]`` input. False requires ``[T,Hkv,K]`` and prepares one
        union per group and KV head. Q, K, V and output layouts are unchanged.

        ``split_kv=True`` permits automatic useful split fanout; False forces
        one KV partition and omits split scratch. Pass False for prefill and
        True for decode, independently of ``use_packed_q``. Workspace sizing
        must use the same value. Change it only by replanning outside capture.
        """

        from .decode import (
            _resolve_cuda_device,
            _validate_head_dim,
            _validate_head_geometry,
            _validate_positive_int,
            _validate_runtime_device,
        )

        batch_size = _validate_positive_int(batch_size, "batch_size")
        seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
        block_topk = _validate_positive_int(block_topk, "block_topk")
        max_seq_len_kv = _validate_positive_int(max_seq_len_kv, "max_seq_len_kv")
        page_size = _validate_positive_int(page_size, "page_size")
        head_dim = _validate_head_dim(head_dim)
        _validate_head_geometry(num_qo_heads, num_kv_heads)
        kv_block_size = _validate_sparse_block_size(kv_block_size)
        _validate_storage_page_size(page_size)
        _validate_shape_parameters(batch_size * seq_len_q, block_topk, seq_len_q)
        if mask_type != "causal":
            raise ValueError(
                "QToken-KvBlock-Sparse-Attention requires mask_type='causal'"
            )
        if not isinstance(use_packed_q, bool):
            raise TypeError("use_packed_q must be a bool")
        for dtype, name in (
            (q_data_type, "q_data_type"),
            (kv_data_type, "kv_data_type"),
            (o_data_type, "o_data_type"),
        ):
            if dtype is not None and not isinstance(dtype, torch.dtype):
                raise TypeError(f"{name} must be a torch.dtype")
        if kv_data_type is None:
            kv_data_type = q_data_type
        if o_data_type is None:
            o_data_type = q_data_type
        planned_device, device_index = _resolve_cuda_device(device)
        _validate_runtime_device(planned_device)
        if not isinstance(workspace_buffer, torch.Tensor):
            raise TypeError("workspace_buffer must be a torch.Tensor")
        if workspace_buffer.device != planned_device:
            raise ValueError("workspace_buffer must be on the planned device")

        max_num_storage_pages = (max_seq_len_kv + page_size - 1) // page_size
        layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
            batch_size * seq_len_q,
            block_topk,
            max_num_storage_pages,
            page_size,
            seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_data_type,
            kv_dtype=kv_data_type,
            out_dtype=o_data_type,
            device=planned_device,
            num_query_groups=batch_size,
            use_packed_q=use_packed_q,
            sparse_block_size=kv_block_size,
            split_kv=split_kv,
            share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        )
        _validate_q_token_kv_block_sparse_attention_workspace(
            workspace_buffer, layout.total_bytes
        )
        with torch.cuda.device(device_index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "QToken-KvBlock-Sparse-Attention planning is unsupported during "
                    "CUDA Graph capture"
                )
        candidate = _QTokenKvBlockSparsePlanConfig(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_block_size=kv_block_size,
            page_size=page_size,
            block_topk=block_topk,
            max_seq_len_kv=max_seq_len_kv,
            device=planned_device,
            workspace_buffer=workspace_buffer,
            use_packed_q=use_packed_q,
            split_kv=split_kv,
            share_pattern_across_kv_heads=share_pattern_across_kv_heads,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
        )
        with self._plan_lock:
            self._config = candidate
            self._prepared_plan = None
            self._prepared_key = None
            self._default_out = None

    def _require_config(self) -> _QTokenKvBlockSparsePlanConfig:
        config = self._config
        if config is None:
            raise RuntimeError("plan() must be called before run()")
        return config

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        block_table: torch.Tensor,
        indexer_block_ids: torch.Tensor,
        token_to_request: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        qo_indptr: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build live routes and launch the current QToken attention plan.

        Packed Q is ``[total_q, Hq, D]`` with ``qo_indptr``. Fixed Q is
        ``[B, Nq, G, Hq, D]`` without ``qo_indptr``. ``indexer_block_ids`` is
        Int32 ``[total_q, block_topk]`` for shared patterns or
        ``[total_q, Hkv, block_topk]`` for independent patterns, with a
        contiguous last dimension. It never uses the BSR name ``block_indices``.
        K/V are separate HND caches shaped ``[num_pages, Hkv, page_size, D]``.

        For every valid query at zero-based position ``p``, the first
        ``min(block_topk, (p + 1) // kv_block_size)`` indexer entries must be
        distinct completed-block IDs in ``[0, (p + 1) // kv_block_size)``.
        Their order is unrestricted; later entries are ignored. In particular,
        Q1 does not compact missing entries or arbitrary ``-1`` holes inside
        this required prefix. The incomplete causal tail is derived from
        ``query_positions`` and must not be inserted into the selected prefix.

        The KV-cache writer must zero-fill unused token slots in allocated
        pages before attention. TMA loads complete fragments, including the
        partial causal tail; score masking does not sanitize NaN/Inf padding
        in V. Attention does not clear cache padding itself.

        Run once eagerly after :meth:`plan`; capture only subsequent calls with
        stable tensor storage. A changed packed extent or retained K/V/offset
        storage triggers a new preparation outside capture. ``sm_scale`` and
        ``v_scale`` are live launch scalars, so frameworks may update Q/K and V
        dequantization scales without rebuilding the capacity plan.

        Parameters
        ----------
        q : torch.Tensor
            Packed ``[total_q, Hq, D]`` or fixed ``[B, Nq, G, Hq, D]`` query,
            with the planned device, dtype and head geometry.
        paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
            Separate HND K/V caches shaped
            ``[num_pages, Hkv, page_size, D]``.
        block_table : torch.Tensor
            Dense CUDA Int32 physical-page table
            ``[num_requests, max_storage_pages]``.
        indexer_block_ids : torch.Tensor
            CUDA Int32 logical block IDs ``[T, block_topk]`` when the plan
            shares patterns, otherwise ``[T, Hkv, block_topk]``. The last
            dimension must be contiguous; each row supplies the distinct
            completed-block prefix above. The causal tail is added internally.
        token_to_request : torch.Tensor
            CUDA Int32 request ID for each flattened query token.
        query_positions : torch.Tensor
            CUDA Int32 or Int64 zero-based causal position for each query token.
        qo_indptr : torch.Tensor, optional
            CUDA Int32 packed-route offsets. Required for packed Q and omitted
            for fixed Q; each route must stay within one request and contain
            at most the planned ``seq_len_q`` tokens.
        sm_scale : float, optional
            Softmax scale, defaulting to ``head_dim**-0.5``. Include Q/K
            dequantization scales here when using FP8 inputs.
        v_scale : float, optional
            Value-cache dequantization scale applied to the output,
            defaulting to one.
        out : torch.Tensor, optional
            Caller-owned output with Q's logical shape and the planned output
            dtype. When omitted, the wrapper retains and reuses its output.
            Must not overlap any input or workspace storage; this is not checked.

        Returns
        -------
        torch.Tensor
            Attention output in Q's logical layout; returns ``out`` when given.
        """

        config = self._require_config()
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        if q.device != config.device or q.dtype != config.q_data_type:
            raise ValueError("q must use the planned device and q_data_type")
        if config.use_packed_q:
            if q.ndim != 3 or qo_indptr is None:
                raise ValueError("packed Q requires [total_q,Hq,D] and qo_indptr")
            _validate_q_token_kv_block_sparse_qo_indptr_tensor(
                qo_indptr, expected_device=config.device
            )
            if qo_indptr.numel() != config.batch_size + 1:
                raise ValueError("qo_indptr length must match the planned route count")
            if q.shape[0] <= 0 or q.shape[0] > config.batch_size * config.seq_len_q:
                raise ValueError("packed q exceeds the planned query-token capacity")
        else:
            if qo_indptr is not None:
                raise ValueError("fixed Q does not accept qo_indptr")
            if (
                q.ndim != 5
                or q.shape[0] * q.shape[1] != config.batch_size
                or q.shape[2] != config.seq_len_q
            ):
                raise ValueError(
                    "fixed q must be [B,Nq,G,Hq,D] with B*Nq and G matching plan"
                )
        if tuple(q.shape[-2:]) != (config.num_qo_heads, config.head_dim):
            raise ValueError("q must use the planned query-head and head dimensions")
        num_query_tokens = (
            int(q.shape[0])
            if config.use_packed_q
            else (config.batch_size * config.seq_len_q)
        )
        expected_ids_shape = (
            (num_query_tokens, config.block_topk)
            if config.share_pattern_across_kv_heads
            else (num_query_tokens, config.num_kv_heads, config.block_topk)
        )
        if (
            not isinstance(indexer_block_ids, torch.Tensor)
            or indexer_block_ids.shape != expected_ids_shape
        ):
            raise ValueError(f"indexer_block_ids must have shape {expected_ids_shape}")
        for tensor, name in (
            (token_to_request, "token_to_request"),
            (query_positions, "query_positions"),
            (block_table, "block_table"),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if token_to_request.shape != (num_query_tokens,):
            raise ValueError("token_to_request must have one entry per query token")
        if query_positions.shape != (num_query_tokens,):
            raise ValueError("query_positions must have one entry per query token")

        k_cache, v_cache = _validate_q_token_kv_block_sparse_paged_kv_cache(
            paged_kv_cache
        )
        if (
            k_cache.device != config.device
            or k_cache.dtype != config.kv_data_type
            or tuple(k_cache.shape[1:])
            != (config.num_kv_heads, config.page_size, config.head_dim)
        ):
            raise ValueError("paged_kv_cache does not match the planned HND cache ABI")
        required_columns = (
            config.max_seq_len_kv + config.page_size - 1
        ) // config.page_size
        if (
            block_table.ndim != 2
            or block_table.dtype != torch.int32
            or block_table.device != config.device
            or block_table.shape[1] < required_columns
        ):
            raise ValueError(
                "block_table must be CUDA Int32 [requests, pages] with planned capacity"
            )
        if out is None:
            if self._default_out is None or self._default_out.shape != q.shape:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "run once eagerly before capture when out is not supplied"
                    )
                self._default_out = torch.empty_like(q, dtype=config.o_data_type)
            out = self._default_out
        elif not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        elif (
            out.shape != q.shape
            or out.device != config.device
            or out.dtype != config.o_data_type
        ):
            raise ValueError("out must match q shape and the planned output dtype")

        prepared_key = (
            _q_token_tensor_abi_key(q),
            _q_token_tensor_abi_key(indexer_block_ids),
            _q_token_tensor_abi_key(block_table),
            _q_token_tensor_abi_key(token_to_request),
            _q_token_tensor_abi_key(query_positions),
            _q_token_tensor_abi_key(out),
            _q_token_tensor_abi_key(k_cache, include_storage=True),
            _q_token_tensor_abi_key(v_cache, include_storage=True),
            _q_token_tensor_abi_key(qo_indptr, include_storage=True),
        )
        if self._prepared_plan is None or prepared_key != self._prepared_key:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "run once eagerly after changing the QToken tensor ABI"
                )
            with self._plan_lock:
                if self._prepared_plan is None or prepared_key != self._prepared_key:
                    self._prepared_plan = _prepare_q_token_kv_block_sparse_attention(
                        q,
                        paged_kv_cache,
                        indexer_block_ids,
                        block_table,
                        token_to_request,
                        query_positions,
                        config.workspace_buffer,
                        out=out,
                        max_seq_len_kv=config.max_seq_len_kv,
                        bmm1_scale=sm_scale,
                        qo_indptr=qo_indptr,
                        max_seq_len_q=(
                            config.seq_len_q if config.use_packed_q else None
                        ),
                        sparse_block_size=config.kv_block_size,
                        split_kv=config.split_kv,
                        share_pattern_across_kv_heads=config.share_pattern_across_kv_heads,
                    )
                    self._prepared_key = prepared_key
        assert self._prepared_plan is not None
        return self._prepared_plan.run(
            q,
            indexer_block_ids,
            block_table,
            token_to_request,
            query_positions,
            out=out,
            sm_scale=sm_scale,
            v_scale=v_scale,
        )


def _flatten_fixed_q_token_kv_block_sparse_groups(
    tensor: torch.Tensor,
    group_size: Optional[int],
) -> torch.Tensor:
    """Return the lower-level decode view for a canonical fixed QToken-KvBlock-Sparse-Attention tensor."""

    if group_size is None:
        return tensor
    flattened = tensor.flatten(0, 1)
    return flattened.squeeze(1) if group_size == 1 else flattened


def _get_q_token_kv_block_sparse_metadata_output_shapes(
    num_query_tokens: int,
    block_topk: int,
    group_size: int,
    *,
    num_query_groups: Optional[int] = None,
    sparse_block_size: int = 4,
    storage_page_size: Optional[int] = None,
    num_pattern_heads: int = 1,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens)`` shapes.

    Physical fragments have size gcd(sparse_block_size, storage_page_size).
    Metadata has one row per query group and pattern head. Grouped masks use
    four packed bytes per Int32 word. Q1 does not consume membership metadata
    and returns a zero-width membership shape.

    Parameters
    ----------
    num_query_tokens : int
        Number of flattened query rows represented by the metadata.
    block_topk : int
        Number of selected logical sparse blocks supplied for each query row.
    group_size : int
        Maximum number of query rows represented by one QToken-KvBlock-Sparse-Attention route.
    num_query_groups : int, optional
        Explicit route count for a packed query. If omitted, query rows are
        partitioned into fixed groups and ``num_query_tokens`` must be
        divisible by ``group_size``.
    sparse_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.
    storage_page_size : int, optional
        Physical storage-page size; defaults to sparse_block_size. A block
        contains sparse_block_size / gcd(sparse_block_size, storage_page_size)
        independently addressable fragments.
    num_pattern_heads : int
        One for shared patterns, otherwise the number of KV heads.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        Shapes for ``q_token_kv_block_sparse_page_indices``, ``q_token_kv_block_sparse_page_memberships``, and
        ``seq_lens``, in that order.
    """

    _validate_sparse_block_size(sparse_block_size)
    _validate_shape_parameters(num_query_tokens, block_topk, group_size)
    groups = _resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    if storage_page_size is None:
        storage_page_size = sparse_block_size
    _validate_storage_page_size(storage_page_size)
    fragments_per_block = sparse_block_size // math.gcd(
        sparse_block_size, storage_page_size
    )
    page_capacity = group_size * (block_topk + 1) * fragments_per_block
    groups *= num_pattern_heads
    membership_words = (
        0
        if group_size == 1
        else (page_capacity + _Q_TOKEN_KV_BLOCK_SPARSE_MEMBERSHIPS_PER_WORD - 1)
        // _Q_TOKEN_KV_BLOCK_SPARSE_MEMBERSHIPS_PER_WORD
    )
    return (groups, page_capacity), (groups, membership_words), (groups,)


def _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
    num_query_tokens: int,
    block_topk: int,
    max_num_storage_pages: int,
    storage_page_size: int,
    group_size: int,
    *,
    max_seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype = torch.float16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str | int] = None,
    num_query_groups: Optional[int] = None,
    use_packed_q: bool = False,
    sparse_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> _PrimsTSQTokenKvBlockSparseWorkspaceLayout:
    """Return the unified allocation layout for QToken-KvBlock-Sparse-Attention metadata and attention.

    The workspace owns the dense QToken-KvBlock-Sparse-Attention page-index table, grouped membership
    words, compact sequence lengths, and kernel scratch. The caller continues
    to provide the model block table, request mappings, and query positions as
    explicit semantic inputs.

    Metadata and attention use disjoint regions. The caller controls whether
    split-KV may be selected, independently of query layout. Split launches
    own dedicated partial-output, statistics and completion-counter storage.
    """

    sparse_block_size = _validate_sparse_block_size(sparse_block_size)
    _validate_shape_parameters(num_query_tokens, block_topk, group_size)
    if num_query_tokens == 0:
        raise ValueError(
            "num_query_tokens must be positive for QToken-KvBlock-Sparse-Attention attention"
        )
    if not isinstance(split_kv, bool):
        raise TypeError("split_kv must be a bool")
    if not isinstance(max_num_storage_pages, int) or isinstance(
        max_num_storage_pages, bool
    ):
        raise TypeError("max_num_storage_pages must be an integer")
    if max_num_storage_pages <= 0:
        raise ValueError("max_num_storage_pages must be positive")
    _validate_storage_page_size(storage_page_size)
    max_seq_len_kv = _validate_q_token_kv_block_sparse_max_seq_len_kv(
        max_seq_len_kv,
        block_table_token_capacity=max_num_storage_pages * storage_page_size,
    )

    from .decode import (
        _resolve_decode_workspace_layout,
        _validate_prims_ts_q_token_kv_block_sparse_group_capacity,
    )

    group_size = _validate_prims_ts_q_token_kv_block_sparse_group_capacity(
        group_size,
        num_qo_heads,
        num_kv_heads,
    )

    groups = _resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    if not isinstance(share_pattern_across_kv_heads, bool):
        raise TypeError("share_pattern_across_kv_heads must be a bool")
    pattern_heads = _num_sparse_pattern_heads(
        num_kv_heads, share_pattern_across_kv_heads
    )
    metadata_rows = groups * pattern_heads
    fragment_size = math.gcd(sparse_block_size, storage_page_size)
    block_capacity = min(
        group_size * (block_topk + 1),
        (max_seq_len_kv + sparse_block_size - 1) // sparse_block_size,
    )
    page_capacity = block_capacity * (sparse_block_size // fragment_size)
    membership_words = (
        0
        if group_size == 1
        else (page_capacity + _Q_TOKEN_KV_BLOCK_SPARSE_MEMBERSHIPS_PER_WORD - 1)
        // _Q_TOKEN_KV_BLOCK_SPARSE_MEMBERSHIPS_PER_WORD
    )
    q_token_kv_block_sparse_page_indices_shape = (metadata_rows, page_capacity)
    q_token_kv_block_sparse_page_indices_bytes = metadata_rows * page_capacity * 4
    q_token_kv_block_sparse_page_memberships_shape = (metadata_rows, membership_words)
    q_token_kv_block_sparse_page_memberships_byte_offset = (
        _align_up_q_token_kv_block_sparse_workspace(
            q_token_kv_block_sparse_page_indices_bytes
        )
    )
    q_token_kv_block_sparse_page_memberships_bytes = (
        metadata_rows * membership_words * 4
    )
    seq_lens_byte_offset = _align_up_q_token_kv_block_sparse_workspace(
        q_token_kv_block_sparse_page_memberships_byte_offset
        + q_token_kv_block_sparse_page_memberships_bytes
    )
    seq_lens_numel = metadata_rows
    seq_lens_bytes = seq_lens_numel * 4
    max_seq_len = (
        block_topk * sparse_block_size + (sparse_block_size - 1)
        if group_size == 1
        else page_capacity * fragment_size
    )
    max_seq_len = min(max_seq_len, max_seq_len_kv)
    if kv_dtype is None:
        kv_dtype = q_dtype
    if out_dtype is None:
        out_dtype = q_dtype
    attention_layout = _resolve_decode_workspace_layout(
        groups,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        fragment_size,
        max_seq_len,
        group_size,
        q_dtype,
        kv_dtype,
        kv_dtype,
        out_dtype,
        "HND",
        "causal",
        use_packed_q,
        -1,
        storage_page_size,
        device,
        use_q_token_kv_block_sparse_route=True,
        use_pdl=True,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
    )
    attention_scratch_bytes = attention_layout.total_bytes
    attention_workspace_byte_offset = _align_up_q_token_kv_block_sparse_workspace(
        seq_lens_byte_offset + seq_lens_bytes
    )
    return _PrimsTSQTokenKvBlockSparseWorkspaceLayout(
        q_token_kv_block_sparse_page_indices_shape=q_token_kv_block_sparse_page_indices_shape,
        q_token_kv_block_sparse_page_indices_bytes=q_token_kv_block_sparse_page_indices_bytes,
        q_token_kv_block_sparse_page_memberships_shape=q_token_kv_block_sparse_page_memberships_shape,
        q_token_kv_block_sparse_page_memberships_byte_offset=q_token_kv_block_sparse_page_memberships_byte_offset,
        q_token_kv_block_sparse_page_memberships_bytes=q_token_kv_block_sparse_page_memberships_bytes,
        seq_lens_byte_offset=seq_lens_byte_offset,
        seq_lens_bytes=seq_lens_bytes,
        attention_workspace_byte_offset=attention_workspace_byte_offset,
        attention_scratch_bytes=attention_scratch_bytes,
        uses_split_kv=attention_layout.uses_split_kv,
        max_seq_len=max_seq_len,
        total_bytes=attention_workspace_byte_offset + attention_scratch_bytes,
    )


@flashinfer_experimental_api
def get_q_token_kv_block_sparse_workspace_size(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_topk: int,
    max_seq_len_kv: int,
    o_data_type: Optional[torch.dtype] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    seq_len_q: Optional[int] = None,
    kv_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> int:
    """Return bytes for QToken-KvBlock-Sparse-Attention workspace storage.

    Packed queries use ``[num_query_tokens, num_qo_heads, head_dim]`` and
    Int32 ``qo_indptr`` partitions their rows into request-safe routes of
    length at most ``seq_len_q``. Fixed queries use
    ``[batch, num_query_groups, group_size, num_qo_heads, head_dim]`` and do
    not require query-offset metadata. The first two fixed axes are flattened
    into the attention route axis without copying tensor storage.

    The dense QToken-KvBlock-Sparse-Attention page indices, grouped membership words, and compact sequence
    lengths are owned by the workspace. These persistent metadata outputs and
    the attention scratch occupy disjoint regions, and callers do not allocate
    any metadata output separately.
    The attention region contains split-KV partials, statistics, and counters
    only when the resolved policy uses split-KV; direct prefill retains only
    small uniform-ABI placeholders.
    ``kv_block_size`` supports 4/8/16/32/64/128 tokens. Shared patterns use
    one metadata row per query group; independent patterns use one per group
    and KV head. Storage-page crossings increase the number of physical
    fragments, not the number of logical candidates sorted.

    Parameters
    ----------
    q : torch.Tensor
        Packed ``[total_q, Hq, D]`` query when ``qo_indptr`` is supplied, or
        contiguous fixed ``[B, Nq, G, Hq, D]`` query otherwise.
    k_cache : torch.Tensor
        HND key cache shaped ``[num_pages, Hkv, storage_page_size, D]``. The
        matching value cache does not affect workspace sizing.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table shaped
        ``[num_requests, max_storage_pages]``.
    block_topk : int
        Number of selected logical sparse blocks supplied for each query row.
    max_seq_len_kv : int
        Static maximum visible logical K/V length in tokens, including current
        query or MTP tokens. It may be smaller than the reserved block-table
        capacity and must not change across replay of a prepared CUDA graph.
    o_data_type : torch.dtype, optional
        Output dtype used to size attention partials. It defaults to the query
        dtype.
    qo_indptr : torch.Tensor, optional
        Contiguous CUDA Int32 cumulative offsets for packed query routes.
    seq_len_q : int, optional
        Maximum packed-route length and QToken-KvBlock-Sparse-Attention group size. It is required when
        ``qo_indptr`` is supplied.
    kv_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.
    split_kv : bool
        Allow automatic split-KV selection (default True), independently of
        query layout. Use False for prefill and True for decode. Pass the same
        value to attention planning; False omits split-KV scratch.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    Returns
    -------
    int
        Required unified workspace size in bytes, including persistent QToken-KvBlock-Sparse-Attention
        metadata outputs and a disjoint attention-scratch region.
    """

    return _get_prims_ts_q_token_kv_block_sparse_workspace_layout_from_tensors(
        q,
        k_cache,
        block_table,
        block_topk,
        max_seq_len_kv=max_seq_len_kv,
        out_dtype=o_data_type,
        qo_indptr=qo_indptr,
        max_seq_len_q=seq_len_q,
        sparse_block_size=kv_block_size,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
    ).total_bytes


def _get_prims_ts_q_token_kv_block_sparse_workspace_layout_from_tensors(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_topk: int,
    *,
    max_seq_len_kv: int,
    out_dtype: Optional[torch.dtype],
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    sparse_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> _PrimsTSQTokenKvBlockSparseWorkspaceLayout:
    sparse_block_size = _validate_sparse_block_size(sparse_block_size)
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("k_cache must be a torch.Tensor")
    if not isinstance(block_table, torch.Tensor):
        raise TypeError("block_table must be a torch.Tensor")
    use_packed_q = qo_indptr is not None
    if not use_packed_q and max_seq_len_q is not None:
        raise ValueError(
            "max_seq_len_q is only valid with packed QToken-KvBlock-Sparse-Attention qo_indptr"
        )
    if use_packed_q:
        if query.ndim != 3:
            raise ValueError(
                "packed QToken-KvBlock-Sparse-Attention query must have shape [total_q,Hq,D]"
            )
        if max_seq_len_q is None:
            raise ValueError(
                "max_seq_len_q is required with QToken-KvBlock-Sparse-Attention qo_indptr"
            )
        _validate_q_token_kv_block_sparse_qo_indptr_layout_tensor(qo_indptr)
        group_size = int(max_seq_len_q)
        groups = int(qo_indptr.numel()) - 1
        num_query_tokens, num_qo_heads, head_dim = query.shape
    elif query.ndim == 5:
        batch_size, groups_per_request, group_size, num_qo_heads, head_dim = query.shape
        if batch_size <= 0 or groups_per_request <= 0:
            raise ValueError(
                "fixed QToken-KvBlock-Sparse-Attention batch and query-group counts must be positive"
            )
        groups = int(batch_size) * int(groups_per_request)
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,G,Hq,D] without qo_indptr"
        )
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [pages, Hkv, storage_page_size, D]")
    if k_cache.device != query.device or k_cache.shape[3] != head_dim:
        raise ValueError("query and k_cache must share device and head dimension")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a rank-two int32 tensor")
    if block_table.device != query.device:
        raise ValueError("block_table must be on the query device")
    _validate_q_token_kv_block_sparse_locator_capacity(
        k_cache,
        sparse_block_size=sparse_block_size,
    )
    if out_dtype is None:
        out_dtype = query.dtype
    return _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
        int(query.shape[0]) if use_packed_q else groups * group_size,
        block_topk,
        block_table.shape[1],
        k_cache.shape[2],
        group_size,
        max_seq_len_kv=max_seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=k_cache.shape[1],
        head_dim=head_dim,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=out_dtype,
        device=query.device,
        num_query_groups=groups,
        use_packed_q=use_packed_q,
        sparse_block_size=sparse_block_size,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
    )


def _build_prims_ts_q_token_kv_block_sparse_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int = 4,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    release_attention_pdl: bool = False,
    share_pattern_across_kv_heads: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense QToken-KvBlock-Sparse-Attention indices and memberships from compact sparse-block IDs.

    ``block_indices`` contains logical sparse-block IDs with shape
    ``[num_query_tokens, block_topk]`` when shared, or ``[num_query_tokens,
    Hkv, block_topk]`` for per-head selections.  Adjacent Q2--Q8 rows must belong to
    the same request and have consecutive positions.  Q1 output locators are
    ordinary encoded subpage locators. Grouped routes return the same plain
    locators plus a separate Int32 table containing four 8-bit membership masks
    per word; membership bit ``i`` marks visibility for query ``i`` in the
    group. Valid locators must fit in nonnegative signed Int32. The combined
    attention APIs prove this from the cache capacity, while callers of this
    advanced metadata-only API own that input-value contract.

    Pass preallocated ``out`` tensors for CUDA graph capture. The page-index
    table has a fixed row width of
    ``group_size * (block_topk + 1) * fragments_per_block``; ``seq_lens``
    selects the live token prefix. Fragments divide both logical blocks and
    storage pages, so physical page IDs never imply logical adjacency.
    The membership output has one Int32 word per four page slots, or zero
    columns for Q1.
    Q1 uses one CUDA C++ CTA per route to map its selected blocks and causal
    tail directly. Q2--Q8 use one CUDA C++ CTA per route to radix-sort the
    bounded ``group_size * (block_topk + 1)`` candidates, unique them while
    OR-reducing membership bits, and map the resulting logical pages through
    the dense block table. The terminal metadata grid releases a following
    PDL-capable QToken-KvBlock-Sparse-Attention attention launch when requested by the combined API.
    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets; this builder checks only its structural tensor contract and does
    not read route values back to the host.
    """

    _validate_inputs(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size,
        storage_page_size,
        max_seq_len_kv,
        sparse_block_size,
        share_pattern_across_kv_heads,
    )
    rows, block_topk = block_indices.shape[0], block_indices.shape[-1]
    pattern_heads = block_indices.shape[1] if block_indices.ndim == 3 else 1
    use_packed_q = qo_indptr is not None
    if use_packed_q:
        _validate_q_token_kv_block_sparse_qo_indptr_tensor(
            qo_indptr,
            expected_device=block_indices.device,
        )
    groups = int(qo_indptr.numel()) - 1 if use_packed_q else None
    expected_shapes = _get_q_token_kv_block_sparse_metadata_output_shapes(
        rows,
        block_topk,
        group_size,
        num_query_groups=groups,
        sparse_block_size=sparse_block_size,
        storage_page_size=storage_page_size,
        num_pattern_heads=pattern_heads,
    )
    if out is None:
        outputs = tuple(
            torch.empty(shape, dtype=torch.int32, device=block_indices.device)
            for shape in expected_shapes
        )
    else:
        if not isinstance(out, tuple) or len(out) != 3:
            raise TypeError(
                "out must be a (q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens) tuple"
            )
        outputs = out
        for tensor, shape in zip(outputs, expected_shapes, strict=True):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.int32
                or tensor.device != block_indices.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "QToken-KvBlock-Sparse-Attention metadata outputs must be contiguous int32 tensors with "
                    f"shapes {expected_shapes} on {block_indices.device}"
                )
    (
        q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships,
        seq_lens,
    ) = outputs
    groups = _resolve_num_query_groups(rows, group_size, groups)
    if groups == 0:
        return outputs

    enable_pdl = device_support_pdl(block_indices.device)
    metadata_module = _get_prims_ts_q_token_kv_block_sparse_metadata_module()
    release_attention_pdl = enable_pdl and release_attention_pdl
    if use_packed_q:
        metadata_module.run_packed(
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            qo_indptr,
            q_token_kv_block_sparse_page_indices,
            q_token_kv_block_sparse_page_memberships,
            seq_lens,
            group_size,
            storage_page_size,
            sparse_block_size,
            max_seq_len_kv,
            release_attention_pdl,
        )
    else:
        metadata_module.run_fixed(
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            q_token_kv_block_sparse_page_indices,
            q_token_kv_block_sparse_page_memberships,
            seq_lens,
            group_size,
            storage_page_size,
            sparse_block_size,
            max_seq_len_kv,
            release_attention_pdl,
        )
    return outputs


def _build_q_token_kv_block_sparse_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int = 4,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    share_pattern_across_kv_heads: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense QToken-KvBlock-Sparse-Attention page indices and memberships from sparse-block IDs.

    ``block_indices`` is ``[T,K]`` when shared, otherwise ``[T,Hkv,K]``.
    ``block_table`` remains the dense ``[requests, max_storage_pages]`` physical
    table. Let F = gcd(sparse_block_size, storage_page_size). Output rows number
    groups * pattern_heads, with width G * (K+1) * (sparse_block_size/F).
    ``seq_lens`` gives each row's live token prefix. Memberships pack four
    fragment bytes per Int32; G1 has zero membership columns. Every locator is
    physical_page * (storage_page_size/F) + subpage. No CSR conversion is needed.

    Q1 uses one CUDA C++ direct-mapping kernel. Q2--Q8 use one CUDA C++
    radix-sort and union kernel per route. Neither path requires
    caller-provided scratch. Advanced callers that capture this raw path must
    preallocate ``out``, warm the same metadata geometry once before capture,
    and retain every tensor at a stable address through replay. The three
    ``out`` storage ranges must be mutually disjoint and must not alias inputs.
    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets and is checked structurally without a host-side value read.

    ``sparse_block_size`` defaults to four. The API validates power-of-two
    values in {4,8,16,32,64,128}. Logical blocks may cross physical cache
    pages; metadata emits only the physical fragments the loader needs.

    Parameters
    ----------
    block_indices : torch.Tensor
        CUDA Int32 selected logical sparse-block IDs shaped ``[T,K]`` or
        ``[T,Hkv,K]``, with a contiguous K dimension.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table shaped
        ``[num_requests, max_storage_pages]`` with contiguous rows.
    token_to_request : torch.Tensor
        Contiguous CUDA Int32 request index for each flattened query row,
        shaped ``[num_query_tokens]``.
    query_positions : torch.Tensor
        Contiguous CUDA Int32 or Int64 absolute position for each flattened
        query row, shaped ``[num_query_tokens]``.
    group_size : int
        Maximum number of query rows represented by one route.
    storage_page_size : int
        Number of tokens in each physical K/V storage page.
    max_seq_len_kv : int
        Static maximum visible logical K/V length in tokens, including current
        query or MTP tokens. It must fit within each dense block-table row.
    sparse_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.
    qo_indptr : torch.Tensor, optional
        Contiguous CUDA Int32 cumulative route offsets for packed queries. If
        omitted, adjacent rows form fixed groups of exactly ``group_size``.
    out : tuple[torch.Tensor, torch.Tensor, torch.Tensor], optional
        Preallocated contiguous CUDA Int32 tensors for page indices, packed
        memberships, and live sequence lengths. Their shapes must match
        :func:`_get_q_token_kv_block_sparse_metadata_output_shapes`. If omitted, the three
        outputs are allocated internally.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens)``. Memberships
        pack four per-page bytes into each Int32 word and have zero columns for
        Q1.
    """

    return _build_prims_ts_q_token_kv_block_sparse_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        sparse_block_size=sparse_block_size,
        qo_indptr=qo_indptr,
        out=out,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
    )


def _prepare_prims_ts_q_token_kv_block_sparse_metadata_plan(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    q_token_kv_block_sparse_page_indices: torch.Tensor,
    q_token_kv_block_sparse_page_memberships: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int = 4,
    qo_indptr: Optional[torch.Tensor] = None,
) -> _PrimsTSQTokenKvBlockSparseMetadataPlan:
    """Freeze validated metadata tensors and launch constants."""

    sparse_block_size = _validate_sparse_block_size(sparse_block_size)
    rows = block_indices.shape[0]
    use_packed_q = qo_indptr is not None
    _resolve_num_query_groups(
        rows,
        group_size,
        int(qo_indptr.numel()) - 1 if use_packed_q else None,
    )
    max_seq_len_kv = _validate_q_token_kv_block_sparse_max_seq_len_kv(
        max_seq_len_kv,
        block_table_token_capacity=block_table.shape[1] * storage_page_size,
    )
    release_attention_pdl = device_support_pdl(block_indices.device)
    metadata_module = _get_prims_ts_q_token_kv_block_sparse_metadata_module()
    metadata_run = (
        metadata_module.run_packed if use_packed_q else metadata_module.run_fixed
    )
    return _PrimsTSQTokenKvBlockSparseMetadataPlan(
        q_token_kv_block_sparse_page_indices=q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
        seq_lens=seq_lens,
        qo_indptr=qo_indptr,
        metadata_run=metadata_run,
        group_size=group_size,
        use_packed_q=use_packed_q,
        sparse_block_size=sparse_block_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        release_attention_pdl=release_attention_pdl,
    )


def _prepare_q_token_kv_block_sparse_attention(
    query: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    out: torch.Tensor,
    max_seq_len_kv: int,
    bmm1_scale: Optional[float] = None,
    bmm2_scale: float = 1.0,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    sparse_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> _PrimsTSQTokenKvBlockSparsePlan:
    """Prepare one graph-stable metadata-plus-attention QToken-KvBlock-Sparse-Attention launch.

    Packed Q/output use ``[num_query_tokens, Hq, D]`` and describe Q1--Q8
    routes with ``qo_indptr`` and ``max_seq_len_q``. A request's final route
    may be shorter than the maximum group size. Fixed Q/output use
    ``[B, Nq, G, Hq, D]`` and omit ``qo_indptr``; ``B * Nq`` becomes the
    internal attention route count through a zero-copy view.

    Frameworks call :meth:`_PrimsTSQTokenKvBlockSparsePlan.run` with current inputs that preserve
    the prepared geometry. The plan owns its dense page-index table, grouped
    membership words, and compact sequence lengths inside ``workspace_buffer``
    and resolves metadata geometry and the PrimTS attention launch only once.
    Packed ``qo_indptr`` must be an
    Int32 device copy of CPU-validated route offsets, such as those from
    :func:`make_q_token_kv_block_sparse_qo_indptr`. Preparation checks only the
    structural tensor contract and does not materialize values on the host.

    Call ``run`` once outside CUDA graph capture to compile and initialize the
    plan, then capture with all semantic inputs, output, and workspace storage
    kept at stable addresses.

    G1 reads the raw indexer inputs directly inside attention and does not
    materialize the workspace's page-index or sequence-length views. The
    workspace size remains a conservative capacity, shared with grouped routes.

    ``sparse_block_size`` defaults to four and selects the semantic block
    specialization in {4, 8, 16, 32, 64, 128}.

    Parameters
    ----------
    query : torch.Tensor
        Packed ``[total_q, Hq, D]`` query when ``qo_indptr`` is supplied, or
        contiguous fixed ``[B, Nq, G, Hq, D]`` query otherwise.
    paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
        Separate HND key and value caches, each shaped
        ``[num_pages, Hkv, storage_page_size, D]`` with matching shape, dtype,
        and device.
    block_indices : torch.Tensor
        CUDA Int32 logical sparse-block IDs shaped ``[T, block_topk]`` when
        shared, otherwise ``[T, Hkv, block_topk]``, with a contiguous last dimension.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table shaped
        ``[num_requests, max_storage_pages]``.
    token_to_request : torch.Tensor
        Contiguous CUDA Int32 request index for each flattened query row.
    query_positions : torch.Tensor
        Contiguous CUDA Int32 or Int64 absolute position for each flattened
        query row.
    workspace_buffer : torch.Tensor
        Caller-owned contiguous CUDA byte workspace with at least the size
        returned by :func:`get_q_token_kv_block_sparse_workspace_size`. Its storage must
        remain stable and disjoint from all inputs and ``out``.
    out : torch.Tensor
        Caller-owned output with the same logical shape as ``query``.
    max_seq_len_kv : int
        Static maximum visible logical K/V length in tokens, including current
        query or MTP tokens. It must fit within each dense block-table row and
        remain unchanged across CUDA graph replay.
    bmm1_scale : float, optional
        Scale applied to QK scores. It defaults to ``head_dim**-0.5``.
    bmm2_scale : float
        Scale applied to the attention output.
    qo_indptr : torch.Tensor, optional
        Contiguous CUDA Int32 cumulative route offsets selecting packed-query
        mode.
    max_seq_len_q : int, optional
        Maximum packed-route length and QToken-KvBlock-Sparse-Attention group size. It is required when
        ``qo_indptr`` is supplied.
    sparse_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    Returns
    -------
    _PrimsTSQTokenKvBlockSparsePlan
        Reusable prepared metadata-plus-attention plan. Call
        :meth:`_PrimsTSQTokenKvBlockSparsePlan.run` once before CUDA graph capture.
    """

    sparse_block_size = _validate_sparse_block_size(sparse_block_size)
    k_cache, v_cache = _validate_q_token_kv_block_sparse_paged_kv_cache(paged_kv_cache)
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a caller-owned torch.Tensor")

    use_packed_q = qo_indptr is not None
    fixed_query_group_size = None
    if use_packed_q:
        if query.ndim != 3:
            raise ValueError(
                "packed QToken-KvBlock-Sparse-Attention query must have shape [total_q,Hq,D]"
            )
        if max_seq_len_q is None:
            raise ValueError(
                "max_seq_len_q is required with QToken-KvBlock-Sparse-Attention qo_indptr"
            )
        group_size = int(max_seq_len_q)
        num_query_tokens = int(query.shape[0])
        _validate_q_token_kv_block_sparse_qo_indptr_tensor(
            qo_indptr,
            expected_device=query.device,
        )
        attention_query = query
        attention_out = out
    elif query.ndim == 5:
        if out.shape != query.shape:
            raise ValueError(
                "fixed QToken-KvBlock-Sparse-Attention output must have the same shape as query"
            )
        if not query.is_contiguous() or not out.is_contiguous():
            raise ValueError(
                "fixed QToken-KvBlock-Sparse-Attention query and output must be contiguous"
            )
        group_size = int(query.shape[2])
        num_query_tokens = int(query.shape[0] * query.shape[1]) * group_size
        attention_query = _flatten_fixed_q_token_kv_block_sparse_groups(
            query, group_size
        )
        attention_out = _flatten_fixed_q_token_kv_block_sparse_groups(out, group_size)
        fixed_query_group_size = group_size
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,G,Hq,D] without qo_indptr"
        )

    _validate_inputs(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size,
        int(k_cache.shape[2]),
        max_seq_len_kv,
        sparse_block_size,
        share_pattern_across_kv_heads,
    )
    if not share_pattern_across_kv_heads and block_indices.shape[1] != k_cache.shape[1]:
        raise ValueError("block_indices pattern head count must match the KV cache")
    if block_indices.shape[0] != num_query_tokens:
        raise ValueError("block_indices must have one row per flattened query token")

    layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout_from_tensors(
        query,
        k_cache,
        block_table,
        int(block_indices.shape[-1]),
        max_seq_len_kv=max_seq_len_kv,
        out_dtype=out.dtype,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        sparse_block_size=sparse_block_size,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
    )
    views = layout.bind(workspace_buffer)
    metadata_plan = _prepare_prims_ts_q_token_kv_block_sparse_metadata_plan(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        views.q_token_kv_block_sparse_page_indices,
        views.q_token_kv_block_sparse_page_memberships,
        views.seq_lens,
        group_size=group_size,
        storage_page_size=int(k_cache.shape[2]),
        max_seq_len_kv=max_seq_len_kv,
        sparse_block_size=sparse_block_size,
        qo_indptr=qo_indptr,
    )
    from .decode import _prepare_prims_ts_batch_decode_plan, _validate_scale

    scale_qk = _validate_scale(
        query.shape[-1] ** -0.5 if bmm1_scale is None else bmm1_scale,
        "bmm1_scale",
    )
    scale_v = _validate_scale(bmm2_scale, "bmm2_scale")
    attention_plan, _ = _prepare_prims_ts_batch_decode_plan(
        attention_query,
        paged_kv_cache,
        views.attention_workspace_buffer,
        views.q_token_kv_block_sparse_page_indices,
        views.seq_lens,
        layout.max_seq_len,
        out=attention_out,
        seq_len_q=group_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size if use_packed_q else None,
        out_dtype=out.dtype,
        mask_type="causal",
        window_left=-1,
        kv_layout="HND",
        page_size=math.gcd(sparse_block_size, int(k_cache.shape[2])),
        direct_q1_sparse_block_size=sparse_block_size,
        use_q_token_kv_block_sparse_route=True,
        use_pdl=True,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        q_token_kv_block_sparse_page_memberships=(
            views.q_token_kv_block_sparse_page_memberships if group_size > 1 else None
        ),
        direct_q1_inputs=(
            (block_indices, block_table, token_to_request, query_positions)
            if group_size == 1
            else ()
        ),
        direct_q1_max_model_len=max_seq_len_kv if group_size == 1 else None,
    )
    return _PrimsTSQTokenKvBlockSparsePlan(
        _metadata_plan=metadata_plan,
        _bmm1_scale=scale_qk,
        _bmm2_scale=scale_v,
        _attention_plan=attention_plan,
        _query=_describe_q_token_kv_block_sparse_tensor(query),
        _block_indices=_describe_q_token_kv_block_sparse_tensor(block_indices),
        _block_table=_describe_q_token_kv_block_sparse_tensor(block_table),
        _token_to_request=_describe_q_token_kv_block_sparse_tensor(token_to_request),
        _query_positions=_describe_q_token_kv_block_sparse_tensor(query_positions),
        _out=_describe_q_token_kv_block_sparse_tensor(out),
        _fixed_query_group_size=fixed_query_group_size,
    )


@flashinfer_experimental_api
def q_token_kv_block_sparse_attention_with_paged_kv_cache(
    q: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    indexer_block_ids: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    max_seq_len_kv: int,
    seq_len_q: Optional[int] = None,
    kv_block_size: int = 4,
    mask_type: Literal["causal"] = "causal",
    sm_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    o_data_type: Optional[torch.dtype] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
) -> torch.Tensor:
    """Plan and run one QToken-KvBlock-Sparse-Attention launch.

    This eager convenience API creates a temporary
    :class:`QTokenKvBlockSparsePagedTSWrapper`, plans capacity, and launches
    once. It is not CUDA-graph capturable. Repeated launches should retain a
    wrapper, call :meth:`QTokenKvBlockSparsePagedTSWrapper.plan` outside
    capture, run once eagerly, and capture only :meth:`~QTokenKvBlockSparsePagedTSWrapper.run`.

    Packed Q is ``[total_q, Hq, D]`` with ``qo_indptr``; ``seq_len_q``
    is its maximum route length. Fixed Q is ``[B, Nq, G, Hq, D]`` without
    ``qo_indptr``, and ``G`` determines ``seq_len_q``. The indexer
    output is named ``indexer_block_ids`` to distinguish it from BSR
    ``block_indices``. ``kv_block_size`` is the semantic indexer atom
    (4/8/16/32/64/128), while the physical cache extent is inferred from
    ``paged_kv_cache``.

    Parameters
    ----------
    q : torch.Tensor
        Packed ``[total_q, Hq, D]`` or fixed ``[B, Nq, G, Hq, D]`` query.
    paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
        Separate HND K/V cache tensors shaped
        ``[num_pages, Hkv, page_size, D]``. The cache writer must zero-fill
        unused token slots in allocated pages, as required by
        :meth:`QTokenKvBlockSparsePagedTSWrapper.run`.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table
        ``[num_requests, max_storage_pages]``.
    indexer_block_ids : torch.Tensor
        CUDA Int32 indexer-selected logical K/V blocks ``[T, block_topk]``
        when shared, otherwise ``[T, Hkv, block_topk]``, with a contiguous last
        dimension. Each valid row provides the completed-block prefix described by
        :meth:`QTokenKvBlockSparsePagedTSWrapper.run`; Q1 does not compact
        missing entries inside that prefix. Tail tokens are derived separately.
    token_to_request : torch.Tensor
        CUDA Int32 request ID for each flattened query token.
    query_positions : torch.Tensor
        CUDA Int32 or Int64 causal position for each query token.
    workspace_buffer : torch.Tensor
        Caller-owned byte workspace sized by
        :func:`get_q_token_kv_block_sparse_workspace_size`.
    max_seq_len_kv : int
        Static maximum logical K/V length in tokens.
    seq_len_q : int, optional
        Maximum packed route length. Fixed Q derives it from ``G``.
    kv_block_size : int
        Semantic sparse K/V block size: 4/8/16/32/64/128, default four.
    mask_type : {"causal"}
        QToken-KvBlock-Sparse-Attention currently supports causal masking only.
    sm_scale : float, optional
        Softmax scale, defaulting to ``head_dim**-0.5``.
    v_scale : float, optional
        Value-cache dequantization scale applied to the attention output,
        defaulting to one. Frameworks using an FP8 K/V cache should pass the
        cache's live V scale.
    out : torch.Tensor, optional
        Caller-owned output with the same logical shape as ``q``.
        Must be disjoint from all inputs, including indexer/request metadata,
        and from ``workspace_buffer``. Storage overlap is not checked.
    o_data_type : torch.dtype, optional
        Output dtype, defaulting to ``out.dtype`` or ``q.dtype``.
    qo_indptr : torch.Tensor, optional
        CUDA Int32 packed-route offsets.
    split_kv : bool
        Allow automatic split-KV selection (default True), independently of
        query layout. Pass False for prefill and True for decode, matching
        the value used to size ``workspace_buffer``.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    Returns
    -------
    torch.Tensor
        Attention output in the same logical layout as ``q``.
    """

    k_cache, v_cache = _validate_q_token_kv_block_sparse_paged_kv_cache(paged_kv_cache)
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    pattern_rank = 2 if share_pattern_across_kv_heads else 3
    if (
        not isinstance(indexer_block_ids, torch.Tensor)
        or indexer_block_ids.ndim != pattern_rank
    ):
        raise ValueError(
            "indexer_block_ids must be [T,K] when shared or [T,Hkv,K] otherwise"
        )
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if o_data_type is None:
        o_data_type = out.dtype if out is not None else q.dtype
    if qo_indptr is None:
        if q.ndim != 5:
            raise ValueError(
                "fixed q must be [B,Nq,G,Hq,D], or supply qo_indptr for packed q"
            )
        fixed_seq_len_q = int(q.shape[2])
        if seq_len_q is not None and seq_len_q != fixed_seq_len_q:
            raise ValueError("fixed seq_len_q must match q.shape[2]")
        seq_len_q = fixed_seq_len_q
        batch_size = int(q.shape[0] * q.shape[1])
        use_packed_q = False
    else:
        if q.ndim != 3:
            raise ValueError("packed q must have shape [total_q,Hq,D]")
        if seq_len_q is None:
            raise ValueError("seq_len_q is required with packed qo_indptr")
        batch_size = int(qo_indptr.numel()) - 1
        use_packed_q = True

    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        batch_size,
        seq_len_q,
        int(q.shape[-2]),
        int(k_cache.shape[1]),
        int(q.shape[-1]),
        kv_block_size,
        int(k_cache.shape[2]),
        int(indexer_block_ids.shape[-1]),
        max_seq_len_kv,
        device=q.device,
        workspace_buffer=workspace_buffer,
        use_packed_q=use_packed_q,
        mask_type=mask_type,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        o_data_type=o_data_type,
    )
    return wrapper.run(
        q,
        (k_cache, v_cache),
        block_table,
        indexer_block_ids,
        token_to_request,
        query_positions,
        qo_indptr=qo_indptr,
        sm_scale=sm_scale,
        v_scale=v_scale,
        out=out,
    )


def _validate_shape_parameters(
    num_query_tokens: int,
    block_topk: int,
    group_size: int,
) -> None:
    for value, name in (
        (num_query_tokens, "num_query_tokens"),
        (block_topk, "block_topk"),
        (group_size, "group_size"),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if num_query_tokens < 0:
        raise ValueError("num_query_tokens must be nonnegative")
    if block_topk <= 0:
        raise ValueError("block_topk must be positive")
    from .decode import _validate_prims_ts_q_token_kv_block_sparse_group_value

    _validate_prims_ts_q_token_kv_block_sparse_group_value(group_size)


def _resolve_num_query_groups(
    num_query_tokens: int,
    group_size: int,
    num_query_groups: Optional[int],
) -> int:
    """Resolve fixed or request-partitioned route count."""

    if num_query_groups is None:
        if num_query_tokens % group_size:
            raise ValueError(
                "num_query_tokens must be divisible by group_size without "
                "packed QToken-KvBlock-Sparse-Attention qo_indptr"
            )
        return num_query_tokens // group_size
    if not isinstance(num_query_groups, int) or isinstance(num_query_groups, bool):
        raise TypeError("num_query_groups must be an integer")
    if num_query_groups <= 0:
        raise ValueError("num_query_groups must be positive")
    min_groups = (num_query_tokens + group_size - 1) // group_size
    if num_query_groups < min_groups or num_query_groups > num_query_tokens:
        raise ValueError(
            "num_query_groups cannot partition num_query_tokens into nonempty "
            f"routes of size at most {group_size}"
        )
    return num_query_groups


def _validate_q_token_kv_block_sparse_qo_indptr_layout_tensor(
    qo_indptr: torch.Tensor,
) -> None:
    """Validate the packed-Q offsets required to size QToken-KvBlock-Sparse-Attention workspace."""

    if (
        not isinstance(qo_indptr, torch.Tensor)
        or qo_indptr.ndim != 1
        or qo_indptr.numel() < 2
        or qo_indptr.dtype != torch.int32
    ):
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention qo_indptr must be a rank-one int32 tensor"
        )


def _validate_q_token_kv_block_sparse_qo_indptr_tensor(
    qo_indptr: torch.Tensor,
    *,
    expected_device: torch.device,
) -> None:
    """Validate the synchronization-free packed-Q offset tensor contract."""

    _validate_q_token_kv_block_sparse_qo_indptr_layout_tensor(qo_indptr)
    if qo_indptr.device != expected_device or not qo_indptr.is_contiguous():
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention qo_indptr must be a contiguous CUDA int32 tensor on the query device"
        )


def _validate_sparse_block_size(sparse_block_size: int) -> int:
    """Validate the generic sparse-block API and current specialization."""

    if not isinstance(sparse_block_size, int) or isinstance(sparse_block_size, bool):
        raise TypeError("sparse_block_size must be an integer")
    if sparse_block_size <= 0 or sparse_block_size & (sparse_block_size - 1):
        raise ValueError("sparse_block_size must be a positive power of two")
    if sparse_block_size not in _Q_TOKEN_KV_BLOCK_SPARSE_BLOCK_SIZES:
        raise NotImplementedError("sparse_block_size must be 4, 8, 16, 32, 64, or 128")
    return sparse_block_size


def _validate_storage_page_size(storage_page_size: int) -> None:
    if not isinstance(storage_page_size, int) or isinstance(storage_page_size, bool):
        raise TypeError("storage_page_size must be an integer")
    if storage_page_size < 4 or storage_page_size % 4:
        raise ValueError("storage_page_size must be a positive multiple of four")


def _validate_q_token_kv_block_sparse_max_seq_len_kv(
    max_seq_len_kv: int,
    *,
    block_table_token_capacity: Optional[int] = None,
) -> int:
    """Validate the static logical K/V bound used by metadata kernels."""

    if not isinstance(max_seq_len_kv, int) or isinstance(max_seq_len_kv, bool):
        raise TypeError("max_seq_len_kv must be an integer")
    if max_seq_len_kv <= 0:
        raise ValueError("max_seq_len_kv must be positive")
    if max_seq_len_kv > _Q_TOKEN_KV_BLOCK_SPARSE_MAX_SEQ_LEN_KV:
        raise ValueError(
            "max_seq_len_kv exceeds the current grouped QToken-KvBlock-Sparse-Attention radix-key bound "
            f"({_Q_TOKEN_KV_BLOCK_SPARSE_MAX_SEQ_LEN_KV})"
        )
    if (
        block_table_token_capacity is not None
        and max_seq_len_kv > block_table_token_capacity
    ):
        raise ValueError(
            "max_seq_len_kv must not exceed the dense block-table token "
            f"capacity ({block_table_token_capacity})"
        )
    return max_seq_len_kv


def _align_up_q_token_kv_block_sparse_workspace(value: int) -> int:
    return (
        (value + _Q_TOKEN_KV_BLOCK_SPARSE_WORKSPACE_ALIGNMENT - 1)
        // _Q_TOKEN_KV_BLOCK_SPARSE_WORKSPACE_ALIGNMENT
        * _Q_TOKEN_KV_BLOCK_SPARSE_WORKSPACE_ALIGNMENT
    )


def _validate_inputs(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int,
    share_pattern_across_kv_heads: bool = True,
) -> None:
    sparse_block_size = _validate_sparse_block_size(sparse_block_size)
    for name, tensor in (
        ("block_indices", block_indices),
        ("block_table", block_table),
        ("token_to_request", token_to_request),
        ("query_positions", query_positions),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if not block_indices.is_cuda:
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention metadata inputs must be CUDA tensors"
        )
    if not isinstance(share_pattern_across_kv_heads, bool):
        raise TypeError("share_pattern_across_kv_heads must be a bool")
    rank = 2 if share_pattern_across_kv_heads else 3
    if block_indices.ndim != rank or block_indices.dtype != torch.int32:
        shape = "[T,K]" if share_pattern_across_kv_heads else "[T,Hkv,K]"
        raise ValueError(f"block_indices must be a rank-{rank} int32 tensor {shape}")
    if rank == 3 and block_indices.shape[1] <= 0:
        raise ValueError("block_indices must have a positive pattern head count")
    rows, block_topk = block_indices.shape[0], block_indices.shape[-1]
    _validate_shape_parameters(rows, block_topk, group_size)
    if block_topk > _Q_TOKEN_KV_BLOCK_SPARSE_MAX_BLOCK_TOPK:
        raise NotImplementedError(
            f"PrimTS QToken-KvBlock-Sparse-Attention currently supports block_topk <= {_Q_TOKEN_KV_BLOCK_SPARSE_MAX_BLOCK_TOPK}"
        )
    _validate_storage_page_size(storage_page_size)
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a nonempty rank-two int32 tensor")
    if not all(block_table.shape):
        raise ValueError("block_table must be nonempty")
    _validate_q_token_kv_block_sparse_max_seq_len_kv(
        max_seq_len_kv,
        block_table_token_capacity=block_table.shape[1] * storage_page_size,
    )
    if token_to_request.shape != (rows,) or token_to_request.dtype != torch.int32:
        raise ValueError("token_to_request must be int32 with one value per row")
    if query_positions.shape != (rows,) or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("query_positions must be int32/int64 with one value per row")
    tensors = (block_table, token_to_request, query_positions)
    if any(tensor.device != block_indices.device for tensor in tensors):
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention metadata inputs must share one CUDA device"
        )
    if block_indices.stride(-1) != 1 or block_table.stride(1) != 1:
        raise ValueError("block_indices and block_table rows must be contiguous")
    if block_table.stride(0) < block_table.shape[1]:
        raise ValueError(
            "block_table must be a non-overlapping dense row-strided page table"
        )
    if token_to_request.stride(0) != 1 or query_positions.stride(0) != 1:
        raise ValueError(
            "per-row QToken-KvBlock-Sparse-Attention metadata must be contiguous"
        )


def _validate_q_token_kv_block_sparse_attention_workspace(
    workspace_buffer: torch.Tensor,
    required_bytes: int,
) -> None:
    if not isinstance(workspace_buffer, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if workspace_buffer.dtype not in (torch.int8, torch.uint8):
        raise TypeError("workspace_buffer must have dtype torch.int8 or torch.uint8")
    if not workspace_buffer.is_cuda:
        raise ValueError("workspace_buffer must be a CUDA tensor")
    if not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    available_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if available_bytes < required_bytes:
        raise ValueError(
            "workspace_buffer is too small: requires at least "
            f"{required_bytes} bytes, got {available_bytes}"
        )
    if workspace_buffer.data_ptr() % 32:
        raise ValueError("workspace_buffer data pointer must be 32-byte aligned")


__all__ = [
    "QTokenKvBlockSparsePagedTSWrapper",
    "get_q_token_kv_block_sparse_workspace_size",
    "q_token_kv_block_sparse_attention_with_paged_kv_cache",
]
