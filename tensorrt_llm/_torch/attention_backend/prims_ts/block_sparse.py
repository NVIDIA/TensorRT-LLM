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

"""Task-scheduled block-sparse attention with a plan/run lifecycle.

``plan()`` fixes geometry, allocates uniform prepared-route capacity, chooses a
Q8/Q16/Q32 SWAPAB or Q64/Q128 KeepsAB specialization, and atomically publishes
an immutable revision. ``run()`` validates compact BSHD Q/K/V and caller-owned
routing tensors, then enqueues route preparation followed by attention in one
compiled adapter on the caller's CUDA stream. The one-shot entry point retains
synchronous canonical-BSR inspection to derive its temporary plan capacity.

``BlockSparsePagedTSWrapper`` shares the same scheduling policy and decode
kernel, but freezes request-level page spans and sequence lengths at plan time
while accepting physical page IDs and sparse routes at run time.
"""

import threading
from typing import Literal

import torch

from flashinfer.api_logging import flashinfer_api

from ._block_sparse.config import (
    _BlockSparseStaticProfile,
    _validate_block_sparse_static_profile,
)
from ._block_sparse.inspection import (
    _BlockSparseInspection,
    _inspect_block_sparse_bsr,
)
from ._block_sparse.plan import (
    _BlockSparsePlanState,
    _build_block_sparse_plan_state,
    _serialize_plan,
    _snapshot_paged_kv_plan_metadata,
    _wait_and_record_block_sparse_plan,
)
from ._block_sparse.runtime import (
    _BlockSparseRunArgs,
    _ContiguousKVStorage,
    _PagedKVStorage,
    launch_block_sparse as _launch_block_sparse,
    record_block_sparse_run_args as _record_block_sparse_run_args,
    validate_block_sparse_metadata as _validate_block_sparse_metadata,
    validate_block_sparse_run as _validate_block_sparse_run,
)
from .decode import (
    PagedKVCache,
    _normalize_paged_kv_cache,
    _resolve_cuda_device,
)


class _BlockSparseWrapperBase:
    """Shared published-state and launch-lifetime mechanics for wrappers."""

    def __init__(self) -> None:
        self._plan_state: _BlockSparsePlanState | None = None
        self._plan_lock = threading.Lock()
        self._capture_pin_lock = threading.Lock()
        self._captured_plan_states: dict[int, _BlockSparsePlanState] = {}

    def _published_state(self) -> _BlockSparsePlanState:
        state = self._plan_state
        if state is None:
            raise AttributeError("plan() has not published a state")
        return state

    @property
    def _policy(self) -> tuple[tuple[str, object], ...]:
        return self._published_state().policy

    def _require_run_state(self) -> _BlockSparsePlanState:
        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        return state

    def _launch_validated_run(
        self,
        state: _BlockSparsePlanState,
        run_args: _BlockSparseRunArgs,
        run_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with torch.cuda.device(state.device), torch.cuda.stream(run_stream):
            if torch.cuda.is_current_stream_capturing():
                with self._capture_pin_lock:
                    self._captured_plan_states.setdefault(id(state), state)
            _wait_and_record_block_sparse_plan(state, run_stream)
            _record_block_sparse_run_args(run_args, run_stream)
            return _launch_block_sparse(run_args, state=state)


class BlockSparseTSWrapper(_BlockSparseWrapperBase):
    """Plan and reuse compact-BSHD block-sparse attention launches.

    Q is ``[B, Sq, Hq, D]`` and K/V are ``[B, Skv, Hkv, D]``. Sparse rows are
    owned per batch, KV head, and query block, so every Q head in one grouped
    KV head consumes the same sparse row. A plan fixes geometry and a per-row
    capacity; every run supplies its own BSR and optional token mask.
    Callers must keep those tensors alive and immutable until the queued run or
    captured graph finishes using them. CUDA Graph capture pins plan-owned
    state only, so captured routing storage remains the caller's responsibility.

    One plan revision owns one mutable route workspace. Its runs must be ordered
    on one stream or externally synchronized; unordered concurrent runs require
    distinct wrappers.
    """

    @_serialize_plan
    def plan(
        self,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_block_size: int,
        kv_block_size: int,
        *,
        device: torch.device | str | int,
        max_blocks_per_row: int,
        use_kv_valid_bits: bool,
        mask_type: Literal["dense", "causal"] = "dense",
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: torch.dtype | None = None,
        o_data_type: torch.dtype | None = None,
    ) -> None:
        """Choose a legal profile and allocate reusable routing capacity.

        The plan owns immutable geometry and a uniform route workspace, not a
        sparse pattern. ``max_blocks_per_row`` bounds each runtime BSR row in
        semantic ``kv_block_size`` blocks. ``use_kv_valid_bits`` selects whether
        every :meth:`run` must supply the shared batch token mask. Callers may
        pass different routing tensor identities and index extents to each run
        as long as they fit this declared capacity.

        MHA, GQA, and MQA are supported with ``Hq / Hkv`` a power of two no
        greater than 32 and ``D=128``. Q, K, V, and O use one matching
        ``torch.float16`` or ``torch.bfloat16`` dtype. Runtime tensor shapes
        are documented by :meth:`run`.

        ``q_block_size`` may be any positive signed-Int32 value for which a
        physical Q tile stays within one BSR row. Equivalently,
        ``q_block_size * (Hq / Hkv)`` must be divisible by 8; therefore Q block
        sizes 1, 2, and 4 require GQA ratios of at least 8, 4, and 2,
        respectively. ``kv_block_size`` may be 8, 16, 32, or a positive
        multiple of 64. The Q tile groups complete Q-head groups and as many Q
        tokens as fit without crossing a semantic Q-block row, up to Q128;
        fine KV blocks cap this at a SWAPAB Q32 tile. Every run prepares
        per-KV-head canonical BSR into compact, profile-selected fixed-width
        route metadata, and the attention core consumes only that metadata.
        This remains true when every KV block is selected;
        callers that know a pattern is dense should choose the dense FMHA API
        explicitly.

        Planning does not inspect routing values and does not synchronize the
        host. Runtime preparation validates ranges and capacity without dynamic
        allocation. The one-shot :func:`block_sparse_attention` entry point
        retains synchronous canonical-BSR inspection so it can derive the
        smallest semantic row bound for that call.

        Concurrent plans are serialized; run keeps using the published state.
        One revision has one mutable route workspace, so its runs must be
        ordered on one stream or externally synchronized. Unordered concurrent
        runs require distinct wrappers.
        """

        static = _validate_block_sparse_static_profile(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            use_kv_valid_bits=use_kv_valid_bits,
            mask_type=mask_type,
            q_dtype=q_data_type,
            kv_dtype=kv_data_type,
            output_dtype=o_data_type,
            max_blocks_per_row=max_blocks_per_row,
        )
        device, device_index = _resolve_cuda_device(device)
        plan_stream = torch.cuda.current_stream(device)
        with torch.cuda.device(device_index), torch.cuda.stream(plan_stream):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "block-sparse planning is unsupported during CUDA Graph capture"
                )
            candidate = _build_block_sparse_plan_state(
                static,
                device=device,
                device_index=device_index,
                plan_stream=plan_stream,
                paged_kv=None,
            )
        # This is the only wrapper mutation. Every failure above leaves the
        # previously published revision intact and runnable.
        self._plan_state = candidate

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indptr: torch.Tensor,
        block_indices: torch.Tensor,
        *,
        kv_valid_bits: torch.Tensor | None = None,
        sm_scale: float | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch the current plan on the caller's current CUDA stream.

        ``q`` and ``out`` use compact ``[B, Sq, Hq, D]`` while ``k`` and ``v``
        use compact ``[B, Skv, Hkv, D]``, with shapes and dtypes fixed by
        ``plan()``. If supplied, ``out`` must match Q's shape and the planned
        output dtype. The returned tensor is exactly ``out`` when one was
        supplied; otherwise it is a newly allocated compact BSHD tensor.
        Only O is returned; this PrimTS API does not return LSE. The launch is
        enqueued asynchronously on the caller's current CUDA stream.

        ``block_indptr`` is compact Int32
        ``[B, Hkv, ceil(Sq / q_block_size) + 1]`` and indexes compact
        ``block_indices``. Every row must fit the planned semantic-block
        capacity; referenced block IDs must be strictly increasing, unique,
        and in range. Runtime value violations cannot raise a synchronous host
        exception: preparation marks only the invalid row with a negative
        header, attention consumes it as empty, and that row's output is finite
        zero without reading route payload. A masked plan requires
        ``kv_valid_bits`` with shape
        ``[B, ceil(Skv / 32)]`` and dtype UInt32; an unmasked plan requires
        ``None``. Routing tensors may have different identities on every run.

        Keep this wrapper alive until every captured CUDA Graph is destroyed.

        Parameters
        ----------
        q : torch.Tensor
            Compact query tensor ``[B, Sq, Hq, D]`` matching the plan.
        k : torch.Tensor
            Compact key tensor ``[B, Skv, Hkv, D]`` matching the plan.
        v : torch.Tensor
            Compact value tensor with the same shape, dtype, and strides as
            ``k``.
        block_indptr : torch.Tensor
            Contiguous Int32 BSR row offsets with shape
            ``[B, Hkv, ceil(Sq / q_block_size) + 1]``.
        block_indices : torch.Tensor
            Contiguous Int32 semantic KV-block IDs referenced by
            ``block_indptr``.
        kv_valid_bits : torch.Tensor, optional
            Contiguous UInt32 token-validity bitmap ``[B, ceil(Skv / 32)]``.
            Supply it exactly when the plan enabled token validity bits.
        sm_scale : float, optional
            Softmax scale. Defaults to ``1 / sqrt(D)``.
        out : torch.Tensor, optional
            Caller-owned compact output buffer ``[B, Sq, Hq, D]`` with the
            planned output dtype.

        Returns
        -------
        torch.Tensor
            The compact output tensor; identical to ``out`` when provided.
        """

        state = self._require_run_state()
        run_stream = torch.cuda.current_stream(state.device)
        run_args = _validate_block_sparse_run(
            q,
            _ContiguousKVStorage(k=k, v=v),
            state=state,
            block_indptr=block_indptr,
            block_indices=block_indices,
            kv_valid_bits=kv_valid_bits,
            sm_scale=sm_scale,
            out=out,
        )
        return self._launch_validated_run(state, run_args, run_stream)


def _inspect_one_shot_block_sparse_routes(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    kv_valid_bits: torch.Tensor | None,
    *,
    static: _BlockSparseStaticProfile,
    device: torch.device,
) -> _BlockSparseInspection:
    """Validate and synchronously inspect one one-shot canonical BSR pattern."""

    _validate_block_sparse_metadata(
        block_indptr,
        block_indices,
        kv_valid_bits,
        device=device,
        batch_size=static.batch_size,
        seq_len_q=static.seq_len_q,
        seq_len_kv=static.seq_len_kv,
        num_kv_heads=static.num_kv_heads,
        q_block_size=static.q_block_size,
        use_kv_valid_bits=static.use_kv_valid_bits,
    )
    return _inspect_block_sparse_bsr(
        block_indptr,
        block_indices,
        batch_size=static.batch_size,
        num_kv_heads=static.num_kv_heads,
        seq_len_q=static.seq_len_q,
        seq_len_kv=static.seq_len_kv,
        q_block_size=static.q_block_size,
        kv_block_size=static.kv_block_size,
        stream=torch.cuda.current_stream(device),
    )


@flashinfer_api
def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    *,
    kv_valid_bits: torch.Tensor | None = None,
    mask_type: Literal["dense", "causal"] = "dense",
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plan and run one compact-BSHD block-sparse attention launch.

    This one-shot form synchronously inspects canonical BSR, derives its largest
    semantic row, creates a capacity-only plan, and passes the original routing
    tensors to :meth:`BlockSparseTSWrapper.run`. It therefore cannot be invoked
    inside CUDA Graph capture; plan a wrapper outside capture and capture only
    ``run()`` instead.

    Parameters
    ----------
    q : torch.Tensor
        Compact query tensor ``[B, Sq, Hq, D]``.
    k : torch.Tensor
        Compact key tensor ``[B, Skv, Hkv, D]``.
    v : torch.Tensor
        Compact value tensor with the same shape, dtype, and strides as ``k``.
    block_indptr : torch.Tensor
        Contiguous Int32 BSR row offsets with shape
        ``[B, Hkv, ceil(Sq / q_block_size) + 1]``.
    block_indices : torch.Tensor
        Contiguous Int32 semantic KV-block IDs referenced by ``block_indptr``.
    q_block_size : int
        Positive number of logical query tokens represented by one BSR row.
        The product with ``Hq / Hkv`` must be divisible by 8 so a physical Q
        tile does not cross row boundaries.
    kv_block_size : int
        Number of logical KV tokens represented by one BSR block ID; it must
        be 8, 16, 32, or a positive multiple of 64.
    kv_valid_bits : torch.Tensor, optional
        Contiguous UInt32 token-validity bitmap ``[B, ceil(Skv / 32)]``.
    mask_type : {"dense", "causal"}, optional
        Attention mask applied inside each selected sparse block.
    sm_scale : float, optional
        Softmax scale. Defaults to ``1 / sqrt(D)``.
    out : torch.Tensor, optional
        Caller-owned compact output buffer ``[B, Sq, Hq, D]``.

    Returns
    -------
    torch.Tensor
        The compact output tensor; identical to ``out`` when provided.
    """

    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be rank 4 compact BSHD")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    batch_size, seq_len_q, num_qo_heads, head_dim = map(int, q.shape)
    kv_batch_size, seq_len_kv, num_kv_heads, kv_head_dim = map(int, k.shape)
    if kv_batch_size != batch_size or kv_head_dim != head_dim:
        raise ValueError("Q and K batch/head dimensions must agree")
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError("K and V must have identical shapes")

    use_kv_valid_bits = kv_valid_bits is not None
    static = _validate_block_sparse_static_profile(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        use_kv_valid_bits=use_kv_valid_bits,
        mask_type=mask_type,
        q_dtype=q.dtype,
        kv_dtype=k.dtype,
        output_dtype=q.dtype if out is None else out.dtype,
    )
    device, _ = _resolve_cuda_device(q.device)
    inspection = _inspect_one_shot_block_sparse_routes(
        block_indptr,
        block_indices,
        kv_valid_bits,
        static=static,
        device=device,
    )

    wrapper = BlockSparseTSWrapper()
    wrapper.plan(
        static.batch_size,
        static.seq_len_q,
        static.seq_len_kv,
        static.num_qo_heads,
        static.num_kv_heads,
        static.head_dim,
        static.q_block_size,
        static.kv_block_size,
        device=device,
        max_blocks_per_row=inspection.max_row_block_count,
        use_kv_valid_bits=static.use_kv_valid_bits,
        mask_type=static.mask_type,
        q_data_type=static.q_dtype,
        kv_data_type=static.kv_dtype,
        o_data_type=static.output_dtype,
    )
    return wrapper.run(
        q,
        k,
        v,
        block_indptr,
        block_indices,
        kv_valid_bits=kv_valid_bits,
        sm_scale=sm_scale,
        out=out,
    )


def _validate_paged_kv_indptr_tensor(
    paged_kv_indptr: torch.Tensor,
) -> tuple[torch.device, int]:
    """Validate the plan-owned page-row source without reading its values."""

    if not isinstance(paged_kv_indptr, torch.Tensor):
        raise TypeError("paged_kv_indptr must be a torch.Tensor")
    if paged_kv_indptr.dtype != torch.int32:
        raise TypeError("paged_kv_indptr must have dtype torch.int32")
    if paged_kv_indptr.ndim != 1:
        raise ValueError("paged_kv_indptr must be one-dimensional")
    if paged_kv_indptr.device.type != "cuda":
        raise ValueError("paged_kv_indptr must be a CUDA tensor")
    if not paged_kv_indptr.is_contiguous():
        raise ValueError("paged_kv_indptr must be contiguous")
    if paged_kv_indptr.data_ptr() % 4 != 0:
        raise ValueError("paged_kv_indptr data pointer must be 4-byte aligned")
    if paged_kv_indptr.numel() < 2:
        raise ValueError("paged_kv_indptr must describe at least one request")
    return paged_kv_indptr.device, paged_kv_indptr.numel() - 1


class BlockSparsePagedTSWrapper(_BlockSparseWrapperBase):
    """Plan stable logical geometry and run with live sparse/page mappings."""

    def _published_state(self) -> _BlockSparsePlanState:
        state = super()._published_state()
        assert state.paged_kv is not None
        return state

    @_serialize_plan
    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        seq_len_q: int,
        seq_len_kv: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_block_size: int,
        kv_block_size: int,
        page_size: int,
        *,
        max_blocks_per_row: int,
        use_kv_valid_bits: bool,
        seq_lens_kv: torch.Tensor | None = None,
        mask_type: Literal["dense", "causal"] = "dense",
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: torch.dtype | None = None,
        o_data_type: torch.dtype | None = None,
    ) -> None:
        """Plan fixed-Q geometry and snapshot stable page and length metadata.

        Without ``seq_lens_kv``, ``seq_len_kv`` is the exact shared K/V length.
        With it, ``seq_len_kv`` is a static maximum and ``seq_lens_kv`` supplies
        one length per request. Both ``paged_kv_indptr`` and ``seq_lens_kv`` are
        copied through a D2H snapshot, validated, and republished as plan-owned
        CUDA tensors. Changing either therefore requires a new plan.

        Each page-table row needs only ``ceil(seq_lens_kv[b] / page_size)``
        live entries; extra entries are spare capacity and are not read. Physical
        page IDs and block-sparse routes remain runtime inputs, so graph replays
        may remap them in place between completed replays.

        ``q_block_size`` may be any positive signed-Int32 value satisfying
        ``q_block_size * (Hq / Hkv) % 8 == 0``. This row-purity condition keeps
        every physical Q tile within exactly one logical BSR row.
        ``kv_block_size`` remains restricted to 8, 16, 32, or a positive
        multiple of 64.
        """

        source_device, batch_size = _validate_paged_kv_indptr_tensor(paged_kv_indptr)
        static = _validate_block_sparse_static_profile(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            use_kv_valid_bits=use_kv_valid_bits,
            mask_type=mask_type,
            q_dtype=q_data_type,
            kv_dtype=kv_data_type,
            output_dtype=o_data_type,
            max_blocks_per_row=max_blocks_per_row,
        )
        assert static.page_size is not None
        device, device_index = _resolve_cuda_device(source_device)
        plan_stream = torch.cuda.current_stream(device)
        with torch.cuda.device(device_index), torch.cuda.stream(plan_stream):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "paged block-sparse planning is unsupported during CUDA Graph capture"
                )
            paged_kv = _snapshot_paged_kv_plan_metadata(
                paged_kv_indptr,
                seq_lens_kv,
                static=static,
                device=device,
                plan_stream=plan_stream,
            )
            candidate = _build_block_sparse_plan_state(
                static,
                device=device,
                device_index=device_index,
                plan_stream=plan_stream,
                paged_kv=paged_kv,
            )
        self._plan_state = candidate

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        paged_kv_indices: torch.Tensor,
        block_indptr: torch.Tensor,
        block_indices: torch.Tensor,
        *,
        kv_valid_bits: torch.Tensor | None = None,
        sm_scale: float | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch with live physical page IDs and per-KV-head sparse routes.

        Q and O are compact ``[B, Sq, Hq, D]`` tensors, including ``Sq=1``.
        The cache is either combined ``[P, 2, Hkv, page, D]`` or a ``(K, V)``
        tuple whose members are ``[P, Hkv, page, D]`` with compact inner HND
        strides and arbitrary non-overlapping outer page strides.

        ``paged_kv_indices`` is compact Int32 with the exact final extent
        snapshotted from ``paged_kv_indptr`` by :meth:`plan`. Its page IDs,
        ``block_indptr``, ``block_indices``, and optional token bits are read at
        run time. Within every sparse row, BSR block IDs must be strictly
        increasing and unique. Each selected block must start before the
        request's frozen live K/V length (the static ``seq_len_kv`` for a
        fixed-length plan). Because the row is sorted, preparation validates
        this live bound from its final block. ``max_blocks_per_row`` applies to
        the provided live row. Referenced physical page IDs must be in
        ``[0, P)``; an invalid BSR or page ID makes its affected row fail closed
        to finite zero without reading a K/V payload out of bounds.

        In eager execution, ``record_stream`` extends the allocator lifetime of
        every launch tensor (Q, normalized K/V, O, BSR metadata, token bits,
        and page metadata) through the asynchronous run. It does not make
        concurrent mutation safe. During CUDA Graph capture and replay, the
        caller must keep Q, the cache, O, and all runtime metadata alive and
        unmodified until replay completes; do not release or overwrite them
        while a replay is outstanding. The wrapper and its captured plan state
        must also outlive the graph. One plan revision owns mutable route
        scratch; unordered concurrent runs require separate wrapper instances.

        Parameters
        ----------
        q : torch.Tensor
            Compact query tensor ``[B, Sq, Hq, D]`` matching the plan.
        paged_kv_cache : PagedKVCache
            Either a combined cache ``[P, 2, Hkv, page_size, D]`` or a
            ``(K, V)`` tuple whose tensors are
            ``[P, Hkv, page_size, D]``.
        paged_kv_indices : torch.Tensor
            Contiguous Int32 physical page IDs. Its length must equal the
            final offset snapshotted from ``paged_kv_indptr`` during planning.
        block_indptr : torch.Tensor
            Contiguous Int32 BSR row offsets with shape
            ``[B, Hkv, ceil(Sq / q_block_size) + 1]``.
        block_indices : torch.Tensor
            Contiguous Int32 logical KV-block IDs referenced by
            ``block_indptr``.
        kv_valid_bits : torch.Tensor, optional
            Contiguous UInt32 logical-token validity bitmap
            ``[B, ceil(seq_len_kv / 32)]``. Supply it exactly when the plan
            enabled token validity bits.
        sm_scale : float, optional
            Softmax scale. Defaults to ``1 / sqrt(D)``.
        out : torch.Tensor, optional
            Caller-owned compact output buffer ``[B, Sq, Hq, D]`` with the
            planned output dtype.

        Returns
        -------
        torch.Tensor
            The compact output tensor; identical to ``out`` when provided.
        """

        state = self._require_run_state()
        run_stream = torch.cuda.current_stream(state.device)
        run_args = _validate_block_sparse_run(
            q,
            _PagedKVStorage(
                paged_kv_cache=paged_kv_cache,
                paged_kv_indices=paged_kv_indices,
            ),
            state=state,
            block_indptr=block_indptr,
            block_indices=block_indices,
            kv_valid_bits=kv_valid_bits,
            sm_scale=sm_scale,
            out=out,
        )
        return self._launch_validated_run(state, run_args, run_stream)


@flashinfer_api
def block_sparse_attention_with_paged_kv_cache(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    *,
    seq_len_kv: int,
    seq_lens_kv: torch.Tensor | None = None,
    kv_valid_bits: torch.Tensor | None = None,
    mask_type: Literal["dense", "causal"] = "dense",
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plan and run one fixed-Q paged block-sparse attention launch.

    This one-shot entry point mirrors :class:`BlockSparsePagedTSWrapper`: the
    logical request-to-page spans come from ``paged_kv_indptr`` and optional
    ``seq_lens_kv``, while live physical page IDs and sparse routes still come
    from ``paged_kv_indices`` plus BSR metadata at run time.

    Parameters
    ----------
    q : torch.Tensor
        Compact query tensor ``[B, Sq, Hq, D]``.
    paged_kv_cache : PagedKVCache
        Either a combined cache ``[P, 2, Hkv, page_size, D]`` or a ``(K, V)``
        tuple whose tensors are ``[P, Hkv, page_size, D]``.
    paged_kv_indptr : torch.Tensor
        Contiguous Int32 request offsets into ``paged_kv_indices``, with shape
        ``[B + 1]``.
    paged_kv_indices : torch.Tensor
        Contiguous Int32 physical page IDs referenced by ``paged_kv_indptr``.
    block_indptr : torch.Tensor
        Contiguous Int32 BSR row offsets with shape
        ``[B, Hkv, ceil(Sq / q_block_size) + 1]``.
    block_indices : torch.Tensor
        Contiguous Int32 logical KV-block IDs referenced by ``block_indptr``.
    q_block_size : int
        Positive number of logical query tokens represented by one BSR row.
        The product with ``Hq / Hkv`` must be divisible by 8 so a physical Q
        tile does not cross row boundaries.
    kv_block_size : int
        Number of logical KV tokens represented by one BSR block ID; it must
        be 8, 16, 32, or a positive multiple of 64.
    seq_len_kv : int
        Exact shared logical KV length, or the maximum logical length when
        ``seq_lens_kv`` is provided.
    seq_lens_kv : torch.Tensor, optional
        Contiguous Int32 per-request logical KV lengths with shape ``[B]``.
    kv_valid_bits : torch.Tensor, optional
        Contiguous UInt32 logical-token validity bitmap
        ``[B, ceil(seq_len_kv / 32)]``.
    mask_type : {"dense", "causal"}, optional
        Attention mask applied inside each selected sparse block.
    sm_scale : float, optional
        Softmax scale. Defaults to ``1 / sqrt(D)``.
    out : torch.Tensor, optional
        Caller-owned compact output buffer ``[B, Sq, Hq, D]``.

    Returns
    -------
    torch.Tensor
        The compact output tensor; identical to ``out`` when provided.
    """

    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.ndim != 4:
        raise ValueError("q must be rank 4 compact BSHD")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")

    metadata_device, batch_size = _validate_paged_kv_indptr_tensor(paged_kv_indptr)
    if metadata_device != q.device:
        raise ValueError(
            f"paged_kv_indptr must be on q.device {q.device}, got {metadata_device}"
        )

    batch_q, seq_len_q, num_qo_heads, head_dim = map(int, q.shape)
    if batch_q != batch_size:
        raise ValueError(
            "the paged metadata batch size must equal q.shape[0]: "
            f"expected {batch_q}, got {batch_size}"
        )

    (
        k_cache,
        _,
        _,
        num_kv_heads,
        page_size,
        runtime_head_dim,
        _,
        _,
    ) = _normalize_paged_kv_cache(paged_kv_cache, expected_device=q.device)
    if runtime_head_dim != head_dim:
        raise ValueError(
            "Q and paged K/V head dimensions must agree: "
            f"expected {head_dim}, got {runtime_head_dim}"
        )

    use_kv_valid_bits = kv_valid_bits is not None
    static = _validate_block_sparse_static_profile(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        page_size=page_size,
        use_kv_valid_bits=use_kv_valid_bits,
        mask_type=mask_type,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        output_dtype=q.dtype if out is None else out.dtype,
    )
    inspection = _inspect_one_shot_block_sparse_routes(
        block_indptr,
        block_indices,
        kv_valid_bits,
        static=static,
        device=metadata_device,
    )

    wrapper = BlockSparsePagedTSWrapper()
    assert static.page_size is not None
    wrapper.plan(
        paged_kv_indptr,
        static.seq_len_q,
        static.seq_len_kv,
        static.num_qo_heads,
        static.num_kv_heads,
        static.head_dim,
        static.q_block_size,
        static.kv_block_size,
        static.page_size,
        max_blocks_per_row=inspection.max_row_block_count,
        use_kv_valid_bits=static.use_kv_valid_bits,
        seq_lens_kv=seq_lens_kv,
        mask_type=static.mask_type,
        q_data_type=static.q_dtype,
        kv_data_type=static.kv_dtype,
        o_data_type=static.output_dtype,
    )
    return wrapper.run(
        q,
        paged_kv_cache,
        paged_kv_indices,
        block_indptr,
        block_indices,
        kv_valid_bits=kv_valid_bits,
        sm_scale=sm_scale,
        out=out,
    )


__all__ = [
    "BlockSparsePagedTSWrapper",
    "BlockSparseTSWrapper",
    "block_sparse_attention",
    "block_sparse_attention_with_paged_kv_cache",
]
