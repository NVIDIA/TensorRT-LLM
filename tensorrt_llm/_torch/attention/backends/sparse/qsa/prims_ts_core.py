# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimTS QToken-KvBlock-Sparse core attention for QSA."""

import bisect
import functools
import threading

import torch

from tensorrt_llm.logger import logger

from .params import QSASparseParams

_PREFILL_GROUP_SIZE = 4
# A plan miss costs ~40 ms per layer; padding prefill rows to a bucket bounds
# the plan keys to max_num_tokens / bucket per layer.
_PREFILL_ROW_BUCKET = 256
# Decode plans are keyed on the row count too; eager decode batches vary.
_DECODE_ROW_BUCKET = 8
# Query rows (requests x query tokens per request, so MTP counts draft tokens)
# below which Triton is faster under CUDA graphs (Rubin SM107 and B300, 32K
# context, TP2): the core ties at one row, the indexer until ~32 rows.
_MIN_DECODE_ROWS = 2
_MIN_INDEXER_DECODE_ROWS = 32
_KV_BLOCK_SIZE = 4
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_SUPPORTED_Q_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_KV_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
_FP8_MAX = 448.0


@functools.cache
def _prims_ts_api():
    try:
        from ...prims_ts import (
            QTokenKvBlockSparsePagedTSWrapper,
            get_q_token_kv_block_sparse_workspace_size,
            make_q_token_kv_block_sparse_qo_indptr,
        )
    except ImportError:
        return None
    return (
        QTokenKvBlockSparsePagedTSWrapper,
        get_q_token_kv_block_sparse_workspace_size,
        make_q_token_kv_block_sparse_qo_indptr,
    )


def prims_ts_available() -> bool:
    return _prims_ts_api() is not None


def _unsupported_reason(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_size: int,
) -> str | None:
    if not q.is_cuda:
        return "requires CUDA tensors"
    if torch.cuda.get_device_capability(q.device) not in _SUPPORTED_CAPABILITIES:
        return "requires an SM100, SM103 or SM107 GPU"
    if not prims_ts_available():
        return "the vendored PrimTS package is not importable"
    if q.dtype not in _SUPPORTED_Q_DTYPES or k_cache.dtype not in _SUPPORTED_KV_DTYPES:
        return f"unsupported dtypes q={q.dtype} kv={k_cache.dtype}"
    if k_cache.dtype != q.dtype and k_cache.dtype != torch.float8_e4m3fn:
        return f"mismatched dtypes q={q.dtype} kv={k_cache.dtype}"
    if page_size % 4 != 0:
        return f"page size {page_size} is not a multiple of the sparse block size 4"
    if q.shape[-1] not in (64, 128, 256):
        return f"unsupported head_dim {q.shape[-1]}"
    return None


@functools.cache
def _prims_ts_indexer_available() -> bool:
    try:
        from . import prims_ts_indexer  # noqa: F401
    except ImportError:
        return False
    return True


def use_prims_ts_indexer(
    params: QSASparseParams,
    q: torch.Tensor,
    *,
    tokens_per_block: int | None = None,
    is_decode: bool = False,
) -> bool:
    """Whether the indexer scoring runs on the PrimTS kernel instead of Triton.

    Follows ``sparse_core_backend`` unless ``indexer_backend`` is set:
    ``triton`` never, ``prims_ts`` always (raising if unsupported), ``auto``
    when the device and inputs qualify and decode batches are large enough.
    """
    backend = params.indexer_backend
    if backend == "follow":
        backend = params.sparse_core_backend
    if backend == "triton":
        return False
    if backend == "auto" and is_decode and q.shape[0] < _MIN_INDEXER_DECODE_ROWS:
        return False
    reason = None
    if not q.is_cuda:
        reason = "requires CUDA tensors"
    elif torch.cuda.get_device_capability(q.device) not in _SUPPORTED_CAPABILITIES:
        reason = "requires an SM100, SM103 or SM107 GPU"
    elif q.dtype != torch.bfloat16:
        reason = f"requires BF16 index queries, got {q.dtype}"
    elif tokens_per_block is not None and (
        tokens_per_block % params.compress_ratio != 0
        or 128 % (tokens_per_block // params.compress_ratio) != 0
    ):
        reason = f"unsupported page size {tokens_per_block}"
    elif not _prims_ts_indexer_available():
        reason = "the CUTLASS task-scheduling DSL is not importable"
    if reason is None:
        return True
    if backend == "prims_ts":
        raise RuntimeError(f"QSA PrimTS indexer {reason}")
    logger.warning_once(
        f"QSA PrimTS indexer unavailable ({reason}); using the Triton scorer",
        key="qsa_prims_ts_indexer_fallback",
    )
    return False


def resolve_sparse_core_backend(
    params: QSASparseParams,
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_size: int,
    is_decode: bool,
    num_rows: int,
) -> str:
    backend = params.sparse_core_backend
    if backend == "triton":
        return "triton"
    reason = _unsupported_reason(q, k_cache, page_size)
    if backend == "prims_ts":
        if reason is not None:
            raise RuntimeError(f"QSA sparse_core_backend=prims_ts {reason}")
        return "prims_ts"
    if reason is not None:
        logger.warning_once(
            f"QSA PrimTS sparse core unavailable ({reason}); using Triton",
            key="qsa_prims_ts_unavailable",
        )
        return "triton"
    # Small decode batches are launch-bound; PrimTS pays warp-specialization
    # barrier costs there and Triton stays faster.
    if is_decode and num_rows < _MIN_DECODE_ROWS:
        return "triton"
    return "prims_ts"


def _request_boundaries(request_indices: torch.Tensor) -> tuple[int, ...]:
    req = request_indices.to(torch.int64).cpu()
    change = torch.nonzero(req[1:] != req[:-1]).flatten() + 1
    return (0, *change.tolist(), req.numel())


def _pad_rows(t: torch.Tensor, rows: int, fill: float) -> torch.Tensor:
    padded = t.new_full((rows, *t.shape[1:]), fill)
    padded[: t.shape[0]] = t
    return padded


def align_prefill_chunk(rows: int) -> int:
    """Largest bucket multiple not above ``rows``, so full chunks need no padding."""
    return max(rows // _PREFILL_ROW_BUCKET * _PREFILL_ROW_BUCKET, min(rows, _PREFILL_ROW_BUCKET))


def slice_request_boundaries(cu_seq_lens: list[int], start: int, end: int) -> tuple[int, ...]:
    """Request boundaries of packed rows ``[start, end)`` from host cumulative lengths."""
    inner = cu_seq_lens[
        bisect.bisect_right(cu_seq_lens, start) : bisect.bisect_left(cu_seq_lens, end)
    ]
    return (0, *(offset - start for offset in inner), end - start)


def _run_abi(run_args) -> tuple:
    return tuple((name, tuple(t.shape), tuple(t.stride()), t.dtype) for name, t in run_args)


class _PrimsTSSparseCore:
    def __init__(self) -> None:
        self._plans: dict[tuple, object] = {}
        self._last_abi: dict[tuple, tuple] = {}
        self._workspace: torch.Tensor | None = None
        self._qo_indptr: dict[tuple, torch.Tensor] = {}
        self._qo_indptr_key: tuple | None = None
        self._qo_indptr_buffer_key: tuple | None = None
        self._lock = threading.Lock()

    def _workspace_for(self, nbytes: int, device: torch.device) -> torch.Tensor:
        ws = self._workspace
        if ws is None or ws.device != device or ws.numel() < nbytes:
            # Plans bind typed views into the workspace; growth invalidates them.
            self._plans.clear()
            ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            self._workspace = ws
        return ws

    def _route_offsets(
        self,
        make_qo_indptr,
        request_boundaries: tuple[int, ...],
        rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        # The FlashInfer wrapper re-prepares, with host readbacks, whenever the
        # qo_indptr storage changes. All QSA layers of a forward share one
        # request layout, so offsets live in a persistent buffer per route count
        # that is rewritten, stream-ordered, only when the layout changes.
        layout = (device.index, rows, request_boundaries)
        if layout == self._qo_indptr_key:
            return self._qo_indptr[self._qo_indptr_buffer_key]
        host = make_qo_indptr(
            torch.tensor(request_boundaries, dtype=torch.int32),
            rows,
            group_size=_PREFILL_GROUP_SIZE,
            device="cpu",
        )
        buffer_key = (device.index, host.numel())
        buffer = self._qo_indptr.get(buffer_key)
        if buffer is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("QSA PrimTS route offsets must be allocated before capture")
            buffer = torch.empty(host.numel(), dtype=torch.int32, device=device)
            self._qo_indptr[buffer_key] = buffer
        buffer.copy_(host, non_blocking=True)
        self._qo_indptr_key = layout
        self._qo_indptr_buffer_key = buffer_key
        return buffer

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        block_ids: torch.Tensor,
        request_indices: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        page_size: int,
        max_seq_len_kv: int,
        softmax_scale: float,
        is_decode: bool,
        request_boundaries: tuple[int, ...] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wrapper_cls, workspace_size, make_qo_indptr = _prims_ts_api()
        rows, num_qo_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        block_topk = block_ids.shape[-1]
        out_dtype = q.dtype

        if k_cache.dtype == torch.float8_e4m3fn and q.dtype != torch.float8_e4m3fn:
            # The sparse route has no BF16-Q/FP8-KV kernel, so Q is cast to FP8
            # with unit scale: the normed Q of this model sits well inside the
            # E4M3 range, and a per-call scale would need a host readback that
            # CUDA-graph capture forbids. This differs numerically from the
            # Triton path, which upcasts K/V.
            q = q.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)

        request_indices = request_indices.to(torch.int32).contiguous()
        q = q.contiguous()
        bucket = _DECODE_ROW_BUCKET if is_decode else _PREFILL_ROW_BUCKET
        padded_rows = -(-rows // bucket) * bucket
        if padded_rows != rows:
            # Padding rows attend to nothing but token 0 of request 0 and are
            # sliced off the output; in packed prefill they form inert routes.
            q = _pad_rows(q, padded_rows, 0)
            block_ids = _pad_rows(block_ids, padded_rows, -1)
            request_indices = _pad_rows(request_indices, padded_rows, 0)
            query_positions = _pad_rows(query_positions, padded_rows, 0)
        if is_decode:
            route_q = q.view(padded_rows, 1, 1, num_qo_heads, head_dim)
            qo_indptr = None
            group_size = 1
            num_routes = padded_rows
            split_kv = True
        else:
            route_q = q
            if request_boundaries is None:
                # Host readback; callers with host sequence lengths pass the
                # boundaries to keep the stream running.
                request_boundaries = _request_boundaries(request_indices)
            with self._lock:
                qo_indptr = self._route_offsets(
                    make_qo_indptr, request_boundaries, padded_rows, q.device
                )
            group_size = _PREFILL_GROUP_SIZE
            num_routes = qo_indptr.numel() - 1
            split_kv = False

        # The wrapper retains K/V storage, and every QSA layer has its own cache
        # pool view, so plans are per layer buffer.
        key = (
            q.device.index,
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            padded_rows,
            num_routes,
            group_size,
            is_decode,
            max_seq_len_kv,
            q.dtype,
            k_cache.dtype,
            out_dtype,
            split_kv,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            block_topk,
        )
        result = out
        if out is None or padded_rows != rows:
            out = torch.empty(
                (padded_rows, num_qo_heads, head_dim), dtype=out_dtype, device=q.device
            )
        with self._lock:
            wrapper = self._plans.get(key)
            if wrapper is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("QSA PrimTS plan must be prepared before CUDA-graph capture")
                nbytes = workspace_size(
                    route_q,
                    k_cache,
                    block_table,
                    block_topk=block_topk,
                    max_seq_len_kv=max_seq_len_kv,
                    o_data_type=out_dtype,
                    qo_indptr=qo_indptr,
                    seq_len_q=None if is_decode else group_size,
                    kv_block_size=_KV_BLOCK_SIZE,
                    split_kv=split_kv,
                )
                workspace = self._workspace_for(nbytes, q.device)
                wrapper = wrapper_cls()
                # The vendored API names the route width seq_len_q.
                wrapper.plan(
                    num_routes,
                    group_size,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    _KV_BLOCK_SIZE,
                    page_size,
                    block_topk,
                    max_seq_len_kv,
                    device=q.device,
                    workspace_buffer=workspace,
                    use_packed_q=not is_decode,
                    split_kv=split_kv,
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                    o_data_type=out_dtype,
                )
                self._plans[key] = wrapper
            run_out = out.view_as(route_q) if is_decode else out
            run_args = (
                ("q", route_q),
                ("block_ids", block_ids),
                ("block_table", block_table),
                ("request_indices", request_indices),
                ("query_positions", query_positions),
                ("out", run_out),
            )
            try:
                wrapper.run(
                    route_q,
                    (k_cache, v_cache),
                    block_table,
                    block_ids,
                    request_indices,
                    query_positions,
                    qo_indptr=qo_indptr,
                    sm_scale=softmax_scale,
                    out=run_out,
                )
            except RuntimeError as error:
                previous = self._last_abi.get(key)
                changed = (
                    [(a, b) for a, b in zip(previous, _run_abi(run_args)) if a != b]
                    if previous
                    else [("no eager run for this plan key", None)]
                )
                raise RuntimeError(f"{error}; ABI changes vs first eager run: {changed}") from error
            if key not in self._last_abi:
                self._last_abi[key] = _run_abi(run_args)
        if result is None:
            return out[:rows]
        if result is not out:
            result.copy_(out[:rows])
        return result


_CORE = _PrimsTSSparseCore()


def prims_ts_sparse_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_ids: torch.Tensor,
    request_indices: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    page_size: int,
    max_seq_len_kv: int,
    softmax_scale: float,
    is_decode: bool,
    request_boundaries: tuple[int, ...] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse GQA over HND paged K/V from compact indexer block IDs.

    ``block_ids`` is the ``[rows, block_topk]`` Int32 Top-K output; the causal
    tail is added by the kernel. Decode rows use the fixed ``[rows,1,1,Hq,D]``
    layout with split-KV; prefill packs rows into request-safe groups of four,
    using host ``request_boundaries`` (row offsets of request starts plus the
    row count) when given. ``out``, when given, receives the result in place.
    """
    return _CORE.run(
        q,
        k_cache,
        v_cache,
        block_table,
        block_ids,
        request_indices,
        query_positions,
        page_size=page_size,
        max_seq_len_kv=max_seq_len_kv,
        softmax_scale=softmax_scale,
        is_decode=is_decode,
        request_boundaries=request_boundaries,
        out=out,
    )


__all__ = [
    "prims_ts_available",
    "prims_ts_sparse_gqa",
    "align_prefill_chunk",
    "resolve_sparse_core_backend",
    "slice_request_boundaries",
]
