# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimTS QToken-KvBlock-Sparse core attention for QSA."""

import functools
import threading

import torch

from tensorrt_llm.logger import logger

from .params import QSASparseParams

_PREFILL_GROUP_SIZE = 4
_MIN_DECODE_ROWS = 64
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_SUPPORTED_Q_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_KV_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
_FP8_MAX = 448.0


@functools.cache
def _prims_ts_api():
    try:
        from flashinfer.attention.prims_ts import (
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
        return "flashinfer.attention.prims_ts is not importable"
    if q.dtype not in _SUPPORTED_Q_DTYPES or k_cache.dtype not in _SUPPORTED_KV_DTYPES:
        return f"unsupported dtypes q={q.dtype} kv={k_cache.dtype}"
    if k_cache.dtype != q.dtype and k_cache.dtype != torch.float8_e4m3fn:
        return f"mismatched dtypes q={q.dtype} kv={k_cache.dtype}"
    if page_size % 4 != 0:
        return f"page size {page_size} is not a multiple of the sparse block size 4"
    if q.shape[-1] not in (64, 128, 256):
        return f"unsupported head_dim {q.shape[-1]}"
    return None


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


def _request_boundaries(request_indices: torch.Tensor) -> torch.Tensor:
    req = request_indices.to(torch.int64).cpu()
    change = torch.nonzero(req[1:] != req[:-1]).flatten() + 1
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            change.to(torch.int32),
            torch.tensor([req.numel()], dtype=torch.int32),
        )
    )


class _PrimsTSSparseCore:
    def __init__(self) -> None:
        self._plans: dict[tuple, object] = {}
        self._last_abi: dict[tuple, tuple] = {}
        self._workspace: torch.Tensor | None = None
        self._lock = threading.Lock()

    def _workspace_for(self, nbytes: int, device: torch.device) -> torch.Tensor:
        ws = self._workspace
        if ws is None or ws.device != device or ws.numel() < nbytes:
            # Plans bind typed views into the workspace; growth invalidates them.
            self._plans.clear()
            ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            self._workspace = ws
        return ws

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
    ) -> torch.Tensor:
        wrapper_cls, workspace_size, make_qo_indptr = _prims_ts_api()
        rows, num_qo_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        block_topk = block_ids.shape[-1]
        out_dtype = q.dtype

        sm_scale = softmax_scale
        if k_cache.dtype == torch.float8_e4m3fn and q.dtype != torch.float8_e4m3fn:
            # The sparse route has no BF16-Q/FP8-KV kernel, so Q is quantized per
            # call with a dynamic per-tensor scale folded into sm_scale. This
            # differs numerically from the Triton path, which upcasts K/V, and
            # the scale readback synchronizes the host.
            q_scale = (q.float().abs().amax() / _FP8_MAX).clamp_min(1e-12)
            q = (q.float() / q_scale).to(torch.float8_e4m3fn)
            sm_scale = softmax_scale * float(q_scale)

        request_indices = request_indices.to(torch.int32).contiguous()
        q = q.contiguous()
        if is_decode:
            route_q = q.view(rows, 1, 1, num_qo_heads, head_dim)
            qo_indptr = None
            group_size = 1
            num_routes = rows
            split_kv = True
        else:
            route_q = q
            qo_indptr = make_qo_indptr(
                _request_boundaries(request_indices),
                rows,
                group_size=_PREFILL_GROUP_SIZE,
                device=q.device,
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
            rows,
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
        out = torch.empty((rows, num_qo_heads, head_dim), dtype=out_dtype, device=q.device)
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
                    group_size=group_size,
                    kv_block_size=4,
                    split_kv=split_kv,
                )
                workspace = self._workspace_for(nbytes, q.device)
                wrapper = wrapper_cls()
                wrapper.plan(
                    num_routes,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    4,
                    page_size,
                    block_topk,
                    max_seq_len_kv,
                    device=q.device,
                    workspace_buffer=workspace,
                    group_size=group_size,
                    use_packed_q=not is_decode,
                    split_kv=split_kv,
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                    o_data_type=out_dtype,
                )
                self._plans[key] = wrapper
            run_out = out.view_as(route_q) if is_decode else out
            abi = tuple(
                (name, tuple(t.shape), tuple(t.stride()), t.dtype)
                for name, t in (
                    ("q", route_q),
                    ("block_ids", block_ids),
                    ("block_table", block_table),
                    ("request_indices", request_indices),
                    ("query_positions", query_positions),
                    ("out", run_out),
                )
            ) + (("k_ptr", k_cache.data_ptr()), ("v_ptr", v_cache.data_ptr()))
            try:
                wrapper.run(
                    route_q,
                    (k_cache, v_cache),
                    block_table,
                    block_ids,
                    request_indices,
                    query_positions,
                    qo_indptr=qo_indptr,
                    sm_scale=sm_scale,
                    out=run_out,
                )
            except RuntimeError as error:
                previous = self._last_abi.get(key)
                changed = (
                    [(a, b) for a, b in zip(previous or (), abi) if a != b]
                    if previous
                    else [("no eager run for this plan key", None)]
                )
                raise RuntimeError(f"{error}; ABI changes vs last eager run: {changed}") from error
            self._last_abi[key] = abi
        return out


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
) -> torch.Tensor:
    """Sparse GQA over HND paged K/V from compact indexer block IDs.

    ``block_ids`` is the ``[rows, block_topk]`` Int32 Top-K output; the causal
    tail is added by the kernel. Decode rows use the fixed ``[rows,1,1,Hq,D]``
    layout with split-KV; prefill packs rows into request-safe groups of four.
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
    )


__all__ = [
    "prims_ts_available",
    "prims_ts_sparse_gqa",
    "resolve_sparse_core_backend",
]
