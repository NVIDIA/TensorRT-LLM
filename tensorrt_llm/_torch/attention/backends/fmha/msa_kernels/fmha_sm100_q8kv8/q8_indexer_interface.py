# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Q8KV4/Q8KV8 paged sparse-attention indexers over the vLLM cache layout.

Each wrapper scores the historical 128-token pages of every query with an E4M3
Q and selects 16 logical pages: 15 ranked by score plus the query's local page,
which is always the final valid entry.

Tensors follow the vLLM indexer layout. All ``num_heads`` index heads (1, 2,
or 4 per rank) share the single index-K head and each head selects its own
pages:

* ``q``: ``[num_tokens, num_heads, 128]`` E4M3, token-major (decode tokens are
  grouped per request, eight MTP tokens each, or ``query_len`` for the Q8KV4
  decode indexer).
* Q8KV8 ``k_cache``: ``[num_blocks, 128, 128]`` E4M3.
* Q8KV4 ``k_cache``: ``[num_blocks, 128, 72]`` uint8. Each page stores the packed
  E2M1 values of all 128 tokens (8192 bytes, 64 bytes per token, the low nibble
  first) followed by the E4M3 scales of 16-element groups (1024 bytes, index
  ``token * 8 + group``).
* ``block_table``: ``[batch, max_blocks]`` int32 logical-to-physical page map;
  ``seq_lens``: ``[batch]`` int32 KV lengths including the current queries.
* Output: ``[num_tokens, num_heads, 16]`` int32 logical page indices.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from .src.common.aot_cache import save_aot, try_load_aot
from .src.sm100.q8kv4_indexer_decode import Q8KV4DecodeIndexerSm100
from .src.sm100.q8kv8_indexer_decode import Q8KV8DecodeIndexerSm100
from .src.sm100.q8kv8_indexer_prefill import MMA_K as _PREFILL_MMA_K
from .src.sm100.q8kv8_indexer_prefill import Q8KV8PrefillIndexerSm100
from .src.sm100.q8kv8_indexer_prefill_plan import (
    Q8KV8PrefillIndexerPlanBuild,
    Q8KV8PrefillIndexerPlanReset,
)

logger = logging.getLogger(__name__)

_PAGE_SIZE = 128
_HEAD_DIM = 128
_TOP_K = 16
_DECODE_QUERY_LENGTH = 8
_SUPPORTED_NUM_HEADS = (1, 2, 4)
_MAXIMUM_PAGES = 8192
_NVFP4_PAGE_BYTES = _PAGE_SIZE * _HEAD_DIM // 2 + _PAGE_SIZE * _HEAD_DIM // 16
_NVFP4_PAGE_WIDTH = _NVFP4_PAGE_BYTES // _PAGE_SIZE
_TMA_ALIGNMENT = 16
# Metadata and scores use scalar accesses, so vLLM slices such as
# seq_lens[lo:hi] only need int32 alignment.
_SCALAR_ALIGNMENT = torch.int32.itemsize
_PREFILL_TASK_CAPACITY_PAGE_CHUNK = 4
# Target tasks per persistent cluster: clusters walk the largest-first task order
# round-robin, so several smaller tasks per cluster even out the causal page
# counts across SMs.
_PREFILL_TASKS_PER_CLUSTER = 4
_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3), (10, 7)})
# Kernels plus this interface, which fixes the compiled argument alignments.
_CODEGEN_SOURCES = (
    "q8_indexer_interface.py",
    "src/sm100/q8kv4_indexer_decode.py",
    "src/sm100/q8kv8_indexer_decode.py",
    "src/sm100/q8kv8_indexer_prefill.py",
    "src/sm100/q8kv8_indexer_prefill_plan.py",
)
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


@cache
def _kernel_source_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in _CODEGEN_SOURCES:
        digest.update((root / source).read_bytes())
    return digest.hexdigest()[:16]


def _require_supported_device(device: torch.device) -> tuple[int, int]:
    capability = torch.cuda.get_device_capability(device)
    if capability not in _SUPPORTED_CAPABILITIES:
        raise RuntimeError(
            "Q8KV4/Q8KV8 indexers support only SM100, SM103 and SM107, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return capability


def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _prefill_score_pages(max_pages: int) -> int:
    """Columns of the prefill score view: max_pages, rounded up to a multiple of
    four when max_pages % 4 is 2 or 3. With the row pitch a multiple of four
    (_prefill_score_pitch), this keeps the TopK on its float4 row kernels (see
    m3_topk.cuh, narrow::usable). The TopK family limits sit at widths of 1 mod
    4, so the rounding never moves a launch to another family."""
    return max_pages if max_pages % 4 < 2 else -(-max_pages // 4) * 4


def _prefill_score_pitch(max_pages: int) -> int:
    """Row pitch of the prefill score buffer, in floats: a multiple of four."""
    return -(-max_pages // 4) * 4


def _is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _stream_ptr(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def _to_cute_tensor(tensor: torch.Tensor, *, assumed_align: int) -> cute.Tensor:
    return from_dlpack(
        tensor.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _cute_arguments(
    tma_operands: tuple[torch.Tensor, ...], scalar_operands: tuple[torch.Tensor, ...]
) -> list[cute.Tensor]:
    return [
        *(_to_cute_tensor(tensor, assumed_align=_TMA_ALIGNMENT) for tensor in tma_operands),
        *(_to_cute_tensor(tensor, assumed_align=_SCALAR_ALIGNMENT) for tensor in scalar_operands),
    ]


def _compile_or_load(key: tuple[object, ...], compile_fn):
    """Return a compiled kernel from the process cache, the AOT cache, or JIT."""

    compiled = _COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"warm up {key[0]} before CUDA Graph capture")
    compiled = try_load_aot(key)
    if compiled is None:
        started_at = time.perf_counter()
        compiled = compile_fn()
        logger.info("[%s] Compiled in %.3fs", key[0], time.perf_counter() - started_at)
        save_aot(key, compiled, sources=_CODEGEN_SOURCES)
    _COMPILE_CACHE[key] = compiled
    return compiled


def _cute_kernel_key(name: str, capability: tuple[int, int], *static) -> tuple[object, ...]:
    # MiniMax-M3.1 vendoring: the MMA K (the exact K32 vs the SM107 K64 instruction) is part of every key, in-process and
    # AOT, so a K32 run never loads a K64 binary (the 840442 trap: the key ignored it).
    return (
        name,
        cutlass.__version__,
        str(cutlass.CUDA_VERSION),
        _kernel_source_digest(),
        capability,
        ("mma_k", _PREFILL_MMA_K),
        *static,
    )


def _check_cuda_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device | None = None,
    contiguous: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _check_block_table(block_table: torch.Tensor, batch: int, device: torch.device) -> None:
    _check_cuda_tensor(block_table, name="block_table", dtype=torch.int32, device=device)
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError(f"block_table must have shape [{batch}, max_blocks]")
    if not 0 < block_table.shape[1] <= _MAXIMUM_PAGES:
        raise ValueError(f"block_table max_blocks must be in [1, {_MAXIMUM_PAGES}]")


def _check_num_heads(num_heads: int, supported: tuple[int, ...] = _SUPPORTED_NUM_HEADS) -> int:
    if num_heads not in supported:
        raise ValueError(f"num_heads must be one of {supported}, got {num_heads!r}")
    return num_heads


def _check_query(q: torch.Tensor, num_tokens: int, num_heads: int, device: torch.device) -> None:
    _check_cuda_tensor(q, name="q", dtype=torch.float8_e4m3fn, device=device)
    expected = (num_tokens, num_heads, _HEAD_DIM)
    if tuple(q.shape) != expected:
        raise ValueError(f"q must have shape {list(expected)}, got {tuple(q.shape)}")
    if q.data_ptr() % _TMA_ALIGNMENT != 0:
        raise ValueError(f"q data pointer must be {_TMA_ALIGNMENT}-byte aligned")


def _check_paged_cache(
    k_cache: torch.Tensor,
    *,
    dtype: torch.dtype,
    page_width: int,
    device: torch.device,
) -> None:
    _check_cuda_tensor(k_cache, name="k_cache", dtype=dtype, device=device, contiguous=False)
    page_shape = (_PAGE_SIZE, page_width)
    if k_cache.ndim != 3 or k_cache.shape[0] <= 0 or tuple(k_cache.shape[1:]) != page_shape:
        raise ValueError(f"k_cache must have shape [num_blocks, {_PAGE_SIZE}, {page_width}]")
    if k_cache.stride(2) != 1 or k_cache.stride(1) != page_width:
        raise ValueError("k_cache pages must be contiguous")
    page_bytes = _PAGE_SIZE * page_width * k_cache.element_size()
    page_stride_bytes = k_cache.stride(0) * k_cache.element_size()
    if page_stride_bytes < page_bytes or page_stride_bytes % _TMA_ALIGNMENT != 0:
        raise ValueError(
            f"k_cache page stride must be at least {page_bytes} bytes and "
            f"{_TMA_ALIGNMENT}-byte aligned, got {page_stride_bytes}"
        )
    if k_cache.data_ptr() % _TMA_ALIGNMENT != 0:
        raise ValueError(f"k_cache data pointer must be {_TMA_ALIGNMENT}-byte aligned")


def _check_topk_output(
    out: torch.Tensor, num_tokens: int, num_heads: int, device: torch.device
) -> None:
    _check_cuda_tensor(out, name="out", dtype=torch.int32, device=device)
    expected = (num_tokens, num_heads, _TOP_K)
    if tuple(out.shape) != expected:
        raise ValueError(f"out must have shape {list(expected)}, got {tuple(out.shape)}")


def _load_indexer_module_from_source_tree(name: str):
    from ..fmha_sm100.jit import get_indexer_module

    return get_indexer_module(name)


# The csrc JIT lives in the parent package, whose import name depends on how
# fmha_sm100 is vendored (for example vllm.third_party.fmha_sm100), so the
# package's sparse.py binds its own loader. Direct imports of this module from
# the source tree or an installed fmha_sm100 use the fallback.
_indexer_module_loader = _load_indexer_module_from_source_tree


def bind_indexer_module_loader(loader) -> None:
    """Load csrc indexer modules through the parent package's ``jit.get_indexer_module``."""

    global _indexer_module_loader
    _indexer_module_loader = loader


def _check_score_out(
    score_out: torch.Tensor, rows: int, max_pages: int, device: torch.device
) -> None:
    """Validate a caller-owned prefill score buffer: fp32 ``[rows, >= max_pages]``."""

    _check_cuda_tensor(
        score_out, name="score_out", dtype=torch.float32, device=device, contiguous=False
    )
    if score_out.ndim != 2 or score_out.shape[0] != rows or score_out.shape[1] < max_pages:
        raise ValueError(
            f"score_out must have shape [{rows}, >= {max_pages}], got {tuple(score_out.shape)}"
        )
    if score_out.stride(1) != 1 or score_out.stride(0) < score_out.shape[1]:
        raise ValueError("score_out must keep each row's pages contiguous and rows disjoint")
    if score_out.data_ptr() % _SCALAR_ALIGNMENT:
        raise ValueError(f"score_out must be {_SCALAR_ALIGNMENT}-byte aligned")


def _topk_select(
    scores: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor, *, use_pdl: bool = True
) -> torch.Tensor:
    """Write 15 score-ranked pages and the forced local page for every row.

    ``scores`` is ``[rows, max_pages]``, or a ``[groups, rows_per_group,
    max_pages]`` view (a strided subset of the rows), ranked in row order;
    ``lengths[row]`` counts the row's candidates including its local page,
    which lands in the last slot. Ranked pages are score-descending with ties
    broken toward the lower page. Rows with at most 16 candidates emit
    ``0 .. lengths[row] - 1`` followed by ``-1``. It launches with programmatic
    dependent launch (it waits for the scores' producer on the device) unless
    ``use_pdl`` is False.
    """

    _indexer_module_loader("indexer_topk_select").indexer_topk_select(
        scores,
        lengths,
        out.view(lengths.shape[0], _TOP_K),
        use_pdl,
        _stream_ptr(scores.device),
    )
    return out


def _decode_num_valid_pages(
    seq_lens: torch.Tensor, max_pages: int, num_heads: int, query_len: int = _DECODE_QUERY_LENGTH
) -> torch.Tensor:
    """Candidate pages per (query, head) row, including the query's local page."""

    query_offsets = torch.arange(-query_len, 0, dtype=torch.int32, device=seq_lens.device)
    positions = seq_lens[:, None] + query_offsets[None, :]
    lengths = torch.div(positions, _PAGE_SIZE, rounding_mode="floor").add_(1)
    lengths.clamp_(min=1, max=max_pages)
    return lengths[:, :, None].expand(-1, -1, num_heads).reshape(-1)


class _BatchDecodeIndexerBase:
    """Shared plan/run state for the eight-token MTP decode indexers.

    ``seq_lens`` includes the current eight MTP tokens, so query ``i`` of
    request ``b`` sits at position ``seq_lens[b] - 8 + i`` and its local page is
    that position divided by 128. The eight tokens of all ``num_heads`` heads
    are scored together, reading each K page once. A wrapper instance owns its
    workspace and buffers and must not be shared by concurrently running
    streams. ``score_buffer`` (fp32, at least ``batch * 8 * num_heads *
    max_pages`` elements) replaces the per-wrapper page-score scratch, so
    wrappers of different batch sizes that never run concurrently can share it.

    Wrappers with ``_supports_query_len`` also plan requests of fewer queries
    (``query_len``): ``q`` and the output then hold only those, which take the
    last ``query_len`` of the eight slots.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        *,
        num_heads: int = 1,
        use_cuda_graph: bool = False,
        block_table_buffer: torch.Tensor | None = None,
        seq_lens_buffer: torch.Tensor | None = None,
        score_buffer: torch.Tensor | None = None,
    ) -> None:
        if score_buffer is not None:
            _check_cuda_tensor(score_buffer, name="score_buffer", dtype=torch.float32)
            if score_buffer.ndim != 1:
                raise ValueError("score_buffer must be one-dimensional")
        if workspace_buffer is not None:
            _check_cuda_tensor(workspace_buffer, name="workspace_buffer", dtype=torch.uint8)
            if workspace_buffer.ndim != 1:
                raise ValueError("workspace_buffer must be one-dimensional")
            if workspace_buffer.data_ptr() % _SCALAR_ALIGNMENT != 0:
                raise ValueError(f"workspace_buffer must be {_SCALAR_ALIGNMENT}-byte aligned")
        if use_cuda_graph:
            if block_table_buffer is None or seq_lens_buffer is None:
                raise ValueError("CUDA Graph mode requires block_table_buffer and seq_lens_buffer")
            self._check_metadata(block_table_buffer, seq_lens_buffer)
        elif block_table_buffer is not None or seq_lens_buffer is not None:
            raise ValueError(
                "block_table_buffer and seq_lens_buffer are only valid with use_cuda_graph=True"
            )
        self._num_heads = _check_num_heads(num_heads, self._supported_num_heads)
        self._workspace = workspace_buffer
        self._owns_workspace = workspace_buffer is None
        self._use_cuda_graph = use_cuda_graph
        self._block_table_buffer = block_table_buffer
        self._seq_lens_buffer = seq_lens_buffer
        self._block_table: torch.Tensor | None = None
        self._seq_lens: torch.Tensor | None = None
        self._score_buffer = score_buffer
        self._scores: torch.Tensor | None = None
        self._num_valid_pages: torch.Tensor | None = None
        self._topk_indices: torch.Tensor | None = None
        self._query_len = _DECODE_QUERY_LENGTH

    _supported_num_heads = _SUPPORTED_NUM_HEADS
    _supports_query_len = False
    _kernel_name: str
    _kernel_class: type

    def _check_k_cache(self, k_cache: torch.Tensor, device: torch.device) -> None:
        raise NotImplementedError

    @classmethod
    def _scheduler_workspace_size(cls, batch_size: int, num_heads: int) -> int:
        return (batch_size + 1) * torch.int32.itemsize

    def _plan_scheduler(self, block_table: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """Prefix-sum the historical pages each request scores and count the
        candidate pages of every query row."""

        batch_size, max_pages = block_table.shape
        self._num_valid_pages.copy_(
            _decode_num_valid_pages(seq_lens, max_pages, self._num_heads, self._query_len)
        )
        scheduler_bytes = self._scheduler_workspace_size(batch_size, self._num_heads)
        scheduler = self._workspace[:scheduler_bytes].view(torch.int32)
        history_pages = torch.div(seq_lens - 1, _PAGE_SIZE, rounding_mode="floor").clamp_(
            min=0, max=max_pages
        )
        scheduler[:1].zero_()
        torch.cumsum(history_pages, dim=0, out=scheduler[1:])

    def _launch_scores(self, q: torch.Tensor, k_cache: torch.Tensor) -> None:
        device = q.device
        sm_count = _sm_count(device)
        key = _cute_kernel_key(
            self._kernel_name,
            _require_supported_device(device),
            sm_count,
            self._num_heads,
        )
        metadata = (self._block_table, self._seq_lens, self._scores, self._workspace)
        compiled = _compile_or_load(
            key,
            lambda: cute.compile(
                self._kernel_class(sm_count=sm_count, num_heads=self._num_heads),
                *_cute_arguments((q, k_cache), metadata),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi --opt-level 2",
            ),
        )
        compiled(q, k_cache, *metadata)

    @classmethod
    def workspace_size(cls, batch_size: int, *, num_heads: int = 1) -> int:
        """Return the opaque scheduler workspace size in bytes."""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return cls._scheduler_workspace_size(
            batch_size, _check_num_heads(num_heads, cls._supported_num_heads)
        )

    @classmethod
    def score_buffer_size(cls, batch_size: int, max_pages: int, *, num_heads: int = 1) -> int:
        """Return the fp32 elements ``score_buffer`` needs for a batch."""

        num_heads = _check_num_heads(num_heads, cls._supported_num_heads)
        return batch_size * _DECODE_QUERY_LENGTH * num_heads * max_pages

    @staticmethod
    def _check_metadata(block_table: torch.Tensor, seq_lens: torch.Tensor) -> None:
        _check_cuda_tensor(seq_lens, name="seq_lens", dtype=torch.int32)
        if seq_lens.ndim != 1 or seq_lens.shape[0] <= 0:
            raise ValueError("seq_lens must have shape [batch]")
        _check_block_table(block_table, seq_lens.shape[0], seq_lens.device)

    def plan(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        query_len: int = _DECODE_QUERY_LENGTH,
    ) -> None:
        """Bind metadata and build the device schedule without host synchronization.

        Call outside CUDA Graph capture whenever ``seq_lens`` or the batch
        changes. In CUDA Graph mode the metadata is copied into the fixed
        constructor buffers, whose shapes are fixed for the wrapper lifetime.
        ``query_len`` is the number of queries per request ``q`` holds.
        """

        self._check_metadata(block_table, seq_lens)
        if query_len != _DECODE_QUERY_LENGTH and (
            not self._supports_query_len
            or isinstance(query_len, bool)
            or not isinstance(query_len, int)
            or not 1 <= query_len <= _DECODE_QUERY_LENGTH
        ):
            raise ValueError(
                f"{type(self).__name__} supports query_len "
                + ("in [1, 8]" if self._supports_query_len else "8")
                + f", got {query_len!r}"
            )
        device = block_table.device
        _require_supported_device(device)
        if _is_capturing(device):
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if self._use_cuda_graph:
            for name, tensor, buffer in (
                ("block_table", block_table, self._block_table_buffer),
                ("seq_lens", seq_lens, self._seq_lens_buffer),
            ):
                if tensor.shape != buffer.shape or tensor.device != buffer.device:
                    raise ValueError(f"CUDA Graph mode fixes {name} shape and device")
                if tensor.data_ptr() != buffer.data_ptr():
                    buffer.copy_(tensor, non_blocking=True)
            block_table = self._block_table_buffer
            seq_lens = self._seq_lens_buffer

        batch_size, max_pages = block_table.shape
        required_bytes = self._scheduler_workspace_size(batch_size, self._num_heads)
        workspace = self._workspace
        if workspace is None or workspace.device != device or workspace.numel() < required_bytes:
            if not self._owns_workspace:
                raise ValueError(
                    f"workspace_buffer must be on {device} with at least {required_bytes} bytes"
                )
            self._workspace = torch.empty(required_bytes, dtype=torch.uint8, device=device)
        self._block_table = block_table
        self._seq_lens = seq_lens
        self._query_len = query_len

        tokens = batch_size * query_len
        score_shape = (batch_size, _DECODE_QUERY_LENGTH * self._num_heads, max_pages)
        if self._score_buffer is not None:
            score_elems = self.score_buffer_size(batch_size, max_pages, num_heads=self._num_heads)
            if self._score_buffer.device != device or self._score_buffer.numel() < score_elems:
                raise ValueError(
                    f"score_buffer must be on {device} with at least {score_elems} elements"
                )
            self._scores = self._score_buffer[:score_elems].view(score_shape)
        elif (
            self._scores is None
            or self._scores.shape != score_shape
            or self._scores.device != device
        ):
            self._scores = torch.empty(score_shape, dtype=torch.float32, device=device)
        if (
            self._topk_indices is None
            or self._topk_indices.shape[0] != tokens
            or self._topk_indices.device != device
        ):
            self._num_valid_pages = torch.empty(
                (tokens * self._num_heads,), dtype=torch.int32, device=device
            )
            self._topk_indices = torch.empty(
                (tokens, self._num_heads, _TOP_K), dtype=torch.int32, device=device
            )
        self._plan_scheduler(block_table, seq_lens)

    def _run_scores(self, q: torch.Tensor, k_cache: torch.Tensor) -> torch.Tensor:
        """Write historical page scores; the local page and later pages stay untouched."""

        if self._scores is None:
            raise RuntimeError("plan() must be called before run()")
        device = self._scores.device
        _check_query(q, self._scores.shape[0] * self._query_len, self._num_heads, device)
        self._check_k_cache(k_cache, device)
        self._launch_scores(q, k_cache)
        return self._scores

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[batch * query_len, num_heads, 16]`` logical page indices for one layer."""

        scores = self._run_scores(q, k_cache)
        if out is None:
            out = self._topk_indices
        else:
            _check_topk_output(out, self._topk_indices.shape[0], self._num_heads, scores.device)
        # The rows of the queries q holds, which fill the last slots of the eight.
        query_rows = scores[:, (_DECODE_QUERY_LENGTH - self._query_len) * self._num_heads :]
        return _topk_select(query_rows, self._num_valid_pages, out)


class BatchDecodeIndexerQ8KV8Wrapper(_BatchDecodeIndexerBase):
    """E4M3 Q and E4M3 K paged decode indexer.

    Example::

        wrapper = BatchDecodeIndexerQ8KV8Wrapper(num_heads=4)
        wrapper.plan(block_table, seq_lens)
        topk_indices = wrapper.run(q, k_cache)
    """

    _kernel_name = "q8kv8_indexer_decode_sm100"
    _kernel_class = Q8KV8DecodeIndexerSm100

    def _check_k_cache(self, k_cache: torch.Tensor, device: torch.device) -> None:
        _check_paged_cache(k_cache, dtype=torch.float8_e4m3fn, page_width=_HEAD_DIM, device=device)


class BatchDecodeIndexerQ8KV4Wrapper(_BatchDecodeIndexerBase):
    """E4M3 Q and NVFP4 K paged decode indexer for one, two or four index heads.

    ``k_cache`` is the vLLM packed NVFP4 page; the per-tensor global scale is
    positive and does not change the ranking, so it is not an input. One head
    runs the CUTLASS C++ kernel and two or four heads run the CuTe DSL kernel. Builds
    on CUDA 13.4 or newer use the public QMUL4 instruction on SM100/SM103, and
    older ones select the exact FP16 dequantization. SM107 has no QMUL4: both of its
    kernels keep every NVFP4 x scale product exact in FP16 instead of rounding it to
    E4M3, so its scores differ from SM100's. Requests may carry fewer than
    eight queries (``plan(..., query_len=n)``), so a caller needs no zero-padded
    copy of its queries and gets ``[batch * n, num_heads, 16]`` back.

    Example::

        wrapper = BatchDecodeIndexerQ8KV4Wrapper(num_heads=4)
        wrapper.plan(block_table, seq_lens)
        topk_indices = wrapper.run(q, k_cache)
    """

    _supported_num_heads = (1, *Q8KV4DecodeIndexerSm100.supported_num_heads)
    _supports_query_len = True
    _kernel_name = "q8kv4_indexer_decode_sm100"
    _kernel_class = Q8KV4DecodeIndexerSm100

    @staticmethod
    def _module():
        return _indexer_module_loader("q8kv4_indexer_decode")

    @classmethod
    def _scheduler_workspace_size(cls, batch_size: int, num_heads: int) -> int:
        if num_heads == 1:
            return int(cls._module().q8kv4_indexer_workspace_size(batch_size))
        return super()._scheduler_workspace_size(batch_size, num_heads)

    def _plan_scheduler(self, block_table: torch.Tensor, seq_lens: torch.Tensor) -> None:
        if self._num_heads == 1:
            # One launch also writes the candidate page counts.
            self._module().q8kv4_indexer_plan(
                block_table,
                seq_lens,
                self._workspace,
                self._num_valid_pages,
                self._query_len,
                _stream_ptr(block_table.device),
            )
        else:
            super()._plan_scheduler(block_table, seq_lens)

    def _check_k_cache(self, k_cache: torch.Tensor, device: torch.device) -> None:
        _check_paged_cache(k_cache, dtype=torch.uint8, page_width=_NVFP4_PAGE_WIDTH, device=device)

    def run_scores_hpr(self, q: torch.Tensor, k_cache: torch.Tensor, kernel_out: torch.Tensor) -> None:
        """MiniMax-M3.1 (MSA #7): the scores of the planned step written straight into the select layout
        (``q8kv4_hpr_buffer``'s ``kernel_out``; read them through its ``view``). Two or four heads (the CuTe DSL kernel)."""
        if self._num_heads == 1 or self._scores is None:
            raise RuntimeError("run_scores_hpr needs plan() and two or four heads")
        device = q.device
        _check_query(q, self._scores.shape[0] * self._query_len, self._num_heads, device)
        self._check_k_cache(k_cache, device)
        sm_count = _sm_count(device)
        key = _cute_kernel_key(
            self._kernel_name, _require_supported_device(device), sm_count, self._num_heads, ("hpr", self._query_len)
        )
        metadata = (self._block_table, self._seq_lens, kernel_out, self._workspace)
        compiled = _compile_or_load(
            key,
            lambda: cute.compile(
                self._kernel_class(sm_count=sm_count, num_heads=self._num_heads, out_query_len=self._query_len),
                *_cute_arguments((q, k_cache), metadata),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi --opt-level 2",
            ),
        )
        compiled(q, k_cache, *metadata)

    def _launch_scores(self, q: torch.Tensor, k_cache: torch.Tensor) -> None:
        if self._num_heads != 1:
            super()._launch_scores(q, k_cache)
            return
        device = q.device
        self._module().q8kv4_indexer_run(
            q,
            k_cache,
            self._block_table,
            self._seq_lens,
            self._workspace,
            _sm_count(device),
            self._scores,
            _stream_ptr(device),
        )


def q8kv4_hpr_buffer(batch_size: int, max_pages: int, num_heads: int, query_len: int, device) -> tuple:
    """MiniMax-M3.1 (MSA #7): the score buffer of ``BatchDecodeIndexerQ8KV4Wrapper.run_scores_hpr`` and its select view
    [num_heads, max_pages, batch_size * query_len] (contiguous, at offset 8 - query_len of the buffer)."""
    rows = batch_size * query_len
    pad = _DECODE_QUERY_LENGTH - query_len
    buf = torch.empty(num_heads * max_pages * rows + pad, dtype=torch.float32, device=device)
    view = buf.as_strided((num_heads, max_pages, rows), (max_pages * rows, rows, 1), pad)
    kernel_out = buf.as_strided((num_heads, max_pages, rows), (max_pages * rows, rows, 1), 0)
    return buf, view, kernel_out


@dataclass(frozen=True, slots=True)
class _PrefillPlanState:
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    task_descriptors: torch.Tensor
    task_counts: torch.Tensor
    plan_error: torch.Tensor
    task_capacity: int
    num_candidate_q_tiles: int
    max_page_chunk: int
    total_q: int
    num_heads: int
    scores: torch.Tensor
    num_valid_pages: torch.Tensor
    topk_indices: torch.Tensor


def _run_prefill_plan(state: _PrefillPlanState) -> None:
    capability = _require_supported_device(state.block_table.device)
    plan_static = (
        Q8KV8PrefillIndexerPlanBuild.num_buckets,
        Q8KV8PrefillIndexerPlanBuild.q_tile,
        Q8KV8PrefillIndexerPlanBuild.descriptor_words,
    )
    reset_compiled = _compile_or_load(
        _cute_kernel_key("q8kv8_indexer_prefill_plan_reset_sm100", capability, *plan_static),
        lambda: cute.compile(
            Q8KV8PrefillIndexerPlanReset(),
            *_cute_arguments((), (state.task_counts, state.plan_error)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        ),
    )
    build_args = (
        state.cu_seqlens_q,
        state.seq_lens,
        state.num_valid_pages,
        state.task_descriptors,
        state.task_counts,
        state.plan_error,
    )
    build_compiled = _compile_or_load(
        _cute_kernel_key("q8kv8_indexer_prefill_plan_build_sm100", capability, *plan_static),
        lambda: cute.compile(
            Q8KV8PrefillIndexerPlanBuild(),
            *_cute_arguments((), build_args),
            Int32(state.num_candidate_q_tiles),
            Int32(state.task_capacity),
            Int32(state.num_heads),
            Int32(state.max_page_chunk),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        ),
    )
    reset_compiled(state.task_counts, state.plan_error)
    build_compiled(
        *build_args,
        state.num_candidate_q_tiles,
        state.task_capacity,
        state.num_heads,
        state.max_page_chunk,
    )


class BatchPrefillIndexerQ8KV8Wrapper:
    """E4M3 Q and E4M3 K true-varlen paged prefill indexer (CuTe DSL).

    Queries use bottom-right causal alignment: query ``i`` of request ``b``
    sits at position ``seq_lens[b] - query_len[b] + i``. All ``num_heads`` heads
    are scored in one launch as Q rows ``token * num_heads + head``.

    Example::

        wrapper = BatchPrefillIndexerQ8KV8Wrapper()
        wrapper.plan(
            cu_seqlens_q, seq_lens, block_table,
            total_q=total_q, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        )
        topk_indices = wrapper.run(q, k_cache)

    A caller that selects pages itself passes its own score buffer to ``plan``
    (``score_out``) and calls ``run_scores``, which skips the TopK.
    """

    def __init__(self, *, num_heads: int = 1) -> None:
        self._num_heads = _check_num_heads(num_heads)
        self._state: _PrefillPlanState | None = None

    def plan(
        self,
        cu_seqlens_q: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        *,
        total_q: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        score_out: torch.Tensor | None = None,
    ) -> None:
        """Size reusable buffers from host bounds and build device tasks.

        ``cu_seqlens_q`` is ``[batch + 1]`` int32 starting at zero. The host
        bounds must cover the device metadata; they size buffers and the plan
        grid but never enter a compile key. Call outside CUDA Graph capture.

        ``score_out`` optionally replaces the wrapper's score buffer: fp32
        ``[total_q * num_heads, >= ceil(max_seqlen_k / 128)]`` with contiguous
        pages per row; the row pitch may be wider, so a slice of a larger
        buffer works. Scores are written in place; entries the kernel does not
        write keep their contents.
        """

        _check_cuda_tensor(cu_seqlens_q, name="cu_seqlens_q", dtype=torch.int32)
        device = cu_seqlens_q.device
        _require_supported_device(device)
        if _is_capturing(device):
            raise RuntimeError("plan() must be called outside CUDA Graph capture")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2:
            raise ValueError("cu_seqlens_q must have shape [batch + 1]")
        batch = cu_seqlens_q.numel() - 1
        _check_cuda_tensor(seq_lens, name="seq_lens", dtype=torch.int32, device=device)
        if tuple(seq_lens.shape) != (batch,):
            raise ValueError(f"seq_lens must have shape [{batch}]")
        _check_block_table(block_table, batch, device)
        total_q, max_seqlen_q, max_seqlen_k = int(total_q), int(max_seqlen_q), int(max_seqlen_k)
        if total_q <= 0 or max_seqlen_q <= 0 or max_seqlen_k <= 0:
            raise ValueError("total_q, max_seqlen_q, and max_seqlen_k must be positive")
        if max_seqlen_k < max_seqlen_q:
            raise ValueError("bottom-right causal alignment requires max_seqlen_k >= max_seqlen_q")
        max_pages = -(-max_seqlen_k // _PAGE_SIZE)
        if block_table.shape[1] < max_pages:
            raise ValueError("block_table max_blocks is smaller than ceil(max_seqlen_k / 128)")

        q_tile = Q8KV8PrefillIndexerPlanBuild.q_tile
        total_rows = total_q * self._num_heads
        q_tile_capacity = -(-total_rows // q_tile) + batch - 1
        task_capacity = q_tile_capacity * -(-max_pages // _PREFILL_TASK_CAPACITY_PAGE_CHUNK)
        # On SM107, cap the planner's page chunk so every Q tile splits into at
        # most clusters // q_tiles tasks: short-query batches then fill the
        # machine in one wave instead of leaving most clusters idle. Host-known
        # upper bounds only (no D2H); long-query batches keep the default chunk.
        if _require_supported_device(device) == (10, 7):
            num_clusters = (
                _sm_count(device)
                * Q8KV8PrefillIndexerSm100.launch_config((10, 7))[0]
                // Q8KV8PrefillIndexerSm100.cta_group_size
            )
            chunks_per_q_tile = max(
                1, num_clusters * _PREFILL_TASKS_PER_CLUSTER // q_tile_capacity
            )
            max_page_chunk = max(
                Q8KV8PrefillIndexerPlanBuild.min_page_chunk,
                min(
                    Q8KV8PrefillIndexerPlanBuild.large_page_chunk,
                    -(-max_pages // chunks_per_q_tile),
                ),
            )
        else:
            max_page_chunk = Q8KV8PrefillIndexerPlanBuild.large_page_chunk
        if score_out is not None:
            _check_score_out(score_out, total_rows, max_pages, device)
        options = {"device": device}
        self._state = _PrefillPlanState(
            cu_seqlens_q=cu_seqlens_q,
            seq_lens=seq_lens,
            block_table=block_table,
            task_descriptors=torch.empty(
                (
                    Q8KV8PrefillIndexerPlanBuild.num_buckets,
                    task_capacity,
                    Q8KV8PrefillIndexerPlanBuild.descriptor_words,
                ),
                dtype=torch.int32,
                **options,
            ),
            task_counts=torch.empty(
                (Q8KV8PrefillIndexerPlanBuild.num_buckets,), dtype=torch.int32, **options
            ),
            plan_error=torch.empty((1,), dtype=torch.int32, **options),
            task_capacity=task_capacity,
            num_candidate_q_tiles=batch * -(-max_seqlen_q * self._num_heads // q_tile),
            max_page_chunk=max_page_chunk,
            total_q=total_q,
            num_heads=self._num_heads,
            scores=(
                score_out
                if score_out is not None
                else torch.empty(
                    (total_rows, _prefill_score_pitch(max_pages)), dtype=torch.float32, **options
                )[:, : _prefill_score_pages(max_pages)]
            ),
            num_valid_pages=torch.empty((total_rows,), dtype=torch.int32, **options),
            topk_indices=torch.empty(
                (total_q, self._num_heads, _TOP_K), dtype=torch.int32, **options
            ),
        )
        _run_prefill_plan(self._state)

    def replan(self) -> None:
        """Rebuild device tasks after in-place updates of the planned metadata."""

        if self._state is None:
            raise RuntimeError("plan() must be called before replan()")
        _run_prefill_plan(self._state)

    def run_scores(self, q: torch.Tensor, k_cache: torch.Tensor) -> torch.Tensor:
        """Score every row's historical pages without the TopK; return the score buffer.

        Each row's pages before its local page are written. The local page and
        later pages keep their previous contents, so a caller selecting its own
        pages pre-fills the buffer (for example with ``-inf``) and forces the
        local page.
        """

        return self._run_scores(q, k_cache)

    def _run_scores(self, q: torch.Tensor, k_cache: torch.Tensor) -> torch.Tensor:
        """Write historical page scores; the local page and later pages stay untouched."""

        state = self._state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        device = state.block_table.device
        _check_query(q, state.total_q, state.num_heads, device)
        _check_paged_cache(k_cache, dtype=torch.float8_e4m3fn, page_width=_HEAD_DIM, device=device)
        capability = _require_supported_device(device)
        ctas_per_sm, k_stages, acc_stages = Q8KV8PrefillIndexerSm100.launch_config(capability)
        num_persistent_clusters = (
            _sm_count(device) * ctas_per_sm // Q8KV8PrefillIndexerSm100.cta_group_size
        )
        # Q rows are token * num_heads + head, matching the scores and TopK rows.
        q_rows = q.view(state.total_q * state.num_heads, 1, _HEAD_DIM)
        metadata = (
            state.block_table,
            state.scores,
            state.task_descriptors,
            state.task_counts,
        )
        compiled = _compile_or_load(
            _cute_kernel_key(
                "q8kv8_indexer_prefill_sm100",
                capability,
                num_persistent_clusters,
                Q8KV8PrefillIndexerSm100.q_tile,
                k_stages,
                acc_stages,
            ),
            lambda: cute.compile(
                Q8KV8PrefillIndexerSm100(
                    compute_capability=capability,
                    num_persistent_clusters=num_persistent_clusters,
                ),
                *_cute_arguments((q_rows, k_cache), metadata),
                Int32(state.num_heads),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            ),
        )
        compiled(q_rows, k_cache, *metadata, state.num_heads)
        return state.scores

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[total_q, num_heads, 16]`` logical page indices for one layer."""

        scores = self._run_scores(q, k_cache)
        state = self._state
        if out is None:
            out = state.topk_indices
        else:
            _check_topk_output(out, state.total_q, state.num_heads, scores.device)
        # The prefill chain runs eagerly and is not PDL-chained.
        return _topk_select(scores, state.num_valid_pages, out, use_pdl=False)


__all__ = [
    "q8kv4_hpr_buffer",
    "BatchDecodeIndexerQ8KV4Wrapper",
    "BatchDecodeIndexerQ8KV8Wrapper",
    "BatchPrefillIndexerQ8KV8Wrapper",
]
