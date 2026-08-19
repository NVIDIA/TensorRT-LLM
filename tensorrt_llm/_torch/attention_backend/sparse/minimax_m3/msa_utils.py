# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA (fmha_sm100) specific helpers for the MiniMax-M3 sparse backend."""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import torch

from tensorrt_llm._utils import maybe_pin_memory

from .common import write_kv_slots

# fmha_sm100 ships only head_dim 128 variants and the MiniMax-M3 checkpoint
# selects topk 16. Callers enforce these early so a misconfiguration fails
# with a clear message rather than a cryptic shape error inside the kernel.
MSA_REQUIRED_TOPK = 16
MSA_REQUIRED_HEAD_DIM = 128

# Path of the fmha_sm100 package inside the MSA git submodule relative to the
# repository root (see 3rdparty/MSA/LICENSE and 3rdparty/MSA/NOTICE).
_MSA_PYTHON_RELPATH = Path("3rdparty") / "MSA" / "python"


def msa_ported_decode_active(metadata) -> bool:
    """Whether this step's generation rows run on the ported decode kernels.

    They always do, when the step has generation rows at all: the Triton sparse
    kernel, the trtllm-gen dense kernel and the CuTe DSL indexer scorer own the
    generation span together (the whole of a pure-decode batch, or the row and
    token suffix of a mixed one) and fmha_sm100 is left with the context prefix.
    There is no per-kernel switch and no fallback for a generation row: a
    geometry the span's kernels cannot serve raises in prepare() instead of
    being routed back to fmha_sm100 at several times the decode cost. See
    _resolve_decode_kernels.

    So False means only that this step has no span: a pure-prefill step, or one
    that opted out by leaving the query length or the buffers the ported kernels
    address unset, which the standalone kernel tests do since their metadata
    never runs prepare().

    One predicate serves every site that has to agree on the span: both
    attention branches and the indexer, which additionally orients its top-k
    table for whichever of them consumes it.
    """
    return (
        getattr(metadata, "msa_decode_query_len", None) is not None
        and getattr(metadata, "msa_block_table", None) is not None
        and getattr(metadata, "msa_seq_lens_cuda", None) is not None
    )


class MsaSpanBounds(NamedTuple):
    """The generation span's bounds.

    Labelled so that a call site taking only some of the five does not depend
    on their order. Unpacks positionally.
    """

    token_first: int
    token_last: int
    row_first: int
    row_last: int
    query_len: int


def msa_decode_span_bounds(metadata, num_tokens: int) -> MsaSpanBounds:
    """Bounds of the generation span the ported decode kernels own this step.

    See _MsaDecodeSpan in msa_backend for what the span is. Reading it through
    getattr keeps this module free of an import back into the backend, and
    covers the standalone kernel tests, whose metadata never ran prepare() and
    so carries no span: there the whole batch is the span, derived from the
    query length alone.

    token_last is where the span's tokens end, and so where the step's live
    tokens end: the span is the whole token axis minus the context prefix.

    Returns zeros when no query length is resolved either, in which case every
    caller is on the fmha_sm100 path and ignores these bounds.
    """
    span = getattr(metadata, "msa_decode_span", None)
    if span is not None:
        return MsaSpanBounds(
            span.token_first, span.token_last, span.row_first, span.row_last, span.query_len
        )
    query_len = getattr(metadata, "msa_decode_query_len", None)
    if query_len is None:
        return MsaSpanBounds(0, 0, 0, 0, 0)
    query_len = int(query_len)
    rows = num_tokens // query_len
    return MsaSpanBounds(0, rows * query_len, 0, rows, query_len)


def check_decode_span_shape(kernel: str, total_q: int, batch: int, query_len: int) -> None:
    """Reject a q that does not cover exactly the batch it was handed.

    The ported decode kernels derive the request id as token // query_len, so
    a longer q reads page table rows and lengths past the batch's last one.
    """
    if total_q != batch * query_len:
        raise ValueError(
            f"{kernel}: total_q ({total_q}) must be batch ({batch}) * "
            f"decode_query_len ({query_len})."
        )


@functools.lru_cache(maxsize=1)
def _find_msa_python_dir() -> Optional[Path]:
    """Locate the fmha_sm100 package dir by walking up from this file.

    Returns None in installed layouts where the 3rdparty submodule is not
    shipped. Walking up avoids hardcoding this module's depth below the
    repository root.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _MSA_PYTHON_RELPATH
        if candidate.is_dir():
            return candidate
    return None


def _ensure_msa_on_path() -> None:
    """Prepend the MSA python package directory to sys.path if present."""
    msa_python = _find_msa_python_dir()
    if msa_python is not None and str(msa_python) not in sys.path:
        sys.path.insert(0, str(msa_python))


def msa_package_available() -> bool:
    """True if fmha_sm100 can be imported (submodule checkout or installed)."""
    _ensure_msa_on_path()
    return importlib.util.find_spec("fmha_sm100") is not None


# One symbol per module touched by 3rdparty/patches/msa_strided_paged_kv.patch.
_MSA_PATCH_MARKERS = (
    ("fmha_sm100.api", "_mixed_batch_split"),
    ("fmha_sm100.cute.interface", "_prepare_paged_hnd_input"),
    ("fmha_sm100.sparse_fmha_adapter", "_page_table_for_plan"),
)


@functools.lru_cache(maxsize=1)
def _require_msa_patch() -> None:
    """Fail fast if the loaded fmha_sm100 is missing the downstream patch.

    The patch keeps prefill from materializing the paged K/V cache and builds
    the sparse page table with one gather per step. scripts/build_wheel.py
    applies it in place on the 3rdparty/MSA submodule. A copy missing either
    marker would silently regress prefill or per-step host time, so report
    every absent symbol with the file it was expected in.
    """
    missing = []
    for module_name, symbol in _MSA_PATCH_MARKERS:
        module = importlib.import_module(module_name)
        if not hasattr(module, symbol):
            missing.append(f"'{symbol}' in '{getattr(module, '__file__', '<unknown>')}'")

    if missing:
        raise RuntimeError(
            "The imported fmha_sm100 is missing the TensorRT-LLM MSA patch "
            f"(expected {', '.join(missing)}). Rebuild with "
            "scripts/build_wheel.py, which applies "
            "3rdparty/patches/msa_strided_paged_kv.patch in place, or apply the "
            "patch to 3rdparty/MSA manually."
        )


def require_msa_module():
    """Import fmha_sm100 from the MSA submodule or raise a clear error.

    The import is deferred to first kernel use so the MSA backend can be
    advertised in the config schema on systems where the kernels cannot load.
    The 3rdparty/MSA/python directory is added to sys.path first, so a source
    checkout with the submodule initialized resolves without a separate install.
    A missing package is a hard error, never a silent fallback to another backend.
    """
    _ensure_msa_on_path()
    try:
        import fmha_sm100
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-M3 MSA attention requires the fmha_sm100 kernels from the "
            "MSA git submodule at 3rdparty/MSA. Initialize it with "
            "'git submodule update --init --recursive', or install fmha_sm100."
        ) from exc
    _require_msa_patch()
    return fmha_sm100


def msa_paged_kv(kv_cache_manager, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer paged K and V in fmha_sm100 HND layout, zero-copy.

    The cache is stored head-major (see `write_msa_main_kv`), so the "HND"
    buffer view is already the [num_slots, num_kv_heads, page_size, head_dim]
    layout fmha_sm100 expects. The kernel reads the page and head strides at
    runtime and needs only each page's [page_size, head_dim] block to be
    contiguous, which this view satisfies, so no copy is required.
    """
    if getattr(kv_cache_manager, "is_fp8_subpaged_layer", lambda _layer_idx: False)(layer_idx):
        raise RuntimeError(
            "hybrid FP8 dense/Eagle cache is physically P32; use the direct "
            "TRTLLM-Gen dense adapter instead of msa_paged_kv"
        )
    buffers = kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
    return buffers[:, 0], buffers[:, 1]


def write_msa_main_kv(
    kv_cache_manager,
    layer_idx: int,
    out_cache_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Write new-token K and V into the paged main cache at out_cache_loc.

    fmha_sm100 reads the paged cache directly, so the new-token K and V must be
    resident before the sparse GQA runs. The write uses the head-major HND view
    so `msa_paged_kv` can return a zero-copy view.
    """
    if getattr(kv_cache_manager, "is_fp8_subpaged_layer", lambda _layer_idx: False)(layer_idx):
        from .msa_scatter import fused_write_subpaged_layer_caches

        k_view, v_view = kv_cache_manager.get_fp8_dense_buffers(layer_idx)
        if not fused_write_subpaged_layer_caches(k_view, v_view, out_cache_loc, k, v):
            raise RuntimeError(
                "MiniMax-M3 hybrid FP8 dense/Eagle cache write requires CUDA "
                "P32 sub-page views and contiguous K/V rows"
            )
        return
    buffers = kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
    k_view, v_view = buffers[:, 0], buffers[:, 1]
    num_kv_heads = int(k_view.shape[1])
    head_dim = int(k_view.shape[3])
    num_tokens = int(k.shape[0])
    write_kv_slots(
        k_view, out_cache_loc, k.reshape(num_tokens, num_kv_heads, head_dim), layout="HND"
    )
    write_kv_slots(
        v_view, out_cache_loc, v.reshape(num_tokens, num_kv_heads, head_dim), layout="HND"
    )


def build_kv_page_indices(
    block_ids_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Build the flattened per-request page table fmha_sm100 consumes, on the host.

    Returns int32 global page ids concatenated per request, request b
    contributing the first ceil(kv_len / page_size) entries of its
    block_ids_cpu row. A request with kv_len <= 0 contributes nothing, matching
    the page count the plan derives from the same lengths. Page ids are global
    and non-contiguous in production, so they are not clamped to a per-request
    bound.

    block_ids_cpu is the [batch, max_blocks] host table
    build_paged_kv_slot_mapping obtains from the cache manager. Both use the
    manager's tokens_per_block as the page size, so
    req_to_token[b, p * page_size] // page_size equals block_ids_cpu[b, p] and
    the page table needs no device work. The result is pinned where that helps,
    so the caller stages it with one asynchronous copy.
    """
    pages = (kv_lens_cpu.to(torch.long) + (page_size - 1)) // page_size
    pages.clamp_(min=0)
    total_pages = int(pages.sum())
    if total_pages == 0:
        return torch.empty(0, dtype=torch.int32)
    batch = int(pages.shape[0])
    row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), pages)
    starts = torch.cumsum(pages, 0) - pages
    col = torch.arange(total_pages, dtype=torch.long) - starts[row]
    return maybe_pin_memory(block_ids_cpu[row, col].to(torch.int32))


def per_token_valid_blocks(
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    *,
    causal: bool,
    block_size: int,
) -> torch.Tensor:
    """Return the per-query number of valid KV blocks, on CPU.

    Expands per-request lengths and offsets to a per-token vector so block
    selection can honour each query token's own causal extent.
    """
    qo = qo_lens_cpu.to(torch.long)
    kv = kv_lens_cpu.to(torch.long)
    batch = int(qo.shape[0])
    total = int(qo.sum().item())
    if total == 0:
        return torch.zeros(0, dtype=torch.long)
    batch_row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), qo)
    starts = torch.zeros(batch, dtype=torch.long)
    if batch > 1:
        starts[1:] = torch.cumsum(qo, 0)[:-1]
    intra = torch.arange(total, dtype=torch.long) - starts[batch_row]
    kv_per = kv[batch_row]
    if causal:
        if qo_offset_cpu is not None:
            off = qo_offset_cpu.to(torch.long)[batch_row]
        else:
            off = (kv - qo)[batch_row]
        eff = torch.minimum(off + intra + 1, kv_per)
    else:
        eff = kv_per
    return (eff + block_size - 1) // block_size


def select_blocks_from_maxscore(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    head_major_output: bool = False,
) -> torch.Tensor:
    """Select per-query top-k blocks from per-KV-head block scores.

    Applies init and local forced blocks and per-query valid-block masking
    on the amax-reduced scores [num_kv_heads, n_blocks, total_q]. Returns
    [total_q, num_kv_heads, topk] int32 ascending block ids with -1 tail
    padding. head_major_output backs that result head-major instead, so
    result.permute(1, 0, 2) is contiguous without a copy.
    """
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    return torch.ops.trtllm.minimax_m3_select_blocks(
        max_score_kv,
        nvb,
        topk,
        init_blocks,
        local_blocks,
        head_major_output,
    )


__all__ = [
    "MSA_REQUIRED_HEAD_DIM",
    "MSA_REQUIRED_TOPK",
    "MsaSpanBounds",
    "build_kv_page_indices",
    "check_decode_span_shape",
    "msa_decode_span_bounds",
    "msa_package_available",
    "msa_paged_kv",
    "msa_ported_decode_active",
    "per_token_valid_blocks",
    "require_msa_module",
    "select_blocks_from_maxscore",
    "write_msa_main_kv",
]
