# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA (fmha_sm100) specific helpers for the MiniMax-M3 sparse backend."""

from __future__ import annotations

import functools
import importlib.util
import types
from typing import Optional, Tuple

import torch

from tensorrt_llm._utils import maybe_pin_memory

from .common import write_kv_slots

# fmha_sm100 ships only head_dim 128 variants and the MiniMax-M3 checkpoint
# selects topk 16. Callers enforce these early so a misconfiguration fails
# with a clear message rather than a cryptic shape error inside the kernel.
MSA_REQUIRED_TOPK = 16
MSA_REQUIRED_HEAD_DIM = 128


def _install_msa_cutlass_compatibility() -> None:
    """Provide the CUTLASS 4.5 names still referenced by the packaged MSA sources."""
    try:
        import cutlass.cute as cute
    except ImportError:
        return

    # MSA has not yet migrated these aliases to their newer CUTLASS DSL
    # names. Keep this shim local to the MSA import path and remove it once the
    # packaged sources use the top-level thread classes and
    # cute.make_rmem_tensor directly.
    for thread_class_name in ("ThrCopy", "ThrMma"):
        if not hasattr(cute.core, thread_class_name):
            setattr(cute.core, thread_class_name, getattr(cute, thread_class_name))
    if not hasattr(cute, "make_fragment"):
        setattr(cute, "make_fragment", cute.make_rmem_tensor)


@functools.lru_cache(maxsize=1)
def msa_package_available() -> bool:
    """Prepare and report whether the packaged fmha_sm100 module can be imported.

    Cached: each call scans sys.path until fmha_sm100 is first imported, and
    FmhaManager construction checks once per attention layer. Whether the
    package is installed cannot change within a process.
    """
    if importlib.util.find_spec("fmha_sm100") is None:
        return False
    _install_msa_cutlass_compatibility()
    return True


def require_msa_module() -> types.ModuleType:
    """Import the packaged fmha_sm100 module or raise a clear error.

    The import is deferred to first kernel use so the MSA backend can be
    advertised in the config schema on systems where the kernels cannot load.
    A missing package is a hard error, never a silent fallback to another backend.
    """
    _install_msa_cutlass_compatibility()
    try:
        import fmha_sm100
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-M3 MSA attention requires the fmha_sm100 kernels packaged "
            "with TensorRT-LLM. Reinstall TensorRT-LLM from a complete build."
        ) from exc
    return fmha_sm100


def msa_paged_kv(kv_cache_manager, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer paged K and V in fmha_sm100 HND layout, zero-copy.

    The cache is stored head-major (see `write_msa_main_kv`), so the "HND"
    buffer view is already the [num_slots, num_kv_heads, page_size, head_dim]
    layout fmha_sm100 expects. The kernel reads the page and head strides at
    runtime and needs only each page's [page_size, head_dim] block to be
    contiguous, which this view satisfies, so no copy is required.
    """
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
    padding. When ``head_major_output`` is set, the logical result uses a
    head-major backing so ``result.permute(1, 0, 2)`` is contiguous without a
    copy.
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
    "build_kv_page_indices",
    "msa_package_available",
    "msa_paged_kv",
    "per_token_valid_blocks",
    "require_msa_module",
    "select_blocks_from_maxscore",
    "write_msa_main_kv",
]
