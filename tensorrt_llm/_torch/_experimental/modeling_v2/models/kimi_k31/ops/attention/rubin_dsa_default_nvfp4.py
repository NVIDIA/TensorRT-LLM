# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent-KV part of the default Rubin target's DSA metadata (FP8 or NVFP4 KV cache).

trtllm-gen reads no NVFP4 KV cache on sm_107. With an NVFP4 cache, the attention gathers the latent
rows each query row attends to into an FP8 scratch and runs the sparse MLA on it: a generation step
through one stable scratch, a context step through one scratch per layer that its gather allocates.

``RubinDsaDefaultNvfp4Mixin`` defines the hooks the default target's metadata classes call; each
is a no-op unless the cache stores the latent KV as NVFP4. The generation gather's buffers are the
engine DSA metadata's own (``nvfp4_mla_fp8_scratch``, about 2.4 GB per rank at 256 sequences and 7
draft tokens, and ``nvfp4_mla_compact_indices``). They exist from the metadata's creation, so the KV
cache estimate's profiling run counts them, and no step allocates. A step's gather plan reads the
fields the specialized preparation fills, never the cache's unique primary pool: the cache manager
refuses that pool when the last MLA layers keep FP8 pages beside the NVFP4 ones.
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple, Optional, Tuple

import torch

from tensorrt_llm.bindings import DataType

# The generation gather's buffers: FP8 latent rows and int32 offsets into them.
_SCRATCH_DTYPE = torch.float8_e4m3fn
_COMPACT_DTYPE = torch.int32


class Nvfp4GenerationGather(NamedTuple):
    """One step's generation gather (``nvfp4_gen_gather``): leading rows of the stable buffers."""

    rows: int  # the step's generation query rows (num_tokens - num_ctx_tokens)
    out: torch.Tensor  # float8_e4m3fn [rows, Top-K, head_dim]: each row's gathered latent rows
    compact_indices: torch.Tensor  # int32 [rows, Top-K]: each Top-K entry's row of ``out``, or -1


class Nvfp4ContextGather(NamedTuple):
    """One step's context gather inputs (``nvfp4_ctx_gather``), views of the step's metadata."""

    query_req_indices: torch.Tensor  # int32 [context rows]: each row's context request
    block_table: torch.Tensor  # int32 [context requests, pages]: their latent-pool slots
    cu_kv_lengths: torch.Tensor  # int64 [context requests + 1]: cumulative cached + new keys
    total_kv_tokens: int  # host int: the last entry of ``cu_kv_lengths``


def nvfp4_latent_cache(kv_cache_manager) -> bool:
    """Whether a cache manager stores the latent KV as NVFP4 (its last MLA layers may keep FP8)."""
    return (
        kv_cache_manager is not None and getattr(kv_cache_manager, "dtype", None) == DataType.NVFP4
    )


def generation_scratch_rows(max_num_sequences: int, max_draft_tokens: int) -> int:
    """Rows of the generation gather's buffers: every query row of a full verify step."""
    return int(max_num_sequences) * (1 + int(max_draft_tokens))


def generation_scratch_bytes(
    max_num_sequences: int, max_draft_tokens: int, topk: int, head_dim: int
) -> int:
    """Bytes of the generation gather's two buffers (FP8 latent rows and their int32 offsets)."""
    rows = generation_scratch_rows(max_num_sequences, max_draft_tokens)
    per_entry = int(head_dim) * _SCRATCH_DTYPE.itemsize + _COMPACT_DTYPE.itemsize
    return rows * int(topk) * per_entry


def _describe(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "none"
    return f"{tuple(tensor.shape)} {tensor.dtype}"


def _check_generation_buffers(md) -> None:
    """The generation gather's buffers at the engine's size for the metadata's draft length."""
    rows = generation_scratch_rows(md.max_num_sequences, md.max_draft_tokens)
    shape = (rows, int(md.num_sparse_topk), int(md.kv_cache_manager.head_dim))
    scratch = getattr(md, "nvfp4_mla_fp8_scratch", None)
    compact = getattr(md, "nvfp4_mla_compact_indices", None)
    if (
        scratch is None
        or compact is None
        or (tuple(scratch.shape), scratch.dtype) != (shape, _SCRATCH_DTYPE)
        or (tuple(compact.shape), compact.dtype) != (shape[:2], _COMPACT_DTYPE)
        or not (scratch.is_contiguous() and compact.is_contiguous())
    ):
        raise RuntimeError(
            f"an NVFP4 KV cache needs the generation gather's FP8 scratch {shape} and its int32 "
            f"offsets {shape[:2]} from the metadata's creation (contiguous); the metadata has "
            f"{_describe(scratch)} and {_describe(compact)}"
        )


def _generation_gather(md, rows: int) -> Nvfp4GenerationGather:
    """The step's generation rows and their slices of the stable buffers (no allocation)."""
    scratch = getattr(md, "nvfp4_mla_fp8_scratch", None)
    compact = getattr(md, "nvfp4_mla_compact_indices", None)
    if scratch is None or compact is None:
        raise RuntimeError("the NVFP4 generation gather's buffers were not allocated")
    capacity = min(int(scratch.shape[0]), int(compact.shape[0]))
    if rows > capacity:
        raise RuntimeError(
            f"the step's {rows} generation rows exceed the NVFP4 generation gather's {capacity} "
            "rows (max_num_sequences x (1 + max_draft_tokens))"
        )
    return Nvfp4GenerationGather(rows, scratch[:rows], compact[:rows])


def _context_gather(md) -> Nvfp4ContextGather:
    """The context gather's inputs, from the fields the specialized preparation fills."""
    num_contexts, num_ctx_tokens = md.num_contexts, md.num_ctx_tokens
    total = int(md.num_ctx_mla_kv_tokens)
    indexed = int(md.host_ctx_kv_indptr[num_contexts])
    if total <= 0 or total != indexed:
        raise RuntimeError(
            f"the NVFP4 context gather of {num_contexts} context requests needs their KV tokens: "
            f"the step counts {total}, the cumulative KV lengths end at {indexed}"
        )
    return Nvfp4ContextGather(
        md.req_idx_per_token[:num_ctx_tokens],
        md.block_table[:num_contexts],
        md.ctx_kv_indptr[: num_contexts + 1],
        total,
    )


class RubinDsaDefaultNvfp4Mixin:
    """The latent-KV hooks of the default target's DSA metadata (see the module docstring)."""

    # The hooks the metadata classes call, in their order.
    HOOKS: ClassVar[Tuple[str, ...]] = (
        "_nvfp4_post_init",
        "_nvfp4_prepare",
        "_nvfp4_on_update_kv_lens",
    )

    def _nvfp4_post_init(self) -> None:
        """Once per metadata object, outside capture: the generation gather's buffers."""
        self.nvfp4_kv_cache = not self.is_cross and nvfp4_latent_cache(self.kv_cache_manager)
        self.nvfp4_gen_gather: Optional[Nvfp4GenerationGather] = None
        self.nvfp4_ctx_gather: Optional[Nvfp4ContextGather] = None
        if not self.nvfp4_kv_cache:
            return
        if (
            getattr(self, "nvfp4_mla_fp8_scratch", None) is None
            or getattr(self, "nvfp4_mla_compact_indices", None) is None
        ):
            self._create_nvfp4_mla_generation_buffers(capture_graph=self.is_cuda_graph)
        _check_generation_buffers(self)

    def _nvfp4_prepare(self) -> None:
        """Host side, every step, after the specialized preparation: the step's gather plan."""
        if not self.nvfp4_kv_cache:
            return
        rows = self.num_tokens - self.num_ctx_tokens
        self.nvfp4_gen_gather = _generation_gather(self, rows) if rows > 0 else None
        self.nvfp4_ctx_gather = _context_gather(self) if self.num_contexts > 0 else None

    def _nvfp4_on_update_kv_lens(self) -> None:
        """Device work only, captured and replayed with the decode metadata: nothing to refresh."""


__all__ = [
    "Nvfp4ContextGather",
    "Nvfp4GenerationGather",
    "RubinDsaDefaultNvfp4Mixin",
    "generation_scratch_bytes",
    "generation_scratch_rows",
    "nvfp4_latent_cache",
]
