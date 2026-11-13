# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's block attention on its paged context: store the block's K / V after each request's context,
then every block row attends, non-causally, to the request's context and the whole block (flashinfer TRTLLM-Gen decode,
the block's rows folded into the query heads)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    drafter_config as _drafter_config,
)
from tensorrt_llm._torch.speculative.dflash_attention import DFLASH_FOLDED_DECODE_LOWERING
from tensorrt_llm._utils import get_sm_version

#: How the DFlash worker names this lowering of a draft layer (its ``validate_block_attention_windows`` contract): the
#: block's rows are folded into the query heads of the decode kernel, so the worker checks the fold's chunk plan.
LOWERING = DFLASH_FOLDED_DECODE_LOWERING

# The trtllm-gen kernels index the multi-CTA KV counters in groups of eight.
_MULTI_CTAS_KV_COUNTER_ALIGNMENT = 8

# Query heads per KV head the TRTLLM-Gen generation kernels serve (the attention backend's
# ``FlashInferTrtllmGenFmha.MAX_HEADS_RATIO_GENERATION``).
MAX_HEADS_RATIO_GENERATION = 32

# Folded query rows per KV head in one decode launch: matching Q / KV dtypes and one query per request select a Q tile
# of up to 128 rows (the DFlash runtime's ``DFLASH_FOLDED_DECODE_MAX_RATIO_PER_LAUNCH``).
FOLDED_DECODE_MAX_RATIO_PER_LAUNCH = 128


def unavailability_reason() -> Optional[str]:
    """Why this build cannot run the op (no flashinfer, missing bindings, not sm_100 / sm_103), or None."""
    return _drafter_config.trtllm_gen_unavailability_reason(get_sm_version())


def folded_decode_chunk_size(block_size: int, num_heads: int, num_kv_heads: int) -> int:
    """Block rows per folded decode launch: the whole block where its rows times the heads per KV head fit the Q tile,
    else chunks of the block (each chunk still attends to the whole context and block). The DFlash runtime's
    ``get_dflash_folded_decode_chunk_size``.

    Raises:
        RuntimeError: a non-positive block, head counts that do not divide, or more heads per KV head than the
            generation kernels serve.
    """
    if block_size <= 0:
        raise RuntimeError(
            f"DFlash folded decode requires a positive block size; got {block_size}."
        )
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise RuntimeError(
            "DFlash folded decode requires positive head counts with num_heads divisible by num_kv_heads; got "
            f"{num_heads} and {num_kv_heads}."
        )
    group = num_heads // num_kv_heads
    if group > MAX_HEADS_RATIO_GENERATION:
        raise RuntimeError(
            f"DFlash folded decode's base Q/KV head ratio {group} exceeds the validated generation ratio "
            f"{MAX_HEADS_RATIO_GENERATION}."
        )
    return min(block_size, FOLDED_DECODE_MAX_RATIO_PER_LAUNCH // group)


@lru_cache(maxsize=128)
def _context_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    """Bytes of the TRTLLM-Gen context preprocessing workspace (the bindings' layout)."""
    if max_num_tokens == 0:
        return 0
    from tensorrt_llm.bindings.internal import thop

    layout = thop.get_trtllm_gen_context_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        True,
        fp8_context_fmha,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def _generation_workspace_size(
    dtype: torch.dtype,
    max_num_seq: int,
    max_num_tokens: int,
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    rotary_embedding_dim: int,
) -> int:
    """Bytes of the TRTLLM-Gen generation preprocessing workspace (the bindings' layout)."""
    if max_num_tokens == 0:
        return 0
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    from tensorrt_llm.bindings.internal import thop

    layout = thop.get_trtllm_gen_generation_workspace_layout(
        dtype,
        max_num_seq,
        max_num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        num_kv_heads,
    )
    return int(layout["total_size"])


@lru_cache(maxsize=128)
def workspace_size(
    dtype: torch.dtype,
    num_tokens: int,
    num_gen_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    max_num_requests: int,
    rotary_embedding_dim: int,
    fp8_context_fmha: bool,
) -> int:
    """Bytes of the op's ``workspace``: the larger of the context and generation workspaces at these sizes."""
    context_size = _context_workspace_size(
        dtype,
        max_num_requests,
        num_tokens,
        num_heads,
        head_size,
        rotary_embedding_dim,
        fp8_context_fmha,
    )
    generation_size = _generation_workspace_size(
        dtype,
        max_num_requests,
        num_gen_tokens,
        num_heads,
        head_size,
        num_kv_heads,
        rotary_embedding_dim,
    )
    return max(context_size, generation_size)


def multi_ctas_kv_counter_size(
    num_heads: int,
    max_num_sequences: int,
    multi_processor_count: int,
) -> int:
    """Bytes of the op's ``counters``: one int32 counter per head per sequence, at least one per SM, rounded up to
    the kernels' group of eight."""
    num_counters = max(num_heads * max_num_sequences, multi_processor_count)
    aligned_num_counters = (
        (num_counters + _MULTI_CTAS_KV_COUNTER_ALIGNMENT - 1)
        // _MULTI_CTAS_KV_COUNTER_ALIGNMENT
        * _MULTI_CTAS_KV_COUNTER_ALIGNMENT
    )
    return aligned_num_counters * torch.int32.itemsize


@dataclass(frozen=True)
class TrtllmGenOps:
    """flashinfer's TRTLLM-Gen paged append and decode attention, and the sizes of their scratch buffers."""

    append_paged_kv_cache: Callable[..., None]
    batch_decode_with_kv_cache: Callable[..., None]
    get_multi_ctas_kv_counter_size: Callable[..., int]
    get_workspace_size: Callable[..., int]


@lru_cache(maxsize=1)
def trtllm_gen_ops() -> TrtllmGenOps:
    """The op's kernels and buffer sizes (raises RuntimeError where this build cannot run them)."""
    reason = unavailability_reason()
    if reason is not None:
        raise RuntimeError(f"DFlash TRTLLM attention backend is unavailable: {reason}.")
    import flashinfer

    return TrtllmGenOps(
        append_paged_kv_cache=flashinfer.page.append_paged_kv_cache,
        batch_decode_with_kv_cache=flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        get_multi_ctas_kv_counter_size=multi_ctas_kv_counter_size,
        get_workspace_size=workspace_size,
    )


class K3DSparkBlockAttn(OpWrapper):
    """For B requests of a T-row block (``q`` [B T, H, D], ``k`` / ``v`` [B T, KV, D], bf16; ``k``, ``v`` and ``out``
    contiguous):

    1. store row ``i`` of ``k`` / ``v`` in ``cache`` (HND pages [P, 2, KV, page, D]) at request
       ``append_batch_indices[i]``'s position ``append_positions[i]`` (pages ``kv_indices[kv_indptr[b]:]``);
    2. write ``out`` [B T, H, D]: request b's rows attend to its first ``seq_lens[b]`` cached rows (pages
       ``block_tables[b]``, ``max_kv_len`` the rows the table spans), every row to every key, scaled by ``D ** -0.5``.

    ``workspace`` and ``counters`` are the kernel's scratch, sized by `workspace_size` and
    `multi_ctas_kv_counter_size` at the larger of B T tokens of H heads and B tokens of
    H * `folded_decode_chunk_size` heads. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{heads}/{kv} heads of 64, B={b}, T={t}: {why}",
            spec=dict(heads=heads, kv=kv, batch=b, block=t),
        )
        for heads, kv, why in (
            (4, 1, "RadixArk drafter, one TP16 rank"),
            (6, 1, "the drafter entries' layout, the splits they decline"),
        )
        for b, t in ((1, 7), (8, 7), (4, 4), (3, 8), (16, 7))
    )

    note = """
    The flashinfer calls the built-in DFlash drafter makes per layer on its
    TRTLLM backend for an unwindowed non-causal layer, with the same
    arguments (dflash_trtllm_folded_block_attention): append_paged_kv_cache
    (HND), then per chunk of folded_decode_chunk_size block rows
    trtllm_batch_decode_with_kv_cache with the chunk's rows folded into the
    query heads (one query per request, window_left -1, bmm1_scale D^-0.5,
    shared paged KV index, no sinks, PDL off). The cells' blocks fit one
    chunk. Page size 64, head dim 64. seq_lens already counts the block
    (context + T).

    The block's K / V stay in the cache after the call (the worker
    overwrites them with the accepted rows' K / V on the next step). The
    workspace and counters must be allocated before CUDA-graph capture at the
    largest batch the graph replays; the only allocations here are each
    chunk's folded q and output. No metadata is consumed.
    The reference accumulates in fp32; the kernel rounds P to bf16.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: torch.Tensor,
        block_tables: torch.Tensor,
        append_batch_indices: torch.Tensor,
        append_positions: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        workspace: torch.Tensor,
        counters: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        ops = trtllm_gen_ops()
        ops.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=append_batch_indices,
            positions=append_positions,
            paged_kv_cache=cache,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout="HND",
        )
        batch = block_tables.shape[0]
        num_heads, head_dim = q.shape[1], q.shape[2]
        num_kv_heads = cache.shape[2]
        block = q.shape[0] // batch
        group = num_heads // num_kv_heads
        chunk = folded_decode_chunk_size(block, num_heads, num_kv_heads)
        grouped_q = q.reshape(batch, block, num_kv_heads, group, head_dim)
        grouped_out = out.view(batch, block, num_kv_heads, group, head_dim)
        for start in range(0, block, chunk):
            end = min(start + chunk, block)
            rows = end - start
            # Request b's rows [start, end) of KV head h are query heads [h * rows * group, (h + 1) * rows * group).
            q_folded = (
                grouped_q[:, start:end]
                .permute(0, 2, 1, 3, 4)
                .reshape(batch, num_kv_heads * rows * group, head_dim)
                .contiguous()
            )
            out_folded = q_folded.new_empty(q_folded.shape, dtype=out.dtype)
            ops.batch_decode_with_kv_cache(
                query=q_folded,
                kv_cache=(cache[:, 0], cache[:, 1]),
                workspace_buffer=workspace,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_kv_len,
                bmm1_scale=head_dim**-0.5,
                bmm2_scale=1.0,
                window_left=-1,
                out=out_folded,
                sinks=None,
                enable_pdl=False,
                kv_layout="HND",
                backend="trtllm-gen",
                q_len_per_req=1,
                max_q_len=None,
                cum_seq_lens_q=None,
                kv_cache_sf=None,
                uses_shared_paged_kv_idx=True,
                bmm1_scale_log2=None,
                multi_ctas_kv_counter_buffer=counters,
            )
            grouped_out[:, start:end].copy_(
                out_folded.view(batch, num_kv_heads, rows, group, head_dim).permute(0, 2, 1, 3, 4)
            )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: torch.Tensor,
        block_tables: torch.Tensor,
        append_batch_indices: torch.Tensor,
        append_positions: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        workspace: torch.Tensor,
        counters: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Plain torch; ignores ``kv_last_page_len``, ``max_kv_len``, ``workspace`` and ``counters``."""
        page = cache.shape[3]
        num_heads, head_dim = q.shape[1], q.shape[2]
        num_kv = k.shape[1]
        for i in range(k.shape[0]):
            b = int(append_batch_indices[i])
            pos = int(append_positions[i])
            pg = int(kv_indices[int(kv_indptr[b]) + pos // page])
            cache[pg, 0, :, pos % page] = k[i]
            cache[pg, 1, :, pos % page] = v[i]
        rows = q.shape[0] // block_tables.shape[0]
        # torch's default fp32 matmul is TF32 here (~1e-3), which would make the reference the inaccurate side.
        with true_fp32_matmul():
            for b in range(block_tables.shape[0]):
                n = int(seq_lens[b])
                pages = block_tables[b, : (n + page - 1) // page].long()
                keys = (
                    cache[pages, 0].permute(1, 0, 2, 3).reshape(num_kv, -1, head_dim)[:, :n].float()
                )
                vals = (
                    cache[pages, 1].permute(1, 0, 2, 3).reshape(num_kv, -1, head_dim)[:, :n].float()
                )
                qb = q[b * rows : (b + 1) * rows].float().transpose(0, 1)  # [H, T, D]
                group = num_heads // num_kv
                keys = keys.repeat_interleave(group, 0)
                vals = vals.repeat_interleave(group, 0)
                p = torch.softmax(qb @ keys.transpose(1, 2) * head_dim**-0.5, -1)
                out[b * rows : (b + 1) * rows] = (
                    (p @ vals)
                    .transpose(0, 1)
                    .reshape(out[b * rows : (b + 1) * rows].shape)
                    .to(out.dtype)
                )

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: torch.Tensor,
        block_tables: torch.Tensor,
        append_batch_indices: torch.Tensor,
        append_positions: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        workspace: torch.Tensor,
        counters: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        assert q.dtype == k.dtype == v.dtype == cache.dtype == out.dtype == torch.bfloat16, (
            "bf16 only"
        )
        assert k.is_contiguous() and v.is_contiguous(), "the appended K / V must be contiguous"
        assert cache.dim() == 5 and cache.shape[1] == 2 and cache.shape[2] == k.shape[1], (
            f"the cache must be HND pages [P, 2, KV, page, D]; got {tuple(cache.shape)}"
        )
        assert q.shape[1] % k.shape[1] == 0, "query heads must be a multiple of KV heads"
        assert q.shape[0] % block_tables.shape[0] == 0, (
            "every request must have the same block rows"
        )
        assert out.is_contiguous(), "out is written through a folded view"
        assert q.shape[1] // k.shape[1] <= MAX_HEADS_RATIO_GENERATION, (
            f"at most {MAX_HEADS_RATIO_GENERATION} query heads per KV head"
        )


k3_dspark_block_attn = K3DSparkBlockAttn()
