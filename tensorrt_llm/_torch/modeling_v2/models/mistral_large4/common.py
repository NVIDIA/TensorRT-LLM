# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constants and step/RoPE/MoE helpers for Mistral Large 4 targets."""

from __future__ import annotations

import itertools
import math
import os
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_workspace_bytes,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

# The latent pool layout this family addresses (`MistralLarge4CacheManager.latent_layout`).
_LATENT_LAYOUT = "fp8_ds_mla"
# The query heads FlashMLA's sm100 sparse kernels run; this rank's heads go first.
_FLASH_MLA_HEADS = 64
# Generation rows' sparse MLA. False: cute_dsl_ds_mla_sparse_decode at this rank's heads.
_GEN_ATTN_FLASH_MLA = False
# The pre-routed MoE call: the runner's routing method is inert once ids and weights are
# given; 1 (renormalize) is the value its cells drive.
_MOE_ROUTING_METHOD = 1
# The routed experts' SwiGLU, the runner's `act_type` 0.
_MOE_ACT_SWIGLU = 0
# NVFP4: one e4m3 scale per 16 activations, laid out row-major (linear).
_NVFP4_SF_VEC = 16
_NVFP4_SF_LINEAR = False
# The context path's crossover and held context workspace ceiling (`_ctx_workspace_rows`).
_CTX_WORKSPACE_ROWS = 1 << 19
# The most context rows one sparse-decode call takes on that path.
_CTX_DECODE_ROWS = 8192
# The decode indexer's top-k: True runs cute_dsl_radix_topk_decode over [4096, 65536] cols.
_TOPK_DECODE_CUTE_DSL = True
_TOPK_DECODE_COLUMNS = (4096, 65536)
# Optional unfused attention glue path for A/B verification; off by default.
_UNFUSED_GLUE = os.environ.get("TRTLLM_MODELING_V2_UNFUSED_GLUE", "0") == "1"


def _prefill_tiles(
    token_start: int, token_end: int, n_k: int, budget: int
) -> list[tuple[int, int]]:
    """The query-row tiles `[lo, hi)` of one indexer prefill chunk over `n_k` keys under a
    logits `budget`: `floor(budget / n_k)` rows each (at least one), the last one short."""
    tile = max(1, budget // n_k)
    return [(lo, min(lo + tile, token_end)) for lo in range(token_start, token_end, tile)]


def _prefill_logits_budget(max_num_tokens: int, max_seq_len: int, ceiling: int = 1 << 28) -> int:
    """The most fp32 logits one indexer prefill tile holds, from the engine's max_num_tokens and
    max_seq_len within `ceiling` and within max_seq_len / 64 query rows at max_seq_len keys."""
    reachable = get_indexer_mqa_logits_workspace_bytes(max_num_tokens, max_seq_len) // 4
    return min(reachable, ceiling, max_seq_len * (max_seq_len >> 6))


def _most_ctx_kv(md) -> int:
    """The most context KV (cached and new rows over a step's context requests) one step can
    attend, from the attention metadata alone."""
    features = md.runtime_features
    if not (features.chunked_prefill or features.cache_reuse):
        return md.max_num_tokens
    return min(md.max_num_requests, md.max_num_tokens) * md.max_seq_len


def _ctx_workspace_rows(md) -> int:
    """The held context workspace's rows: up to `_CTX_WORKSPACE_ROWS`."""
    return min(2 * _most_ctx_kv(md), _CTX_WORKSPACE_ROWS)


def _step_common(
    md: TrtllmAttentionMetadata,
    indexer_layers: list[int],
    *,
    include_all_rank_num_tokens: bool = False,
) -> dict:
    """The batch state both targets read: KV lengths and pool addressing for the MLA ops, the
    indexer's key pools and slot mappings, the top-k conversion tables and the decode scorer's
    schedule."""
    mgr = md.kv_cache_manager
    latent_pool = mgr.get_latent_pool()
    state = dict(
        num_seqs=md.num_seqs,
        num_tokens=md.num_tokens,
        kv_lens_cuda=md.kv_lens_cuda_runtime,
        kv_lens_host=md.kv_lens_runtime,
        prompt_lens_host=md.prompt_lens_cpu_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        tokens_per_block=md.tokens_per_block,
        max_seq_len=md.max_seq_len,
        latent_pool=latent_pool,
        latent_kv=latent_pool.view(-1, latent_pool.shape[2], 1, latent_pool.shape[3]),
        ix_pools={i: mgr.get_indexer_k_cache_buffers(i) for i in indexer_layers},
        block_table=md.block_table,
        req_idx_per_token=md.req_idx_per_token,
        slot_mapping_fp8=md.slot_mapping_fp8,
        slot_mapping_scale=md.slot_mapping_scale,
        skip_gen=bool(md.skip_indexer_for_gen_reqs),
        topk_indices_buffer=md.topk_indices_buffer if md.enable_indexer_skip else None,
        indexer_k_cache_block_offsets=md.indexer_k_cache_block_offsets,
        scheduler_metadata_buffer=md.scheduler_metadata_buffer,
        radix_aux_indices=md.radix_aux_indices,
        radix_aux_logits=md.radix_aux_logits,
    )
    if include_all_rank_num_tokens:
        state["all_rank_num_tokens"] = [int(n) for n in md.all_rank_num_tokens]
    return state


def _padded_rows(core, step: dict, rows: int) -> int:
    """The row count every rank pads its block to before attention-DP collectives."""
    counts = step["all_rank_num_tokens"]
    assert len(counts) == len(core.dp_group) and counts[core.dp_rank] == rows, (
        f"rank {core.dp_rank} holds {rows} rows; all_rank_num_tokens is {counts}"
    )
    return max(counts)


def _padding_rows(counts: list[int], padded: int, device) -> torch.Tensor | None:
    """Which of the gathered rows are padding: bool `[len(counts) * padded]` on `device`, or None."""
    if all(n == padded for n in counts):
        return None
    own = torch.tensor(counts, dtype=torch.int64, device=device)
    return (torch.arange(padded, device=device)[None, :] >= own[:, None]).reshape(-1)


def _ctx_kv_offsets(md: TrtllmAttentionMetadata, num_contexts: int) -> dict:
    """The context rope/append op's per-request offsets."""
    if md.enable_context_mla_with_cached_kv:
        return dict(
            cu_ctx_cached=md.ctx_cached_token_indptr[: num_contexts + 1],
            cu_ctx_kv=md.ctx_kv_indptr[: num_contexts + 1],
            max_ctx_new=int(md.max_ctx_seq_len),
            ctx_cached_tokens=int(md.num_ctx_cached_tokens),
        )
    cu = md.mla_prepare_ctx_cu_seqlens()
    return dict(
        cu_ctx_cached=torch.zeros(num_contexts + 1, dtype=torch.int64, device=cu.device),
        cu_ctx_kv=cu.to(torch.int64),
        max_ctx_new=int(md.seq_lens[:num_contexts].max()),
        ctx_cached_tokens=0,
    )


def _ctx_host_offsets(md: TrtllmAttentionMetadata, num_contexts: int) -> dict:
    """The host prefix sums of the new and total (cached + new) lengths over context requests."""
    if md.enable_context_mla_with_cached_kv:
        return dict(
            host_cu_ctx_new=md.host_ctx_uncached_token_indptr[: num_contexts + 1].tolist(),
            host_cu_ctx_kv=md.host_ctx_kv_indptr[: num_contexts + 1].tolist(),
        )
    host_cu = [0, *itertools.accumulate(md.seq_lens[:num_contexts].tolist())]
    return dict(host_cu_ctx_new=host_cu, host_cu_ctx_kv=host_cu)


def _indexer_blocks(chunks, step: dict) -> list:
    """The context indexer's scoring blocks for one step, cut at request boundaries past
    `max_seq_len` keys."""
    tc = step["num_ctx_tokens"]
    if chunks is None:
        chunks = [
            SimpleNamespace(
                token_start=0,
                token_end=tc,
                k_token_start=0,
                k_token_end=step["num_ctx_kv_tokens"],
                cu_seqlen_ks=step["cu_seqlen_ks"][:tc],
                cu_seqlen_ke=step["cu_seqlen_ke"][:tc],
            )
        ]
    max_keys = step["max_seq_len"]
    cu_new, cu_kv = step["host_cu_ctx_new"], step["host_cu_ctx_kv"]
    blocks = []
    for chunk in chunks:
        if chunk.k_token_end - chunk.k_token_start <= max_keys:
            blocks.append(chunk)
            continue
        for r in range(len(cu_new) - 1):
            lo, hi = max(chunk.token_start, cu_new[r]), min(chunk.token_end, cu_new[r + 1])
            if lo >= hi:
                continue
            k_lo, k_hi = max(chunk.k_token_start, cu_kv[r]), min(chunk.k_token_end, cu_kv[r + 1])
            assert k_hi - k_lo <= max_keys, (
                f"context request {r} holds keys [{k_lo}, {k_hi}) of an indexer chunk, past "
                f"max_seq_len {max_keys}"
            )
            shift = k_lo - chunk.k_token_start
            rows = slice(lo - chunk.token_start, hi - chunk.token_start)
            blocks.append(
                SimpleNamespace(
                    token_start=lo,
                    token_end=hi,
                    k_token_start=k_lo,
                    k_token_end=k_hi,
                    cu_seqlen_ks=torch.sub(chunk.cu_seqlen_ks[rows], shift),
                    cu_seqlen_ke=torch.sub(chunk.cu_seqlen_ke[rows], shift),
                )
            )
    return blocks


def _prefill_blocks(
    chunk, cu_new: list[int], cu_kv: list[int], max_keys: int
) -> list[tuple[int, int, int, int]]:
    """The indexer's scoring blocks over one engine prefill chunk for attention-DP targets."""
    n_k = chunk.k_token_end - chunk.k_token_start
    if n_k <= max_keys:
        return [(chunk.token_start, chunk.token_end, chunk.k_token_start, n_k)]
    blocks = []
    for r in range(len(cu_new) - 1):
        lo, hi = max(chunk.token_start, cu_new[r]), min(chunk.token_end, cu_new[r + 1])
        if lo >= hi:
            continue
        k_lo, k_hi = max(chunk.k_token_start, cu_kv[r]), min(chunk.k_token_end, cu_kv[r + 1])
        assert 0 < k_hi - k_lo <= max_keys, (
            f"context request {r} of an indexer chunk holds keys [{k_lo}, {k_hi}), past "
            f"max_seq_len {max_keys}"
        )
        blocks.append((lo, hi, k_lo, k_hi - k_lo))
    assert blocks and blocks[0][0] == chunk.token_start and blocks[-1][1] == chunk.token_end, (
        f"the requests {cu_new} do not tile the chunk's rows "
        f"[{chunk.token_start}, {chunk.token_end})"
    )
    return blocks


def _yarn_lerope_inv_freq(cfg, alpha: torch.Tensor) -> torch.Tensor:
    """The rotary inverse frequencies, fp32: YaRN blend scaled by LeRoPE `exp(alpha)`."""
    rope = cfg.rope_parameters
    dim = cfg.qk_rope_head_dim
    theta = float(rope["rope_theta"])
    factor = float(rope["factor"])
    original = float(rope["original_max_position_embeddings"])
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    lerope = torch.exp(alpha.detach().to(device="cpu", dtype=torch.float32))
    extrapolated = 1.0 / pos_freqs * lerope
    interpolated = 1.0 / (factor * pos_freqs) * lerope

    def correction_dim(rotations: float) -> float:
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(correction_dim(float(rope.get("beta_fast", 32)))), 0)
    high = min(math.ceil(correction_dim(float(rope.get("beta_slow", 1)))), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation = 1 - ramp
    return interpolated * (1 - extrapolation) + extrapolated * extrapolation


def _rope_tables(cfg, alpha: torch.Tensor, positions: int, dtype, device) -> SimpleNamespace:
    """The rotary cos/sin tables built from one fp32 angle grid and rounded to `dtype`."""
    inv = _yarn_lerope_inv_freq(cfg, alpha)
    t = torch.arange(positions, dtype=torch.float32)
    angle = torch.einsum("i,j -> ij", t, inv)
    cos = angle.cos().to(dtype).float()
    sin = angle.sin().to(dtype).float()
    identity = torch.cat([torch.ones_like(cos), torch.zeros_like(sin)], dim=1)
    return SimpleNamespace(
        rows=positions,
        indexer=torch.cat([cos, sin], dim=1).to(device).contiguous(),
        identity=identity.to(device).contiguous(),
    )


def _moe_scalars(isc1, isc2, ws2_13, ws2_2) -> SimpleNamespace:
    """The NVFP4 MoE call scalars derived from the checkpoint's reciprocal per-tensor scales."""
    gate1 = (isc1 * ws2_13).contiguous()
    return SimpleNamespace(
        gscale=(1.0 / isc1).contiguous(),
        out1=(gate1 / isc2).contiguous(),
        out1_gate=gate1,
        out2=(isc2 * ws2_2).contiguous(),
    )
