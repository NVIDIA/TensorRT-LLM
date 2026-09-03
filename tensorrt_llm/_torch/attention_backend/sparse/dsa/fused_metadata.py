# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DSA decode-metadata Triton kernel.

TRT-LLM builds the DSA sparse-attention decode metadata as a chain of ~30 eager
pointwise/scan launches per step inside
``DSAtrtllmAttentionMetadata.on_update_kv_lens`` (cumsum / arange / searchsorted
for ``req_idx_per_token``; ``seq_starts`` / ``token_offsets`` / ``global_positions``;
``_compute_slot_mappings`` floor-div / remainder / clamp / block-offset gather for
``slot_mapping_fp8`` and ``slot_mapping_scale``; two int64 generation cumsums for
``gen_kv_indptr`` / ``gen_cached_token_indptr``).  This module collapses that
whole block into a **single Triton launch** to cut the per-step host/launch
overhead of the decode metadata preparation.

The kernel is env-gated (``TRTLLM_FUSED_DSA_METADATA=1``) and produces integer
metadata that must be bit-identical to the eager chain; the differential unit
test verifies that contract.  It only handles the pure-decode / generation step
(``num_contexts == 0``); the caller falls back to the eager chain otherwise.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_seqs", "num_tokens"])
def _fused_dsa_decode_metadata_kernel(
    seq_lens_ptr,  # int32 [num_seqs]
    kv_lens_ptr,  # int32 [num_seqs]
    block_offsets_ptr,  # int32 [num_seqs, max_blocks]
    req_idx_per_token_ptr,  # int32 [num_tokens] (out)
    slot_fp8_ptr,  # int64 [num_tokens] (out)
    slot_scale_ptr,  # int64 [num_tokens] (out)
    gen_kv_indptr_ptr,  # int64 [num_seqs + 1] (out)
    gen_cached_indptr_ptr,  # int64 [num_seqs + 1] (out)
    block_offsets_stride_0,
    block_offsets_stride_1: tl.constexpr,
    num_seqs,
    num_tokens,
    max_blocks: tl.constexpr,
    tokens_per_block: tl.constexpr,
    data_bytes_per_token: tl.constexpr,
    scale_size: tl.constexpr,
    block_stride: tl.constexpr,
    scale_base_offset: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)

    # ----- header program: the two int64 generation cumsums ------------------
    if pid == num_seqs:
        offs_b = tl.arange(0, BLOCK_S)
        mask_b = offs_b < num_seqs
        seq = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        kv = tl.load(kv_lens_ptr + offs_b, mask=mask_b, other=0).to(tl.int64)
        cached = kv - seq
        cu_kv = tl.cumsum(kv, 0)
        cu_cached = tl.cumsum(cached, 0)
        tl.store(gen_kv_indptr_ptr, tl.full((), 0, tl.int64))
        tl.store(gen_cached_indptr_ptr, tl.full((), 0, tl.int64))
        tl.store(gen_kv_indptr_ptr + 1 + offs_b, cu_kv, mask=mask_b)
        tl.store(gen_cached_indptr_ptr + 1 + offs_b, cu_cached, mask=mask_b)
        return

    # ----- per-request program: req_idx + slot mappings ----------------------
    r = pid
    # flat_start = sum(seq_lens[0:r]) computed in registers (num_seqs tiny)
    offs_s = tl.arange(0, BLOCK_S)
    seq_all = tl.load(seq_lens_ptr + offs_s, mask=offs_s < num_seqs, other=0)
    flat_start = tl.sum(tl.where(offs_s < r, seq_all, 0)).to(tl.int64)

    seq_r = tl.load(seq_lens_ptr + r).to(tl.int64)
    kv_r = tl.load(kv_lens_ptr + r).to(tl.int64)
    start_pos = kv_r - seq_r

    offs_j = tl.arange(0, BLOCK_T).to(tl.int64)
    g = flat_start + offs_j  # global (flattened) token idx
    # Bound every store by the host-known num_tokens in addition to the
    # per-request length. g is derived from a device-side prefix sum of
    # seq_lens, so a stale/corrupt seq_lens_cuda must never let a write escape
    # the [0, num_tokens) live extent -- the eager chain writes host-sliced
    # buffers and is bounded by construction, and this restores that guarantee.
    mask_j = (offs_j < seq_r) & (g < num_tokens)

    # req_idx_per_token[g] = r
    tl.store(req_idx_per_token_ptr + g, tl.full((BLOCK_T,), r, tl.int32), mask=mask_j)

    gpos = start_pos + offs_j  # global KV position
    blk = gpos // tokens_per_block
    pos = gpos % tokens_per_block
    # Match torch's floor-division / non-negative remainder for gpos < 0: Triton
    # lowers signed //,% to truncate-toward-zero with a sign-following
    # remainder, so normalize pos into [0, tokens_per_block). The lower clamp
    # below already erases the corresponding quotient difference. gpos < 0 only
    # arises from stale token-to-seq mappings (the eager _compute_slot_mappings
    # cuda path defends against the same case).
    pos = tl.where(pos < 0, pos + tokens_per_block, pos)
    # clamp to prevent OOB from stale token-to-seq mappings under CUDA graph
    # replay with MTP + DSA (matches _compute_slot_mappings cuda path).
    blk = tl.minimum(tl.maximum(blk, 0), max_blocks - 1)
    bid = tl.load(
        block_offsets_ptr + r * block_offsets_stride_0 + blk * block_offsets_stride_1,
        mask=mask_j,
        other=0,
    ).to(tl.int64)

    fp8 = bid * block_stride + pos * data_bytes_per_token
    scale = bid * block_stride + scale_base_offset + pos * scale_size
    tl.store(slot_fp8_ptr + g, fp8, mask=mask_j)
    tl.store(slot_scale_ptr + g, scale, mask=mask_j)


def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,  # int32 [num_seqs]
    kv_lens: torch.Tensor,  # int32 [num_seqs]
    block_offsets: torch.Tensor,  # int32 [num_seqs, max_blocks]
    req_idx_per_token: torch.Tensor,  # int32 [>= num_tokens] (out, sliced view)
    slot_mapping_fp8: torch.Tensor,  # int64 [>= num_tokens] (out, sliced view)
    slot_mapping_scale: torch.Tensor,  # int64 [>= num_tokens] (out, sliced view)
    gen_kv_indptr: torch.Tensor,  # int64 [>= num_seqs + 1] (out, sliced view)
    gen_cached_token_indptr: torch.Tensor,  # int64 [>= num_seqs + 1] (out, sliced)
    num_tokens: int,
    max_query_len: int,
    tokens_per_block: int,
    index_head_dim: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> None:
    """Fill the DSA decode metadata (req_idx + slot mappings + gen indptrs) in one
    Triton launch.  Bit-identical to the eager chain in on_update_kv_lens.

    ``max_query_len`` is the per-request upper bound on query tokens at decode
    (== next_n == 1 + max_draft_tokens). It sizes the per-request lane count
    (BLOCK_T); it must be >= every ``seq_lens[r]``.

    Views must already be sliced to their live extents:
    ``req_idx_per_token[:num_tokens]``, ``slot_mapping_*[:num_tokens]``,
    ``gen_*_indptr[:num_seqs + 1]``.
    """
    num_seqs = seq_lens.shape[0]
    assert num_seqs > 0 and num_tokens > 0
    assert block_offsets.shape[0] == num_seqs
    # Coarse aggregate sanity check. It does NOT prove per-request coverage
    # (each seq_len <= max_query_len): e.g. seq_lens=[10, 1, 1, 1], next_n=8
    # passes yet request 0 needs 10 lanes. The pure-decode contract guarantees
    # every gen request carries exactly next_n query tokens (so sum == num_tokens
    # and each seq_len == next_n <= max_query_len); the per-store g < num_tokens
    # mask in the kernel keeps any invariant violation in-bounds rather than
    # corrupting neighboring buffers.
    assert num_tokens <= num_seqs * max_query_len, (
        f"fused DSA metadata: num_tokens={num_tokens} exceeds "
        f"num_seqs={num_seqs} * max_query_len={max_query_len}"
    )

    if data_bytes_per_token is None:
        data_bytes_per_token = index_head_dim
    scale_size = index_head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token
    max_blocks = block_offsets.shape[1]

    # Per-request lane count: bound by the per-request query length (next_n),
    # NOT num_tokens. Sizing by num_tokens would make each of the num_seqs
    # programs iterate ~num_tokens lanes (O(num_seqs^2) work) and blow up the
    # constexpr BLOCK_T at high concurrency (e.g. bs=256 -> 2048: register
    # spill + a distinct compiled signature per graph bucket). next_n is a
    # batch-independent host constant (typically 8), keeping work O(num_tokens).
    block_t = triton.next_power_of_2(max_query_len)
    block_s = triton.next_power_of_2(num_seqs)
    grid = (num_seqs + 1,)

    _fused_dsa_decode_metadata_kernel[grid](
        seq_lens,
        kv_lens,
        block_offsets,
        req_idx_per_token,
        slot_mapping_fp8,
        slot_mapping_scale,
        gen_kv_indptr,
        gen_cached_token_indptr,
        block_offsets.stride(0),
        block_offsets.stride(1),
        num_seqs,
        num_tokens,
        max_blocks,
        tokens_per_block,
        data_bytes_per_token,
        scale_size,
        block_stride,
        scale_base_offset,
        BLOCK_S=block_s,
        BLOCK_T=block_t,
    )
