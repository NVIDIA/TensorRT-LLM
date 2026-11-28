# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side planning of the Kimi K3.1 DSA indexer: its per-step parameters, the slot mappings of the index-K
cache, the prefill chunking and query split, and the CuTe DSL paged-logits work split.

These are copies of the engine DSA indexer's planning helpers (``attention/backends/sparse/dsa/indexer.py``)
for K3.1's index keys: uncompressed (one key per token) and FP8 (one byte per element); the family indexer
and DSA metadata use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_workspace_bytes,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

if TYPE_CHECKING:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
        K31DSAtrtllmAttentionMetadata,
    )
    from tensorrt_llm.mapping import Mapping

# Cap the per-call indexer MQA-logits transient (in elements). fp8_mqa_logits
# allocates its [q x kv] logits output via torch.empty; the KV dimension is the
# full context and is unbounded, so for a large query chunk on a
# long-context prefill this single allocation can reach tens of GB. Tiling the
# query dimension caps the transient to q_tile x kv with identical results
# (each query row's logits/top-k are independent). The KV cache budget reserves
# the same cap (the runtime's indexer logits workspace, FP32 elements).
_INDEXER_MQA_LOGITS_ELEM_BUDGET = get_indexer_mqa_logits_workspace_bytes() // torch.float32.itemsize

#: Columns of the split-work workspace the CUDA radix decode Top-K takes for FP32 scores.
_MAX_RADIX_BLOCKS_PER_ROW = 10

# Cap on the [sum(q) x sum(kv)] logits of context requests packed into one
# indexer prefill chunk (in elements; 1 << 28 is 1 GiB of fp32). A packed
# chunk gathers the KV of every request in it and scores every query against
# all of it, so its logits grow with the product of the packed totals while
# each query row only needs its own request's window. Prefix-reuse hits make
# this explode: each brings few new tokens but its whole cached context. The
# cap only decides which requests share a chunk; _packed_logits_elem_budget
# also clamps it to the largest single-request logits tile the configuration
# can reach. The memory of each logits call is bounded separately by the query
# tiling under _INDEXER_MQA_LOGITS_ELEM_BUDGET, which also tiles a request whose
# own logits exceed this cap in a chunk of its own.
_INDEXER_PACKED_LOGITS_ELEM_BUDGET = 1 << 28


_DG_SCHEDULE_BLOCK_KV = 64


def use_cute_dsl_paged_mqa_logits(requested: bool) -> bool:
    """Whether the CuTe DSL paged MQA logits kernel runs instead of DeepGEMM.

    The indexer and the metadata must both use this: the metadata builds
    DSL-only buffers (atom-split expansion, padded logits row stride) and
    builds its next_n schedules off the same decision. is_sm_100f() is the
    custom op's own precondition.
    """
    return requested and IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()


def _pick_dsl_expand(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3, 4),
) -> Tuple[int, int]:
    """Pick (expand_factor, effective_next_n) for the DSL paged kernel
    using a wave-aware strategy.

    The DSL kernel natively supports ``effective_next_n ∈ kernel_atoms``
    (FP8 index keys: ``(1, 2, 3, 4)``; the SM 10.3 perf targets' native tiles
    ``(1, ..., 8)``). For ``next_n`` not natively supported or when SM
    utilization can be improved, reshape
    ``[B, next_n, ...]`` -> ``[B * expand_factor, effective_next_n, ...]``
    caller-side.

    Strategy: enumerate ``(expand_factor, effective_next_n)`` pairs with
    ``expand_factor * effective_next_n == next_n`` and ``effective_next_n
    in kernel_atoms``. Score each by ``(waves, -expand_factor)`` where
    ``waves = ceil(B * expand_factor * ceil(max_ctx/256) / num_sms)``.
    Pick min waves; on tie, prefer LARGER expand_factor (more SMs busy per
    wave; pays HBM cost of expand_factor x KV re-reads).

    When ``batch_size == 0`` or ``max_ctx == 0`` (workload unknown), fall
    back to the HBM-minimizing pick: largest effective_next_n that divides
    next_n cleanly (still constrained to ``kernel_atoms``).

    Examples (``kernel_atoms=(1, 2, 3, 4)``, num_sms=148, SPLIT_KV=256 tokens):
        next_n=4, B=1,  ctx=4096   -> (4, 1): ntask=64<148, 1 wave, max factor
        next_n=4, B=64, ctx=4096   -> (1, 4): ntask=1024>148 at factor 1, fewest waves
        next_n=2, B=1,  ctx=4096   -> (2, 1): wave-tie, larger factor
    """
    # Workload unknown: the fewest KV re-reads.
    if batch_size <= 0 or max_ctx <= 0:
        for eff in sorted(kernel_atoms, reverse=True):
            if next_n % eff == 0:
                return next_n // eff, eff
        return next_n, 1

    SPLIT_KV_TOKENS = 256
    cands = []
    for eff in kernel_atoms:
        if next_n % eff == 0:
            factor = next_n // eff
            ntask = batch_size * factor * ((max_ctx + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, factor, eff))
    if not cands:
        return next_n, 1
    cands.sort(key=lambda x: (x[0], -x[1]))  # min waves, max factor
    _, factor, eff = cands[0]
    return factor, eff


def _atom_ctx_offsets(next_n: int, num_atoms: int) -> List[int]:
    """Per-atom causal stagger of an in-kernel next_n split: atom i (0 = oldest)
    sees the context shortened by the tokens of the atoms after it. Same split
    as the kernel (first ``next_n % num_atoms`` atoms get one extra token)."""
    base, rem = divmod(next_n, num_atoms)
    return [
        next_n - (i * base + min(i, rem) + base + (1 if i < rem else 0)) for i in range(num_atoms)
    ]


def _pick_dsl_split(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3, 4),
) -> Tuple[int, int, int]:
    """``_pick_dsl_expand`` plus the kernel's in-kernel uneven split.

    Returns ``(expand_factor, atom, num_kernel_atoms)``. When no atom > 1 in
    ``kernel_atoms`` divides ``next_n`` (e.g. 5 or 7), the caller-side split can
    only run atom 1 and reads the KV ``next_n`` times; the in-kernel split runs
    ``ceil(next_n / max(kernel_atoms))`` atoms instead. It is taken when atom 1
    would need many waves and the split needs fewer, or when the workload is unknown.
    """
    factor, atom = _pick_dsl_expand(next_n, num_sms, batch_size, max_ctx, kernel_atoms)
    if atom != 1 or next_n <= 1 or any(next_n % a == 0 for a in kernel_atoms if a > 1):
        return factor, atom, 1
    num_atoms = -(-next_n // max(kernel_atoms))
    if batch_size > 0 and max_ctx > 0:
        tiles = -(-max_ctx // 256)
        waves_split = -(-batch_size * num_atoms * tiles // num_sms)
        waves_atom1 = -(-batch_size * next_n * tiles // num_sms)
        # Below ~16 atom-1 waves the split's wider tiles cost more than the KV
        # re-reads they save (GB300, next_n=7: B=1 0.87-1.00x; >= 24 waves 1.04-2.6x).
        if waves_split >= waves_atom1 or waves_atom1 < 16:
            return factor, atom, 1
    return 1, -(-next_n // num_atoms), num_atoms


def _compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute flat byte indices for FP8 data and scales from global token positions.

    Shared by K31Indexer.prepare() (CPU) and on_update_kv_lens() (GPU) to avoid
    duplicating the slot mapping arithmetic.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (logical element count).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.

    Returns:
        (fp8_indices, scale_indices): Flat byte offsets into the cache pool.
    """
    data_bytes_per_token = head_dim  # one FP8 byte per element
    scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token

    block_indices_in_seq = global_positions // tokens_per_block
    pos_in_blocks = global_positions % tokens_per_block

    max_blocks = block_offsets.shape[1]
    if block_indices_in_seq.is_cuda:
        # Clamp to prevent OOB from stale token-to-seq mappings in the padded
        # rows of a CUDA graph capture / replay.
        block_indices_in_seq = block_indices_in_seq.clamp(0, max_blocks - 1)
    else:
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = block_ids * block_stride + scale_base_offset + pos_in_blocks * scale_size
    return fp8_indices, scale_indices


def split_prefill_chunks(
    seq_lens: torch.Tensor,
    max_chunk_size: int,
    start_idx: int = 0,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Split prefill requests into chunks based on max_chunk_size.
    Supports two-level chunking:
    1. Request-boundary chunking: group multiple small requests into one chunk
    2. Intra-request chunking: split large requests into multiple Q-block chunks

    Args:
        seq_lens: Sequence lengths for all requests
        max_chunk_size: Maximum number of tokens per chunk
        start_idx: Starting index for prefill requests

    Returns:
        List of chunk groups, where each group is a list of chunk specs.
        Each chunk spec is (req_idx, token_start_in_req, token_end_in_req, req_cum_start)

        - For multi-request chunks: group contains multiple specs (one per request)
        - For intra-request chunks: each Q-block is a separate group with single spec
    """
    chunk_groups = []
    num_reqs = len(seq_lens)

    current_req = start_idx
    # Compute cumulative token positions
    query_start_loc_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device="cpu"), seq_lens.cumsum(dim=0).to(torch.int32)]
    )

    while current_req < num_reqs:
        seq_len = seq_lens[current_req].item()
        req_cum_start = query_start_loc_cpu[current_req].item()

        if seq_len <= max_chunk_size:
            # This request fits in one chunk - try to pack with others
            current_size = seq_len
            chunk_specs = [(current_req, 0, seq_len, req_cum_start)]
            next_req = current_req + 1

            # Try to add more requests to this chunk
            while next_req < num_reqs:
                next_seq_len = seq_lens[next_req].item()
                if next_seq_len > max_chunk_size:
                    # Next request is large, stop packing
                    break
                if current_size + next_seq_len <= max_chunk_size:
                    next_cum_start = query_start_loc_cpu[next_req].item()
                    chunk_specs.append((next_req, 0, next_seq_len, next_cum_start))
                    current_size += next_seq_len
                    next_req += 1
                else:
                    break

            # Add as one multi-request chunk group
            chunk_groups.append(chunk_specs)
            current_req = next_req
        else:
            # Large request - split into Q-blocks
            # Each Q-block is a separate chunk group (processed in separate iteration)
            num_q_blocks = (seq_len + max_chunk_size - 1) // max_chunk_size
            for q_block_idx in range(num_q_blocks):
                token_start = q_block_idx * max_chunk_size
                token_end = min(token_start + max_chunk_size, seq_len)
                q_block_spec = [(current_req, token_start, token_end, req_cum_start)]
                chunk_groups.append(q_block_spec)

            current_req += 1

    return chunk_groups


def group_prefill_requests_by_logits_budget(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    elem_budget: int,
) -> List[List[int]]:
    """Group consecutive context requests into indexer prefill chunks.

    A request joins the current group only while the group's packed logits,
    ``sum(q_lens) * sum(kv_lens)``, stay within ``elem_budget``. A request that
    exceeds the budget on its own forms a single-request group.

    Args:
        q_lens: New (query) tokens per request.
        kv_lens: Compressed KV tokens per request, cached plus new.
        elem_budget: Maximum packed logits elements for a multi-request group.

    Returns:
        Groups of request indices, in order, covering every request once.
    """
    assert len(q_lens) == len(kv_lens)
    groups: List[List[int]] = []
    group_q = group_kv = 0
    for req_idx, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        if groups and (group_q + q_len) * (group_kv + kv_len) <= elem_budget:
            groups[-1].append(req_idx)
            group_q += q_len
            group_kv += kv_len
        else:
            groups.append([req_idx])
            group_q, group_kv = q_len, kv_len
    return groups


def plan_mla_chunked_prefill_chunks(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    elem_budget: int,
    q_begin: int = 0,
    q_end: Optional[int] = None,
) -> List[List[Tuple[int, int, int, int]]]:
    """Plan the indexer prefill chunks covering context query tokens ``[q_begin, q_end)``.

    Requests that lie fully inside the range are grouped with
    ``group_prefill_requests_by_logits_budget``. A request cut by the range
    boundary contributes only its covered query tokens, as a chunk of its own.

    Args:
        q_lens: New (query) tokens per request.
        kv_lens: Compressed KV tokens per request, cached plus new.
        elem_budget: Maximum packed logits elements for a multi-request group.
        q_begin: First context query token to cover.
        q_end: End of the covered context query tokens; defaults to all of them.

    Returns:
        Chunk specs for ``K31Indexer.prepare_one_prefill_chunk``, in token order:
        ``(req_idx, token_start_in_req, token_end_in_req, req_cum_start)``.
    """
    assert len(q_lens) == len(kv_lens)
    if q_end is None:
        q_end = sum(q_lens)
    q_starts = []
    q_start = 0
    for q_len in q_lens:
        q_starts.append(q_start)
        q_start += q_len

    chunks: List[List[Tuple[int, int, int, int]]] = []
    full_run: List[int] = []

    def flush_full_run() -> None:
        groups = group_prefill_requests_by_logits_budget(
            [q_lens[i] for i in full_run], [kv_lens[i] for i in full_run], elem_budget
        )
        for group in groups:
            chunks.append(
                [(full_run[g], 0, q_lens[full_run[g]], q_starts[full_run[g]]) for g in group]
            )
        full_run.clear()

    for req_idx, q_len in enumerate(q_lens):
        begin = max(q_begin, q_starts[req_idx])
        end = min(q_end, q_starts[req_idx] + q_len)
        if begin >= end:
            continue
        if end - begin == q_len:
            full_run.append(req_idx)
        else:
            flush_full_run()
            chunks.append(
                [
                    (
                        req_idx,
                        begin - q_starts[req_idx],
                        end - q_starts[req_idx],
                        q_starts[req_idx],
                    )
                ]
            )
    flush_full_run()
    return chunks


# Shrink the indexer prefill chunk size for very long requests to bound the
# fp8_mqa_logits activation memory (~ chunk_size * KV length), keyed on the
# largest index-key length in the batch (one key per token). Entries are
# (kv_len_lower_bound_exclusive, chunk_size); >512K -> 8K, [256K, 512K]
# -> 16K, otherwise unchanged.
_INDEXER_CHUNK_SIZE_HEURISTIC = (
    (512 * 1024, 8 * 1024),
    (256 * 1024 - 1, 16 * 1024),
)


def select_indexer_chunk_size(configured_chunk_size: int, max_kv_len: int) -> int:
    """Pick the indexer prefill chunk size from the batch's largest index-key length.

    Only reduces ``configured_chunk_size`` (never increases it).
    """
    for threshold, chunk_size in _INDEXER_CHUNK_SIZE_HEURISTIC:
        if max_kv_len > threshold:
            return min(configured_chunk_size, chunk_size)
    return configured_chunk_size


def compute_cu_seqlen_kv_bounds_with_cache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
    cached_token_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention,
    accounting for cached KV tokens.

    Args:
        seq_lens: current token lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of context tokens across all sequences in current batch
        cached_token_lens: Cached KV token lengths [num_contexts], dtype=torch.int32 (optional)
        kv_lens: KV token lengths [num_contexts], dtype=torch.int32 (optional)

    Returns:
        cu_seqlen_ks: Start index in KV for each Q token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each Q token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Total KV lengths per request
    if kv_lens is None:
        kv_lens = (
            seq_lens if cached_token_lens is None else cached_token_lens + seq_lens
        )  # [num_contexts]

    # Cumulative KV offsets: where each request's KV sequence starts in global KV space
    cu_kv_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(kv_lens, dim=0).to(torch.int32),
        ]
    )  # [num_contexts + 1]

    # Map each Q token to its request: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32), seq_lens
    )  # [num_ctx_tokens]

    # Each Q token's KV window starts at its request's KV sequence start
    cu_seqlen_ks = cu_kv_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local Q position within each request (0-based, relative to current batch context tokens)
    cu_q_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0).to(torch.int32),
        ]
    )  # [num_contexts + 1]

    global_q_positions = torch.arange(num_ctx_tokens, device=device, dtype=torch.int32)
    local_q_positions = global_q_positions - torch.repeat_interleave(
        cu_q_offsets[:-1], seq_lens
    )  # [num_ctx_tokens]

    if cached_token_lens is not None:
        cached_per_token = torch.repeat_interleave(cached_token_lens, seq_lens)  # [num_ctx_tokens]
        cu_seqlen_ke = cu_seqlen_ks + (cached_per_token + local_q_positions + 1)  # [num_ctx_tokens]
    else:
        cu_seqlen_ke = cu_seqlen_ks + (local_q_positions + 1)  # [num_ctx_tokens]

    return cu_seqlen_ks, cu_seqlen_ke


@dataclass
class K31IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk in the indexer"""

    cu_seqlen_ks: torch.Tensor  # Attention window start for each token
    cu_seqlen_ke: torch.Tensor  # Attention window end for each token
    token_start: int  # Q token start index in batch
    token_end: int  # Q token end index in batch
    k_token_start: int  # K token start index in batch
    k_token_end: int  # K token end index in batch


def _split_prefill_queries(
    count: int, mapping: Optional[Mapping], threshold: int
) -> Tuple[int, int, Optional[List[int]]]:
    """Return local query bounds and optional sizes for a TP prefill gather."""
    if (
        threshold < 0
        or count < threshold
        or mapping is None
        or mapping.enable_attention_dp
        or mapping.tp_size <= 1
    ):
        return 0, count, None
    size, rank = mapping.tp_size, mapping.tp_rank
    sizes = [(r + 1) * count // size - r * count // size for r in range(size)]
    return count * rank // size, count * (rank + 1) // size, sizes


def _prefill_q_split_threshold(metadata: K31DSAtrtllmAttentionMetadata) -> int:
    """Query-token count from which prefill indexer queries are split across TP ranks."""
    sparse_metadata_params = metadata.sparse_metadata_params
    return sparse_metadata_params.q_split_threshold if sparse_metadata_params is not None else 8192


def _packed_logits_elem_budget(metadata: K31DSAtrtllmAttentionMetadata) -> int:
    """Packed-logits cap for grouping context requests into one prefill chunk.

    A single request's logits tile is at most ``min(_INDEXER_MQA_LOGITS_ELEM_BUDGET,
    max_num_tokens * max_seq_len)`` elements, the size the KV cache estimate
    reserves for the indexer logits workspace where it reserves one. Clamping
    the packed cap to it keeps a packed chunk within that reserve.
    """
    single_request_tile = min(
        _INDEXER_MQA_LOGITS_ELEM_BUDGET,
        metadata.max_num_tokens * metadata.kv_cache_manager.max_seq_len,
    )
    return min(_INDEXER_PACKED_LOGITS_ELEM_BUDGET, single_request_tile)


@dataclass
class K31IndexerParams:
    """
    Parameters for indexer.
    """

    num_contexts: int
    num_generations: int
    num_ctx_tokens: int
    head_dim: int
    quant_block_size: int
    tokens_per_block: int
    request_ids: List[int]
    num_past_tokens: List[int]
    seq_lens: torch.Tensor

    def __post_init__(self):
        # One index key per token: the cached keys are the cached tokens.
        self.cached_kv_tokens = torch.tensor(self.num_past_tokens, dtype=torch.int32)
        self.kv_lens = self.seq_lens + self.cached_kv_tokens
        self.new_kv_tokens = self.kv_lens - self.cached_kv_tokens

    @property
    def batch_size(self):
        return len(self.request_ids)
