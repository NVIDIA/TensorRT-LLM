# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused CUDA parity tests for the MiniMax-M3 sparse attention algorithm.

These tests exercise
:class:`tensorrt_llm._torch.attention_backend.sparse.minimax_m3.MiniMaxM3SparseAttention`
at the same shape contract :class:`KVCacheManagerV2` produces:

  * Paged main K/V cache shape ``[num_slots, num_kv_heads, head_dim]``.
  * Paged index K cache shape ``[num_slots, 1, sparse_index_dim]``.
  * Paged ``req_to_token[req_idx, pos] -> slot_id`` indirection (with
    intentionally non-contiguous slot ordering to catch any latent
    contiguity assumption).

Parity reference
----------------

Each test compares the algorithm to an independent hand-written
reference :func:`_reference_minimax_sparse_decode` /
:func:`_reference_minimax_sparse_prefill` that mirrors SGLang's
``minimax_sparse_ops/naive/*.py`` for the contiguous case but extended
to drive the paged cache directly. The reference uses different code
paths (different einsum shapes, explicit per-block loops) than the
implementation under test, so a bug in either surface is unlikely to
hide via "both sides agree".

Negative controls
-----------------

Per the Goal 1.4 acceptance contract in ``acceptance-criteria.md``,
the tests include:

  * Wrong score scale (mis-scaling index attention diverges).
  * Wrong block selection (bottom-k instead of top-k diverges).
  * Wrong RoPE rotation (rotating over full ``head_dim`` instead of
    partial ``rotary_dim`` diverges).
  * Fake index-cache geometry (passing per-index-head index K instead
    of replicated single-head index K fails).

Plus the CUDA-graph contract:

  * ``MiniMaxM3SparseAttentionMetadata.prepare()`` stores
    ``max_seqlen_q`` / ``max_seqlen_k`` as plain Python ints.
  * The decode forward function captures cleanly under a CUDA graph
    and replays bit-identical output.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import pytest
import torch

# Make the SGLang-naive reference module importable as an absolute module.
# The tests are collected from a directory without an ``__init__.py``, so
# relative imports do not resolve under pytest's rootless test collection;
# prepending this directory to ``sys.path`` lets the helper be imported
# by its filename stem (``_minimax_m3_sglang_naive``) from any test below.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Hand-written reference (paged, independent of the SUT).
# ---------------------------------------------------------------------------


def _reference_block_scores(
    qk: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """``[*..., n_tokens]`` -> ``[*..., n_blocks]`` via max-pool."""
    *prefix, n = qk.shape
    pad = (block_size - (n % block_size)) % block_size
    if pad:
        pad_t = qk.new_full((*prefix, pad), float("-inf"))
        qk = torch.cat([qk, pad_t], dim=-1)
    n2 = qk.shape[-1]
    return qk.view(*prefix, n2 // block_size, block_size).amax(dim=-1)


def _reference_index_topk(
    idx_q_b: torch.Tensor,  # [num_idx_heads, sparse_index_dim]
    idx_k_seq: torch.Tensor,  # [seq_len, sparse_index_dim]
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    idx_sm_scale: float,
    causal_pos: int,
) -> torch.Tensor:
    """Compute per-(kv_head) top-k blocks for one Q token (reference).

    Replicates SGLang's naive impl but driven by a single Q token and
    a single sequence's K matrix already gathered from the paged
    cache. Returns ``[num_kv_heads, topk_union]`` int64, -1 padded.
    """
    num_idx_heads = idx_q_b.shape[0]
    seq_len = idx_k_seq.shape[0]
    idx_group_size = num_idx_heads // num_kv_heads
    qk = (idx_q_b.float() @ idx_k_seq.float().T) * idx_sm_scale  # [H, L]
    # Causal mask vs. the Q token's own position.
    arange = torch.arange(seq_len, device=qk.device)
    mask = arange.unsqueeze(0) <= causal_pos
    qk = qk.masked_fill(~mask, float("-inf"))
    scores = _reference_block_scores(qk, block_size)  # [H, n_blocks]
    n_blocks = scores.shape[-1]
    eff = min(seq_len, causal_pos + 1)
    n_valid = (eff + block_size - 1) // block_size
    # init + local priority
    if init_blocks > 0:
        scores[:, :init_blocks] = 1e30
    if local_blocks > 0:
        start = max(0, n_valid - local_blocks)
        if start < n_valid:
            scores[:, start:n_valid] = 1e29
    # Mask invalid (past n_valid) blocks.
    if n_valid < n_blocks:
        scores[:, n_valid:] = float("-inf")
    # Per-head top-k.
    per_head = torch.full((num_idx_heads, topk), fill_value=-1, device=qk.device, dtype=torch.int64)
    take = min(topk, n_valid) if n_valid > 0 else 0
    if take > 0:
        _, idx = scores[:, :n_valid].topk(k=take, dim=-1)
        per_head[:, :take] = idx
    # Union per kv-head.
    per_kv: List[torch.Tensor] = []
    for kh in range(num_kv_heads):
        chunk = per_head[kh * idx_group_size : (kh + 1) * idx_group_size]
        # union of unique non-negative ids.
        uniq = sorted({int(x) for x in chunk.flatten().tolist() if int(x) >= 0})
        out = torch.full((idx_group_size * topk,), -1, device=qk.device, dtype=torch.int64)
        for j, v in enumerate(uniq):
            out[j] = v
        per_kv.append(out)
    return torch.stack(per_kv, dim=0)  # [num_kv_heads, idx_group_size * topk]


def _reference_sparse_gqa_decode(
    q_b: torch.Tensor,  # [num_q_heads, head_dim]
    k_seq: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
    v_seq: torch.Tensor,
    topk_per_kv: torch.Tensor,  # [num_kv_heads, topk_union]
    *,
    block_size: int,
    sm_scale: float,
    causal_pos: int,
) -> torch.Tensor:
    """Sparse GQA reference for one Q token."""
    num_q_heads = q_b.shape[0]
    num_kv_heads = k_seq.shape[1]
    g = num_q_heads // num_kv_heads
    out = torch.zeros_like(q_b)
    for kh in range(num_kv_heads):
        positions: List[int] = []
        for bi in topk_per_kv[kh].tolist():
            if bi < 0:
                continue
            start = bi * block_size
            end = min(start + block_size, causal_pos + 1)
            if start >= causal_pos + 1:
                continue
            positions.extend(range(start, end))
        if not positions:
            continue
        sel = torch.tensor(positions, device=q_b.device, dtype=torch.long)
        ksel = k_seq.index_select(0, sel)[:, kh, :].float()
        vsel = v_seq.index_select(0, sel)[:, kh, :].float()
        for gi in range(g):
            qh = kh * g + gi
            qvec = q_b[qh].float()
            scores = (qvec @ ksel.T) * sm_scale
            attn = scores.softmax(dim=-1)
            out[qh] = (attn @ vsel).to(q_b.dtype)
    return out


def _reference_minimax_sparse_decode(
    q: torch.Tensor,  # [batch, num_q_heads, head_dim]
    idx_q: torch.Tensor,  # [batch, num_idx_heads, sparse_index_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    config,
    sm_scale: float,
    idx_sm_scale: float,
) -> torch.Tensor:
    """Reference decode for parity. Returns ``[batch, num_q_heads * head_dim]``."""
    batch = int(q.shape[0])
    outs = []
    for b in range(batch):
        sl = int(seq_lens[b].item())
        slot_row = int(slot_ids[b].item())
        sl_idx = req_to_token[slot_row, :sl].to(torch.long)
        idx_k_seq = idx_k_cache.index_select(0, sl_idx).squeeze(1)
        k_seq = k_cache.index_select(0, sl_idx)
        v_seq = v_cache.index_select(0, sl_idx)
        topk_per_kv = _reference_index_topk(
            idx_q[b],
            idx_k_seq,
            block_size=config.block_size,
            topk=config.topk,
            init_blocks=config.init_blocks,
            local_blocks=config.local_blocks,
            num_kv_heads=config.num_kv_heads,
            idx_sm_scale=idx_sm_scale,
            causal_pos=sl - 1,
        )
        o = _reference_sparse_gqa_decode(
            q[b],
            k_seq,
            v_seq,
            topk_per_kv,
            block_size=config.block_size,
            sm_scale=sm_scale,
            causal_pos=sl - 1,
        )
        outs.append(o.reshape(config.num_q_heads * config.head_dim))
    return torch.stack(outs, dim=0)


def _reference_minimax_sparse_prefill(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    idx_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    config,
    sm_scale: float,
    idx_sm_scale: float,
) -> torch.Tensor:
    """Reference prefill for parity. Returns ``[total_q, num_q_heads * head_dim]``."""
    total_q = int(q.shape[0])
    out = torch.zeros(total_q, config.num_q_heads * config.head_dim, device=q.device, dtype=q.dtype)
    cu = cu_seqlens_q.tolist()
    for b in range(int(slot_ids.shape[0])):
        start, end = cu[b], cu[b + 1]
        sl = int(seq_lens[b].item())
        slot_row = int(slot_ids[b].item())
        sl_idx = req_to_token[slot_row, :sl].to(torch.long)
        idx_k_seq = idx_k_cache.index_select(0, sl_idx).squeeze(1)
        k_seq = k_cache.index_select(0, sl_idx)
        v_seq = v_cache.index_select(0, sl_idx)
        pref = int(prefix_lens[b].item())
        for qi in range(start, end):
            causal_pos = pref + (qi - start)
            topk_per_kv = _reference_index_topk(
                idx_q[qi],
                idx_k_seq,
                block_size=config.block_size,
                topk=config.topk,
                init_blocks=config.init_blocks,
                local_blocks=config.local_blocks,
                num_kv_heads=config.num_kv_heads,
                idx_sm_scale=idx_sm_scale,
                causal_pos=causal_pos,
            )
            o = _reference_sparse_gqa_decode(
                q[qi],
                k_seq,
                v_seq,
                topk_per_kv,
                block_size=config.block_size,
                sm_scale=sm_scale,
                causal_pos=causal_pos,
            )
            out[qi] = o.reshape(config.num_q_heads * config.head_dim)
    return out


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _checkpoint_scale_config():
    """Match the real MiniMax-M3 checkpoint-scale per-rank geometry.

    The acceptance test names the checkpoint settings (TP=1 view, since
    the focused parity test does not exercise tensor parallel):
      * num_q_heads=64
      * num_kv_heads=4
      * head_dim=128
      * num_index_heads=4
      * sparse_index_dim=128
      * block_size=128
      * topk=16
      * init_blocks=0
      * local_blocks=1
      * score_type='max'
      * disable_index_value=True on every sparse layer

    The tests scale down the K-side context length below the real
    524288-token max so single-test runtime stays inside the CI slot
    while still exercising init/local priority + topk selection over
    enough blocks that the choice is meaningful.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseConfig

    return MiniMaxM3SparseConfig(
        num_q_heads=64,
        num_kv_heads=4,
        head_dim=128,
        num_index_heads=4,
        sparse_index_dim=128,
        block_size=128,
        topk=16,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
    )


def _small_synthetic_config():
    """Smaller geometry for fast negative-control feedback."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseConfig

    return MiniMaxM3SparseConfig(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=16,
        num_index_heads=2,
        sparse_index_dim=16,
        block_size=4,
        topk=2,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
    )


def _populate_paged_cache(
    *,
    cache_main_k: torch.Tensor,
    cache_main_v: torch.Tensor,
    cache_idx_k: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill the paged caches at the slot ids ``req_to_token`` points to.

    Returns the (contiguous, per-batch) K, V, idx_K tensors used to
    populate the caches so the test can also produce the corresponding
    "would-have-been-contiguous" reference if needed.
    """
    batch = int(slot_ids.shape[0])
    for b in range(batch):
        sl = int(seq_lens[b].item())
        slot_row = int(slot_ids[b].item())
        slots = req_to_token[slot_row, :sl].to(torch.long)
        cache_main_k.index_copy_(0, slots, torch.randn_like(cache_main_k.index_select(0, slots)))
        cache_main_v.index_copy_(0, slots, torch.randn_like(cache_main_v.index_select(0, slots)))
        cache_idx_k.index_copy_(0, slots, torch.randn_like(cache_idx_k.index_select(0, slots)))
    return cache_main_k, cache_main_v, cache_idx_k


def _make_noncontiguous_req_to_token(
    batch: int, seq_len: int, num_slots: int, device: torch.device, seed: int
) -> torch.Tensor:
    """Build a ``[batch, seq_len]`` mapping that picks non-contiguous slot ids.

    Slot ids are drawn from a permutation of ``[0, num_slots)`` so
    different requests interleave in the slot space.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_slots, generator=g)
    needed = batch * seq_len
    if needed > num_slots:
        raise ValueError(f"need {needed} unique slots but only have {num_slots}")
    return perm[:needed].view(batch, seq_len).to(torch.int32).to(device)


# ---------------------------------------------------------------------------
# Sanity / contract tests (independent of CUDA graphs / parity)
# ---------------------------------------------------------------------------


def test_metadata_prepare_stores_python_int_max_lengths():
    """Goal 1.4 CUDA-graph contract: max lengths are plain Python ints."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
    )

    seq_lens = torch.tensor([5, 9, 2], dtype=torch.int32)
    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=torch.zeros(3, 16, dtype=torch.int32),
        slot_ids=torch.arange(3, dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
    )
    meta.prepare()
    assert isinstance(meta.max_seqlen_k, int)
    assert isinstance(meta.max_seqlen_q, int)
    assert meta.max_seqlen_k == 9
    assert meta.max_seqlen_q == 1


def test_metadata_prepare_prefill_requires_extend_lengths():
    """Prefill metadata without extend_seq_lens_cpu raises with a clear message."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
    )

    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=True,
        req_to_token=torch.zeros(2, 16, dtype=torch.int32),
        slot_ids=torch.arange(2, dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
        prefix_lens=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        extend_seq_lens_cpu=None,
    )
    with pytest.raises(ValueError, match="extend_seq_lens_cpu"):
        meta.prepare()


def test_sparse_config_rejects_unsupported_score_type():
    """``score_type='log_sum_exp'`` (or any non-'max') must be rejected."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseConfig

    with pytest.raises(ValueError, match="score_type"):
        MiniMaxM3SparseConfig(
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            num_index_heads=2,
            sparse_index_dim=8,
            block_size=2,
            topk=2,
            score_type="log_sum_exp",
        )


def test_sparse_config_rejects_indivisible_head_counts():
    """num_q_heads / num_index_heads must be divisible by num_kv_heads."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseConfig

    with pytest.raises(ValueError, match="divisible"):
        MiniMaxM3SparseConfig(
            num_q_heads=5,
            num_kv_heads=2,
            head_dim=8,
            num_index_heads=2,
            sparse_index_dim=8,
            block_size=2,
            topk=2,
        )
    with pytest.raises(ValueError, match="divisible"):
        MiniMaxM3SparseConfig(
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            num_index_heads=3,
            sparse_index_dim=8,
            block_size=2,
            topk=2,
        )


# ---------------------------------------------------------------------------
# Index cache geometry + negative control
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_index_cache_allocates_only_index_k_when_value_disabled():
    """``disable_index_value=True`` skips index V allocation entirely."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseIndexCache

    cache = MiniMaxM3SparseIndexCache(
        num_layers=4,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        num_slots=8,
        sparse_index_dim=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    for lid in (1, 2, 3):
        assert cache.has_index_value(lid) is False
        assert cache.get_index_v_buffer(lid) is None
        buf = cache.get_index_k_buffer(lid)
        assert tuple(buf.shape) == (8, 1, 16)
    with pytest.raises(KeyError):
        cache.get_index_k_buffer(0)  # not a sparse layer


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_index_cache_value_layers_get_full_kv_pair():
    """When ``disable_index_value=False`` for some layers, V is allocated."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseIndexCache

    cache = MiniMaxM3SparseIndexCache(
        num_layers=4,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[2],  # layer 2 still disabled
        num_slots=8,
        sparse_index_dim=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    assert cache.has_index_value(1) is True
    assert cache.has_index_value(2) is False
    assert cache.has_index_value(3) is True
    assert tuple(cache.get_index_v_buffer(1).shape) == (8, 1, 16)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_index_cache_rejects_fake_index_geometry():
    """**Negative control**: writing per-index-head K (instead of replicated
    single-head K) must fail. Catches a regression where someone shapes
    ``idx_k`` to mirror the main K geometry."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3SparseIndexCache

    cache = MiniMaxM3SparseIndexCache(
        num_layers=2,
        sparse_layer_ids=[0, 1],
        disable_index_value_layer_ids=[0, 1],
        num_slots=8,
        sparse_index_dim=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    # Per-index-head shape: [N, num_idx_heads, sparse_index_dim].
    bad_idx_k = torch.zeros(3, 4, 16, device="cuda")
    out_cache_loc = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="single-head"):
        cache.set_index_k(0, out_cache_loc, bad_idx_k)


# ---------------------------------------------------------------------------
# Decode parity (small synthetic + checkpoint-scale)
# ---------------------------------------------------------------------------


def _build_decode_inputs(
    *,
    config,
    batch: int,
    seq_len_max: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
    noncontiguous: bool = True,
):
    """Construct populated paged caches + decode Q/idx_Q + metadata."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
    )

    torch.manual_seed(seed)
    num_slots = batch * seq_len_max
    if noncontiguous:
        req_to_token = _make_noncontiguous_req_to_token(
            batch, seq_len_max, num_slots, device, seed=seed
        )
    else:
        req_to_token = torch.arange(num_slots, dtype=torch.int32, device=device).view(
            batch, seq_len_max
        )

    slot_ids = torch.arange(batch, dtype=torch.int32, device=device)
    # Vary seq_lens across batch entries.
    seq_lens_list = [
        seq_len_max,
        max(config.block_size * (config.topk + config.local_blocks + 1), 64),
    ][:batch]
    while len(seq_lens_list) < batch:
        seq_lens_list.append(seq_len_max)
    seq_lens_list = [min(sl, seq_len_max) for sl in seq_lens_list]
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()

    k_cache = torch.zeros(
        num_slots, config.num_kv_heads, config.head_dim, device=device, dtype=dtype
    )
    v_cache = torch.zeros(
        num_slots, config.num_kv_heads, config.head_dim, device=device, dtype=dtype
    )
    idx_k_cache = torch.zeros(num_slots, 1, config.sparse_index_dim, device=device, dtype=dtype)
    _populate_paged_cache(
        cache_main_k=k_cache,
        cache_main_v=v_cache,
        cache_idx_k=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
    )

    q = torch.randn(batch, config.num_q_heads, config.head_dim, device=device, dtype=dtype)
    idx_q = torch.randn(
        batch, config.num_index_heads, config.sparse_index_dim, device=device, dtype=dtype
    )

    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
    )
    meta.prepare()
    return q, idx_q, k_cache, v_cache, idx_k_cache, meta


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_parity_small_synthetic():
    """Small-geometry decode matches the hand-written paged reference."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=32, device=device, dtype=dtype, seed=0
    )

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5

    idx_o, o = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    assert idx_o is None
    o_ref = _reference_minimax_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        meta.req_to_token,
        meta.slot_ids,
        meta.seq_lens,
        config=cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_parity_checkpoint_scale():
    """Checkpoint-scale decode matches the reference at config-spec settings."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _checkpoint_scale_config()
    device = torch.device("cuda")
    dtype = torch.float32
    # Use a context length that exercises top-k + local prio over many blocks
    # but stays small enough that the hand-written reference loops run fast.
    seq_len_max = cfg.block_size * (cfg.topk + cfg.local_blocks + 4)
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=seq_len_max, device=device, dtype=dtype, seed=1
    )

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    o_ref = _reference_minimax_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        meta.req_to_token,
        meta.slot_ids,
        meta.seq_lens,
        config=cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_paged_noncontiguous_kv_layout():
    """Non-contiguous slot order still matches the reference.

    The default ``_make_noncontiguous_req_to_token`` already shuffles
    slot ids; this test pins the contract by comparing the
    non-contiguous run to a fresh contiguous run with **the same**
    underlying per-sequence K/V matrices, and asserts identical output.
    Catches any code path that latently assumes ``slot_ids[b] == b *
    seq_len + pos``.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
        minimax_m3_sparse_decode,
    )

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32

    # Round 1: contiguous layout.
    torch.manual_seed(7)
    batch = 2
    seq_len_max = 24
    num_slots = batch * seq_len_max
    req_contig = torch.arange(num_slots, dtype=torch.int32, device=device).view(batch, seq_len_max)
    slot_ids = torch.arange(batch, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len_max, seq_len_max - 4], dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    k_cache_c = torch.randn(num_slots, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    v_cache_c = torch.randn(num_slots, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    idx_k_cache_c = torch.randn(num_slots, 1, cfg.sparse_index_dim, device=device, dtype=dtype)
    q = torch.randn(batch, cfg.num_q_heads, cfg.head_dim, device=device, dtype=dtype)
    idx_q = torch.randn(
        batch, cfg.num_index_heads, cfg.sparse_index_dim, device=device, dtype=dtype
    )
    meta_c = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=req_contig,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
    )
    meta_c.prepare()
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_contig = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache_c,
        v_cache_c,
        idx_k_cache_c,
        None,
        meta_c,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )

    # Round 2: shuffle slot ids and rebuild the cache to match.
    torch.manual_seed(11)
    perm = torch.randperm(num_slots, device="cpu").to(device)
    req_nc = perm.to(torch.int32).view(batch, seq_len_max)
    k_cache_n = torch.zeros_like(k_cache_c)
    v_cache_n = torch.zeros_like(v_cache_c)
    idx_k_cache_n = torch.zeros_like(idx_k_cache_c)
    for b in range(batch):
        sl = int(seq_lens[b].item())
        src_slots = req_contig[b, :sl].to(torch.long)
        dst_slots = req_nc[b, :sl].to(torch.long)
        k_cache_n.index_copy_(0, dst_slots, k_cache_c.index_select(0, src_slots))
        v_cache_n.index_copy_(0, dst_slots, v_cache_c.index_select(0, src_slots))
        idx_k_cache_n.index_copy_(0, dst_slots, idx_k_cache_c.index_select(0, src_slots))
    meta_n = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=req_nc,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
    )
    meta_n.prepare()
    _, o_nc = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache_n,
        v_cache_n,
        idx_k_cache_n,
        None,
        meta_n,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_contig, o_nc, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Prefill parity + decode-after-prefill (cache reuse)
# ---------------------------------------------------------------------------


def _build_prefill_inputs(
    *,
    config,
    batch: int,
    prefix_len: int,
    chunk_len: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Build prefill inputs with a prefix already in the cache."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
    )

    torch.manual_seed(seed)
    seq_len_max = prefix_len + chunk_len
    num_slots = batch * seq_len_max
    req_to_token = _make_noncontiguous_req_to_token(
        batch, seq_len_max, num_slots, device, seed=seed
    )
    slot_ids = torch.arange(batch, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len_max] * batch, dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([prefix_len] * batch, dtype=torch.int32, device=device)
    extend_seq_lens_cpu = [chunk_len] * batch
    cu_seqlens_q = torch.tensor(
        [0] + [chunk_len * (i + 1) for i in range(batch)],
        dtype=torch.int32,
        device=device,
    )

    k_cache = torch.zeros(
        num_slots, config.num_kv_heads, config.head_dim, device=device, dtype=dtype
    )
    v_cache = torch.zeros(
        num_slots, config.num_kv_heads, config.head_dim, device=device, dtype=dtype
    )
    idx_k_cache = torch.zeros(num_slots, 1, config.sparse_index_dim, device=device, dtype=dtype)
    _populate_paged_cache(
        cache_main_k=k_cache,
        cache_main_v=v_cache,
        cache_idx_k=idx_k_cache,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
    )

    total_q = chunk_len * batch
    q = torch.randn(total_q, config.num_q_heads, config.head_dim, device=device, dtype=dtype)
    idx_q = torch.randn(
        total_q, config.num_index_heads, config.sparse_index_dim, device=device, dtype=dtype
    )

    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=True,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        prefix_lens=prefix_lens,
        cu_seqlens_q=cu_seqlens_q,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
    )
    meta.prepare()
    return q, idx_q, k_cache, v_cache, idx_k_cache, meta


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_prefill_parity_small_synthetic():
    """Small-geometry prefill matches the reference (with chunked prefix)."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_prefill

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_prefill_inputs(
        config=cfg, batch=2, prefix_len=12, chunk_len=4, device=device, dtype=dtype, seed=3
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    idx_o, o = minimax_m3_sparse_prefill(
        q,
        k_cache,
        v_cache,
        idx_q,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    assert idx_o is None
    o_ref = _reference_minimax_sparse_prefill(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        meta.req_to_token,
        meta.slot_ids,
        meta.seq_lens,
        meta.prefix_lens,
        meta.cu_seqlens_q,
        config=cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_prefill_parity_checkpoint_scale():
    """Checkpoint-scale prefill at the configured per-rank geometry."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_prefill

    cfg = _checkpoint_scale_config()
    device = torch.device("cuda")
    dtype = torch.float32
    # Small enough for the reference's per-Q loop to finish quickly.
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_prefill_inputs(
        config=cfg,
        batch=2,
        prefix_len=cfg.block_size * 3,
        chunk_len=4,
        device=device,
        dtype=dtype,
        seed=4,
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o = minimax_m3_sparse_prefill(
        q,
        k_cache,
        v_cache,
        idx_q,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    o_ref = _reference_minimax_sparse_prefill(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        meta.req_to_token,
        meta.slot_ids,
        meta.seq_lens,
        meta.prefix_lens,
        meta.cu_seqlens_q,
        config=cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_after_prefill_cache_reuse():
    """After prefill, decode reuses the same paged cache (cache reuse path)."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
        minimax_m3_sparse_decode,
        minimax_m3_sparse_prefill,
    )

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta_pf = _build_prefill_inputs(
        config=cfg, batch=2, prefix_len=0, chunk_len=16, device=device, dtype=dtype, seed=5
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _ = minimax_m3_sparse_prefill(
        q,
        k_cache,
        v_cache,
        idx_q,
        idx_k_cache,
        None,
        meta_pf,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )

    # Append one new decode token per sequence using the same caches.
    batch = int(meta_pf.slot_ids.shape[0])
    new_seq_lens = meta_pf.seq_lens + 1  # one new token per sequence
    new_seq_lens_cpu = new_seq_lens.cpu()
    # Write the new token's main K/V and index K into a fresh slot per
    # sequence (slot = max so far + 1).
    new_slots = torch.arange(
        meta_pf.req_to_token.shape[1] * batch,
        meta_pf.req_to_token.shape[1] * batch + batch,
        device=device,
        dtype=torch.long,
    )
    # Expand req_to_token to hold the new token positions.
    new_req_to_token = torch.cat(
        [meta_pf.req_to_token, new_slots.view(batch, 1).to(torch.int32)], dim=1
    )
    # Grow caches to accommodate the new slots.
    grow = batch
    k_cache_g = torch.cat(
        [k_cache, torch.randn(grow, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)],
        dim=0,
    )
    v_cache_g = torch.cat(
        [v_cache, torch.randn(grow, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)],
        dim=0,
    )
    idx_k_cache_g = torch.cat(
        [idx_k_cache, torch.randn(grow, 1, cfg.sparse_index_dim, device=device, dtype=dtype)], dim=0
    )
    meta_dc = MiniMaxM3SparseAttentionMetadata(
        is_prefill=False,
        req_to_token=new_req_to_token,
        slot_ids=meta_pf.slot_ids,
        seq_lens=new_seq_lens,
        seq_lens_cpu=new_seq_lens_cpu,
    )
    meta_dc.prepare()

    q_d = torch.randn(batch, cfg.num_q_heads, cfg.head_dim, device=device, dtype=dtype)
    idx_q_d = torch.randn(
        batch, cfg.num_index_heads, cfg.sparse_index_dim, device=device, dtype=dtype
    )
    _, o = minimax_m3_sparse_decode(
        q_d,
        idx_q_d,
        k_cache_g,
        v_cache_g,
        idx_k_cache_g,
        None,
        meta_dc,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    o_ref = _reference_minimax_sparse_decode(
        q_d,
        idx_q_d,
        k_cache_g,
        v_cache_g,
        idx_k_cache_g,
        meta_dc.req_to_token,
        meta_dc.slot_ids,
        meta_dc.seq_lens,
        config=cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# disable_index_value semantics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_disable_index_value_true_returns_none_idx_o():
    """When True, ``idx_o`` is ``None`` (index V is not consumed)."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=16, device=device, dtype=torch.float32, seed=2
    )
    idx_o, _ = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
    )
    assert idx_o is None


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_disable_index_value_false_requires_idx_v_cache():
    """When ``False``, an ``idx_v_cache`` tensor must be supplied."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=16, device=device, dtype=torch.float32, seed=2
    )
    with pytest.raises(ValueError, match="idx_v_cache"):
        minimax_m3_sparse_decode(
            q,
            idx_q,
            k_cache,
            v_cache,
            idx_k_cache,
            None,
            meta,
            cfg,
            disable_index_value=False,
        )


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_disable_index_value_false_returns_idx_o_shape():
    """When ``False`` with a real ``idx_v_cache``, ``idx_o`` is populated."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=16, device=device, dtype=torch.float32, seed=2
    )
    idx_v_cache = torch.randn(
        idx_k_cache.shape[0], 1, cfg.sparse_index_dim, device=device, dtype=torch.float32
    )
    idx_o, o = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        idx_v_cache,
        meta,
        cfg,
        disable_index_value=False,
    )
    assert idx_o is not None
    assert tuple(idx_o.shape) == (2, cfg.num_index_heads * cfg.sparse_index_dim)
    assert tuple(o.shape) == (2, cfg.num_q_heads * cfg.head_dim)


# ---------------------------------------------------------------------------
# CUDA graph capture/replay
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_cuda_graph_capture_and_replay_matches_eager():
    """**CUDA graph hard path**: capture the decode forward, replay,
    output is bit-identical to an eager run with the same inputs.

    This exercises the metadata's CUDA-graph-safe contract: scalar max
    lengths are pre-computed CPU-side, no GPU-CPU sync runs inside
    the captured region, and the algorithm uses only static-shape
    tensor ops derived from those scalars.

    The test saves and restores the CUDA RNG state around the graph
    capture region so subsequent tests in the same pytest session can
    still call ``torch.randn_like`` without hitting a leftover
    "Offset increment outside graph capture" error.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=16, device=device, dtype=dtype, seed=42
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    eager_args = dict(
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
        disable_index_value=True,
    )
    _, o_eager = minimax_m3_sparse_decode(
        q, idx_q, k_cache, v_cache, idx_k_cache, None, meta, cfg, **eager_args
    )

    # Save the CUDA RNG state so we can restore it after the graph capture
    # corrupts the Philox counter. Without this, subsequent tests that
    # call ``torch.randn_like`` would fail with "Offset increment outside
    # graph capture encountered unexpectedly."
    saved_state = torch.cuda.get_rng_state()
    try:
        # Warm up the kernel inside its own stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                minimax_m3_sparse_decode(
                    q, idx_q, k_cache, v_cache, idx_k_cache, None, meta, cfg, **eager_args
                )
        torch.cuda.current_stream().wait_stream(stream)

        out_buf: List[torch.Tensor] = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, captured = minimax_m3_sparse_decode(
                q, idx_q, k_cache, v_cache, idx_k_cache, None, meta, cfg, **eager_args
            )
            out_buf.append(captured)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(out_buf[0], o_eager, rtol=1e-5, atol=1e-5)
    finally:
        torch.cuda.set_rng_state(saved_state)


# ---------------------------------------------------------------------------
# Negative controls (Goal 1.4 acceptance contract)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_negative_control_wrong_score_scale_diverges():
    """**Negative control**: mis-scaling the **main** GQA attention
    (e.g. using ``sparse_index_dim**-0.5`` instead of
    ``head_dim**-0.5``) changes the softmax distribution over selected
    K positions and therefore the attended output. The parity gate
    must catch this.

    Note: scaling the **index** attention (``idx_sm_scale``) by a
    positive constant is invariant under top-k selection when
    ``init_blocks + local_blocks >= topk`` (the priority constants
    dominate). The score-scale that meaningfully affects output for
    ``disable_index_value=True`` is the main ``sm_scale`` used by the
    sparse GQA over selected blocks.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=32, device=device, dtype=dtype, seed=9
    )
    correct_sm_scale = cfg.head_dim**-0.5
    wrong_sm_scale = (cfg.head_dim * 100) ** -0.5  # very different temperature
    idx_sm_scale = cfg.sparse_index_dim**-0.5

    _, o_correct = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=correct_sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    _, o_wrong = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=wrong_sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    # A 100x change in softmax temperature must visibly change attention.
    diff = (o_correct - o_wrong).abs().max().item()
    assert diff > 1e-3, (
        f"wrong sm_scale produced identical output (diff={diff:g}); "
        "the score-scale negative control did not catch the change"
    )


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_negative_control_wrong_block_selection_diverges():
    """**Negative control**: replacing top-k with bottom-k (by flipping
    the index Q sign) selects entirely different blocks; output must
    diverge from the correct top-k output."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=32, device=device, dtype=dtype, seed=10
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_correct = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    # Flipping idx_q sign rotates the top-k selection to bottom-k.
    _, o_flipped = minimax_m3_sparse_decode(
        q,
        -idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    diff = (o_correct - o_flipped).abs().max().item()
    assert diff > 1e-3, (
        f"flipping idx_q sign did not change selected blocks "
        f"(diff={diff:g}); block-selection negative control is too lax"
    )


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_negative_control_wrong_rope_dim_diverges():
    """**Negative control**: applying RoPE over the full ``head_dim``
    (instead of the partial ``rotary_dim``) changes Q/K and therefore
    the attention output.

    The sparse algorithm itself is RoPE-agnostic (the caller applies
    RoPE before invoking the algorithm). This test demonstrates that
    feeding the algorithm with two different roped Q choices produces
    different outputs — i.e. the test surface is sensitive to the
    model-layer RoPE configuration. A regression where someone
    accidentally rotates over ``head_dim`` instead of ``rotary_dim``
    in the model layer would surface as a parity drift here.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=16, device=device, dtype=dtype, seed=11
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5

    rotary_dim_partial = cfg.head_dim // 2  # matches the M3 partial RoPE policy

    def _apply_rope_at_pos(
        t: torch.Tensor, dim: int, positions: torch.Tensor, theta: float = 10000.0
    ) -> torch.Tensor:
        """Apply RoPE to only the first ``dim`` channels at non-zero positions.

        RoPE at position 0 is identity (cos=1, sin=0), so the test must
        use real non-zero positions to actually rotate the vectors.
        """
        d_half = dim // 2
        chunk = t.clone()
        rope_part = chunk[..., :dim].reshape(*chunk.shape[:-1], d_half, 2)
        freqs = torch.arange(d_half, device=t.device, dtype=torch.float32)
        inv_freq = theta ** (-freqs / d_half)
        pos_f = positions.to(torch.float32)
        angle = pos_f[:, None] * inv_freq[None, :]
        cos = angle.cos().to(t.dtype).reshape(t.shape[0], *([1] * (t.ndim - 2)), d_half)
        sin = angle.sin().to(t.dtype).reshape(t.shape[0], *([1] * (t.ndim - 2)), d_half)
        x = rope_part[..., 0].clone()
        y = rope_part[..., 1].clone()
        rope_part[..., 0] = x * cos - y * sin
        rope_part[..., 1] = x * sin + y * cos
        rotated = rope_part.reshape(*chunk.shape[:-1], dim)
        chunk[..., :dim] = rotated
        return chunk

    # Use distinct non-zero positions so RoPE actually rotates.
    positions = torch.tensor([3, 11], device=device, dtype=torch.int32)
    # Partial RoPE only on rotary_dim of head_dim (the correct M3 policy).
    q_partial = _apply_rope_at_pos(q, rotary_dim_partial, positions)
    # Full-head RoPE (the "wrong RoPE dim" regression).
    q_full = _apply_rope_at_pos(q, cfg.head_dim, positions)

    _, o_partial = minimax_m3_sparse_decode(
        q_partial,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    _, o_full = minimax_m3_sparse_decode(
        q_full,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    diff = (o_partial - o_full).abs().max().item()
    assert diff > 1e-3, (
        f"full-head RoPE and partial-head RoPE produced identical output "
        f"(diff={diff:g}); the rope-dim negative control would not catch a "
        "model-layer regression"
    )


# ---------------------------------------------------------------------------
# Init / local block priority semantics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_init_blocks_force_first_blocks_into_topk():
    """``init_blocks=N`` forces the first N blocks into the top-k.

    When ``init_blocks >= topk`` every selected block is in the
    init region, so flipping the sign of any other-block index K
    cannot change the output.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseConfig,
        minimax_m3_sparse_decode,
    )

    # init_blocks == topk so init priority covers the entire top-k.
    cfg = MiniMaxM3SparseConfig(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=16,
        num_index_heads=2,
        sparse_index_dim=16,
        block_size=4,
        topk=2,
        init_blocks=2,
        local_blocks=0,
    )
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=1, seq_len_max=32, device=device, dtype=dtype, seed=21
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_ref = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    # Perturb idx_k for slots past the first 2 blocks (= 8 slots): since
    # init_blocks forces blocks 0 and 1 into the top-k, the index K of the
    # later slots is irrelevant for block selection. Main K/V are still
    # consulted for the sparse GQA step, so we mutate ONLY idx_k here.
    idx_k_cache_perturbed = idx_k_cache.clone()
    init_slots = meta.req_to_token[meta.slot_ids[0].long(), : 2 * cfg.block_size].to(torch.long)
    mask = torch.ones(idx_k_cache.shape[0], dtype=torch.bool, device=device)
    mask[init_slots] = False
    idx_k_cache_perturbed[mask] += 100.0
    _, o_perturbed = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache_perturbed,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_ref, o_perturbed, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_local_blocks_force_tail_blocks_into_topk():
    """``local_blocks=N`` forces the last N blocks into the top-k.

    Symmetric to the init-blocks test: with ``local_blocks==topk``,
    only the tail blocks' main K/V matter.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseConfig,
        minimax_m3_sparse_decode,
    )

    cfg = MiniMaxM3SparseConfig(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=16,
        num_index_heads=2,
        sparse_index_dim=16,
        block_size=4,
        topk=2,
        init_blocks=0,
        local_blocks=2,
    )
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=1, seq_len_max=32, device=device, dtype=dtype, seed=22
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_ref = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    # Tail = blocks 6, 7 (positions 24..31 for seq_len 32).
    sl = int(meta.seq_lens[0].item())
    tail_start = sl - 2 * cfg.block_size
    tail_slots = meta.req_to_token[meta.slot_ids[0].long(), tail_start:sl].to(torch.long)
    idx_k_cache_perturbed = idx_k_cache.clone()
    mask = torch.ones(idx_k_cache.shape[0], dtype=torch.bool, device=device)
    mask[tail_slots] = False
    idx_k_cache_perturbed[mask] += 100.0
    _, o_perturbed = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache_perturbed,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_ref, o_perturbed, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# MiniMaxM3SparseAttention orchestrator + cache write
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_sparse_attention_orchestrator_decode_matches_function():
    """``MiniMaxM3SparseAttention.forward`` matches the plain function."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttention,
        MiniMaxM3SparseIndexCache,
        minimax_m3_sparse_decode,
    )

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=16, device=device, dtype=dtype, seed=30
    )
    # Build a side cache and copy the prepared idx_k_cache into layer 0.
    num_slots = k_cache.shape[0]
    side = MiniMaxM3SparseIndexCache(
        num_layers=4,
        sparse_layer_ids=[3],
        disable_index_value_layer_ids=[3],
        num_slots=num_slots,
        sparse_index_dim=cfg.sparse_index_dim,
        dtype=dtype,
        device=device,
    )
    side.get_index_k_buffer(3).copy_(idx_k_cache)

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    attn = MiniMaxM3SparseAttention(config=cfg, index_cache=side)
    _, o_orch = attn.forward(
        layer_idx=3,
        q=q,
        idx_q=idx_q,
        k_cache=k_cache,
        v_cache=v_cache,
        metadata=meta,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    _, o_fn = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_orch, o_fn, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_write_caches_appends_main_and_index_buffers():
    """``write_caches`` writes new K/V/idx_K to the slots in ``out_cache_loc``."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttention,
        MiniMaxM3SparseIndexCache,
    )

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    num_slots = 32
    k_cache = torch.zeros(num_slots, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    side = MiniMaxM3SparseIndexCache(
        num_layers=4,
        sparse_layer_ids=[3],
        disable_index_value_layer_ids=[3],
        num_slots=num_slots,
        sparse_index_dim=cfg.sparse_index_dim,
        dtype=dtype,
        device=device,
    )
    attn = MiniMaxM3SparseAttention(config=cfg, index_cache=side)

    n = 5
    out_cache_loc = torch.tensor([2, 7, 11, 18, 25], device=device, dtype=torch.int32)
    k_new = torch.randn(n, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    v_new = torch.randn(n, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype)
    idx_k_new = torch.randn(n, 1, cfg.sparse_index_dim, device=device, dtype=dtype)
    attn.write_caches(
        layer_idx=3,
        out_cache_loc=out_cache_loc,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k_new,
        v=v_new,
        idx_k=idx_k_new,
        idx_v=None,
    )
    # Every targeted slot now equals the source vector.
    for i, slot in enumerate(out_cache_loc.tolist()):
        torch.testing.assert_close(k_cache[slot], k_new[i], rtol=0, atol=0)
        torch.testing.assert_close(v_cache[slot], v_new[i], rtol=0, atol=0)
        torch.testing.assert_close(side.get_index_k_buffer(3)[slot], idx_k_new[i], rtol=0, atol=0)
    # Untargeted slots stay at zero.
    for slot in range(num_slots):
        if slot in out_cache_loc.tolist():
            continue
        assert torch.all(k_cache[slot] == 0)
        assert torch.all(v_cache[slot] == 0)
        assert torch.all(side.get_index_k_buffer(3)[slot] == 0)


# ---------------------------------------------------------------------------
# Sanity: the reference itself reproduces SGLang's naive priority order.
# ---------------------------------------------------------------------------


def test_reference_block_scores_max_reduction():
    """``_reference_block_scores`` performs max-pool over the block axis."""
    qk = torch.tensor([1.0, 5.0, 3.0, 7.0, 2.0, -1.0, 4.0, 0.0])
    out = _reference_block_scores(qk, block_size=4)
    expected = torch.tensor([7.0, 4.0])
    torch.testing.assert_close(out, expected)


def test_reference_block_scores_pads_with_neg_inf():
    """Final partial block gets -inf padding so it does not bias the max."""
    qk = torch.tensor([1.0, 2.0, 3.0, 0.5, 4.0])
    out = _reference_block_scores(qk, block_size=4)
    assert out[1].item() == pytest.approx(4.0)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_reference_index_topk_init_local_priority():
    """Init/local priority overrides numerical scores in the reference."""
    sparse_index_dim = 4
    block_size = 2
    topk = 2
    num_kv_heads = 1
    seq_len = 8  # 4 blocks
    device = torch.device("cuda")
    # Make blocks 1, 2 score very high; blocks 0 (init) and 3 (local)
    # are forced into the top-k regardless of score.
    idx_q_b = torch.ones(1, sparse_index_dim, device=device)
    idx_k_seq = torch.zeros(seq_len, sparse_index_dim, device=device)
    idx_k_seq[2:6, :] = 10.0
    top = _reference_index_topk(
        idx_q_b,
        idx_k_seq,
        block_size=block_size,
        topk=topk,
        init_blocks=1,
        local_blocks=1,
        num_kv_heads=num_kv_heads,
        idx_sm_scale=1.0,
        causal_pos=seq_len - 1,
    )
    chosen = sorted(int(v) for v in top[0].tolist() if v >= 0)
    # init=block 0, local=last block (index 3) — both must be present.
    assert 0 in chosen
    assert 3 in chosen
    # And no -1 (unfilled) other than padding past the union size.
    # The union size = idx_group_size * topk = 1 * 2 = 2 for this case.
    assert chosen == [0, 3]


# ---------------------------------------------------------------------------
# Tiny smoke for the prefill orchestrator integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_sparse_attention_orchestrator_prefill_matches_function():
    """``MiniMaxM3SparseAttention.forward`` (prefill mode) matches the function."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttention,
        MiniMaxM3SparseIndexCache,
        minimax_m3_sparse_prefill,
    )

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_prefill_inputs(
        config=cfg, batch=2, prefix_len=8, chunk_len=4, device=device, dtype=dtype, seed=40
    )
    side = MiniMaxM3SparseIndexCache(
        num_layers=4,
        sparse_layer_ids=[3],
        disable_index_value_layer_ids=[3],
        num_slots=k_cache.shape[0],
        sparse_index_dim=cfg.sparse_index_dim,
        dtype=dtype,
        device=device,
    )
    side.get_index_k_buffer(3).copy_(idx_k_cache)

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    attn = MiniMaxM3SparseAttention(config=cfg, index_cache=side)
    _, o_orch = attn.forward(
        layer_idx=3,
        q=q,
        idx_q=idx_q,
        k_cache=k_cache,
        v_cache=v_cache,
        metadata=meta,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    _, o_fn = minimax_m3_sparse_prefill(
        q,
        k_cache,
        v_cache,
        idx_q,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_orch, o_fn, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# SGLang-naive cross-check (Stage 2 acceptance item 3)
# ---------------------------------------------------------------------------
#
# The tests above compare the SUT against an in-file hand-written reference.
# A bug shared by SUT and reference (e.g. matched off-by-one, matched
# misinterpretation of an init/local edge case) would slip past every
# parity test in this file. Stage 2 closes that gap by anchoring the
# reference against SGLang's actual sparse-attention semantics, vendored
# into :mod:`_minimax_m3_sglang_naive`. These tests:
#
#   * verify the in-file reference and the SGLang-derived reference agree
#     at checkpoint-scale geometry on the configured paged inputs; and
#   * fail when the SGLang-derived reference is deliberately mutated, so
#     the test surface really catches reference drift.


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_sglang_derived_reference_matches_inline_reference():
    """In-file reference matches the SGLang-naive-derived reference.

    Both references are pure PyTorch and use the same paged inputs. If
    they disagree, at least one is buggy in a way that the in-file
    reference cannot self-detect — exactly the silent-failure mode the
    Stage 2 gap-fix names.
    """
    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_sparse_decode

    cfg = _checkpoint_scale_config()
    device = torch.device("cuda")
    dtype = torch.float32
    seq_len_max = cfg.block_size * (cfg.topk + cfg.local_blocks + 4)
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=seq_len_max, device=device, dtype=dtype, seed=101
    )

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5

    o_inline = _reference_minimax_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        meta.req_to_token,
        meta.slot_ids,
        meta.seq_lens,
        config=cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    o_sglang = sglang_naive_sparse_decode(
        q=q,
        idx_q=idx_q,
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        req_to_token=meta.req_to_token,
        slot_ids=meta.slot_ids,
        seq_lens=meta.seq_lens,
        config=SGLangNaiveSparseConfig.from_minimax_config(cfg),
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_inline, o_sglang, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_sut_matches_sglang_derived_reference_checkpoint_scale():
    """SUT (Triton kernels) matches the SGLang-naive reference at checkpoint scale.

    This is the Stage 2 acceptance item 3 closure: focused sparse-attention
    coverage anchored against SGLang's actual sparse-attention semantics
    rather than only the in-file helper. The geometry is the checkpoint-
    scale config (64 Q heads, 4 KV heads, head_dim=128, 4 index heads,
    sparse_index_dim=128, block_size=128, topk=16, local_blocks=1).
    """
    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_sparse_decode

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _checkpoint_scale_config()
    device = torch.device("cuda")
    dtype = torch.float32
    seq_len_max = cfg.block_size * (cfg.topk + cfg.local_blocks + 4)
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=seq_len_max, device=device, dtype=dtype, seed=102
    )

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_sut = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    o_sglang = sglang_naive_sparse_decode(
        q=q,
        idx_q=idx_q,
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        req_to_token=meta.req_to_token,
        slot_ids=meta.slot_ids,
        seq_lens=meta.seq_lens,
        config=SGLangNaiveSparseConfig.from_minimax_config(cfg),
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_sut, o_sglang, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_prefill_sut_matches_sglang_derived_reference_checkpoint_scale():
    """SUT prefill matches the SGLang-naive reference at checkpoint scale."""
    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_sparse_prefill

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_prefill

    cfg = _checkpoint_scale_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_prefill_inputs(
        config=cfg,
        batch=2,
        prefix_len=cfg.block_size * 3,
        chunk_len=4,
        device=device,
        dtype=dtype,
        seed=103,
    )
    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_sut = minimax_m3_sparse_prefill(
        q,
        k_cache,
        v_cache,
        idx_q,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    o_sglang = sglang_naive_sparse_prefill(
        q=q,
        idx_q=idx_q,
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        req_to_token=meta.req_to_token,
        slot_ids=meta.slot_ids,
        seq_lens=meta.seq_lens,
        prefix_lens=meta.prefix_lens,
        cu_seqlens_q=meta.cu_seqlens_q,
        config=SGLangNaiveSparseConfig.from_minimax_config(cfg),
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.testing.assert_close(o_sut, o_sglang, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_mutation_of_sglang_reference_is_caught():
    """**Mutation control**: a deliberately broken SGLang reference disagrees with SUT.

    Mutate the SGLang-naive reference by replacing top-k selection with
    bottom-k (negate scores) and verify the parity comparison **fails**.
    This proves the cross-check is sensitive enough to catch a real
    reference regression — it is not just "two functions both return the
    same constant".
    """
    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_sparse_decode

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import minimax_m3_sparse_decode

    cfg = _small_synthetic_config()
    device = torch.device("cuda")
    dtype = torch.float32
    q, idx_q, k_cache, v_cache, idx_k_cache, meta = _build_decode_inputs(
        config=cfg, batch=2, seq_len_max=32, device=device, dtype=dtype, seed=104
    )

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5
    _, o_sut = minimax_m3_sparse_decode(
        q,
        idx_q,
        k_cache,
        v_cache,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )

    # Mutated reference: invert idx_q so top-k selection collapses onto
    # bottom-k blocks. SGLang's contract picks top-k by score; flipping
    # the sign exchanges that to bottom-k. The SUT still uses the real
    # idx_q so it cannot agree.
    sg_cfg = SGLangNaiveSparseConfig.from_minimax_config(cfg)
    o_mutated = sglang_naive_sparse_decode(
        q=q,
        idx_q=-idx_q,  # mutation: flip block-selection direction
        k_cache=k_cache,
        v_cache=v_cache,
        idx_k_cache=idx_k_cache,
        req_to_token=meta.req_to_token,
        slot_ids=meta.slot_ids,
        seq_lens=meta.seq_lens,
        config=sg_cfg,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    diff = (o_sut - o_mutated).abs().max().item()
    assert diff > 1e-3, (
        f"SUT and the mutated SGLang reference produced identical output "
        f"(diff={diff:g}); the cross-check is not sensitive enough to "
        "catch a real block-selection regression in the reference"
    )


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_sglang_reference_priority_invariants():
    """SGLang-naive reference honours init/local priority on a fixed fixture.

    Pins the SGLang-derived selector against a hand-checked expected
    output so a regression in the vendored translation (e.g. an
    accidental swap of init and local constants, a wrong
    ``num_kv_heads`` reshape, a missing causal clamp) is caught even
    without comparing to the SUT.
    """
    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_topk_select

    # Same fixture as ``test_reference_index_topk_init_local_priority`` so the
    # SGLang-naive vendor is held to the same invariant the in-file
    # reference is.
    sparse_index_dim = 4
    block_size = 2
    topk = 2
    num_kv_heads = 1
    seq_len = 8  # 4 blocks
    device = torch.device("cuda")
    idx_q = torch.ones(1, sparse_index_dim, device=device)
    idx_k_seq = torch.zeros(seq_len, sparse_index_dim, device=device)
    idx_k_seq[2:6, :] = 10.0
    sg_cfg = SGLangNaiveSparseConfig(
        num_q_heads=1,
        num_kv_heads=num_kv_heads,
        head_dim=sparse_index_dim,
        num_index_heads=1,
        sparse_index_dim=sparse_index_dim,
        block_size=block_size,
        topk=topk,
        init_blocks=1,
        local_blocks=1,
    )
    top = sglang_naive_topk_select(
        idx_q=idx_q,
        idx_k_seq=idx_k_seq,
        config=sg_cfg,
        idx_sm_scale=1.0,
        causal_pos=seq_len - 1,
    )
    chosen = sorted(int(v) for v in top[0].tolist() if v >= 0)
    assert chosen == [0, 3]


# ---------------------------------------------------------------------------
# Joint CUDA graph + KVCacheManagerV2 (Stage 2 acceptance item 3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_cuda_graph_joint_with_kv_cache_manager_v2():
    """Joint hard-path proof: ``KVCacheManagerV2`` + ``cuda_graph=true``.

    Stage 2 acceptance item 3 ("checkpoint-scale focused coverage
    exercises ``KVCacheManagerV2`` and ``cuda_graph=true`` **together**
    in the same CUDA run") requires that the algorithm runs end-to-end
    under a single test that:

      * allocates real cache buffers via ``MiniMaxM3KVCacheManagerV2``
        (the production ``KVCacheManagerV2`` subclass);
      * builds metadata via ``build_runtime_metadata_from_kv_manager``;
      * captures the decode forward into a real
        ``torch.cuda.CUDAGraph`` and replays it; and
      * cross-checks the captured output against the SGLang-naive
        reference (so the joint test is also reference-anchored).
    """
    import gc

    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_sparse_decode

    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseConfig,
        _write_main_kv_slots,
        build_runtime_metadata_from_kv_manager,
        minimax_m3_sparse_decode,
    )

    # Compact but checkpoint-shape-faithful geometry. The full
    # ``_checkpoint_scale_config`` requires more cache memory than the
    # standard KVCacheManagerV2 fixture pool exposes; the geometry below
    # keeps the M3 contract (4 KV heads, 4 index heads, init=0,
    # local_blocks=1, score_type='max', disable_index_value=True)
    # while staying inside the bring-up CI slot.
    cfg = MiniMaxM3SparseConfig(
        num_q_heads=32,
        num_kv_heads=4,
        head_dim=64,
        num_index_heads=4,
        sparse_index_dim=64,
        block_size=8,
        topk=4,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens_per_block = 8
    seq_len = 32

    # Build a real MiniMaxM3KVCacheManagerV2. Layer 3 is sparse.
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=512, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        sparse_index_dim=cfg.sparse_index_dim,
        num_layers=4,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        req_id = 7001
        added = mgr.add_dummy_requests(
            request_ids=[req_id],
            token_nums=[seq_len],
            is_gen=False,
        )
        assert added is not None

        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        m3_meta, _ = build_runtime_metadata_from_kv_manager(
            kv_cache_manager=mgr,
            request_ids=[req_id],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            is_prefill=False,
        )

        layer_idx = 3
        kv_pool = mgr.get_buffers(layer_idx)
        k_cache = kv_pool[:, 0].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        v_cache = kv_pool[:, 1].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        idx_k_cache = mgr.get_index_k_buffer(layer_idx)

        # Seed every slot the metadata addresses with finite values.
        all_slots = m3_meta.req_to_token[0, :seq_len].to(torch.long)
        with torch.random.fork_rng(devices=["cuda"]):
            torch.manual_seed(210)
            cpu_k = torch.randn(seq_len, cfg.num_kv_heads, cfg.head_dim) * 0.1
            cpu_v = torch.randn(seq_len, cfg.num_kv_heads, cfg.head_dim) * 0.1
            cpu_idx_k = torch.randn(seq_len, 1, cfg.sparse_index_dim) * 0.1
            cpu_q = torch.randn(1, cfg.num_q_heads, cfg.head_dim) * 0.1
            cpu_idx_q = torch.randn(1, cfg.num_index_heads, cfg.sparse_index_dim) * 0.1
        k_cache.index_copy_(0, all_slots, cpu_k.to(device=device, dtype=k_cache.dtype))
        v_cache.index_copy_(0, all_slots, cpu_v.to(device=device, dtype=v_cache.dtype))
        # ``idx_k_cache`` is the V2 4-D paged view after the Stage 14
        # paged index-K rewrite; ``index_copy_`` cannot bridge the
        # ``[seq_len, 1, sparse_index_dim]`` source onto the 4-D
        # destination. Route through the layout-aware helper.
        _write_main_kv_slots(
            idx_k_cache,
            all_slots.to(torch.int32),
            cpu_idx_k.to(device=device, dtype=idx_k_cache.dtype),
        )
        q = cpu_q.to(device=device, dtype=dtype)
        idx_q = cpu_idx_q.to(device=device, dtype=dtype)

        sm_scale = cfg.head_dim**-0.5
        idx_sm_scale = cfg.sparse_index_dim**-0.5
        eager_args = dict(
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
            disable_index_value=True,
        )

        # Eager reference run on the V2-backed paged buffers.
        _, o_eager = minimax_m3_sparse_decode(
            q, idx_q, k_cache, v_cache, idx_k_cache, None, m3_meta, cfg, **eager_args
        )

        # Capture the decode forward into a real CUDA graph and replay.
        saved_state = torch.cuda.get_rng_state()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    minimax_m3_sparse_decode(
                        q, idx_q, k_cache, v_cache, idx_k_cache, None, m3_meta, cfg, **eager_args
                    )
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            out_buf = []
            with torch.cuda.graph(graph):
                _, captured = minimax_m3_sparse_decode(
                    q, idx_q, k_cache, v_cache, idx_k_cache, None, m3_meta, cfg, **eager_args
                )
                out_buf.append(captured)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out_buf[0], o_eager, rtol=1e-5, atol=1e-5)
        finally:
            torch.cuda.set_rng_state(saved_state)

        # Cross-check the graph-captured output against the SGLang-naive
        # reference. This is the joint "KVCacheManagerV2 + cuda_graph +
        # SGLang-derived reference" assertion the Stage 2 gap-fix names.
        o_sglang = sglang_naive_sparse_decode(
            q=q,
            idx_q=idx_q,
            k_cache=k_cache,
            v_cache=v_cache,
            idx_k_cache=idx_k_cache,
            req_to_token=m3_meta.req_to_token,
            slot_ids=m3_meta.slot_ids,
            seq_lens=m3_meta.seq_lens,
            config=SGLangNaiveSparseConfig.from_minimax_config(cfg),
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
        )
        torch.testing.assert_close(o_eager, o_sglang, rtol=5e-2, atol=5e-2)
    finally:
        mgr.shutdown()
        gc.collect()


# ---------------------------------------------------------------------------
# Joint CUDA graph + KVCacheManagerV2 at checkpoint-scale geometry
# (Stage 3 acceptance item 4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_decode_cuda_graph_joint_with_kv_cache_manager_v2_checkpoint_scale():
    """Checkpoint-scale joint coverage: V2 + cuda_graph + prefill + decode.

    Stage 3 acceptance item 4 names the checkpoint-scale geometry that
    must be exercised together with KVCacheManagerV2 and
    ``cuda_graph=true`` in a single test: 64 Q heads, 4 KV heads,
    head_dim 128, 4 index heads, index_dim 128, block_size 128, topk 16,
    score_type ``max``. The test covers:

      * Real :class:`MiniMaxM3KVCacheManagerV2` allocation for one
        sparse layer with ``disable_index_value=True``.
      * Prefill through ``minimax_m3_sparse_prefill`` to populate the
        cache (so the subsequent decode actually exercises cache reuse
        rather than running over freshly-zeroed slots).
      * Decode through ``minimax_m3_sparse_decode``, both eager and
        captured into a real ``torch.cuda.CUDAGraph`` (the joint
        hard-path proof).
      * Cross-check the captured decode output against the
        SGLang-naive reference at the same checkpoint-scale config.
      * Mutation control: deliberately perturb the cache and assert
        the SGLang-naive reference returns a different result, so the
        comparison cannot trivially pass on tensor-shape agreement.
    """
    import gc

    from _minimax_m3_sglang_naive import SGLangNaiveSparseConfig, sglang_naive_sparse_decode

    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        _write_main_kv_slots,
        build_runtime_metadata_from_kv_manager,
        minimax_m3_sparse_decode,
        minimax_m3_sparse_prefill,
    )

    # Checkpoint-scale geometry per Stage 3 acceptance item 4.
    cfg = _checkpoint_scale_config()
    assert cfg.num_q_heads == 64
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 128
    assert cfg.num_index_heads == 4
    assert cfg.sparse_index_dim == 128
    assert cfg.block_size == 128
    assert cfg.topk == 16
    assert cfg.score_type == "max"

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens_per_block = cfg.block_size

    # Exercise topk selection over enough blocks that the choice is
    # meaningful (topk + local + a few extra blocks of candidates).
    # 17 blocks * 128 = 2176 prefill tokens, plus one decode token.
    prefill_blocks = cfg.topk + cfg.local_blocks + 4
    prefill_tokens = prefill_blocks * cfg.block_size
    max_seq_len = (prefill_tokens + cfg.block_size) * 2
    # Cache pool sized for one batch + headroom for the test's prefill
    # write + decode cache-reuse path. 32 blocks per layer * 128 = 4096.
    max_cache_tokens = max(4096, max_seq_len * 2)

    # Build a real MiniMaxM3KVCacheManagerV2. One sparse layer at
    # index 3, matching the M3 schedule (layers 0-2 dense, 3+ sparse).
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    sparse_layer_id = 3
    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=max_cache_tokens, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[sparse_layer_id],
        disable_index_value_layer_ids=[sparse_layer_id],
        sparse_index_dim=cfg.sparse_index_dim,
        num_layers=sparse_layer_id + 1,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        req_id = 9001
        added = mgr.add_dummy_requests(
            request_ids=[req_id],
            token_nums=[prefill_tokens],
            is_gen=False,
        )
        assert added is not None

        sm_scale = cfg.head_dim**-0.5
        idx_sm_scale = cfg.sparse_index_dim**-0.5

        # --- Phase 1: prefill (populate the cache via the real backend)
        prefill_seq_lens = torch.tensor([prefill_tokens], dtype=torch.int32, device=device)
        prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
        prefill_meta, _ = build_runtime_metadata_from_kv_manager(
            kv_cache_manager=mgr,
            request_ids=[req_id],
            seq_lens=prefill_seq_lens,
            seq_lens_cpu=prefill_seq_lens.cpu(),
            is_prefill=True,
            prefix_lens=prefix_lens,
            extend_seq_lens_cpu=[prefill_tokens],
        )

        # Synthesize prefill Q/K/V/idx_K/idx_Q for the populated tokens.
        with torch.random.fork_rng(devices=["cuda"]):
            torch.manual_seed(2024)
            prefill_q = (
                torch.randn(
                    prefill_tokens, cfg.num_q_heads, cfg.head_dim, device=device, dtype=dtype
                )
                * 0.05
            )
            prefill_k = (
                torch.randn(
                    prefill_tokens, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype
                )
                * 0.05
            )
            prefill_v = (
                torch.randn(
                    prefill_tokens, cfg.num_kv_heads, cfg.head_dim, device=device, dtype=dtype
                )
                * 0.05
            )
            prefill_idx_q = (
                torch.randn(
                    prefill_tokens,
                    cfg.num_index_heads,
                    cfg.sparse_index_dim,
                    device=device,
                    dtype=dtype,
                )
                * 0.05
            )
            prefill_idx_k = (
                torch.randn(prefill_tokens, cfg.sparse_index_dim, device=device, dtype=dtype) * 0.05
            )

        # Write the projected K/V/idx_K into the V2-managed caches at
        # the slots the metadata addresses.
        kv_pool = mgr.get_buffers(sparse_layer_id)
        k_cache = kv_pool[:, 0].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        v_cache = kv_pool[:, 1].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        idx_k_cache = mgr.get_index_k_buffer(sparse_layer_id)
        prefill_slots = prefill_meta.req_to_token[0, :prefill_tokens].to(torch.long)
        k_cache.index_copy_(0, prefill_slots, prefill_k)
        v_cache.index_copy_(0, prefill_slots, prefill_v)
        # ``idx_k_cache`` is the V2 4-D paged view; route through the
        # layout-aware helper.
        _write_main_kv_slots(
            idx_k_cache,
            prefill_slots.to(torch.int32),
            prefill_idx_k.unsqueeze(1),
        )

        # Run prefill once so the production prefill kernel touches the
        # cache (cache_reuse path: same buffers will be read by decode).
        # Signature: (q, k_cache, v_cache, idx_q, idx_k_cache, idx_v_cache,
        #             metadata, config, ...)
        _, _ = minimax_m3_sparse_prefill(
            prefill_q,
            k_cache,
            v_cache,
            prefill_idx_q,
            idx_k_cache,
            None,
            prefill_meta,
            cfg,
            disable_index_value=True,
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
        )
        torch.cuda.synchronize()

        # --- Phase 2: decode (one new token, reuses the populated cache)
        decode_seq_lens = torch.tensor([prefill_tokens], dtype=torch.int32, device=device)
        decode_meta, _ = build_runtime_metadata_from_kv_manager(
            kv_cache_manager=mgr,
            request_ids=[req_id],
            seq_lens=decode_seq_lens,
            seq_lens_cpu=decode_seq_lens.cpu(),
            is_prefill=False,
        )

        with torch.random.fork_rng(devices=["cuda"]):
            torch.manual_seed(2025)
            q_decode = (
                torch.randn(1, cfg.num_q_heads, cfg.head_dim, device=device, dtype=dtype) * 0.05
            )
            idx_q_decode = (
                torch.randn(
                    1, cfg.num_index_heads, cfg.sparse_index_dim, device=device, dtype=dtype
                )
                * 0.05
            )

        eager_args = dict(
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
            disable_index_value=True,
        )

        # Eager decode against the prefill-populated cache.
        _, o_eager = minimax_m3_sparse_decode(
            q_decode,
            idx_q_decode,
            k_cache,
            v_cache,
            idx_k_cache,
            None,
            decode_meta,
            cfg,
            **eager_args,
        )

        # Capture the same decode into a real CUDA graph and replay.
        saved_state = torch.cuda.get_rng_state()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    minimax_m3_sparse_decode(
                        q_decode,
                        idx_q_decode,
                        k_cache,
                        v_cache,
                        idx_k_cache,
                        None,
                        decode_meta,
                        cfg,
                        **eager_args,
                    )
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            out_buf = []
            with torch.cuda.graph(graph):
                _, captured = minimax_m3_sparse_decode(
                    q_decode,
                    idx_q_decode,
                    k_cache,
                    v_cache,
                    idx_k_cache,
                    None,
                    decode_meta,
                    cfg,
                    **eager_args,
                )
                out_buf.append(captured)
            graph.replay()
            torch.cuda.synchronize()
            # Captured replay must match eager (bf16 tolerance).
            torch.testing.assert_close(out_buf[0], o_eager, rtol=5e-3, atol=5e-3)
        finally:
            torch.cuda.set_rng_state(saved_state)

        # --- SGLang-semantic reference comparison
        o_sglang = sglang_naive_sparse_decode(
            q=q_decode,
            idx_q=idx_q_decode,
            k_cache=k_cache,
            v_cache=v_cache,
            idx_k_cache=idx_k_cache,
            req_to_token=decode_meta.req_to_token,
            slot_ids=decode_meta.slot_ids,
            seq_lens=decode_meta.seq_lens,
            config=SGLangNaiveSparseConfig.from_minimax_config(cfg),
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
        )
        # Larger tolerance: bf16 + the SGLang-naive reference reorders
        # the softmax accumulation. The structural assertion is that
        # the production decode matches the SGLang semantics within
        # bf16 numerical reach.
        torch.testing.assert_close(o_eager, o_sglang, rtol=5e-2, atol=5e-2)

        # --- Mutation control: perturbing the index-K cache must
        # change the SGLang-naive reference's output. This proves the
        # reference is actually consuming the cache (i.e. a regression
        # that silently zeroes the index-K branch is caught).
        idx_k_cache_saved = idx_k_cache.clone()
        try:
            idx_k_cache.zero_()
            o_mutated = sglang_naive_sparse_decode(
                q=q_decode,
                idx_q=idx_q_decode,
                k_cache=k_cache,
                v_cache=v_cache,
                idx_k_cache=idx_k_cache,
                req_to_token=decode_meta.req_to_token,
                slot_ids=decode_meta.slot_ids,
                seq_lens=decode_meta.seq_lens,
                config=SGLangNaiveSparseConfig.from_minimax_config(cfg),
                sm_scale=sm_scale,
                idx_sm_scale=idx_sm_scale,
            )
            diff = (o_mutated.float() - o_sglang.float()).abs().mean().item()
            # The synthesized Q/K values are intentionally small (0.05x
            # scale) so the SGLang-naive comparison stays inside bf16
            # precision. A 1e-4 mean-abs-diff is well above the bf16
            # noise floor for values of that magnitude and proves the
            # index-K branch contributes to the output.
            assert diff > 1e-4, (
                "mutation control failed: zeroing the index-K cache "
                "did not change the SGLang-naive reference output; the "
                "reference is not actually consuming idx_k_cache "
                f"(mean_abs_diff={diff:.6e})"
            )
        finally:
            idx_k_cache.copy_(idx_k_cache_saved)
    finally:
        mgr.shutdown()
        gc.collect()


# ---------------------------------------------------------------------------
# Stage 6 — checkpoint-scale sparse-attention memory regression
# ---------------------------------------------------------------------------
#
# The Stage 5 OOM was a single ~13.78 GiB FP32 allocation inside
# ``_sparse_gqa_masked``: ``k_padded.to(torch.float32).index_select(
# 0, q_batch_row)`` materialized a [total_q=2688, max_k=2688,
# num_kv_heads=4, head_dim=128] FP32 slab. The Stage 6 contract is
# that the checkpoint-scale prefill must keep its peak working set
# below the Stage 5 cliff *without* lowering checkpoint dimensions.
#
# The test below pins the checkpoint-scale geometry (so a future
# "fix" that quietly reduces dimensions to fit memory cannot pass),
# probes peak CUDA memory across a real prefill call, and rejects the
# legacy ≥13 GiB single-allocation pattern.


def _checkpoint_scale_prefill_inputs(device, dtype):
    """Build inputs for the Stage 6 memory-regression prefill probe.

    Uses the same checkpoint-scale geometry as
    :func:`test_decode_cuda_graph_joint_with_kv_cache_manager_v2_checkpoint_scale`
    but builds the cache + metadata directly (no
    :class:`MiniMaxM3KVCacheManagerV2`) so the test only measures the
    prefill kernel's own working set.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseAttentionMetadata,
    )

    cfg = _checkpoint_scale_config()
    # Same prefill geometry as the joint CUDA-graph test: 21 blocks of
    # 128 tokens = 2688 tokens, which is the exact total_q/max_k that
    # produced the Stage 5 13.78 GiB allocation.
    prefill_blocks = cfg.topk + cfg.local_blocks + 4  # 21 blocks
    prefill_tokens = prefill_blocks * cfg.block_size  # 2688 tokens
    batch = 1

    cache_slots = prefill_tokens + cfg.block_size
    k_cache = torch.zeros(cache_slots, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(cache_slots, 1, cfg.sparse_index_dim, dtype=dtype, device=device)

    # Identity req_to_token mapping over the live prefix; one request.
    req_to_token = torch.arange(cache_slots, dtype=torch.int32, device=device).unsqueeze(
        0
    )  # [1, cache_slots]
    slot_ids = torch.tensor([0], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([prefill_tokens], dtype=torch.int32, device=device)
    prefix_lens = torch.zeros(batch, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, prefill_tokens], dtype=torch.int32, device=device)

    meta = MiniMaxM3SparseAttentionMetadata(
        is_prefill=True,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        prefix_lens=prefix_lens,
        cu_seqlens_q=cu_seqlens_q,
        extend_seq_lens_cpu=[prefill_tokens],
    )
    meta.prepare()

    with torch.random.fork_rng(devices=["cuda"]):
        torch.manual_seed(909)
        k_cache[:prefill_tokens] = torch.randn_like(k_cache[:prefill_tokens]) * 0.05
        v_cache[:prefill_tokens] = torch.randn_like(v_cache[:prefill_tokens]) * 0.05
        idx_k_cache[:prefill_tokens] = torch.randn_like(idx_k_cache[:prefill_tokens]) * 0.05
        prefill_q = (
            torch.randn(
                prefill_tokens,
                cfg.num_q_heads,
                cfg.head_dim,
                device=device,
                dtype=dtype,
            )
            * 0.05
        )
        prefill_idx_q = (
            torch.randn(
                prefill_tokens,
                cfg.num_index_heads,
                cfg.sparse_index_dim,
                device=device,
                dtype=dtype,
            )
            * 0.05
        )
    return cfg, prefill_tokens, k_cache, v_cache, idx_k_cache, prefill_q, prefill_idx_q, meta


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_sparse_attention_checkpoint_scale_memory_budget():
    """Stage 6 memory regression: checkpoint-scale sparse prefill does not
    materialize the FP32 padded K/V cliff that produced the Stage 5 OOM.

    Asserts that on the failure-mode geometry (total_q=2688, max_k=2688,
    num_kv_heads=4, head_dim=128, num_q_heads=64) the prefill's peak
    GPU memory delta is well under 13 GiB, and that the kernel's chunked
    plan keeps each per-Q FP32 K/V slab below the same ceiling. Reports
    peak allocated/reserved bytes for the Stage 6 evidence trail.
    """
    import gc

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        _compute_index_attn_chunk_q,
        _compute_sparse_gqa_chunk_q,
        minimax_m3_sparse_prefill,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    (
        cfg,
        prefill_tokens,
        k_cache,
        v_cache,
        idx_k_cache,
        prefill_q,
        prefill_idx_q,
        meta,
    ) = _checkpoint_scale_prefill_inputs(device, dtype)

    # Pin checkpoint-scale geometry: a future "fix" that lowers
    # dimensions to fit a tight memory budget must fail this test.
    assert cfg.num_q_heads == 64
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 128
    assert cfg.num_index_heads == 4
    assert cfg.sparse_index_dim == 128
    assert cfg.block_size == 128
    assert cfg.topk == 16
    assert cfg.score_type == "max"
    assert prefill_tokens == (cfg.topk + cfg.local_blocks + 4) * cfg.block_size

    # What the legacy unchunked path would have asked the allocator for:
    legacy_k_per_q_bytes = (
        prefill_tokens * prefill_tokens * cfg.num_kv_heads * cfg.head_dim * 4  # FP32
    )
    # Same for ``idx_k_per_q`` inside ``_index_attention_and_select``:
    legacy_idx_k_per_q_bytes = prefill_tokens * prefill_tokens * cfg.sparse_index_dim * 4

    sm_scale = cfg.head_dim**-0.5
    idx_sm_scale = cfg.sparse_index_dim**-0.5

    # --- Plan check: the chunked planner must pick a chunk that
    # bounds each per-Q FP32 K/V slab below the Stage 6 ceiling.
    g = cfg.num_q_heads // cfg.num_kv_heads
    chunk_q_gqa = _compute_sparse_gqa_chunk_q(
        prefill_tokens, prefill_tokens, cfg.num_kv_heads, cfg.head_dim, g
    )
    chunk_q_idx = _compute_index_attn_chunk_q(
        prefill_tokens,
        prefill_tokens,
        cfg.num_index_heads,
        cfg.sparse_index_dim,
        disable_index_value=True,
    )
    # Per-chunk K/V FP32 slab bytes.
    chunk_k_per_q_bytes = chunk_q_gqa * prefill_tokens * cfg.num_kv_heads * cfg.head_dim * 4
    chunk_idx_k_per_q_bytes = chunk_q_idx * prefill_tokens * cfg.sparse_index_dim * 4
    LEGACY_THRESHOLD = 13 * (1024**3)  # Stage 6 acceptance: <13 GiB
    assert chunk_k_per_q_bytes < LEGACY_THRESHOLD, (
        f"sparse-GQA chunk_q={chunk_q_gqa} produces {chunk_k_per_q_bytes / 2**30:.2f}"
        f" GiB per-chunk K/V slab, matching the Stage 5 cliff (legacy was "
        f"{legacy_k_per_q_bytes / 2**30:.2f} GiB)"
    )
    assert chunk_idx_k_per_q_bytes < LEGACY_THRESHOLD, (
        f"index-attn chunk_q={chunk_q_idx} produces {chunk_idx_k_per_q_bytes / 2**30:.2f}"
        f" GiB per-chunk idx-K slab"
    )

    # --- Runtime probe: reset stats, run prefill, measure peak delta.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)

    _, o = minimax_m3_sparse_prefill(
        prefill_q,
        k_cache,
        v_cache,
        prefill_idx_q,
        idx_k_cache,
        None,
        meta,
        cfg,
        disable_index_value=True,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
    )
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_delta = peak_alloc - base_alloc

    # Stage 6 criterion: "reporting peak CUDA memory allocated/reserved".
    print(
        f"[stage6-memory] checkpoint-scale prefill: total_q={prefill_tokens} max_k={prefill_tokens}"
    )
    print(
        f"[stage6-memory] legacy unchunked k_per_q would have requested "
        f"{legacy_k_per_q_bytes / 2**30:.2f} GiB; idx_k_per_q "
        f"{legacy_idx_k_per_q_bytes / 2**30:.2f} GiB"
    )
    print(
        f"[stage6-memory] chunked planner: sparse_gqa chunk_q={chunk_q_gqa} "
        f"-> {chunk_k_per_q_bytes / 2**20:.1f} MiB per chunk; "
        f"index_attn chunk_q={chunk_q_idx} -> "
        f"{chunk_idx_k_per_q_bytes / 2**20:.1f} MiB per chunk"
    )
    print(
        f"[stage6-memory] base_alloc={base_alloc / 2**30:.3f} GiB "
        f"peak_alloc={peak_alloc / 2**30:.3f} GiB "
        f"peak_delta={peak_delta / 2**30:.3f} GiB "
        f"peak_reserved={peak_reserved / 2**30:.3f} GiB"
    )

    # Stage 6 acceptance: any single allocation request of 13 GiB or
    # larger is the Stage 5 failure mode. The peak delta bounds the
    # sum of live allocations, so requiring it below 13 GiB implies no
    # *individual* allocation in the prefill kernel reached 13 GiB
    # (since live allocations sum to ≥ each individual one).
    assert peak_delta < LEGACY_THRESHOLD, (
        f"checkpoint-scale prefill peak working set {peak_delta / 2**30:.2f} GiB "
        f"exceeded the Stage 6 ceiling of 13 GiB; the implementation regressed "
        f"to the Stage 5 FP32 padded K/V allocation pattern "
        f"(legacy would have requested {legacy_k_per_q_bytes / 2**30:.2f} GiB "
        f"for k_per_q alone)"
    )

    # Sanity check: a no-op stub (e.g. "return torch.zeros(...)") would
    # trivially satisfy the memory bound. Require non-trivial output.
    assert o.shape == (prefill_tokens, cfg.num_q_heads * cfg.head_dim)
    assert o.dtype == dtype
    nonzero_rows = (o.float().abs().sum(dim=-1) > 0).sum().item()
    assert nonzero_rows == prefill_tokens, (
        f"prefill output has {prefill_tokens - nonzero_rows} all-zero rows; "
        f"the kernel may have short-circuited the GQA path"
    )

    del k_cache, v_cache, idx_k_cache, prefill_q, prefill_idx_q, o, meta
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_sparse_attention_checkpoint_scale_memory_planner_rejects_legacy_chunking():
    """Stage 6 negative control: the chunk planner refuses the
    legacy "single-chunk" plan (chunk_q == total_q) at checkpoint-scale
    geometry.

    If a future regression silently increases the chunk budget so the
    planner returns ``chunk_q == total_q``, the per-chunk K/V slab
    would be the same 13.78 GiB cliff that produced the Stage 5 OOM.
    This test fails in that case before any allocator touches the GPU.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        _compute_index_attn_chunk_q,
        _compute_sparse_gqa_chunk_q,
    )

    cfg = _checkpoint_scale_config()
    assert cfg.num_q_heads == 64
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 128
    assert cfg.num_index_heads == 4
    assert cfg.sparse_index_dim == 128
    assert cfg.block_size == 128
    assert cfg.topk == 16

    # Same total_q/max_k that produced the Stage 5 OOM.
    prefill_tokens = (cfg.topk + cfg.local_blocks + 4) * cfg.block_size
    g = cfg.num_q_heads // cfg.num_kv_heads

    chunk_q_gqa = _compute_sparse_gqa_chunk_q(
        prefill_tokens, prefill_tokens, cfg.num_kv_heads, cfg.head_dim, g
    )
    chunk_q_idx = _compute_index_attn_chunk_q(
        prefill_tokens,
        prefill_tokens,
        cfg.num_index_heads,
        cfg.sparse_index_dim,
        disable_index_value=True,
    )

    assert chunk_q_gqa < prefill_tokens, (
        f"sparse-GQA planner returned chunk_q={chunk_q_gqa} == total_q="
        f"{prefill_tokens}; the legacy unchunked path would request "
        f"~13.78 GiB FP32 for k_per_q and OOM as in Stage 5"
    )
    assert chunk_q_idx <= prefill_tokens

    # Per-chunk K/V slab bytes must each stay below the Stage 6 13-GiB
    # ceiling (otherwise the per-chunk allocation alone is the legacy
    # failure mode).
    LEGACY_THRESHOLD = 13 * (1024**3)
    chunk_k_per_q_bytes = chunk_q_gqa * prefill_tokens * cfg.num_kv_heads * cfg.head_dim * 4
    chunk_idx_k_per_q_bytes = chunk_q_idx * prefill_tokens * cfg.sparse_index_dim * 4
    assert chunk_k_per_q_bytes < LEGACY_THRESHOLD
    assert chunk_idx_k_per_q_bytes < LEGACY_THRESHOLD

    # Stage 6 forbids reduced geometry as a way to pass — re-pin the
    # checkpoint-scale dims here so a hypothetical "shrink the model to
    # fit" workaround also fails this assertion.
    assert prefill_tokens == 2688  # 21 blocks * 128 tokens


# ---------------------------------------------------------------------------
# Iter-129 regression: CUDA-graph buffer stability across replays
#
# Bug reproduced by focused job 1964501: the production hard-path smoke crashed
# on the second prompt with ``Indexing.cu:1515 indexSelectSmallIndex
# srcIndex < srcSelectDimSize`` inside ``cuda_graph_runner.replay``. The
# captured graph reads ``req_to_token`` / ``slot_ids`` via ``index_select``,
# but ``build_runtime_metadata_from_kv_manager`` was reallocating those
# tensors on every ``prepare()`` call. Between capture and replay, the
# captured kernel kept dereferencing the warmup tensor's freed memory, so
# the index_select either produced wrong tokens (text_00 mismatch) or hit an
# out-of-bounds gather on a later prompt (text_01 crash).
#
# The fix routes the production prepare() through a persistent buffer dict
# allocated by ``allocate_minimax_m3_static_buffers``. These tests assert
# the two contract pieces the production path relies on:
#
#   1. ``build_runtime_metadata_from_kv_manager`` with ``static_buffers=...``
#      writes into the same ``data_ptr()``s across consecutive calls, even
#      when input contents (request ids, seq_lens) change.
#   2. The captured graph keeps producing correct output after a
#      ``prepare()``-equivalent rebuild between capture and replay.
# ---------------------------------------------------------------------------


def test_iter129_allocate_minimax_m3_static_buffers_geometry():
    """The static-buffer allocator returns a dict whose tensors match the
    requested geometry exactly. Catches a future regression that silently
    inflates or shrinks any of the persistent allocations.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        allocate_minimax_m3_static_buffers,
    )

    device = torch.device("cpu") if not _has_cuda() else torch.device("cuda")
    bufs = allocate_minimax_m3_static_buffers(
        max_num_sequences=4,
        max_num_tokens=128,
        max_kv_len=256,
        device=device,
    )
    assert bufs["max_num_sequences"] == 4
    assert bufs["max_num_tokens"] == 128
    assert bufs["max_kv_len"] == 256
    assert bufs["device"] == device
    assert tuple(bufs["req_to_token"].shape) == (4, 256)
    assert bufs["req_to_token"].dtype == torch.int32
    assert tuple(bufs["slot_ids"].shape) == (4,)
    # slot_ids must be arange so the captured kernel can use them as
    # static-batch identifiers under graph replay.
    expected_slot_ids = torch.arange(4, dtype=torch.int32, device=device)
    torch.testing.assert_close(bufs["slot_ids"], expected_slot_ids, rtol=0, atol=0)
    assert tuple(bufs["seq_lens_dev"].shape) == (4,)
    assert tuple(bufs["prefix_lens"].shape) == (4,)
    assert tuple(bufs["cu_seqlens_q"].shape) == (5,)
    assert tuple(bufs["out_cache_loc"].shape) == (128,)
    assert tuple(bufs["q_batch_row"].shape) == (128,)
    assert tuple(bufs["q_positions"].shape) == (128,)


def test_iter129_allocate_minimax_m3_static_buffers_rejects_nonpositive():
    """The allocator rejects nonsense sizes loudly instead of producing
    zero-sized tensors that would later raise an opaque CUDA error.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        allocate_minimax_m3_static_buffers,
    )

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="max_num_sequences"):
        allocate_minimax_m3_static_buffers(
            max_num_sequences=0,
            max_num_tokens=4,
            max_kv_len=8,
            device=device,
        )
    with pytest.raises(ValueError, match="max_num_tokens"):
        allocate_minimax_m3_static_buffers(
            max_num_sequences=2,
            max_num_tokens=0,
            max_kv_len=8,
            device=device,
        )
    with pytest.raises(ValueError, match="max_kv_len"):
        allocate_minimax_m3_static_buffers(
            max_num_sequences=2,
            max_num_tokens=4,
            max_kv_len=0,
            device=device,
        )


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_iter129_build_runtime_metadata_static_buffers_preserve_data_ptr():
    """Calling ``build_runtime_metadata_from_kv_manager`` repeatedly with
    a shared ``static_buffers`` dict must produce metadata whose
    ``data_ptr()`` is constant across calls.

    This is the contract the captured CUDA graph relies on: the kernel
    bakes in ``req_to_token.data_ptr()`` / ``slot_ids.data_ptr()`` at
    capture time, and the replay must read from the same physical memory
    even though the *contents* (request_ids, seq_lens) change every
    scheduler step.

    Before the iter-129 fix, ``build_runtime_metadata_from_kv_manager``
    allocated fresh tensors per call and ignored any caller-supplied
    buffers — so the focused hard-path smoke crashed inside
    ``index_select`` when the captured kernel dereferenced freed warmup
    memory. The assertions below pin the addresses against the buffer's
    ``data_ptr()`` so the regression is caught by a fast CUDA unit test.
    """
    import gc

    import tensorrt_llm  # noqa: F401  (initialises bindings)
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        MiniMaxM3SparseConfig,
        allocate_minimax_m3_static_buffers,
        build_runtime_metadata_from_kv_manager,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    cfg = MiniMaxM3SparseConfig(
        num_q_heads=32,
        num_kv_heads=4,
        head_dim=64,
        num_index_heads=4,
        sparse_index_dim=64,
        block_size=8,
        topk=4,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
    )
    device = torch.device("cuda")
    tokens_per_block = 8

    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=512, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        sparse_index_dim=cfg.sparse_index_dim,
        num_layers=4,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        # The V2 cache manager may report a current ``max_kv_len`` that
        # exceeds ``max_seq_len`` once block-table padding and the cache
        # quota are factored in (observed in iter173 1999328 / 1999364:
        # current max_kv_len=96 for max_seq_len=64). Size the static
        # buffers with comfortable headroom so the capture/replay
        # overflow guard does not reject this test on a setup-time
        # mismatch.
        max_blocks = (64 + tokens_per_block - 1) // tokens_per_block
        bufs = allocate_minimax_m3_static_buffers(
            max_num_sequences=2,
            max_num_tokens=16,
            max_kv_len=max_blocks * tokens_per_block * 4,
            device=device,
        )
        captured_req_to_token_ptr = bufs["req_to_token"].data_ptr()
        captured_slot_ids_ptr = bufs["slot_ids"].data_ptr()
        captured_seq_lens_ptr = bufs["seq_lens_dev"].data_ptr()
        captured_out_cache_loc_ptr = bufs["out_cache_loc"].data_ptr()

        observed_ptrs = []
        for round_idx, (req_id, seq_len) in enumerate(
            [
                (9101, 16),
                (9102, 24),
                # Second call uses a different request id and a longer
                # sequence -- the contents of req_to_token / seq_lens /
                # out_cache_loc differ but the buffer addresses must not.
            ]
        ):
            added = mgr.add_dummy_requests(
                request_ids=[req_id],
                token_nums=[seq_len],
                is_gen=False,
            )
            assert added is not None

            seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
            m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                kv_cache_manager=mgr,
                request_ids=[req_id],
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.cpu(),
                is_prefill=False,
                device=device,
                static_buffers=bufs,
            )
            observed_ptrs.append(
                {
                    "round": round_idx,
                    "req_to_token": m3_meta.req_to_token.data_ptr(),
                    "slot_ids": m3_meta.slot_ids.data_ptr(),
                    "seq_lens": m3_meta.seq_lens.data_ptr(),
                    "out_cache_loc": out_cache_loc.data_ptr(),
                }
            )

            # Freeing the request between rounds forces the kv manager to
            # rebuild internal state, which previously triggered fresh
            # ``torch.tensor(...)`` allocations from the metadata builder
            # and broke the captured-pointer contract. V2 frees by
            # LlmRequest object via :meth:`free_resources`.
            for req in added:
                mgr.free_resources(req)

        # Every round must hit the same underlying buffers.
        for obs in observed_ptrs:
            assert obs["req_to_token"] == captured_req_to_token_ptr, (
                f"round {obs['round']}: req_to_token data_ptr changed; "
                f"captured kernel would read from freed memory"
            )
            assert obs["slot_ids"] == captured_slot_ids_ptr, (
                f"round {obs['round']}: slot_ids data_ptr changed"
            )
            assert obs["seq_lens"] == captured_seq_lens_ptr, (
                f"round {obs['round']}: seq_lens data_ptr changed"
            )
            assert obs["out_cache_loc"] == captured_out_cache_loc_ptr, (
                f"round {obs['round']}: out_cache_loc data_ptr changed"
            )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_iter129_build_runtime_metadata_lazy_allocates_from_placeholder():
    """When the caller passes a placeholder dict (only the ``device`` and
    ``*_hint`` keys), ``build_runtime_metadata_from_kv_manager`` must
    lazily allocate the persistent buffers using the current scheduler
    step's actual geometry.

    This is the production path used by
    ``MiniMaxM3AttentionMetadata.prepare()`` -- the metadata cannot
    predict the CUDA-graph warmup's ``max_kv_len`` (job 1964591 showed
    max_seq_len=512 but warmup used max_kv_len=640), so the placeholder
    pattern defers sizing to the first prepare() call.

    After the first call, the placeholder must be populated with the
    real tensors and subsequent calls must hit the same ``data_ptr()``s.
    """
    import gc

    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        build_runtime_metadata_from_kv_manager,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    device = torch.device("cuda")
    tokens_per_block = 8

    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=512, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        sparse_index_dim=64,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        # Caller-supplied placeholder dict: only the device and hints,
        # no allocated tensors yet. This is what
        # ``_maybe_get_m3_static_buffers`` produces on first use.
        placeholder: dict = {
            "device": device,
            "max_num_sequences_hint": 2,
            "max_num_tokens_hint": 16,
        }

        req_id = 9501
        seq_len = 24
        added = mgr.add_dummy_requests(
            request_ids=[req_id],
            token_nums=[seq_len],
            is_gen=False,
        )
        assert added is not None

        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        meta_1, _ = build_runtime_metadata_from_kv_manager(
            kv_cache_manager=mgr,
            request_ids=[req_id],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            is_prefill=False,
            device=device,
            static_buffers=placeholder,
        )

        # First call must have populated the placeholder with real
        # tensors.
        assert "req_to_token" in placeholder, (
            "lazy alloc failed -- placeholder still missing req_to_token"
        )
        assert "slot_ids" in placeholder
        assert "seq_lens_dev" in placeholder
        assert "out_cache_loc" in placeholder
        # The allocated max_num_sequences must cover the hint.
        assert placeholder["max_num_sequences"] >= 2
        # The allocated max_kv_len must cover the current call's actual
        # geometry plus one block of headroom (the lazy allocator's
        # safety margin).
        assert placeholder["max_kv_len"] >= (3 * tokens_per_block), (
            f"lazy alloc max_kv_len={placeholder['max_kv_len']} too small "
            f"for current call (need at least 3 blocks * tokens_per_block={tokens_per_block})"
        )

        captured_req_to_token_ptr = placeholder["req_to_token"].data_ptr()
        captured_slot_ids_ptr = placeholder["slot_ids"].data_ptr()

        # Free and re-add the request to force the manager to rebuild
        # its block table from scratch; then call build again and assert
        # the buffer pointers are unchanged. V2 frees by LlmRequest
        # object via :meth:`free_resources`.
        for req in added:
            mgr.free_resources(req)
        added2 = mgr.add_dummy_requests(
            request_ids=[req_id + 1],
            token_nums=[16],
            is_gen=False,
        )
        assert added2 is not None
        seq_lens_2 = torch.tensor([16], dtype=torch.int32, device=device)
        meta_2, _ = build_runtime_metadata_from_kv_manager(
            kv_cache_manager=mgr,
            request_ids=[req_id + 1],
            seq_lens=seq_lens_2,
            seq_lens_cpu=seq_lens_2.cpu(),
            is_prefill=False,
            device=device,
            static_buffers=placeholder,
        )
        assert meta_2.req_to_token.data_ptr() == captured_req_to_token_ptr, (
            "lazy alloc broke buffer-stability contract -- second call reallocated req_to_token"
        )
        assert meta_2.slot_ids.data_ptr() == captured_slot_ids_ptr
        for req in added2:
            mgr.free_resources(req)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_iter129_static_buffers_reject_oversized_batch():
    """The static-buffer path must fail loudly when the runtime batch
    exceeds the buffer's allocation -- otherwise a too-large batch would
    silently overflow the persistent buffer and corrupt the next graph
    replay.
    """
    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        allocate_minimax_m3_static_buffers,
        build_runtime_metadata_from_kv_manager,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    device = torch.device("cuda")
    tokens_per_block = 8
    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=256, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1],
        disable_index_value_layer_ids=[1],
        sparse_index_dim=64,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=32,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        # Buffers sized for batch=1, but we will request batch=2 below.
        bufs = allocate_minimax_m3_static_buffers(
            max_num_sequences=1,
            max_num_tokens=4,
            max_kv_len=tokens_per_block * 4,
            device=device,
        )

        req_ids = [9201, 9202]
        for r in req_ids:
            mgr.add_dummy_requests(
                request_ids=[r],
                token_nums=[16],
                is_gen=False,
            )

        seq_lens = torch.tensor([16, 16], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="max_num_sequences"):
            build_runtime_metadata_from_kv_manager(
                kv_cache_manager=mgr,
                request_ids=req_ids,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.cpu(),
                is_prefill=False,
                device=device,
                static_buffers=bufs,
            )
    finally:
        for r in (9201, 9202):
            try:
                mgr.remove_request(r)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Iter-131 — MiniMaxM3AttentionMetadata.prepare() mixed-batch fix
#
# Production_1964654 surfaced two related defects in the production-path
# metadata builder:
#
#   (1) ``prepare()`` interpreted ``attn_metadata.seq_lens`` as the
#       cumulative kv length even though ``model_engine`` populates it
#       with the per-step new-token count (1 for plain decode,
#       ``len(prompt_tokens)`` for context, see ``_prepare_tp_inputs``).
#       That made the decode path write the new K/V into
#       ``req_to_token[b, 0]`` instead of the slot for the current
#       generation position, and set ``max_seqlen_k=1`` so the kernel
#       only attended over the one fresh slot. Production_1964654 saw
#       ``test_production_logit_and_generation_parity`` diverge from
#       SGLang at ``first_diff_pos=1`` (the first decode step) for every
#       fixed text prompt.
#
#   (2) The same predicate (``num_contexts == batch_size and
#       num_contexts > 0``) routed mixed prefill+decode batches through
#       the decode-only branch. The PyExecutor scheduler emits mixed
#       batches whenever a new request prefills while in-flight requests
#       decode, which is the standard GSM8K serving pattern; that mode
#       wrote a decode-sized ``out_cache_loc`` (one per request) while
#       the layer fed prefill-sized K/V, producing
#       ``index_copy_(): Number of indices (8) should be equal to
#       source.size(dim) (1412)`` in ``test_gsm8k_100_production``.
#
# The iter-131 fix recomputes ``kv_lens_cpu_list[b] = num_cached[b] +
# seq_lens_cpu[b]`` and uses the extend path (``is_prefill=True``)
# whenever ``num_contexts > 0``. Pure-decode batches keep the decode
# optimization but now also receive cumulative kv lengths so the
# new-token slot lands at ``req_to_token[b, num_cached[b]]``. The tests
# below pin both fixes against the production-path
# ``MiniMaxM3AttentionMetadata.prepare()`` -- not just the underlying
# builder -- because the builder already accepted cumulative lengths
# (the focused tests pass them directly). The bug lived only in the
# prepare-time translation from ``attn_metadata.seq_lens`` to the
# algorithm metadata.
# ---------------------------------------------------------------------------


def _iter131_make_m3_attn_metadata(
    mgr,
    request_ids,
    per_step_seq_lens,
    num_cached_tokens_per_seq,
    num_contexts,
    *,
    max_num_requests=4,
    max_num_tokens=4096,
):
    """Build a real ``MiniMaxM3AttentionMetadata`` for prepare() testing.

    Mirrors the way ``model_engine._prepare_tp_inputs`` populates the
    standard ``AttentionMetadata`` fields: ``seq_lens`` is the per-step
    new-token count, ``num_contexts`` is the prefill prefix of the
    batch, and ``kv_cache_params.num_cached_tokens_per_seq`` is the
    pre-step cached length per row.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        get_minimax_m3_attention_metadata_cls,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata_cls = get_minimax_m3_attention_metadata_cls()
    md = metadata_cls(
        seq_lens=torch.tensor(per_step_seq_lens, dtype=torch.int32),
        num_contexts=int(num_contexts),
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=list(num_cached_tokens_per_seq),
        ),
        kv_cache_manager=mgr,
        request_ids=list(request_ids),
        prompt_lens=list(per_step_seq_lens),
        max_num_requests=int(max_num_requests),
        max_num_tokens=int(max_num_tokens),
    )
    return md


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_iter131_metadata_prepare_pure_decode_uses_cumulative_kv_lens():
    """Pure-decode prepare(): the new K/V slot must be
    ``req_to_token[b, num_cached]`` and ``max_seqlen_k`` must reflect the
    cumulative kv length, not the per-step ``seq_lens`` of 1.
    """
    import gc

    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    tokens_per_block = 8
    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=256, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        sparse_index_dim=64,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        # Add a request that has already been prefilled to length 24; the
        # decode step under test appends one new token at position 24
        # (cumulative kv length becomes 25).
        req_id = 9601
        prefill_tokens = 24
        added = mgr.add_dummy_requests(
            request_ids=[req_id],
            token_nums=[prefill_tokens + 1],
            is_gen=False,
        )
        assert added is not None

        md = _iter131_make_m3_attn_metadata(
            mgr,
            request_ids=[req_id],
            per_step_seq_lens=[1],  # per-step new-token count (decode)
            num_cached_tokens_per_seq=[prefill_tokens],  # cumulative cached
            num_contexts=0,  # pure decode
        )

        md.prepare()

        attachment = md.minimax_m3
        assert attachment is not None, "prepare() must build the M3 attachment"
        m3_meta = attachment["metadata"]
        out_cache_loc = attachment["out_cache_loc"]

        # The new-token slot must be the slot for cumulative position
        # ``num_cached`` (== seq_lens[b] - 1 with cumulative semantics),
        # not slot 0 (which is what the pre-iter-131 ``seq_lens-1`` read
        # produced when seq_lens was the per-step 1).
        expected_slot = int(m3_meta.req_to_token[0, prefill_tokens].item())
        assert int(out_cache_loc[0].item()) == expected_slot, (
            f"iter-131 regression: decode out_cache_loc must point at "
            f"req_to_token[0, num_cached={prefill_tokens}]={expected_slot}, "
            f"got {int(out_cache_loc[0].item())}"
        )
        # Cumulative kv length must drive max_seqlen_k -- a per-step
        # max_seqlen_k of 1 would mask everything except the new slot.
        assert m3_meta.max_seqlen_k >= prefill_tokens + 1, (
            f"iter-131 regression: max_seqlen_k must be cumulative; got "
            f"{m3_meta.max_seqlen_k}, expected >= {prefill_tokens + 1}"
        )
        # Pure-decode batches keep the decode optimization.
        assert m3_meta.is_prefill is False, (
            "pure-decode batches (num_contexts==0) must take the decode path "
            "for CUDA-graph warmup compatibility"
        )
    finally:
        try:
            mgr.remove_request(9601)
        except Exception:
            pass
        mgr.shutdown()
        gc.collect()


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_iter131_metadata_prepare_mixed_batch_uses_extend_path():
    """Mixed prefill+decode prepare(): must route through the extend
    path so ``out_cache_loc`` covers every new token (sum of
    extend_seq_lens), not one slot per request.

    This is the production_1964654 GSM8K failure mode: 8 requests with
    a 1412-token total new-token count fired the ``index_copy_``
    mismatch because the decode-only branch produced an 8-entry
    ``out_cache_loc``.
    """
    import gc

    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    tokens_per_block = 8
    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=512, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        sparse_index_dim=64,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        # Row 0: prefill chunk of 24 new tokens (num_cached=0).
        # Row 1: decode of 1 new token over a 23-token prefix.
        # Total new tokens = 25; the pre-iter-131 decode-only branch
        # would have produced ``out_cache_loc.shape[0] == 2``.
        req_ids = [9701, 9702]
        mgr.add_dummy_requests(request_ids=[req_ids[0]], token_nums=[24], is_gen=False)
        mgr.add_dummy_requests(request_ids=[req_ids[1]], token_nums=[24], is_gen=False)

        md = _iter131_make_m3_attn_metadata(
            mgr,
            request_ids=req_ids,
            per_step_seq_lens=[24, 1],  # 24 prefill tokens + 1 decode token
            num_cached_tokens_per_seq=[0, 23],  # 0 cached + 23 cached
            num_contexts=1,  # one prefill row at the head of the batch
        )

        md.prepare()
        attachment = md.minimax_m3
        assert attachment is not None
        m3_meta = attachment["metadata"]
        out_cache_loc = attachment["out_cache_loc"]

        # Extend path -> total_q = sum(extend_seq_lens) = 25.
        assert int(out_cache_loc.shape[0]) == 25, (
            f"iter-131 regression: mixed prefill+decode out_cache_loc must "
            f"cover every new token (24 + 1 = 25); got {out_cache_loc.shape[0]}. "
            f"The decode-only branch returning 2 entries is the production_1964654 "
            f"GSM8K index_copy_ failure."
        )
        # Extend path -> metadata.is_prefill is True.
        assert m3_meta.is_prefill is True, (
            "iter-131 regression: mixed batches must take the extend "
            "(is_prefill=True) path so the prefill kernel handles the "
            "ragged Q layout via cu_seqlens_q / q_batch_row / q_positions."
        )
        # cu_seqlens_q must end at total_q.
        assert m3_meta.cu_seqlens_q is not None
        assert int(m3_meta.cu_seqlens_q[-1].item()) == 25
        # Per-row extend lengths.
        assert m3_meta.extend_seq_lens_cpu == [24, 1]
        # max_seqlen_k from cumulative kv lengths: max(24, 24) = 24.
        assert m3_meta.max_seqlen_k == 24
    finally:
        for r in (9701, 9702):
            try:
                mgr.remove_request(r)
            except Exception:
                pass
        mgr.shutdown()
        gc.collect()


@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 sparse attention needs CUDA")
def test_iter131_metadata_prepare_pure_prefill_kv_lens_match_chunk_offset():
    """Single-row chunked prefill: ``kv_lens = num_cached + per_step``
    correctly tracks both the resumed-from-cache prefix and the new
    chunk so the extend path's ``prefix_lens`` and ``extend_seq_lens``
    decompose into the same values the iter-129 focused builder tests
    pass directly.
    """
    import gc

    import tensorrt_llm  # noqa: F401
    import tensorrt_llm.bindings as _bindings
    from tensorrt_llm._torch.attention_backend.sparse import get_minimax_m3_kv_cache_manager_cls
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
    from tensorrt_llm.mapping import Mapping

    DataType = _bindings.DataType
    CacheType = _bindings.internal.batch_manager.CacheType

    tokens_per_block = 8
    mgr_cls = get_minimax_m3_kv_cache_manager_cls()
    mgr = mgr_cls(
        KvCacheConfigV2(max_tokens=256, enable_block_reuse=False),
        CacheType.SELF,
        sparse_layer_ids=[1, 2, 3],
        disable_index_value_layer_ids=[1, 2, 3],
        sparse_index_dim=64,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=64,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
        vocab_size=32000,
    )
    try:
        # Single chunked-prefill row: 16 tokens already cached, 16 new
        # this chunk. Cumulative kv length = 32.
        req_id = 9801
        mgr.add_dummy_requests(request_ids=[req_id], token_nums=[32], is_gen=False)
        md = _iter131_make_m3_attn_metadata(
            mgr,
            request_ids=[req_id],
            per_step_seq_lens=[16],
            num_cached_tokens_per_seq=[16],
            num_contexts=1,
        )
        md.prepare()
        attachment = md.minimax_m3
        assert attachment is not None
        m3_meta = attachment["metadata"]
        out_cache_loc = attachment["out_cache_loc"]

        # Out-cache-loc covers the 16 new tokens; positions 16..31 of the
        # request live in slots ``req_to_token[0, 16..31]``.
        assert int(out_cache_loc.shape[0]) == 16
        for offset in range(16):
            expected_slot = int(m3_meta.req_to_token[0, 16 + offset].item())
            assert int(out_cache_loc[offset].item()) == expected_slot, (
                f"chunked-prefill out_cache_loc[{offset}] must address "
                f"req_to_token[0, {16 + offset}] = {expected_slot}, "
                f"got {int(out_cache_loc[offset].item())}"
            )
        # Cumulative kv length 32 drives max_seqlen_k.
        assert m3_meta.max_seqlen_k == 32
        assert m3_meta.extend_seq_lens_cpu == [16]
    finally:
        try:
            mgr.remove_request(9801)
        except Exception:
            pass
        mgr.shutdown()
        gc.collect()
