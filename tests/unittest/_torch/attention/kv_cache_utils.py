# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the unified attention-backend test suite.

These utilities bridge the *logical* per-sequence view of K/V used by tests and
references and the *physical* paged layout managed by ``KVCacheManager`` /
``KVCacheManagerV2``. They also wrap the production ``RotaryEmbedding`` so the
test reference applies RoPE identically to the non-fused backends.
"""

import functools
from typing import Sequence

import torch

from tensorrt_llm._torch.attention_backend.interface import RopeParams
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding


def fill_kv_cache_logical(
    kv_cache_manager,
    layer_idx: int,
    request_ids: Sequence[int],
    k_per_seq: Sequence[torch.Tensor],
    v_per_seq: Sequence[torch.Tensor],
    *,
    kv_layout: str = "NHD",
) -> None:
    """Write logical per-sequence cached K/V into a real KV cache manager.

    Generalizes the per-test block-copy loop in ``test_attention.py`` so callers
    never juggle paged layouts. Each ``k_per_seq[i]`` / ``v_per_seq[i]`` is the
    cached prefix of sequence ``i`` shaped ``[cached_len_i, num_kv_heads,
    head_dim]`` in compute dtype; the backend appends the new tokens itself.

    Args:
        kv_cache_manager: a real ``KVCacheManager`` / ``KVCacheManagerV2``.
        layer_idx: the layer whose pool to write.
        request_ids: request ids previously passed to ``add_dummy_requests``.
        k_per_seq, v_per_seq: per-sequence logical cached K and V.
        kv_layout: ``"NHD"`` (TRTLLM/Vanilla) or ``"HND"`` (FlashInfer). Must
            match the layout the backend reads with, since the two interpret the
            physical block memory differently.
    """
    try:
        buf = kv_cache_manager.get_buffers(layer_idx, kv_layout=kv_layout)
    except TypeError:
        # Older signature without kv_layout (defaults to NHD).
        assert kv_layout == "NHD", "manager.get_buffers does not accept kv_layout"
        buf = kv_cache_manager.get_buffers(layer_idx)

    # Zero the pool first: the freshly-allocated KV pool is uninitialized, and
    # the decode kernel reads whole blocks (masking positions beyond kv_len).
    # Masking suppresses values but not NaNs, and an uninitialized fp8 (e4m3)
    # byte can decode to NaN, which then poisons the softmax. Real runs populate
    # the cache incrementally via the kernel's own append, so they never hit
    # this; a manual prefill must initialize the unused block tail to zero.
    buf.zero_()

    # buf: NHD -> [pages, 2, tokens_per_block, num_kv_heads, head_dim]
    #      HND -> [pages, 2, num_kv_heads, tokens_per_block, head_dim]
    tokens_per_block = buf.shape[2] if kv_layout == "NHD" else buf.shape[3]
    blocks_per_req = kv_cache_manager.get_batch_cache_indices(list(request_ids), layer_idx)

    for i, blocks in enumerate(blocks_per_req):
        blocks = [b for b in blocks if b != -1]
        k_i, v_i = k_per_seq[i], v_per_seq[i]
        cached_len = k_i.shape[0]
        written = 0
        for blk in blocks:
            if written >= cached_len:
                break
            n = min(tokens_per_block, cached_len - written)
            kk = k_i[written : written + n].to(buf.dtype)
            vv = v_i[written : written + n].to(buf.dtype)
            if kv_layout == "NHD":
                buf[blk, 0, :n].copy_(kk)
                buf[blk, 1, :n].copy_(vv)
            else:  # HND: store as [num_kv_heads, n, head_dim]
                buf[blk, 0, :, :n].copy_(kk.transpose(0, 1))
                buf[blk, 1, :, :n].copy_(vv.transpose(0, 1))
            written += n


def make_position_ids(
    seq_lens: Sequence[int],
    num_cached_tokens: Sequence[int],
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Build per-token RoPE position ids for a packed (varlen) batch.

    For sequence ``i`` the query tokens occupy absolute positions
    ``[cached_i, cached_i + seq_len_i)``. Returns a 1-D int tensor of shape
    ``[sum(seq_lens)]`` matching the packed q layout used by the backends.
    """
    pieces = []
    for s_len, cached in zip(seq_lens, num_cached_tokens):
        pieces.append(torch.arange(cached, cached + s_len, dtype=torch.int32))
    return torch.cat(pieces).to(device)


@functools.lru_cache(maxsize=None)
def _rotary_embedding(rope_params: RopeParams, head_dim: int, is_neox: bool) -> RotaryEmbedding:
    # ``RopeParams`` is hashable (``unsafe_hash=True``), so this caches the
    # (expensive) cos/sin table construction across cases sharing a config.
    return RotaryEmbedding(rope_params, head_dim=head_dim, is_neox=is_neox)


def apply_rope(
    tensor: torch.Tensor,
    position_ids: torch.Tensor,
    rope_params: RopeParams,
    head_dim: int,
    *,
    is_neox: bool = True,
) -> torch.Tensor:
    """Apply RoPE to a single packed ``[num_tokens, num_heads * head_dim]`` tensor.

    Passing a single-element target list forces the pure-PyTorch rotation path in
    ``RotaryEmbedding`` (the 2-target branch dispatches to a fused FlashInfer
    in-place op), keeping the reference deterministic and side-effect free.
    """
    emb = _rotary_embedding(rope_params, head_dim, is_neox)
    return emb(position_ids, [tensor])[0]
