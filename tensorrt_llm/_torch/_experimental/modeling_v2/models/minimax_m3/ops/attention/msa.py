# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MSA dispatch and grouped-head reference; target cache readers remain bound locally."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    SparseBackendForwardArgs,
)


class MsaAttentionBase(OpWrapper):
    def __call__(
        self,
        backend,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        idx_q: Optional[torch.Tensor],
        metadata,
        output: torch.Tensor,
        sparse: bool,
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if k is not None:
            backend.write_layer_caches(
                k, v, None, metadata, kv_scale_orig_quant=kv_scale_orig_quant
            )
        if sparse:
            blocks = backend.run_indexer(idx_q, None, metadata, idx_k_prewritten=True)
            args = AttentionForwardArgs(
                output=output, sparse_backend_args=SparseBackendForwardArgs(topk_indices=blocks)
            )
        else:
            args = AttentionForwardArgs(output=output)
        args.kv_scale_orig_quant = kv_scale_orig_quant
        args.kv_scale_quant_orig = kv_scale_quant_orig
        backend.forward(q, None, None, metadata, forward_args=args)
        return output


class GroupedMsaAttentionBase(MsaAttentionBase):
    def reference(
        self,
        backend,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        idx_q: Optional[torch.Tensor],
        metadata,
        output: torch.Tensor,
        sparse: bool,
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``output`` as the call is supposed to write it, [T, heads * 128] in ``output``'s dtype.

        Every attended row -- the step's own included -- is read from the layer's paged cache, at the slots the
        metadata's KV cache manager holds for each request, so this is evaluated after the call (whose write of
        ``k`` / ``v`` it then reads back). ``k``, ``v`` and ``kv_scale_orig_quant`` are therefore not read; the
        test checks the write itself. Query token ``j`` of request ``r`` sits at position ``cached_r + j`` and
        attends positions up to it; on a sparse layer only those in its top-k blocks. Mirrors grouped KV heads (query
        head h reads KV head h // (heads / kv heads); two per rank at tp2), each KV head selecting its own top-k
        blocks from its own index heads against the rank's single index-K head.
        """
        manager = metadata.kv_cache_manager
        layer = backend.layer_idx
        heads, dim = (backend.num_heads, backend.head_dim)
        kv_heads = int(backend.num_kv_heads)
        assert heads % kv_heads == 0
        group = heads // kv_heads
        page = int(manager.tokens_per_block)
        sm_scale = dim ** (-0.5) / float(backend.q_scaling)
        num_seqs = int(metadata.num_seqs)
        qlens = [int(n) for n in metadata.seq_lens[:num_seqs].tolist()]
        cached = [int(c) for c in metadata.kv_cache_params.num_cached_tokens_per_seq[:num_seqs]]
        total = sum(qlens)
        assert q.shape[0] == total and output.shape[0] == total
        pool = manager.get_buffers(layer, "HND")
        nvfp4 = manager.is_nvfp4_layer(layer)
        if nvfp4:
            scale_pool = manager.get_block_scale_buffers(layer, "HND")
            k_global = float(kv_scale_quant_orig.reshape(-1)[1])
            v_global = float(kv_scale_quant_orig.reshape(-1)[2])
        q_rows = q.reshape(total, heads, dim)
        if q_rows.dtype != torch.float8_e4m3fn and pool.dtype == torch.float8_e4m3fn:
            q_rows = q_rows.to(torch.float8_e4m3fn)
        q64 = q_rows.to(torch.float64)
        if sparse:
            config = backend.m3_config
            idx_cache = metadata.msa_idx_k_cache(layer)
            idx64 = idx_q.reshape(total, config.num_index_heads, config.sparse_index_dim).to(
                torch.float64
            )
            assert config.num_index_heads % kv_heads == 0
            idx_group = config.num_index_heads // kv_heads
        out = torch.zeros(total, heads, dim, dtype=torch.float64, device=q.device)
        first = 0
        for row, request_id in enumerate(list(metadata.request_ids)[:num_seqs]):
            n, length = (qlens[row], cached[row] + qlens[row])
            blocks = (length + page - 1) // page
            pages = torch.tensor(
                manager.get_batch_cache_indices([request_id], layer_idx=layer)[0][:blocks],
                dtype=torch.int64,
                device=q.device,
            )
            qpos = cached[row] + torch.arange(n, device=q.device)
            causal = torch.arange(length, device=q.device)[None, :] <= qpos[:, None]
            for g in range(kv_heads):
                if nvfp4:
                    keys = self._dequant_pages(
                        pool[:, 0].view(torch.uint8),
                        scale_pool[:, 0],
                        pages,
                        quad_scales=False,
                        kv_head=g,
                    )
                    values = self._dequant_pages(
                        pool[:, 1].view(torch.uint8),
                        scale_pool[:, 1],
                        pages,
                        quad_scales=True,
                        kv_head=g,
                    )
                    keys, values = (keys * k_global, values * v_global)
                else:
                    keys = pool[pages, 0, g].to(torch.float64)
                    values = pool[pages, 1, g].to(torch.float64)
                keys = keys.reshape(-1, dim)[:length]
                values = values.reshape(-1, dim)[:length]
                allowed = causal
                if sparse:
                    idx_keys = idx_cache[pages, 0].to(torch.float64).reshape(-1, dim)[:length]
                    chosen = self._select_blocks(
                        idx64[first : first + n, g * idx_group : (g + 1) * idx_group],
                        idx_keys,
                        causal,
                        qpos,
                        page,
                        config,
                    )
                    allowed = causal & chosen.repeat_interleave(page, dim=1)[:, :length]
                h0, h1 = (g * group, (g + 1) * group)
                for lo in range(0, n, 256):
                    hi = min(n, lo + 256)
                    logits = (
                        torch.einsum("thd,kd->thk", q64[first + lo : first + hi, h0:h1], keys)
                        * sm_scale
                    )
                    logits = logits.masked_fill(~allowed[lo:hi, None, :], float("-inf"))
                    out[first + lo : first + hi, h0:h1] = torch.softmax(logits, dim=-1) @ values
            first += n
        return out.reshape(total, heads * dim).to(output.dtype)


def _selected_blocks(
    idx_q: torch.Tensor,
    idx_keys: torch.Tensor,
    allowed: torch.Tensor,
    qpos: torch.Tensor,
    page: int,
    config,
) -> torch.Tensor:
    """The blocks each query token of one request attends, [n, blocks] bool.

    A block's score is the maximum, over the KV head's index heads and the block's causally visible keys, of
    ``idx_q . idx_k``. Of the token's ``ceil((pos + 1) / page)`` valid blocks the first ``init_blocks`` and the last
    ``local_blocks`` are forced in, and the ``topk`` highest-scoring are kept, ties to the lower block (the
    backend's ``select_blocks_from_maxscore`` rule). The positive score scale the scorers apply does not change the
    ranking, so none is applied.
    """
    n, length = allowed.shape
    blocks = (length + page - 1) // page
    scores = torch.einsum("thd,kd->thk", idx_q, idx_keys).amax(dim=1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    padded = torch.full((n, blocks * page), float("-inf"), dtype=scores.dtype, device=scores.device)
    padded[:, :length] = scores
    block_scores = padded.view(n, blocks, page).amax(dim=-1)
    ids = torch.arange(blocks, device=scores.device)[None, :]
    n_valid = torch.div(qpos + page, page, rounding_mode="floor")[:, None]
    block_scores = block_scores.masked_fill(ids >= n_valid, float("-inf"))
    forced = (ids < torch.clamp(n_valid, max=config.init_blocks)) | (
        ids >= n_valid - config.local_blocks
    ) & (ids < n_valid)
    block_scores = block_scores.masked_fill(forced, float("inf"))
    order = torch.sort(block_scores, dim=1, descending=True, stable=True).indices[:, : config.topk]
    keep = torch.arange(order.shape[1], device=scores.device)[None, :] < n_valid
    chosen = torch.zeros(n, blocks, dtype=torch.bool, device=scores.device)
    rows = torch.arange(n, device=scores.device)[:, None].expand_as(order)
    chosen[rows[keep], order[keep]] = True
    return chosen
