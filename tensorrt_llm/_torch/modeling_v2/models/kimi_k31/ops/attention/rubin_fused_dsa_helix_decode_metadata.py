# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA decode metadata of a Helix CP generation step in one launch (helix8).

Request index per token, the rank-local index-key slots (-1 where another CP rank owns the token),
the generation indptrs, the 2D indexer KV lengths and the per-token verify bounds, for a step of
uniform verify blocks with Helix bounds. The family kernel replaces the eager chain and the Helix
ownership rewrite of the engine's ``on_update_kv_lens``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper

from ...kernels.attention.rubin_dsa_kernels import fused_dsa_helix_decode_metadata


def _spec(batch, next_n, kv_lens, tokens_per_block=64, **overrides) -> dict:
    """``batch`` generation requests of ``next_n`` verify tokens on a CP8 rank's pages."""
    spec = dict(
        batch=batch, next_n=next_n, kv_lens=tuple(kv_lens), tokens_per_block=tokens_per_block, cp=8
    )
    spec.update(overrides)
    return spec


class RubinFusedDsaHelixDecodeMetadata(OpWrapper):
    """For ``S`` generation requests of ``seq_lens[r]`` tokens and ``T = num_tokens`` tokens,
    writes in place: ``req_idx_per_token[:T]`` (int32, the request of token ``t``);
    ``slot_mapping_fp8[:T]`` / ``slot_mapping_scale[:T]`` (int64): the byte offsets of token
    ``t``'s index key and scale at its rank-local position ``helix_local_slots[t]`` in
    ``block_offsets[req]``, or -1 where ``helix_local_slots[t] < 0``; ``gen_kv_indptr[1:S + 1]``
    and ``gen_cached_token_indptr[1:S + 1]`` (int64): the running sums of ``kv_lens`` and of
    ``kv_lens - seq_lens`` (entry 0 is not written); ``kv_lens_2d[:S, :width]``: ``kv_lens``
    broadcast; ``kv_lens_expanded[:T]``: ``max(helix_kv_bounds, 1)``. Returns None.
    """

    # No architecture listed until a GPU run drives the cells on sm_107.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="M: DFlash2 k5 verify at batch 16 (6 tokens each)", spec=_spec(16, 6, (64, 131072))
        ),
        Cell(why="one request, one page", spec=_spec(1, 6, (6, 64))),
        Cell(why="k7 verify at batch 12, tokens across page edges", spec=_spec(12, 8, (60, 9000))),
        Cell(why="plain decode under Helix (one token each)", spec=_spec(64, 1, (1, 200000))),
    )
    note = """
    helix8's DSA metadata calls this from on_update_kv_lens on a generation step with uniform
    verify blocks and Helix bounds; the decode schedules follow through get_paged_mqa_logits_metadata.
    Integer arithmetic only, so the outputs are bitwise those of the eager chain.
    """

    def __call__(
        self,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        helix_local_slots: torch.Tensor,
        helix_kv_bounds: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        gen_kv_indptr: torch.Tensor,
        gen_cached_token_indptr: torch.Tensor,
        kv_lens_2d: torch.Tensor,
        kv_lens_expanded: torch.Tensor,
        num_tokens: int,
        width: int,
        tokens_per_block: int,
        index_head_dim: int,
        quant_block_size: int,
        data_bytes_per_token: int,
    ) -> None:
        fused_dsa_helix_decode_metadata(
            seq_lens,
            kv_lens,
            block_offsets,
            helix_local_slots,
            helix_kv_bounds,
            req_idx_per_token,
            slot_mapping_fp8,
            slot_mapping_scale,
            gen_kv_indptr,
            gen_cached_token_indptr,
            kv_lens_2d,
            kv_lens_expanded,
            num_tokens=num_tokens,
            width=width,
            tokens_per_block=tokens_per_block,
            index_head_dim=index_head_dim,
            quant_block_size=quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

    def reference(
        self,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        helix_local_slots: torch.Tensor,
        helix_kv_bounds: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        gen_kv_indptr: torch.Tensor,
        gen_cached_token_indptr: torch.Tensor,
        kv_lens_2d: torch.Tensor,
        kv_lens_expanded: torch.Tensor,
        num_tokens: int,
        width: int,
        tokens_per_block: int,
        index_head_dim: int,
        quant_block_size: int,
        data_bytes_per_token: int,
    ) -> None:
        """The eager chain and the Helix ownership rewrite, into the same outputs."""
        device = seq_lens.device
        scale_size = index_head_dim // quant_block_size * 4
        block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
        scale_base = tokens_per_block * data_bytes_per_token
        num_seqs = seq_lens.shape[0]
        cu_seq = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        token = torch.arange(num_tokens, device=device, dtype=torch.int32)
        req = torch.searchsorted(cu_seq, token, right=True)
        req_idx_per_token[:num_tokens] = req.to(req_idx_per_token.dtype)
        local = helix_local_slots[:num_tokens].long()
        owned = local >= 0
        position = local.clamp(min=0)
        page = (position // tokens_per_block).clamp(max=block_offsets.shape[1] - 1)
        in_page = position % tokens_per_block
        block = block_offsets[req.long(), page].long()
        fp8 = block * block_stride + in_page * data_bytes_per_token
        scale = block * block_stride + scale_base + in_page * scale_size
        slot_mapping_fp8[:num_tokens] = torch.where(owned, fp8, -1)
        slot_mapping_scale[:num_tokens] = torch.where(owned, scale, -1)
        torch.cumsum(kv_lens.long(), dim=0, out=gen_kv_indptr[1 : num_seqs + 1])
        torch.cumsum(
            kv_lens.long() - seq_lens.long(), dim=0, out=gen_cached_token_indptr[1 : num_seqs + 1]
        )
        kv_lens_2d[:num_seqs, :width] = kv_lens.unsqueeze(1).to(kv_lens_2d.dtype)
        kv_lens_expanded[:num_tokens] = helix_kv_bounds[:num_tokens].clamp(min=1)

    def is_valid(
        self,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        helix_local_slots: torch.Tensor,
        helix_kv_bounds: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        gen_kv_indptr: torch.Tensor,
        gen_cached_token_indptr: torch.Tensor,
        kv_lens_2d: torch.Tensor,
        kv_lens_expanded: torch.Tensor,
        num_tokens: int,
        width: int,
        tokens_per_block: int,
        index_head_dim: int,
        quant_block_size: int,
        data_bytes_per_token: int,
    ) -> None:
        # Tokens past sum(seq_lens) get no request; a row's position past its table is clamped.
        assert int(seq_lens.sum()) == num_tokens, "num_tokens is not the step's token count"
        assert block_offsets.shape[0] >= seq_lens.shape[0], "the block table has too few rows"
        pages = -(-(int(helix_local_slots[:num_tokens].max()) + 1) // tokens_per_block)
        assert pages <= block_offsets.shape[1], "a local slot lies past the block table row"
        assert kv_lens_2d.shape[1] >= width, "kv_lens_2d is narrower than the verify width"


rubin_fused_dsa_helix_decode_metadata = RubinFusedDsaHelixDecodeMetadata()
