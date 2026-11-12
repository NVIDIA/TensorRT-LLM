# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The head of a speculative decode step in one Triton launch: the overlap-scheduler correction
of position ids and KV lengths and the Helix verify-group buffers derived from the positions."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# The module, not the function: kernels.spec_decode_head imports this entry for its hook.
from ...kernels.k3r_spec import spec_decode_head as _head


def _helix_local_len(g: torch.Tensor, tokens_per_block: int, cp_size: int, cp_rank: int):
    """Rank-local prefix length of the first ``g`` global tokens of a Helix-sharded sequence."""
    ledger = tokens_per_block * cp_size
    full = torch.div(g, ledger, rounding_mode="floor")
    rest = g - full * ledger - cp_rank * tokens_per_block
    return full * tokens_per_block + rest.clamp(min=0, max=tokens_per_block)


class K3SpecDecodeHead(OpWrapper):
    """In place, returns None. With ``has_overlap``: ``position_ids[:num_pos] +=
    pos_offsets[:num_pos]`` and ``kv_lens[kv_start : kv_start + num_kv] += kv_offsets[:num_kv]``.
    With ``num_helix > 0``, after ``helix_positions[:num_helix] += pos_offsets[:num_helix]``
    (with ``has_overlap`` only), what ``TrtllmAttentionMetadata.recompute_helix_spec_buffers``
    writes: ``helix_slots`` (the rank-local write slot, -1 on other ranks' pages),
    ``helix_bounds`` (the rank-local KV bound of each token), ``helix_zero_kv = bounds == 0`` and
    ``kv_lens[kv_bound_start + s]`` = the bound of sequence ``s``'s last token.

    Integer tensors are int32 with unit stride; ``helix_zero_kv`` is bool; ``num_helix`` is a
    multiple of ``tokens_per_seq``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS = (
        Cell(
            why="helix16 ULL c4: 4 requests x 8 tokens (DSpark 7), 16 ranks, 128-token pages, "
            "with the overlap scheduler's offsets, on the first rank",
            spec=dict(num_seqs=4, tokens_per_seq=8, cp_size=16, cp_rank=0, overlap=True),
        ),
        Cell(
            why="helix16 ULL c1: one request on the last rank",
            spec=dict(num_seqs=1, tokens_per_seq=8, cp_size=16, cp_rank=15, overlap=True),
        ),
        Cell(
            why="helix8 c240: 20 requests x 6 tokens (DSpark 5), 8 ranks",
            spec=dict(num_seqs=20, tokens_per_seq=6, cp_size=8, cp_rank=3, overlap=True),
        ),
        Cell(
            why="helix8 c75 / c174: 8 requests x 8 tokens without previous-step offsets",
            spec=dict(num_seqs=8, tokens_per_seq=8, cp_size=8, cp_rank=7, overlap=False),
        ),
    )

    note = """
    The ``RuntimeHooks.decode_prologue`` of helix8 / helix16 (``SpecDecodeHead``) calls it with
    the preparer's previous-step offsets and the metadata's Helix buffers; the zero-bound flags go
    to the hook's scratch. One program of 4 warps, 512 lanes per pass; the KV-bound pass reads the
    bounds back after a barrier. The scalars are not specialized, so one compile serves every
    step.
    """

    def __call__(
        self,
        position_ids: torch.Tensor,
        pos_offsets: torch.Tensor,
        num_pos: int,
        kv_lens: torch.Tensor,
        kv_offsets: torch.Tensor,
        kv_start: int,
        num_kv: int,
        *,
        has_overlap: bool,
        helix_positions: Optional[torch.Tensor] = None,
        helix_slots: Optional[torch.Tensor] = None,
        helix_bounds: Optional[torch.Tensor] = None,
        helix_zero_kv: Optional[torch.Tensor] = None,
        num_helix: int = 0,
        tokens_per_block: int = 1,
        cp_size: int = 1,
        cp_rank: int = 0,
        kv_bound_start: int = 0,
        tokens_per_seq: int = 1,
    ) -> None:
        _head.fused_spec_head_prep(
            position_ids,
            pos_offsets,
            num_pos,
            kv_lens,
            kv_offsets,
            kv_start,
            num_kv,
            has_overlap=has_overlap,
            helix_positions=helix_positions,
            helix_slots=helix_slots,
            helix_bounds=helix_bounds,
            helix_zero_kv=helix_zero_kv,
            num_helix=num_helix,
            tokens_per_block=tokens_per_block,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_bound_start=kv_bound_start,
            tokens_per_seq=tokens_per_seq,
        )

    def reference(
        self,
        position_ids: torch.Tensor,
        pos_offsets: torch.Tensor,
        num_pos: int,
        kv_lens: torch.Tensor,
        kv_offsets: torch.Tensor,
        kv_start: int,
        num_kv: int,
        *,
        has_overlap: bool,
        helix_positions: Optional[torch.Tensor] = None,
        helix_slots: Optional[torch.Tensor] = None,
        helix_bounds: Optional[torch.Tensor] = None,
        helix_zero_kv: Optional[torch.Tensor] = None,
        num_helix: int = 0,
        tokens_per_block: int = 1,
        cp_size: int = 1,
        cp_rank: int = 0,
        kv_bound_start: int = 0,
        tokens_per_seq: int = 1,
    ) -> dict:
        """The written tensors after the call, by argument name (clones; only those the call
        writes)."""
        pos, kv = position_ids.clone(), kv_lens.clone()
        out = dict(position_ids=pos, kv_lens=kv)
        if has_overlap:
            pos[:num_pos] += pos_offsets[:num_pos]
            kv[kv_start : kv_start + num_kv] += kv_offsets[:num_kv]
        if num_helix > 0:
            n = num_helix
            hpos = helix_positions.clone()
            if has_overlap:
                hpos[:n] += pos_offsets[:n]
            g = hpos[:n].to(torch.int64)
            before = _helix_local_len(g, tokens_per_block, cp_size, cp_rank)
            bound = _helix_local_len(g + 1, tokens_per_block, cp_size, cp_rank)
            owned = torch.div(g, tokens_per_block, rounding_mode="floor") % cp_size == cp_rank
            slots, bounds, zero = helix_slots.clone(), helix_bounds.clone(), helix_zero_kv.clone()
            slots[:n] = torch.where(owned, before, -1).to(slots.dtype)
            bounds[:n] = bound.to(bounds.dtype)
            zero[:n] = bound == 0
            num_seqs = n // tokens_per_seq
            last = bounds[tokens_per_seq - 1 : num_seqs * tokens_per_seq : tokens_per_seq]
            kv[kv_bound_start : kv_bound_start + num_seqs] = last
            out.update(
                helix_positions=hpos, helix_slots=slots, helix_bounds=bounds, helix_zero_kv=zero
            )
        return out

    def is_valid(
        self,
        position_ids: torch.Tensor,
        pos_offsets: torch.Tensor,
        num_pos: int,
        kv_lens: torch.Tensor,
        kv_offsets: torch.Tensor,
        kv_start: int,
        num_kv: int,
        **kwargs,
    ) -> None:
        # The launcher checks dtypes and strides; the ranges it reads are not checked.
        assert position_ids.numel() >= num_pos and pos_offsets.numel() >= num_pos
        assert kv_lens.numel() >= kv_start + num_kv and kv_offsets.numel() >= num_kv
        num_helix = kwargs.get("num_helix", 0)
        if num_helix > 0:
            assert 0 <= kwargs.get("cp_rank", 0) < kwargs.get("cp_size", 1), "cp_rank"
            num_seqs = num_helix // kwargs.get("tokens_per_seq", 1)
            assert kv_lens.numel() >= kwargs.get("kv_bound_start", 0) + num_seqs
            assert pos_offsets.numel() >= num_helix or not kwargs["has_overlap"]


k3_spec_decode_head = K3SpecDecodeHead()
