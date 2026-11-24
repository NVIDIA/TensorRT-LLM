# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's sparse MLA decode over this rank's Helix KV shard, pushing each owner's heads
into its DcpValue mailbox (vLLM e3ddbdbd60's ``SparseDcpDecode``,
``kernels/vendor/vllm_main/.../dsa/attn_sparse.py``, push mode)."""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HEADS, ROW, LATENT = 96, 576, 512


def _cell(why: str, rows: int, kept: tuple[int, ...]) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, kept=kept))


class DsaSparseDcpDecode(OpWrapper):
    """For each row ``b < rows`` and each of the 96 gathered query heads ``h``: softmax attention
    of the FP8 query ``q[b, h]`` (``core.mla_query.mailbox[parity, b]``, 576 bytes) over this rank's
    kept FP8 KV rows ``kv_rows_u8[indices[b, :valid[b]]]`` (logits ``q . k * scale_log2`` in base
    2, V = the 512 latent bytes), pushed into owner ``o = h // 6``'s mailbox:
    ``mail_o.partial[parity, core.rank, b, h % 6]`` = the BF16 latent and ``mail_o.lse[parity,
    core.rank, b, h % 6]`` = its base-2 LSE (a row with ``valid[b] == 0``: only the LSE
    ``EMPTY_LSE``). Re-arms the q mailbox rows it read (0xFF). ``latent`` / ``lse`` are the push
    launch's unused operands.

    ``core``: the prepared ``FlatDsaCore`` (the ULL target's ``modeling.py``; its q mailbox views,
    peer address tables, scale, rank); ``decode``: its compiled kernel for ``rows``
    (``core.kernels[rows][0]``); ``stream``: the current stream's CUstream. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "served b1: 8 rows (2 head tiles per CTA, bulk cluster exchange)",
            8,
            (2048, 0, 7, 517, 130, 1, 2048, 64),
        ),  # fmt: skip
        _cell("q1 row: 1 row, the whole Top-K kept", 1, (2048,)),
        _cell("4 rows, one empty (a rank owning none of the row's keys)", 4, (300, 0, 2048, 65)),
    )

    note = """
    The K3.1 ULL generation target's attention core (rows 1-8, ``dsa.MAX_ROWS``; 16 Helix ranks,
    6 owner heads each). FP8 Q and KV, FP32 logits, E4M3 P at 448 of the running row max, FP32
    sums of the unquantized P, a cluster of 4 CTAs per (16 heads, row) splitting the keys and
    merging in FP32, one BF16 rounding (FlashInfer's sparse MLA decode's precision). Reads nothing
    before its griddepcontrol.wait; launched with PDL; triggers after its wait.
    """

    def __call__(self, core, decode, parity, kv_rows_u8, indices, valid, latent, lse, stream):
        cutlass, cute_view = core.cutlass, core.to_cute
        decode(
            core.q_views[parity],
            cute_view(kv_rows_u8),
            cute_view(indices, 4),
            cute_view(valid, 4),
            cute_view(latent),
            cute_view(lse, 4),
            cutlass.Float32(core.scale_log2),
            cutlass.Float32(core.out_scale),
            *core.peer_views,
            cutlass.Int32(parity),
            cutlass.Int32(core.rank),
            stream,
        )

    def reference(
        self, core, decode, parity, kv_rows_u8, indices, valid, latent, lse, stream
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What this rank pushes: ``(rows [B, 96, 512] FP32, base-2 LSE [B, 96])`` over the
        dequantized FP8 query and kept rows (``-inf`` LSE and zero rows where nothing is kept).
        ``latent``, ``lse`` and ``stream`` are not read; the query is read before the call
        re-arms it, so call this first."""
        rows = latent.shape[0]
        q = core.mla_query.mailbox[parity, :rows].view(torch.float8_e4m3fn).float()
        out = torch.zeros((rows, HEADS, LATENT), dtype=torch.float32, device=q.device)
        lse2 = torch.full((rows, HEADS), -math.inf, dtype=torch.float32, device=q.device)
        for b in range(rows):
            n = int(valid[b])
            if n == 0:
                continue
            k = kv_rows_u8[indices[b, :n].long()].view(torch.float8_e4m3fn).double()
            s = (q[b].double() @ k.t()) * core.scale_log2 * math.log(2.0)
            lse2[b] = (torch.logsumexp(s, dim=1) / math.log(2.0)).float()
            out[b] = (torch.softmax(s, dim=1) @ k[:, :LATENT] * core.out_scale).float()
        return out, lse2

    def is_valid(self, core, decode, parity, kv_rows_u8, indices, valid, latent, lse, stream):
        """``FlatDsaCore.run`` checks the operand layouts; the kernel does not check that every
        kept index addresses a pool row, or that the kept counts fit the indices."""
        rows = latent.shape[0]
        counts = valid[:rows]
        if bool((counts < 0).any()) or bool((counts > indices.shape[1]).any()):
            raise ValueError("dsa_sparse_dcp_decode: kept counts outside [0, 2048]")
        for b in range(rows):
            kept = indices[b, : int(counts[b])]
            if kept.numel() and (int(kept.min()) < 0 or int(kept.max()) >= kv_rows_u8.shape[0]):
                raise ValueError("dsa_sparse_dcp_decode: a kept index outside the KV pool rows")


dsa_sparse_dcp_decode = DsaSparseDcpDecode()
