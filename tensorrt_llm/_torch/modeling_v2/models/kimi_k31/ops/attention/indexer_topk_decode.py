# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode / verify top-k: each generation row's top-k causal keys (CUDA radix)."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: K3.1's DSA top-k width (dsa_config.index_topk).
INDEX_TOPK = 2048


def _spec(batch, next_n, seq_lens, width, **overrides) -> dict:
    """``seq_lens``: (low, high) of the per-request KV lengths, drawn per cell; ``width`` the
    logits row length (the indexer's max sequence length), ``row_pad`` extra row stride."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        seq_lens=tuple(seq_lens),
        width=width,
        row_pad=0,
        index_topk=INDEX_TOPK,
        compress_ratio=1,
    )
    spec.update(overrides)
    return spec


class IndexerTopkDecode(OpWrapper):
    """Writes into ``indices`` (int32 [B * n, index_topk], contiguous) the ``index_topk`` largest
    of the causal keys ``[0, (seq_lens[b] - n + t + 1) // compress_ratio)`` of row ``b * n + t``
    of ``logits`` (fp32 [B * n, W], any row stride; ``n`` = ``next_n``), in unspecified order. A
    row with at most ``index_topk`` keys yields ``0 .. len - 1`` then -1. Columns past a row's
    causal end are never read.

    ``radix_aux_indices`` (int32) / ``radix_aux_logits`` (fp32), contiguous, at least
    ``B * n * blocks * index_topk`` elements, are the caller's split-work scratch: unless ``W``
    is below 4096 the launch splits each row over 2 to 10 blocks (2 when the rows fill half the
    SMs, 10 when ``W`` is 200k or more, a wave-fitted count in between) and merges them through
    the scratch. ``row_kv_lens`` (int32 [rows]) replaces the per-row causal count when given.
    Returns None.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a row narrower than 4096: one block per row (insertion), the short-row shortcut",
            spec=_spec(1, 1, (1500, 1500), 4000),
        ),
        Cell(
            why="one block per row at the top-k edge: 2048 and 2049 keys",
            spec=_spec(2, 1, (2048, 2049), 4000),
        ),
        Cell(
            why="decode, one request below the top-k in an 8k row: split blocks, the merge "
            "drops the padding",
            spec=_spec(1, 1, (1500, 1500), 8192),
        ),
        Cell(
            why="decode, 8 requests of 3k-30k keys in 32k rows: the small-batch split sweep",
            spec=_spec(8, 1, (3000, 30000), 32768),
        ),
        Cell(
            why="decode, 128 requests in 32k rows: two blocks per row (rows >= half the SMs)",
            spec=_spec(128, 1, (2100, 32768), 32768),
        ),
        Cell(
            why="decode, 256 requests: the largest batch, rows below and above the top-k",
            spec=_spec(256, 1, (1000, 16000), 16384),
        ),
        Cell(
            why="DFlash2 w5 verify: 5 causal rows per request, the first rows below the top-k",
            spec=_spec(4, 5, (2045, 9000), 16384),
        ),
        Cell(
            why="DFlash2 w8 verify at batch 64",
            spec=_spec(64, 8, (2040, 30000), 32768),
        ),
        Cell(
            why="a 256k-wide logits row (max sequence length 262144): 10 blocks per row",
            spec=_spec(2, 8, (100000, 262144), 262144),
        ),
        Cell(
            why="the CuTe DSL logits arena: a row stride rounded up to 256 past the width",
            spec=_spec(4, 5, (3000, 7000), 7000, row_pad=168),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target reaches this from the DSA indexer's generation phase (decode and the
    DFlash2 verify) on architectures other than SM 10.3 (where every target takes
    indexer_topk_decode_gvr) and on SM 10.3 for non-fp32 logits:
    sparse_attention_config.use_cute_dsl_topk defaults to False, so the CUDA radix kernel is the
    default decode top-k there and the CuTe DSL one is not on the path. Arguments: logits the [B *
    n, max sequence length] output of the paged logits op (a row-strided view of its arena),
    seq_lens the generation requests' KV lengths (kv_lens_cuda_runtime, including the n new tokens),
    indices the generation rows of the step's top-k buffer, next_n the verify width (1 for plain
    decode), index_topk 2048, compress_ratio 1, the metadata's radix_aux_indices / radix_aux_logits
    sliced to [:rows, :10, :2048], and row_kv_lens None: the target never enables ragged
    verification, so the per-row form is not certified here. The indices are request-local key
    positions; the order of the written indices varies from run to run, so the cells compare each
    row as a set over distinct logits, and logits tied at the k-th value may be chosen either way.
    The workspaces are caller-owned with stable addresses: the call captures in a CUDA graph.

    ``indices`` must be exactly ``index_topk`` wide: the kernel writes row r at
    ``r * index_topk``. The split choice reads the logits width and the row count, not the
    rows' lengths.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        index_topk: int,
        compress_ratio: int,
        radix_aux_indices: Optional[torch.Tensor],
        radix_aux_logits: Optional[torch.Tensor],
        row_kv_lens: Optional[torch.Tensor],
    ) -> None:
        torch.ops.trtllm.indexer_topk_decode(
            logits,
            seq_lens,
            indices,
            next_n,
            index_topk,
            compress_ratio=compress_ratio,
            radix_aux_indices=radix_aux_indices,
            radix_aux_logits=radix_aux_logits,
            row_kv_lens=row_kv_lens,
        )

    def reference(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        index_topk: int,
        compress_ratio: int,
        radix_aux_indices: Optional[torch.Tensor],
        radix_aux_logits: Optional[torch.Tensor],
        row_kv_lens: Optional[torch.Tensor],
    ) -> None:
        """torch.topk over each row's causal keys (the rest masked to -inf), written into
        ``indices`` in descending logit order; short rows get the identity then -1. The
        workspaces are ignored."""
        rows = logits.shape[0]
        device = logits.device
        if row_kv_lens is not None:
            lengths = row_kv_lens.long() // compress_ratio
        else:
            r = torch.arange(rows, device=device)
            kv = seq_lens.long()[r // next_n]
            lengths = (kv - next_n + r % next_n + 1) // compress_ratio
        cols = torch.arange(logits.shape[1], device=device).unsqueeze(0)
        masked = torch.where(cols < lengths.unsqueeze(1), logits.float(), float("-inf"))
        k = min(index_topk, logits.shape[1])
        top = masked.topk(k, dim=1).indices
        slots = torch.arange(index_topk, device=device).unsqueeze(0)
        short = torch.where(slots < lengths.unsqueeze(1), slots, -1)
        out = torch.full_like(short, -1)
        out[:, :k] = top
        out = torch.where((lengths > index_topk).unsqueeze(1), out, short)
        indices[:, :index_topk] = out.to(indices.dtype)

    def is_valid(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        index_topk: int,
        compress_ratio: int,
        radix_aux_indices: Optional[torch.Tensor],
        radix_aux_logits: Optional[torch.Tensor],
        row_kv_lens: Optional[torch.Tensor],
    ) -> None:
        # The kernel writes row r at r * index_topk: a wider buffer gets its rows interleaved.
        assert indices.shape[1] == index_topk, (
            f"indices must be [rows, {index_topk}], got {tuple(indices.shape)}"
        )
        # Keys past the logits row are read from the next row.
        lengths = row_kv_lens if row_kv_lens is not None else seq_lens
        if lengths.numel():
            longest = int(lengths.max()) // compress_ratio
            assert longest <= logits.shape[1], "a row's keys run past the logits row"


indexer_topk_decode = IndexerTopkDecode()
