# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-16 MSA block selection for decode rows of up to 9216 blocks."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Blocks per decode row the op covers (9 keys per thread of one 1024-thread CTA).
MAX_BLOCKS = 9216
#: Blocks the selection keeps per (query, KV head).
TOPK = 16
#: Scores the forcing substitutes: initial blocks rank above local ones, and both above any finite score.
INIT_SCORE = 1.0e30
LOCAL_SCORE = 1.0e29

# M3's checkpoint: no forced initial blocks, the one block holding the query forced in.
_M3 = dict(num_kv_heads=1, init_blocks=0, local_blocks=1)


class M3SelectLong(OpWrapper):
    """Fill ``out`` [total_q, num_kv_heads, 16] with each query's top-16 blocks.

    ``scores`` is the per-KV-head block maxima [num_kv_heads, num_blocks,
    total_q] the CuTe DSL index scorer wrote (amax over the KV head's index
    heads). Per (query, KV head) row the op keeps the ``n_valid_blocks[query]``
    leading blocks (clamped to ``[0, num_blocks]``), forces the last
    ``local_blocks`` of them (counted from the *unclamped* valid count) to
    ``LOCAL_SCORE`` and the first ``init_blocks`` to ``INIT_SCORE`` -- local wins
    where both apply -- and selects the 16 largest, ties to the smaller block.
    A pick whose forced score is -inf is emitted as -1. The ids are written
    ascending with a -1 tail.

    Same contract as ``torch.ops.trtllm.minimax_m3_select_blocks`` (the in-tree
    ``select_blocks_from_maxscore``) and bit-identical to it, except on a row
    where more than 1024 valid blocks tie at the 16th-largest score: this op
    breaks that tie to the smallest blocks as the contract says, the in-tree op
    does not. The test checks both against ``reference``. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="32K-context engine (257 worst-case k-tiles), Eagle3 verify at one request: 4 query rows",
            spec=dict(num_blocks=257, total_q=4, valid="near_full", scores="random", **_M3),
        ),
        Cell(
            why="decode without speculation at batch 8 on a 2048-block engine: the 2-key-per-thread path at its edge",
            spec=dict(num_blocks=2048, total_q=8, valid="near_full", scores="random", **_M3),
        ),
        Cell(
            why="1M-context engine at batch 8 x qlen 4: rows on all three key widths, short rows and padding rows",
            spec=dict(num_blocks=8193, total_q=32, valid="mixed", scores="random", **_M3),
        ),
        Cell(
            why="duplicated scores: ties broken to the smaller block, and > 1024 keys at the bound (re-read path)",
            spec=dict(num_blocks=4500, total_q=4, valid="near_full", scores="ties", **_M3),
        ),
        Cell(
            why="non-finite scores: +inf, NaN ranked above +inf, and rows whose picks run into -inf (emitted -1)",
            spec=dict(num_blocks=600, total_q=8, valid="mixed", scores="nonfinite", **_M3),
        ),
    )

    note = """
    Certified at one KV head per rank (TP4) with M3's forcing (`init_blocks=0`,
    `local_blocks=1`). The kernel takes any head count and forcing; neither is
    driven here.

    `scores` may be strided (the op reads it by stride); the target hands over
    the scorer's contiguous view, which is what the cells drive. Rows of at
    most 128 blocks are within the kernel's range, but the target keeps those on
    the in-tree op, so they are not certified here.

    Ordering is by the descending-score key `orderedKey`: a positive NaN ranks
    above +inf, +0 above -0, and a negative NaN below -inf -- which, unlike
    -inf, is emitted as a block id rather than -1. The cells drive positive
    NaN only.
    """

    def __call__(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        early_trigger: bool = False,
    ) -> torch.Tensor:
        torch.ops.trtllm.minimax_m3_select_long(
            scores, n_valid_blocks, out, init_blocks, local_blocks, use_pdl, early_trigger
        )
        return out

    def reference(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        early_trigger: bool = False,
    ) -> torch.Tensor:
        """The [total_q, num_kv_heads, 16] int32 table the op is supposed to write.

        Exact integer ranking, no floating point beyond the forcing: each
        score's bit pattern maps to an unsigned key that is smaller for a larger
        score (NaN above +inf), packed with the block id so one sort ranks by
        (score descending, block ascending). ``out``, ``use_pdl`` and
        ``early_trigger`` only say where and how the op writes; they do not enter the result.
        """
        num_kv_heads, num_blocks, total_q = scores.shape
        device = scores.device
        blocks = torch.arange(num_blocks, device=device, dtype=torch.int64)
        raw = n_valid_blocks[:total_q].to(torch.int64)
        valid = raw.clamp(0, num_blocks)
        local_start = (raw - local_blocks).clamp_min(0)
        result = torch.full((total_q, num_kv_heads, TOPK), -1, dtype=torch.int32, device=device)
        for head in range(num_kv_heads):
            forced = scores[head].float().t().clone()  # [total_q, num_blocks]
            forced = torch.where(blocks[None, :] < init_blocks, INIT_SCORE, forced)
            forced = torch.where(blocks[None, :] >= local_start[:, None], LOCAL_SCORE, forced)
            bits = forced.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            key = torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)
            in_row = blocks[None, :] < valid[:, None]
            packed = torch.where(
                in_row, (key << 16) | blocks[None, :], torch.iinfo(torch.int64).max
            )
            k = min(TOPK, num_blocks)
            picked = torch.topk(packed, k, dim=1, largest=False).values
            picked_key = picked >> 16
            keep = (picked != torch.iinfo(torch.int64).max) & (picked_key != 0xFF800000)
            ids = torch.where(keep, picked & 0xFFFF, torch.iinfo(torch.int64).max)
            ids = ids.sort(dim=1).values
            ids = torch.where(ids == torch.iinfo(torch.int64).max, -1, ids)
            result[:, head, :k] = ids.to(torch.int32)
        return result

    def is_valid(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        early_trigger: bool = False,
    ) -> None:
        """The inputs this op takes and answers wrongly.

        Negative forcing counts are accepted and push the forced ranges past the
        row instead of failing; the in-tree op this one stands in for refuses
        them, so a caller relying on this one would silently diverge from it.
        An ``out`` overlapping ``scores`` is overwritten while it is still read.
        """
        assert init_blocks >= 0 and local_blocks >= 0, (
            f"init_blocks ({init_blocks}) and local_blocks ({local_blocks}) must be non-negative; the "
            "op accepts negative counts and selects as if the forced ranges were elsewhere"
        )
        storage = scores.untyped_storage()
        assert not (storage.data_ptr() <= out.data_ptr() < storage.data_ptr() + storage.nbytes()), (
            "out must not alias scores: the op writes it while scores are still read"
        )


m3_select_long = M3SelectLong()
