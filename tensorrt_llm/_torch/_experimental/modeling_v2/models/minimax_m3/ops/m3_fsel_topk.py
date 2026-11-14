# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSEL top-16 MSA block select for decode rows of up to 9216 blocks (family level).

``mv2_minimax_m3::minimax_m3_sm107tp4_fsel_topk`` runs the low-latency reference's exact page top-16
(``page_topk_nt512r4`` ``fsel_kernel<512, 4, false, 4>``, CUDA C++, ported as is:
``cpp/tensorrt_llm/kernels/minimaxM3Sm107Tp4/fselTopkSm107.cu``) on the per-KV-head
block maxima the index scorer writes, one CTA per (query, KV head) row: a witness-bound candidate pool (per-warp top-4
lane maxima), an exact 4-threads-per-entry rank of at most 64 candidates by (score desc, block asc), and an exact
radix-select fallback when the pool overflows.

It fills the same [total_q, num_kv_heads, 16] table as the targets' ``m3_select_long`` / ``m3_select_long_dsl``
(``reference`` below): ascending block ids, -1 tail. Over the M3 decode domain (finite scores below the forcing
scores, n_valid <= num_blocks or past it) the table is the same bit for bit, as a set and in order. Two input classes
differ (``note``). Targets keep the switch, the row range and the engagement line; this module only adapts the call.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

#: Blocks per decode row the kernel covers (kFselTopkMaxBlocks: 1M tokens + draft rows at 128 tokens per block).
MAX_BLOCKS = 9216
#: Blocks the selection keeps per (query, KV head).
TOPK = 16
#: The forcing scores of the select contract this op matches (m3_select_long): init above local above any finite score.
INIT_SCORE = 1.0e30
LOCAL_SCORE = 1.0e29
#: The traced launch shape (512-thread CTAs; rows of at most 4096 blocks run the 256-thread body).
VARIANT = 6
#: PORT(i8 MS4): variant 6, with every row of more than MS_MIN_BLOCKS valid blocks (512K+ tokens) split over a cluster
#: of 4 CTAs (each ranks a quarter of the row; the first merges the 4 x 16 candidates over DSMEM). The same table bit
#: for bit (the reference page_topk_nt512r4_ms.cu FSEL_MS 2, reference r8.4). With at most MS_MIN_BLOCKS blocks of
#: scores no row can split and the launch is variant 6. Which rows split is decided on the device from each row's valid
#: count, so one captured launch serves every context (graph-safe).
VARIANT_MS4 = 7
MS_MIN_BLOCKS = 4096


def ms4_variant(num_blocks: int, ms4: bool) -> int:
    """The launch variant for scores ``num_blocks`` wide (the engine's static per-row block width): ``VARIANT_MS4`` when
    ``ms4`` and a row can exceed ``MS_MIN_BLOCKS`` valid blocks, else ``VARIANT``."""
    return VARIANT_MS4 if ms4 and num_blocks > MS_MIN_BLOCKS else VARIANT


_MS4_OK: Dict[torch.device, bool] = {}


def ms4_available(device: torch.device) -> bool:
    """Whether this build's FSEL op takes ``VARIANT_MS4`` on ``device``: probed once with one 129-block row (variant 7
    runs there as variant 6, so the probe stores a table and changes nothing else). An op built before PORT(i8 MS4)
    rejects the variant; a target then keeps variant 6 instead of failing its select. Call before CUDA-graph capture;
    False when the op is absent or a capture is running (not cached then)."""
    device = torch.device(device)
    if device not in _MS4_OK:
        if fsel_op() is None or torch.cuda.is_current_stream_capturing():
            return False
        scores = torch.zeros(1, 129, 1, dtype=torch.float32, device=device)
        n_valid = torch.full((1,), 129, dtype=torch.int32, device=device)
        out = torch.empty(1, 1, TOPK, dtype=torch.int32, device=device).permute(1, 0, 2)
        try:
            family_ops().minimax_m3_sm107tp4_fsel_topk(
                scores, n_valid, prepare(device), out, 0, 1, False, 0, VARIANT_MS4
            )
            torch.cuda.synchronize(device)
            ok = True
        except RuntimeError as e:
            print(
                f"[M31_FSEL_MS4] unavailable on {device}: the FSEL op rejects variant {VARIANT_MS4} ({str(e)[:200]})",
                file=sys.stderr,
                flush=True,
            )
            ok = False
        _MS4_OK[device] = ok
    return _MS4_OK[device]


# M3's checkpoint: no forced initial blocks, the one block holding the query forced in.
_M3 = dict(num_kv_heads=1, init_blocks=0, local_blocks=1)

# The identity block table per device: the kernel maps each picked block through a table (the main KV page rows); M3's
# select emits logical block ids, so every row reads arange(MAX_BLOCKS) at a zero row stride.
_IDENTITY: Dict[torch.device, torch.Tensor] = {}


def fsel_op():
    """``family_ops().minimax_m3_sm107tp4_fsel_topk`` when this build carries it, else None."""
    return getattr(family_ops(), "minimax_m3_sm107tp4_fsel_topk", None)


def prepare(device: torch.device) -> torch.Tensor:
    """The identity block table on ``device``, allocated once outside any CUDA-graph capture (a target calls this
    before warmup)."""
    device = torch.device(device)
    if device not in _IDENTITY:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("m3_fsel_topk.prepare must run before CUDA-graph capture")
        _IDENTITY[device] = torch.arange(MAX_BLOCKS, dtype=torch.int32, device=device)
    return _IDENTITY[device]


_ENGAGED = set()


def log_engaged(
    target: str, rank: int, device: torch.device, max_rows: int, trigger_phase: int
) -> None:
    """One engagement line per (target, rank) on stderr: ``[M31_TP4_FSEL] engaged ...`` (printed when a target arms
    the op before warmup; the per-step ``taken`` dispatch line is the target's)."""
    key = (target, rank)
    if key in _ENGAGED:
        return
    _ENGAGED.add(key)
    print(
        f"[M31_TP4_FSEL] engaged: target {target} rank {rank} device {device} rows <= {max_rows} "
        f"trigger phase {trigger_phase} variant {VARIANT} (reference fsel top-16, CUDA C++)",
        file=sys.stderr,
        flush=True,
    )


def identity_table(device: torch.device) -> Optional[torch.Tensor]:
    """The prepared identity block table on ``device``, or None."""
    return _IDENTITY.get(torch.device(device))


class M3FselTopk(OpWrapper):
    """Fill ``out`` [total_q, num_kv_heads, 16] with each query's top-16 blocks by the FSEL kernel.

    ``scores`` is the per-KV-head block maxima [num_kv_heads, num_blocks, total_q] (any strides). Per (query, KV head)
    row the op keeps the ``n_valid_blocks[query]`` leading blocks (clamped to ``[0, num_blocks]``), always selects the
    last ``local_blocks`` of them (counted from the unclamped valid count) and the first ``init_blocks``, and fills the
    rest with the largest scores, ties to the smaller block. The ids are written ascending with a -1 tail (rows of at
    most 16 valid blocks). Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP4, one request, draft length 7 at 150K tokens: 8 query rows of ~1172 valid blocks on the 1M "
            "engine's 8193-block scores",
            spec=dict(num_blocks=8193, total_q=8, valid="c1_150k", scores="random", **_M3),
        ),
        Cell(
            why="decode at batch 8 on a 2048-block engine: the 256-thread 16-key body",
            spec=dict(num_blocks=2048, total_q=8, valid="near_full", scores="random", **_M3),
        ),
        Cell(
            why="1M-context engine at batch 8 x qlen 4: rows on all three bodies, short rows, padding and clamped rows",
            spec=dict(num_blocks=8193, total_q=32, valid="mixed", scores="random", **_M3),
        ),
        Cell(
            why="duplicated scores: ties broken to the smaller block, pool overflow into the exact radix fallback",
            spec=dict(num_blocks=4500, total_q=4, valid="near_full", scores="ties", **_M3),
        ),
        Cell(
            why="rows of 9216 blocks (the port's cap): the 512-thread 18-key body",
            spec=dict(num_blocks=9216, total_q=4, valid="near_full", scores="random", **_M3),
        ),
    )

    CELLS += (
        Cell(
            why="TP2 two KV heads, mixed validity at the decode row cap",
            spec=dict(
                num_blocks=8193,
                total_q=8,
                num_kv_heads=2,
                init_blocks=0,
                local_blocks=1,
                valid="mixed",
                scores="random",
            ),
        ),
        Cell(
            why="TP2 two KV heads, exact radix fallback on ties",
            spec=dict(
                num_blocks=9216,
                total_q=8,
                num_kv_heads=2,
                init_blocks=0,
                local_blocks=1,
                valid="near_full",
                scores="ties",
            ),
        ),
        Cell(
            why="TP2, draft length 7 at 4 requests: 32 query rows x two KV heads, mixed validity on the 1M engine",
            spec=dict(
                num_blocks=8193,
                total_q=32,
                num_kv_heads=2,
                init_blocks=0,
                local_blocks=1,
                valid="mixed",
                scores="random",
            ),
        ),
        Cell(
            why="TP2, draft length 7 at 2 requests: 16 query rows x two KV heads, ties into the exact radix fallback",
            spec=dict(
                num_blocks=9216,
                total_q=16,
                num_kv_heads=2,
                init_blocks=0,
                local_blocks=1,
                valid="near_full",
                scores="ties",
            ),
        ),
    )

    note = """
    Certified at one or two KV heads per rank (TP4 / TP2, SM107) with M3's forcing
    (`init_blocks=0`, `local_blocks=1`), bit for bit against the targets'
    `m3_select_long` / `m3_select_long_dsl` reference on every cell.

    Not the same table on two non-finite input classes (the M3 scorer writes
    finite block maxima for every valid block, so neither reaches the decode):
    a pick whose score is -inf is emitted as its block id (the select contract
    emits -1 and compacts), and the forced local / init blocks are always kept
    where the select contract ranks them at 1e29 / 1e30, so 16 or more blocks
    scoring at or above that (+inf, positive NaN) push them out there. With
    `init_blocks > 0` and more forced blocks than 16, the forced picks can also
    differ; M3 does not force initial blocks.

    The kernel stores the table only after its grid dependency wait (the M3
    index scorer releases its dependents before its own wait); it reads the
    valid count and the block table before the wait (the count is final before
    the step's first layer, the table is static).
    """

    def __call__(
        self,
        scores: torch.Tensor,
        n_valid_blocks: torch.Tensor,
        out: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        use_pdl: bool = False,
        trigger_phase: int = 0,
        variant: int = VARIANT,
    ) -> torch.Tensor:
        table = identity_table(scores.device)
        if table is None:
            table = prepare(scores.device)
        family_ops().minimax_m3_sm107tp4_fsel_topk(
            scores,
            n_valid_blocks,
            table,
            out,
            init_blocks,
            local_blocks,
            use_pdl,
            trigger_phase,
            variant,
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
        trigger_phase: int = 0,
        variant: int = VARIANT,
    ) -> torch.Tensor:
        """The [total_q, num_kv_heads, 16] int32 table of the M3 select contract (the targets' ``m3_select_long``
        reference: forcing by substituted scores, exact integer ranking of the fp32 bit order with the block id,
        -inf picks emitted as -1, ascending ids). ``out``, ``use_pdl``, ``trigger_phase`` and ``variant`` only say
        where and how the op writes."""
        num_kv_heads, num_blocks, total_q = scores.shape
        device = scores.device
        blocks = torch.arange(num_blocks, device=device, dtype=torch.int64)
        raw = n_valid_blocks[:total_q].to(torch.int64)
        valid = raw.clamp(0, num_blocks)
        local_start = (raw - local_blocks).clamp_min(0)
        result = torch.full((total_q, num_kv_heads, TOPK), -1, dtype=torch.int32, device=device)
        big = torch.iinfo(torch.int64).max
        for head in range(num_kv_heads):
            forced = scores[head].float().t().clone()  # [total_q, num_blocks]
            forced = torch.where(blocks[None, :] < init_blocks, INIT_SCORE, forced)
            forced = torch.where(blocks[None, :] >= local_start[:, None], LOCAL_SCORE, forced)
            bits = forced.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            key = torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)
            in_row = blocks[None, :] < valid[:, None]
            packed = torch.where(in_row, (key << 16) | blocks[None, :], big)
            k = min(TOPK, num_blocks)
            picked = torch.topk(packed, k, dim=1, largest=False).values
            picked_key = picked >> 16
            keep = (picked != big) & (picked_key != 0xFF800000)
            ids = torch.where(keep, picked & 0xFFFF, big)
            ids = ids.sort(dim=1).values
            ids = torch.where(ids == big, -1, ids)
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
        trigger_phase: int = 0,
        variant: int = VARIANT,
    ) -> None:
        """The inputs this op takes and answers differently from the select contract (``note``)."""
        assert init_blocks >= 0 and local_blocks >= 0, (
            f"init_blocks ({init_blocks}) and local_blocks ({local_blocks}) must be non-negative"
        )
        assert scores.shape[1] <= MAX_BLOCKS, f"at most {MAX_BLOCKS} blocks, got {scores.shape[1]}"
        total_q = scores.shape[2]
        valid = n_valid_blocks[:total_q].to(torch.int64).clamp(0, scores.shape[1])
        in_row = (
            torch.arange(scores.shape[1], device=scores.device)[None, :] < valid[:, None]
        )  # [total_q, blocks]
        rows = scores.float().permute(0, 2, 1)  # [heads, total_q, blocks]
        assert not bool((torch.isneginf(rows) & in_row[None]).any()), (
            "a valid block scores -inf: the op emits its id where the select contract emits -1"
        )
        at_forcing = (rows >= LOCAL_SCORE) | (torch.isnan(rows) & ~torch.signbit(rows))
        assert int((at_forcing & in_row[None]).sum(dim=2).max()) < TOPK, (
            "16 or more valid blocks score at or above the forcing score: the op keeps the forced blocks"
        )
        storage = scores.untyped_storage()
        assert not (storage.data_ptr() <= out.data_ptr() < storage.data_ptr() + storage.nbytes()), (
            "out must not alias scores"
        )


m3_fsel_topk = M3FselTopk()


def fsel_select_blocks(
    max_score_kv: torch.Tensor,
    *,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    use_pdl: bool = False,
    trigger_phase: int = 0,
    variant: int = VARIANT,
) -> Optional[torch.Tensor]:
    """FSEL over per-KV-head scores [num_kv_heads, n_blocks, total_q] with 129..9216 blocks: the head-major backed
    [total_q, num_kv_heads, 16] int32 table (``long_select_blocks``' layout), or None outside the op's shapes, when the
    op is not built, or when the identity table was not prepared before a capture (callers keep their select)."""
    if (
        fsel_op() is None
        or max_score_kv.dtype != torch.float32
        or not max_score_kv.is_cuda
        or max_score_kv.dim() != 3
        or not 128 < max_score_kv.shape[1] <= MAX_BLOCKS
        or n_valid_blocks.dtype != torch.int32
        or not n_valid_blocks.is_cuda
        or n_valid_blocks.shape[0] < max_score_kv.shape[2]
    ):
        return None
    table = identity_table(max_score_kv.device)
    if table is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        table = prepare(max_score_kv.device)
    num_kv_heads, _, total_q = max_score_kv.shape
    out = torch.empty(
        (num_kv_heads, total_q, TOPK), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    family_ops().minimax_m3_sm107tp4_fsel_topk(
        max_score_kv,
        n_valid_blocks.contiguous(),
        table,
        out,
        init_blocks,
        local_blocks,
        use_pdl,
        trigger_phase,
        variant,
    )
    return out
