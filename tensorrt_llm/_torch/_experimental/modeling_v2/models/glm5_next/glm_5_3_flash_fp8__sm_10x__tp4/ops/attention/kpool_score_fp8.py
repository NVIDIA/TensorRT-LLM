# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool scores of packed context rows in FP8, through DeepGEMM's MQA logits.

The same scores as ``kpool_score``,

    score[r, j] = sum_h (w[r, h] * w_scale) * relu(q_scale * q[r, h, :] . key_j),

for the context rows of one prefill step, computed on E4M3 operands: the pool keys of every
block table are gathered into ``[tables * pools, head_dim]`` and quantized per key, the queries
per (row, head), each with an FP32 scale ``amax / 448`` (``fused_cat_fp8``). The query scales
and ``w_scale * q_scale`` fold into FP32 head weights, and ``fp8_mqa_logits`` scores row ``r``
against the key range of its table ``request_ids[r]``. The launch sequence is the gather, two
quantizations, the head-weight product, the logits and, for several tables, a per-row copy.
"""

from __future__ import annotations

import torch
import triton

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm.deep_gemm import fp8_mqa_logits

#: The score the reference gives a pool a row cannot see; the op leaves those columns unspecified.
FP32_MIN = torch.finfo(torch.float32).min


class KpoolScoreFp8(OpWrapper):
    """``q`` BF16 ``[N, H, 128]``, ``weights`` ``[N, H]`` -> FP32 scores, or None.

    Row ``r`` scores the first ``kv_lens[r] // kpool`` pools of table ``request_ids[r]``; its
    later columns are not meaningful (unwritten or another table's keys), so a reader stops at
    ``kv_lens[r] // kpool``. ``max_kv_len`` is a host bound on every ``kv_lens[r]`` and sets the
    ``pools = max_kv_len // kpool`` gathered per table. The result is

    - one table: ``[N, pools]``, a strided view of DeepGEMM's logits whose row stride is the
      kernel's aligned row width (a multiple of 4 columns, wider than ``pools``);
    - several tables: a fresh ``[N, max(1024, cdiv(pools, 64) * 64)]`` tensor with each row's
      ``pools`` columns copied in front, the rest unwritten;
    - None when there is no row or no complete pool (``max_kv_len < kpool``).
    """

    ARCHS = frozenset({Arch.SM_100})
    # The indexer has 32 heads of 128 and scores pools of 4 over 32-token blocks. Each cell
    # gives (cached, new) per context request and the scored row range (a rank's ceil(T / 4)
    # sequence-parallel rows); max_kv_len is the largest visible length among those rows, as
    # `MlaLayer.context_rows` passes it. The two-request cell takes the copy path and puts the
    # second request's rows at another table's key range.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one rank's sequence-parallel quarter of a 13,312-token prompt (rows 0-3327)",
            spec=dict(context=((0, 13312),), rows=(0, 3328)),
        ),
        Cell(
            why="a 13,312-token prompt's last sequence-parallel quarter (rows 9984-13311)",
            spec=dict(context=((0, 13312),), rows=(9984, 13312)),
        ),
        Cell(
            why="rank 1's quarter (rows 10-19) of a 37-token prompt: two to five pools",
            spec=dict(context=((0, 37),), rows=(10, 20)),
        ),
        Cell(
            why="two packed context requests, a chunk continuation and a fresh prompt",
            spec=dict(context=((4099, 3000), (0, 1313)), rows=(0, 4313)),
        ),
    )

    note = """
    The scores are those of the E4M3-rounded operands. With per-row scales `amax / 448`, an
    operand's relative rounding error is at most 2**-4 and a score moves by a few percent of
    its scale; a score whose operands are on the E4M3 grid is exact up to FP32 accumulation.
    The pools selected from these scores can therefore differ from those of the BF16 scores
    (`kpool_score`) where two pools score within that rounding of each other.

    sm_100 only (DeepGEMM MQA logits). The target scores every packed context row here
    (`MlaLayer.context_rows`); when this returns None the rows have no complete pool and
    attend their incomplete trailing pool only.

    Transients grow with tables * pools: the gathered BF16 keys, their FP8 copy and scale,
    the padded logits and, for several tables, the row copy. The target splits packed context
    rows into calls of bounded size (`_score_calls` in `layers/mla.py`); this entry takes
    whatever it is given.
    """

    def __call__(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        tokens_per_block: int,
        *,
        head_dim: int,
        kpool: int,
        q_scale: float,
        w_scale: float,
        request_ids: torch.Tensor,
        max_kv_len: int,
    ) -> torch.Tensor | None:
        n = q.shape[0]
        pools = max(0, int(max_kv_len)) // kpool
        tables = block_tables.shape[0]
        if n == 0 or pools == 0:
            return None
        positions = torch.arange(pools, device=q.device) * kpool
        page = positions // tokens_per_block
        offset = positions - page * tokens_per_block
        slots = block_tables[:, page].long()  # [tables, pools]
        keys = index_pool[slots, offset, 2 * head_dim : 3 * head_dim].reshape(-1, head_dim)
        half = head_dim // 2
        keys_fp8, key_scale = torch.ops.trtllm.fused_cat_fp8(keys[:, :half], keys[:, half:], False)
        q_rows = q.reshape(-1, head_dim)
        q_fp8, q_row_scale = torch.ops.trtllm.fused_cat_fp8(
            q_rows[:, :half], q_rows[:, half:], False
        )
        head_weights = weights.float() * q_row_scale.view(n, -1) * (w_scale * q_scale)
        start = request_ids.to(torch.int32) * pools
        end = start + (kv_lens // kpool).to(torch.int32)
        logits = fp8_mqa_logits(
            q_fp8.view(q.shape),
            (keys_fp8, key_scale.view(-1)),
            head_weights,
            start,
            end,
            clean_logits=False,
        )
        if tables == 1:
            # DeepGEMM pads the logits rows to an aligned stride, as the pool top-k needs.
            return logits[:, :pools]
        # Keep the copy's rows aligned and wider than the selection, like the logits arena.
        out = torch.empty(
            n, max(1024, triton.cdiv(pools, 64) * 64), dtype=torch.float32, device=q.device
        )
        rows = torch.arange(n, device=q.device)
        out[:, :pools] = logits[:, : tables * pools].view(n, tables, pools)[
            rows, request_ids.long()
        ]
        return out

    def reference(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        tokens_per_block: int,
        *,
        head_dim: int,
        kpool: int,
        q_scale: float,
        w_scale: float,
        request_ids: torch.Tensor,
        max_kv_len: int,
    ) -> torch.Tensor | None:
        """The exact scores the FP8 call approximates, FP32 from the BF16 operands, as a
        contiguous ``[N, pools]``: FP32 minimum past each row's ``kv_lens[r] // kpool``, None
        where the op returns None."""
        n = q.shape[0]
        pools = max(0, int(max_kv_len)) // kpool
        if n == 0 or pools == 0:
            return None
        dev = q.device
        out = torch.full((n, pools), FP32_MIN, dtype=torch.float32, device=dev)
        first = torch.arange(pools, device=dev) * kpool
        slot = block_tables[:, first // tokens_per_block].long()  # [tables, pools]
        keys = index_pool[:, :, 2 * head_dim : 3 * head_dim][slot, (first % tokens_per_block)]
        keys = keys.float()  # [tables, pools, head_dim]
        table = request_ids.long()
        visible = kv_lens.long() // kpool
        col = torch.arange(pools, device=dev)
        for r0 in range(0, n, 64):
            rows = slice(r0, min(r0 + 64, n))
            with true_fp32_matmul():
                dots = torch.einsum("rhd,rpd->rhp", q[rows].float(), keys[table[rows]])
            mixed = (dots * q_scale).relu() * (weights[rows].float() * w_scale)[:, :, None]
            scores = mixed.sum(dim=1)
            out[rows] = torch.where(col < visible[rows, None], scores, FP32_MIN)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        tokens_per_block: int,
        *,
        head_dim: int,
        kpool: int,
        q_scale: float,
        w_scale: float,
        request_ids: torch.Tensor,
        max_kv_len: int,
    ) -> None:
        assert q.dim() == 3 and q.shape[2] == head_dim
        n, num_heads, _ = q.shape
        assert weights.shape == (n, num_heads)
        assert index_pool.dim() == 3 and index_pool.shape[1] == tokens_per_block
        assert index_pool.shape[2] >= 3 * head_dim
        assert block_tables.dim() == 2
        assert kv_lens.shape == (n,) and request_ids.shape == (n,)
        pools = max(0, int(max_kv_len)) // kpool
        if n == 0 or pools == 0:
            return
        # Row r's key range is [request_ids[r] * pools, + kv_lens[r] // kpool): a visible
        # length past max_kv_len, or a table id past the tables, reads another table's keys.
        widest = int(kv_lens.max())
        assert widest <= max_kv_len, f"kv_lens reach {widest} > max_kv_len {max_kv_len}"
        assert int(kv_lens.min()) >= 0
        assert 0 <= int(request_ids.min()) and int(request_ids.max()) < block_tables.shape[0], (
            f"request_ids must index the {block_tables.shape[0]} tables passed"
        )
        last_page = (pools - 1) * kpool // tokens_per_block
        assert last_page < block_tables.shape[1], (
            f"max_kv_len {max_kv_len} reaches page {last_page} of {block_tables.shape[1]}"
        )


kpool_score_fp8 = KpoolScoreFp8()
