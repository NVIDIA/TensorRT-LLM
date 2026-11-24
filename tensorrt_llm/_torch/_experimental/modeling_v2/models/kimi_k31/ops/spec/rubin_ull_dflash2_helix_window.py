# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's helix-layout window on the engine's CuTe DSL MLA decode: one CP rank's
partial over the window positions it owns.

``kernels/spec/rubin_ull_dflash2_helix_window.py`` packs each block row's visible keys (this
rank's history pages of its window, then the block's own keys up to the row) into BF16 pages;
``torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell`` attends every head over them, each row
its own sequence. The caller projects the partial through W_UV and merges the ranks
(``rubin_dflash2_helix_combine`` with the returned mask).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.rubin_ull_dflash2_helix_window import pack_helix_window

#: Kimi K3.1 DFlash2 drafter: kv_lora_rank 512 + rope 64, window 1024, 96 heads.
_LATENT = 512
_DIM = 576


def helix_window_workspace(
    rows: int, heads: int, page_size: int, window: int, cp_size: int, device
):
    """Buffers of :data:`rubin_ull_dflash2_helix_window` for exactly ``rows`` block rows, made
    before graph capture: the packed BF16 pages (``cap`` keys a row: the most a row's window can
    put on one rank, plus a page), their identity page table, the per-row local lengths, the
    empty-row mask and the decode's workspace."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    cap = (triton.cdiv(window, page_size * cp_size) + 1) * page_size
    pages = cap // page_size
    size = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        heads, 1, _LATENT, rows, cutlass.Float32
    )
    return SimpleNamespace(
        rows=rows,
        heads=heads,
        page_size=page_size,
        window=window,
        cap=cap,
        packed=torch.empty((rows * pages, page_size, _DIM), dtype=torch.bfloat16, device=device),
        tables=torch.arange(rows * pages, dtype=torch.int32, device=device).view(rows, pages).t(),
        lengths=torch.empty(rows, dtype=torch.int32, device=device),
        empty=torch.empty(rows, dtype=torch.bool, device=device),
        splits=torch.ones(rows, dtype=torch.int32, device=device),
        workspace=torch.empty(size, dtype=torch.uint8, device=device),
    )


def _spec(lengths, block=8, cp=4, heads=96, far=False) -> dict:
    return dict(lengths=tuple(lengths), block=block, cp=cp, heads=heads, far=far)


class RubinUllDFlash2HelixWindow(OpWrapper):
    """Rank ``cp_rank``'s partial of block row ``r = b * blk + j`` (query at ``L_b + j``) over the
    positions ``x`` of ``[max(0, L_b + j + 1 - W), L_b + j]`` it owns (``(x // P) % cp ==
    cp_rank``): history positions ``x < L_b`` from ``pool`` (FP8 at scale 1 or BF16, page
    ``page_table[table_rows[b], x // P // cp]``), the block's own keys from ``block_kv[b * blk +
    x - L_b]``. Keys are the 576-wide latents, values their first 512 columns, the query BF16.

    Returns ``(partial [R, heads, 512] BF16 normalized, stats [R, heads, 2] float32 (natural
    log-sum-exp of the scaled logits, 1), empty [R] bool)``: the decode folds max and sum into
    one log-sum-exp, which the (max, sum) combine weighs the same (``1 * exp(lse - peak)``). A
    row with no local key decodes one zero key; ``empty`` marks it for the combine. ``ws`` is
    :func:`helix_window_workspace` of exactly ``R`` rows.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="b1 at the served window (190000 history positions)", spec=_spec((190000,))),
        Cell(why="b1, no history: only the block's own keys", spec=_spec((0,))),
        Cell(
            why="b3 short histories: one key, a page less one, a page and one",
            spec=_spec((1, 63, 65)),
        ),
        Cell(
            why="b4 around the window: L = W - 1, W, 8 W and 128 K (some rows empty on a rank)",
            spec=_spec((1023, 1024, 8192, 131072)),
        ),
        Cell(why="b2, block 5 (K = 4)", spec=_spec((4097, 1017), block=5)),
        Cell(
            why="pages past 2**31 bytes of a 5-layer interleaved pool (64-bit pool offsets)",
            spec=_spec((2000,), far=True),
        ),
    )

    note = """
    One pack launch (rows x cap / 8 programs) and one MLA decode (each row a one-token sequence
    of at most ``cap`` keys, 96 heads); BF16 P and V in the decode, FP32 softmax. Not bitwise
    the Triton window entry (``rubin_dflash2_window_triton``): the test bounds the merged result
    against an FP32 reference.
    """

    def __call__(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: torch.Tensor,
        num_ctx: torch.Tensor,
        block_kv: torch.Tensor,
        block: int,
        softmax_scale: float,
        cp_size: int,
        cp_rank: int,
        ws: SimpleNamespace,
    ):
        rows, heads = ws.rows, ws.heads
        fp8 = pool.dtype == torch.float8_e4m3fn
        pack_helix_window[(rows, triton.cdiv(ws.cap, 8))](
            pool.view(torch.uint8) if fp8 else pool,
            block_kv.view(torch.uint8) if fp8 else block_kv,
            page_table,
            table_rows,
            num_ctx,
            ws.packed,
            ws.lengths,
            page_table.stride(0),
            pool.stride(0),
            block,
            ws.page_size,
            cp_size,
            cp_rank,
            ws.window,
            ws.cap,
            fp8,
            ws.empty,
            num_warps=8,
        )
        q = query.view(rows, 1, heads, _DIM)
        out = torch.empty((rows, 1, heads, _LATENT), dtype=query.dtype, device=query.device)
        stats = torch.empty((rows, heads, 2), dtype=torch.float32, device=query.device)
        torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
            q[..., :_LATENT].permute(2, 3, 1, 0),
            q[..., _LATENT:].permute(2, 3, 1, 0),
            ws.packed[..., :_LATENT].permute(1, 2, 0),
            ws.packed[..., _LATENT:].permute(1, 2, 0),
            ws.tables,
            ws.lengths,
            out.permute(2, 3, 1, 0),
            ws.workspace,
            heads,
            1,
            ws.page_size,
            softmax_scale,
            1.0,
            rows,
            stats,
            ws.lengths,
            ws.splits,
        )
        return out[:, 0], stats, ws.empty

    def reference(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: torch.Tensor,
        num_ctx: torch.Tensor,
        block_kv: torch.Tensor,
        block: int,
        softmax_scale: float,
        cp_size: int,
        cp_rank: int,
        ws: SimpleNamespace,
    ):
        """FP32 ``(partial, stats, empty)`` as the call returns them; an empty row is a zero
        partial with stats ``(0, 1)`` (one zero key: log-sum-exp of a single zero logit)."""
        rows, heads = query.shape[0], query.shape[1]
        page = pool.shape[1]
        partial = torch.zeros(rows, heads, _LATENT, dtype=torch.float32, device=query.device)
        stats = torch.zeros(rows, heads, 2, dtype=torch.float32, device=query.device)
        stats[..., 1] = 1
        empty = torch.zeros(rows, dtype=torch.bool, device=query.device)
        for r in range(rows):
            b, j = divmod(r, block)
            length = int(num_ctx[b])
            end = length + j + 1
            pos = torch.arange(max(0, end - ws.window), end, device=query.device)
            pos = pos[(pos // page) % cp_size == cp_rank]
            if pos.numel() == 0:
                empty[r] = True
                continue
            hist = pos[pos < length]
            pages = page_table[int(table_rows[b]), hist // page // cp_size].long()
            own = block_kv.reshape(-1, _DIM)[b * block + pos[pos >= length] - length]
            keys = torch.cat((pool[pages, hist % page].float(), own.float()))
            logits = query[r].float() @ keys.T * softmax_scale
            partial[r] = logits.softmax(-1) @ keys[:, :_LATENT]
            stats[r, :, 0] = logits.logsumexp(-1)
        return partial, stats, empty

    def is_valid(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: torch.Tensor,
        num_ctx: torch.Tensor,
        block_kv: torch.Tensor,
        block: int,
        softmax_scale: float,
        cp_size: int,
        cp_rank: int,
        ws: SimpleNamespace,
    ) -> None:
        assert query.dtype == torch.bfloat16 and query.is_contiguous(), "contiguous BF16 query"
        assert query.shape[0] == ws.rows and query.shape[1] == ws.heads and query.shape[2] == _DIM
        assert pool.dtype in (torch.bfloat16, torch.float8_e4m3fn) and pool.dim() == 3
        assert pool.shape[1] == ws.page_size and pool.shape[2] == _DIM
        assert pool.stride(1) == _DIM and pool.stride(2) == 1, "each page's rows contiguous"
        assert block_kv.dtype == pool.dtype and block_kv.is_contiguous()
        assert ws.rows % block == 0 and table_rows.shape[0] >= ws.rows // block


rubin_ull_dflash2_helix_window = RubinUllDFlash2HelixWindow()

__all__ = ["helix_window_workspace", "rubin_ull_dflash2_helix_window"]
