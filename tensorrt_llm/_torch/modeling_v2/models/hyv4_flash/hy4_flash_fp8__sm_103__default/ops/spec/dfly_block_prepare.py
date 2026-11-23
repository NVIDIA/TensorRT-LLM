# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFly2 draft-block prepare: the block's rows and pool slots, then each draft layer's selection.

The drafter keeps no KV cache: block rows are appended to the target sparse layer's latent pool,
and each draft layer attends over that layer's compacted picks followed by the block's causal
prefix. Selections hold pool rows (``page * block_size + offset``); page 0 is a real page.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

MASK_TOKEN_ID = 120009
BLOCK_WIDTH = 6
#: `dfly_target_index_topk`.
TARGET_TOPK = 2048
PAGE_ROWS = 32
MROPE_PLANES = 3
#: Padding: only valid past a row's `lengths`.
PAD_SLOT_ID = -1


class DraftBlock(NamedTuple):
    """Row ``g * block_width + j`` is request ``g``, column ``j``; column 0 is the bonus token."""

    block_ids: torch.Tensor  # int64 [gens * width]: bonus at column 0, mask tokens after
    positions: torch.Tensor  # int64 [gens * width]: anchor + j for an active request
    mrope_positions: torch.Tensor  # int64 [3, gens * width]: the three planes, coincident for text
    slots: torch.Tensor  # int32 [gens * width]: the row's pool slot, PAD_SLOT_ID if inactive
    seq_lens: torch.Tensor  # int32 [gens]: the block's absolute KV bound, anchor + width
    active: torch.Tensor  # bool [gens]: whether the request drafts this step
    source_rows: torch.Tensor  # int64 [gens]: the last accepted token's row in the target batch
    anchors: torch.Tensor  # int64 [gens]: the block's first position


@triton.jit
def _dfly_block_prepare_kernel(
    BONUS,  # [G] int64, the accepted token after the last accepted draft
    NUM_ACCEPTED,  # [G] int32, drafts accepted this step
    QUERY_START_LOC,  # [G + 1] int64, the target batch's per-request token offsets
    TARGET_POSITIONS,  # [T] int64, the target batch's absolute positions
    BLOCK_TABLE,  # [G, NB] int32, the pool's page table per request
    BLOCK_IDS,  # [G * W] int64
    POSITIONS,  # [G * W] int64
    MROPE,  # [PLANES, G * W] int64
    SLOTS,  # [G * W] int32
    SEQ_LENS,  # [G] int32
    ACTIVE,  # [G] bool
    SOURCE_ROWS,  # [G] int64
    ANCHORS,  # [G] int64
    num_blocks,
    block_size,
    max_model_len,
    mask_token_id,
    rows,
    WIDTH: tl.constexpr,
    WBLOCK: tl.constexpr,
    PLANES: tl.constexpr,
    PAD: tl.constexpr,
):
    """One request per program. A request drafts only if it accepted a draft and the block fits
    under ``max_model_len``; an inactive request gets mask rows at positions ``0 .. W - 1`` and
    ``PAD`` slots.
    """
    g = tl.program_id(0)
    start = tl.load(QUERY_START_LOC + g)
    nacc = tl.load(NUM_ACCEPTED + g).to(tl.int64)
    source = start + nacc
    anchor = tl.load(TARGET_POSITIONS + source) + 1
    active = (nacc > 0) & (anchor + WIDTH <= max_model_len)
    bonus = tl.load(BONUS + g).to(tl.int64)
    tl.store(ACTIVE + g, active)
    tl.store(SOURCE_ROWS + g, source)
    tl.store(ANCHORS + g, anchor)
    tl.store(SEQ_LENS + g, (tl.where(active, anchor, 0) + WIDTH).to(tl.int32))

    j = tl.arange(0, WBLOCK)
    in_block = j < WIDTH
    row = g * WIDTH + j
    positions = tl.where(active, anchor + j.to(tl.int64), j.to(tl.int64))
    mask = tl.full([WBLOCK], mask_token_id, tl.int64)
    block_ids = tl.where(j == 0, tl.where(active, bonus, mask), mask)
    tl.store(BLOCK_IDS + row, block_ids, mask=in_block)
    tl.store(POSITIONS + row, positions, mask=in_block)
    for plane in tl.static_range(PLANES):
        tl.store(MROPE + plane * rows + row, positions, mask=in_block)

    block_idx = positions // block_size
    valid = in_block & active & (positions < max_model_len) & (block_idx < num_blocks)
    page = tl.load(
        BLOCK_TABLE + g * num_blocks + tl.minimum(block_idx, num_blocks - 1),
        mask=valid,
        other=0,
    ).to(tl.int64)
    slot = page * block_size + positions % block_size
    tl.store(SLOTS + row, tl.where(valid, slot, PAD).to(tl.int32), mask=in_block)


class DflyBlockPrepare(OpWrapper):
    """``(bonus, num_accepted, query_start_loc, target_positions, block_table)`` -> ``DraftBlock``.

    ``source_rows = query_start_loc[:G] + num_accepted`` and ``anchors = positions[source] + 1``,
    so both must index the same batch. ``block_table`` is already mapped to the draft layer's slab
    (``page * scale + layer_offset``) and ``block_size`` must be that pool's page.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `accepted` is how many of the 5 drafts each request accepted; `page_scale`/`layer_offset`
    # (default 1/0) put the table in the caller's `page * scale + offset` form.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode request: six block rows and their pool slots",
            spec=dict(gens=1, prefix=0, pos_offset=0, accepted="all", pages=8),
        ),
        Cell(
            why="64 draft requests after a 512-token prefix: 384 block rows, mixed accepted counts",
            spec=dict(gens=64, prefix=512, pos_offset=100, accepted="mixed", pages=256),
        ),
        Cell(
            why="a block crossing a page boundary: the anchor at offset 30 of its page",
            spec=dict(gens=2, prefix=0, pos_offset=24, accepted="all", pages=8),
        ),
        Cell(
            why="the model-length edge: the last request's block does not fit and is inactive",
            spec=dict(gens=4, prefix=0, pos_offset=0, accepted="all", max_model_len=24, pages=8),
        ),
        Cell(
            why="the engine's largest batch after a context prefix: 64 requests x width 6, one "
            "of them accepting none, over the caller's scaled page table",
            spec=dict(
                gens=64,
                prefix=512,
                pos_offset=100,
                accepted="mixed-with-zero",
                pages=256,
                page_scale=12,
                layer_offset=11,
            ),
        ),
        Cell(
            why="one request that accepted none of its five drafts: its whole block row is "
            "inactive",
            spec=dict(
                gens=1,
                prefix=0,
                pos_offset=0,
                accepted="zero",
                pages=8,
                page_scale=12,
                layer_offset=7,
            ),
        ),
    )

    def __call__(
        self,
        bonus: torch.Tensor,
        num_accepted: torch.Tensor,
        query_start_loc: torch.Tensor,
        target_positions: torch.Tensor,
        block_table: torch.Tensor,
        max_model_len: int,
        *,
        mask_token_id: int = MASK_TOKEN_ID,
        block_width: int = BLOCK_WIDTH,
        block_size: int = PAGE_ROWS,
    ) -> DraftBlock:
        gens = bonus.shape[0]
        if block_width < 2:
            raise ValueError(f"dfly_block_prepare: block_width {block_width} must be at least 2")
        if block_size < 1:
            raise ValueError(f"dfly_block_prepare: block_size {block_size} must be positive")
        if bonus.shape != (gens,) or num_accepted.shape != (gens,):
            raise ValueError(
                f"dfly_block_prepare: bonus {tuple(bonus.shape)} and num_accepted "
                f"{tuple(num_accepted.shape)} must both be ({gens},)"
            )
        if tuple(query_start_loc.shape) != (gens + 1,):
            raise ValueError(
                f"dfly_block_prepare: query_start_loc {tuple(query_start_loc.shape)} must be "
                f"({gens + 1},)"
            )
        if block_table.dim() != 2 or block_table.shape[0] != gens:
            raise ValueError(
                f"dfly_block_prepare: block_table {tuple(block_table.shape)} must be ({gens}, pages)"
            )
        num_blocks = block_table.shape[1]
        if num_blocks < 1:
            raise ValueError("dfly_block_prepare: block_table has no pages")
        rows = gens * block_width
        device = bonus.device
        out = DraftBlock(
            block_ids=torch.empty(rows, dtype=torch.int64, device=device),
            positions=torch.empty(rows, dtype=torch.int64, device=device),
            mrope_positions=torch.empty(MROPE_PLANES, rows, dtype=torch.int64, device=device),
            slots=torch.empty(rows, dtype=torch.int32, device=device),
            seq_lens=torch.empty(gens, dtype=torch.int32, device=device),
            active=torch.empty(gens, dtype=torch.bool, device=device),
            source_rows=torch.empty(gens, dtype=torch.int64, device=device),
            anchors=torch.empty(gens, dtype=torch.int64, device=device),
        )
        if gens == 0:
            return out
        _dfly_block_prepare_kernel[(gens,)](
            bonus,
            num_accepted,
            query_start_loc,
            target_positions,
            block_table,
            out.block_ids,
            out.positions,
            out.mrope_positions,
            out.slots,
            out.seq_lens,
            out.active,
            out.source_rows,
            out.anchors,
            num_blocks,
            block_size,
            max_model_len,
            mask_token_id,
            rows,
            WIDTH=block_width,
            WBLOCK=triton.next_power_of_2(block_width),
            PLANES=MROPE_PLANES,
            PAD=PAD_SLOT_ID,
            num_warps=1,
        )
        return out

    def reference(
        self,
        bonus: torch.Tensor,
        num_accepted: torch.Tensor,
        query_start_loc: torch.Tensor,
        target_positions: torch.Tensor,
        block_table: torch.Tensor,
        max_model_len: int,
        *,
        mask_token_id: int = MASK_TOKEN_ID,
        block_width: int = BLOCK_WIDTH,
        block_size: int = PAGE_ROWS,
    ) -> DraftBlock:
        """The same block in plain torch; integer arithmetic, so every field must be exact."""
        gens = bonus.shape[0]
        width = block_width
        device = bonus.device
        num_blocks = block_table.shape[1]
        source = query_start_loc[:gens] + num_accepted.long()
        anchor = target_positions[source] + 1
        active = (num_accepted > 0) & (anchor + width <= max_model_len)
        j = torch.arange(width, dtype=torch.int64, device=device)
        positions = torch.where(active[:, None], anchor[:, None] + j, j.expand(gens, width))
        block_ids = torch.full((gens, width), mask_token_id, dtype=torch.int64, device=device)
        block_ids[:, 0] = torch.where(active, bonus, torch.full_like(bonus, mask_token_id))
        block_idx = positions // block_size
        valid = (
            active[:, None].expand(gens, width)
            & (positions < max_model_len)
            & (block_idx < num_blocks)
        )
        page = block_table.long().gather(1, block_idx.clamp(max=num_blocks - 1))
        slots = page * block_size + positions % block_size
        slots = torch.where(valid, slots, torch.full_like(slots, PAD_SLOT_ID))
        rows = gens * width
        block_ids = block_ids.reshape(rows)
        positions = positions.reshape(rows)
        slots = slots.reshape(rows)
        return DraftBlock(
            block_ids=block_ids,
            positions=positions,
            mrope_positions=positions.expand(MROPE_PLANES, rows).contiguous(),
            slots=slots.to(torch.int32),
            seq_lens=(torch.where(active, anchor, torch.zeros_like(anchor)) + width).to(
                torch.int32
            ),
            active=active,
            source_rows=source,
            anchors=anchor,
        )

    def is_valid(
        self,
        bonus: torch.Tensor,
        num_accepted: torch.Tensor,
        query_start_loc: torch.Tensor,
        target_positions: torch.Tensor,
        block_table: torch.Tensor,
        max_model_len: int,
        *,
        mask_token_id: int = MASK_TOKEN_ID,
        block_width: int = BLOCK_WIDTH,
        block_size: int = PAGE_ROWS,
    ) -> None:
        gens = bonus.shape[0]
        assert bonus.dtype is torch.int64 and bonus.is_contiguous()
        assert num_accepted.dtype is torch.int32 and num_accepted.is_contiguous()
        assert query_start_loc.dtype is torch.int64 and query_start_loc.is_contiguous()
        assert target_positions.dtype is torch.int64 and target_positions.is_contiguous()
        assert block_table.dtype is torch.int32 and block_table.dim() == 2
        # The kernel walks the table at its row width.
        assert block_table.stride(0) == block_table.shape[1], "block_table rows must be dense"
        assert max_model_len >= block_width, (
            f"max_model_len {max_model_len} is shorter than the {block_width}-row block"
        )
        if gens == 0:
            return
        # Past `block_width - 1` the source row reads the next request's position: plausible, wrong.
        assert int(num_accepted.min()) >= 0 and int(num_accepted.max()) <= block_width - 1, (
            f"num_accepted must be in [0, {block_width - 1}]"
        )
        starts = query_start_loc[: gens + 1]
        assert bool((starts[1:] >= starts[:-1]).all()), "query_start_loc must be non-decreasing"
        assert int(starts[-1]) <= target_positions.numel(), (
            "query_start_loc runs past target_positions"
        )


dfly_block_prepare = DflyBlockPrepare()


@triton.jit
def _dfly_block_indices_kernel(
    HISTORY,  # [L, S, TOPK] int32, the target layers' picks, pool rows, -1 padded
    STATE_ROWS,  # [G] int64, each request's row in HISTORY
    BLOCK_SLOTS,  # [G, W] int32, the block rows' pool rows
    ACTIVE,  # [G] bool
    PACKED,  # [L, G * W, OUT] int32, written
    LENGTHS,  # [L, G * W] int32, written
    history_layer_stride,
    history_state_stride,
    packed_layer_stride,
    packed_row_stride,
    lengths_layer_stride,
    WIDTH: tl.constexpr,
    TARGET_TOPK: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    BLOCK_BLOCK: tl.constexpr,
    PAD: tl.constexpr,
):
    """One (request, draft layer) per program: compact the layer's picks, append the causal block.

    The three stores to a row cover disjoint columns (``[0, n)``, ``[n, n + j + 1)``,
    ``[n + j + 1, OUT)``), so no fill pass is needed.
    """
    g = tl.program_id(0)
    layer = tl.program_id(1)
    active = tl.load(ACTIVE + g)
    inactive = ~active
    state = tl.load(STATE_ROWS + g)
    cols = tl.arange(0, TARGET_TOPK)
    history = tl.load(
        HISTORY + layer.to(tl.int64) * history_layer_stride + state * history_state_stride + cols,
        mask=active,
        other=PAD,
    )
    valid = history >= 0
    order = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    n = tl.sum(valid.to(tl.int32), axis=0)
    blk = tl.arange(0, BLOCK_BLOCK)
    slots = tl.load(BLOCK_SLOTS + g * WIDTH + blk, mask=blk < WIDTH, other=PAD)
    full = tl.arange(0, OUT_BLOCK)
    pad = tl.full([OUT_BLOCK], PAD, tl.int32)
    for j in tl.static_range(WIDTH):
        row = g * WIDTH + j
        base = (
            PACKED + layer.to(tl.int64) * packed_layer_stride + row.to(tl.int64) * packed_row_stride
        )
        count = LENGTHS + layer.to(tl.int64) * lengths_layer_stride + row
        tl.store(base + full, pad, mask=(full < OUT_WIDTH) & inactive)
        tl.store(base + order, history, mask=valid & active)
        tl.store(base + n + blk, slots, mask=(blk <= j) & active)
        tail = n + j + 1 + full
        tl.store(base + tail, pad, mask=(tail < OUT_WIDTH) & active)
        tl.store(count, tl.where(active, n + j + 1, 0))


class DflyBlockIndices(OpWrapper):
    """``(history, state_rows, block_slots, active)`` -> ``packed``/``lengths``, written in place.

    ``history`` int32 ``[layers, states, topk]`` holds pool-row picks (``-1`` = none);
    ``state_rows`` picks each request's row. Row ``(layer, g * width + j)`` of ``packed`` gets the
    valid picks compacted, then ``block_slots[g, :j + 1]``, then ``-1``; ``lengths`` is its count
    (0 if inactive). ``packed`` must be at least ``topk + width`` wide.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `states` is the picks-state row count, not the batch. The drafter calls this per layer, so
    # the last cell is `layers=1` at the engine's per-step budget and the caller's packed width.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request, five draft layers: a batch of one against eight state rows",
            spec=dict(layers=5, gens=1, states=8, valid=TARGET_TOPK, active="all"),
        ),
        Cell(
            why="64 requests x 6 rows x 5 layers: the envelope's largest selection",
            spec=dict(layers=5, gens=64, states=64, valid=TARGET_TOPK, active="all"),
        ),
        Cell(
            why="a short sequence: 37 of 2048 picks valid, the rest compacted away",
            spec=dict(layers=5, gens=8, states=64, valid=37, active="all"),
        ),
        Cell(
            why="mixed activity: inactive requests carry pad slots and write -1 rows, count 0",
            spec=dict(layers=5, gens=8, states=64, valid=TARGET_TOPK, active="mixed"),
        ),
        Cell(
            why="an empty pick list: each row's selection is its causal block alone, counts 1..6",
            spec=dict(layers=5, gens=4, states=16, valid=0, active="all"),
        ),
        Cell(
            why="the per-layer selection call: one draft layer's slice, not five, over the "
            "engine's 64-request batch, at the state's per-step row budget and the caller's "
            "packed width",
            spec=dict(
                layers=1,
                gens=64,
                states=8192,
                valid=TARGET_TOPK,
                active="all",
                out_width=2176,
            ),
        ),
    )

    note = """
    Unchecked: `history` must index the same pool as `block_slots`, and the block's slots must be
    distinct from the picks (a duplicate row is attended twice).
    """

    def __call__(
        self,
        history: torch.Tensor,
        state_rows: torch.Tensor,
        block_slots: torch.Tensor,
        active: torch.Tensor,
        packed: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        layers, states, topk = history.shape
        gens, width = block_slots.shape
        if topk < 2 or topk & (topk - 1):
            raise ValueError(f"dfly_block_indices: topk {topk} must be a power of two, at least 2")
        if packed.shape[:2] != (layers, gens * width) or lengths.shape != (layers, gens * width):
            raise ValueError(
                f"dfly_block_indices: packed {tuple(packed.shape)} and lengths "
                f"{tuple(lengths.shape)} must be ({layers}, {gens * width}, out) and "
                f"({layers}, {gens * width})"
            )
        if packed.shape[2] < topk + width:
            raise ValueError(
                f"dfly_block_indices: packed is {packed.shape[2]} wide; a row needs "
                f"topk + width = {topk + width} columns"
            )
        if state_rows.shape != (gens,) or active.shape != (gens,):
            raise ValueError(
                f"dfly_block_indices: state_rows {tuple(state_rows.shape)} and active "
                f"{tuple(active.shape)} must both be ({gens},)"
            )
        if gens == 0 or layers == 0:
            return
        out_width = packed.shape[2]
        _dfly_block_indices_kernel[(gens, layers)](
            history,
            state_rows,
            block_slots,
            active,
            packed,
            lengths,
            history.stride(0),
            history.stride(1),
            packed.stride(0),
            packed.stride(1),
            lengths.stride(0),
            WIDTH=width,
            TARGET_TOPK=topk,
            OUT_WIDTH=out_width,
            OUT_BLOCK=triton.next_power_of_2(out_width),
            BLOCK_BLOCK=triton.next_power_of_2(width),
            PAD=PAD_SLOT_ID,
            num_warps=4,
        )

    def reference(
        self,
        history: torch.Tensor,
        state_rows: torch.Tensor,
        block_slots: torch.Tensor,
        active: torch.Tensor,
        packed: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        """The same selection in plain torch, written into the same two buffers."""
        layers, states, topk = history.shape
        gens, width = block_slots.shape
        device = history.device
        out_width = packed.shape[2]
        picks = history[:, state_rows.long()]  # [L, G, TOPK]
        valid = picks >= 0
        counts = valid.sum(-1)  # [L, G]
        order = valid.cumsum(-1) - 1
        # Compact the valid picks to the front; invalid ones scatter into a spare column `topk`.
        tmp = torch.full(
            (layers, gens, width, topk + 1), PAD_SLOT_ID, dtype=torch.int32, device=device
        )
        dest = torch.where(valid, order, torch.full_like(order, topk))
        tmp.scatter_(
            -1,
            dest[..., None, :].expand(layers, gens, width, topk),
            picks[..., None, :].expand(layers, gens, width, topk),
        )
        rows = torch.full(
            (layers, gens, width, out_width), PAD_SLOT_ID, dtype=torch.int32, device=device
        )
        rows[..., :topk] = tmp[..., :topk]
        for j in range(width):
            for c in range(j + 1):
                # Clamp: an inactive request's count is arbitrary and its row is masked below.
                column = (counts + c).clamp(max=out_width - 1)
                rows[:, :, j, :].scatter_(
                    -1, column[..., None], block_slots[None, :, c, None].expand(layers, gens, 1)
                )
        rows.masked_fill_(~active[None, :, None, None], PAD_SLOT_ID)
        j = torch.arange(width, device=device)
        want_counts = torch.where(
            active[None, :, None], counts[..., None] + j + 1, torch.zeros_like(counts[..., None])
        )
        packed.copy_(rows.reshape(layers, gens * width, out_width))
        lengths.copy_(want_counts.reshape(layers, gens * width).to(torch.int32))

    def is_valid(
        self,
        history: torch.Tensor,
        state_rows: torch.Tensor,
        block_slots: torch.Tensor,
        active: torch.Tensor,
        packed: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        layers, states, topk = history.shape
        gens, width = block_slots.shape
        # The kernel reads each row's picks at stride 1.
        assert history.dtype is torch.int32 and history.stride(-1) == 1, (
            "history must be int32, dense last dim"
        )
        assert state_rows.dtype is torch.int64 and state_rows.shape == (gens,)
        assert block_slots.dtype is torch.int32 and block_slots.is_contiguous()
        assert active.dtype is torch.bool and active.shape == (gens,)
        assert packed.dtype is torch.int32 and packed.stride(-1) == 1
        assert lengths.dtype is torch.int32 and lengths.shape == (layers, gens * width)
        assert lengths.stride(0) == gens * width, "lengths must be dense"
        if gens == 0:
            return
        live = active.nonzero().flatten()
        if live.numel() == 0:
            return
        rows = state_rows[live]
        assert int(rows.min()) >= 0 and int(rows.max()) < states, (
            f"state_rows must index the {states} state rows"
        )
        slots = block_slots[live]
        # A pad row inside the counted region is read as a real latent row.
        assert int(slots.min()) >= 0, "an active request has a -1 block slot"


dfly_block_indices = DflyBlockIndices()
