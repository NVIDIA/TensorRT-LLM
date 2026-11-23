# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFly2 draft-block prepare: the six-row block and its per-layer selections, two launches.

The DFly2 drafter drafts a block of ``block_width`` rows per request -- the accepted anchor's
bonus token followed by ``block_width - 1`` mask-token rows -- and each of its draft layers
attends over the target sparse layer's latent pool: that layer's top-``target_index_topk``
picks for the anchor, compacted, followed by the block's own causal prefix. The drafter keeps
no KV cache of its own (vocabulary row 29, ``own_kv_cache`` false): the block's rows are
appended to the target layer's pool at slots this entry computes, and the target's next verify
step reads them back.

Two entries, because the two halves have different grids and different arguments:

- ``DflyBlockPrepare`` (one program per request) builds the block: ``block_ids`` (bonus, then
  mask tokens), ``positions``, the three mRoPE planes, ``seq_lens``, ``active``,
  ``source_rows``/``anchors``, and the block rows' ``slots`` in the target layer's pool.
- ``DflyBlockIndices`` (one program per request and draft layer) writes every block row's
  selection into ``packed``/``lengths``: the compacted target picks followed by the block's
  causal prefix, in the pool-row-id form ``SparseMlaDecode`` reads.

Adapted from vLLM's checkpoint-native ``_prepare_dfly_inputs_kernel``,
``_dfly_query_slots_kernel`` and ``_build_dfly_reuse_indices_kernel``
(``vllm/models/hy_v4/nvidia/dfly_inputs.py``, Apache-2.0) and SGLang's
``_build_draft_topk_kernel`` (``python/sglang/srt/speculative/dfly_utils.py``, Apache-2.0).
SGLang's is the one that writes pool slots rather than positions, which is the form this
target's sparse attention takes. The in-tree candidate,
``models/kimi_k3_vl/ops/spec/k3_dflash_gen_prepare.py``, prepares a block over the drafter's
*private* paged latent context; Hy4's block is prepared over the *target layer's* pool, so the
pattern is re-derived here rather than copied. Values are this checkpoint's (``config.json``):
mask token 120009, block width 6, ``dfly_target_index_topk`` 2048,
``dfly_target_index_layer``/``dfly_target_kv_layers`` [25, 29, 33, 37, 41], and
``dfly_context_local_window`` null -- so a selection is the picks plus the block and has no
window columns.

Re-derived for this engine, not copied:

- vLLM reserves page 0 as its null block and pads every slot whose page is 0; TRT-LLM's KV
  cache manager has no null page, so page 0 is a real page here and ``DflyBlockPrepare``
  accepts it. A cell drives it.
- vLLM's reuse lists hold *positions*; SGLang's and this entry's hold *pool rows* (the flat
  ``page * block_size + offset`` slot), because ``SparseMlaDecode`` reads row ids. A history
  entry is therefore only ever compared against ``-1``: with a zero local window every pick is
  a past token, so there is no ``< anchor`` filter to apply.
- ``lengths`` is written per layer, not once for layer 0 as SGLang writes it: each layer's pick
  list is compacted by its own valid count, and only this checkpoint's uniform pick count makes
  the two agree.
- vLLM copies the anchor's target hidden into the block's row inside its prepare kernel; that
  gather is the caller's here, because ``dfly_fusion`` takes the target hidden already
  gathered and the vocabulary's step does not name it.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's ``dfly_mask_token_id``: the token id the block's draft rows carry.
MASK_TOKEN_ID = 120009
#: The checkpoint's draft block width: five drafts plus the anchor (``--speculative-dfly-block-size 6``).
BLOCK_WIDTH = 6
#: The checkpoint's ``dfly_target_index_topk``: how many picks each target sparse layer keeps.
TARGET_TOPK = 2048
#: The latent pool's page, the same 32 rows ``sparse_mla_decode`` pages the pool into.
PAGE_ROWS = 32
#: The interleaved mRoPE's position planes (config ``mrope_section`` [12, 10, 10]).
MROPE_PLANES = 3
#: A row no slot is assigned to. ``SparseMlaDecode`` reads ``-1`` as padding and refuses it
#: inside a row's counted region, so it is the only value that may sit past ``lengths``.
PAD_SLOT_ID = -1


class DraftBlock(NamedTuple):
    """The block ``DflyBlockPrepare`` builds, one row per ``(request, block column)``.

    Row ``g * block_width + j`` of the flat tensors belongs to request ``g`` and block column
    ``j``; ``j == 0`` is the bonus token and ``j >= 1`` the mask-token rows.
    """

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
    """One request per program: its block rows, and where they land in the target layer's pool.

    The anchor is the position after the last accepted token -- ``source_rows[g] =
    query_start_loc[g] + num_accepted[g]``, the row whose hidden predicts the bonus -- so a
    request drafts only when it accepted at least one draft (``num_accepted > 0``) and the
    whole block fits under ``max_model_len``. An inactive request gets mask-token rows at
    positions ``0 .. W - 1`` and ``PAD`` slots; nothing may read them.
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

    # The block row's slot: its page from the request's table, its offset inside the page.
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

    One launch, one program per request. ``query_start_loc`` is the target batch's exclusive
    prefix offsets (``[G + 1]``, non-decreasing) and ``target_positions`` its absolute
    positions, both over the *same* batch, so ``source_rows[g] = query_start_loc[g] +
    num_accepted[g]`` is the last accepted token's row and ``anchors[g] = positions[source] +
    1`` the block's first position. ``block_table`` is the request's page table into the
    target layer's latent pool (``[G, NB]``), already mapped to that layer's slab the way the
    manager's ``get_primary_pool_page_index_params`` gives it (``page * scale +
    layer_offset``); ``block_size`` is that pool's page.

    ``block_width`` 6 is this checkpoint's draft block; ``mask_token_id`` is its
    ``dfly_mask_token_id`` (120009). Both are explicit because the call site knows them.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The checkpoint's decode envelope: up to 64 draft requests of 6 rows each (the engine
    # materialises draft rows in multiples of 6), after a context prefix in the same batch.
    # `accepted` is how many of the 5 drafts each request accepted this step; a request with
    # none drafts nothing, and one whose block runs past `max_model_len` is inactive too.
    # `page_scale`/`layer_offset` (1 and 0, raw page ids, unless a cell says otherwise) put the
    # table in the caller's form: `page * scale + offset`, one draft layer's slab of the pool.
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

    note = """
    Called by the target's drafter at `Hyv4FlashDraft.dflash_forward`, once per step, to build
    the six rows it drafts into and the slots it appends them to in the target layer's pool.

    The anchor is the row *after* the last accepted token (`query_start_loc[g] +
    num_accepted[g]`, the last accepted token's own row whose hidden predicts the bonus), which
    is vLLM's `query_end - num_rejected - 1` written in this engine's arguments. A request
    drafts only when it accepted at least one draft and `anchor + block_width <= max_model_len`
    -- the block's last position `anchor + block_width - 1` is then inside the model's length.

    Page 0 is a real page: vLLM's allocator reserves it as a null block and its slot kernel
    pads every slot whose page is 0, while TRT-LLM's KV cache manager has no null page. The
    cell at `gens=1` puts the first request's block in page 0 and is checked against the
    reference for exactly the slot arithmetic, so the difference from vLLM is measured, not
    assumed.

    Preconditions the callable cannot check (stated, not pinned by a test):
      - `block_size` must be the target pool's page. This target's pool is 32 rows
        (`PAGE_ROWS`, the page `sparse_mla_decode` reads), the default; any other value
        computes correctly for a pool of that page, and a value that disagrees with the pool
        writes plausible slots for the wrong rows.
      - `query_start_loc` and `target_positions` must index the same batch. A start location
        from a gen-only slice against a whole-batch positions tensor keeps every shape and
        gives every request a plausible wrong anchor; `is_valid` bounds the start locations
        by the positions' length, which catches the out-of-range form but not the in-range one.
      - `num_accepted` counts accepted *drafts* (0 to `block_width - 1`); the block's column 0
        is the bonus, which the caller has already resolved.
      - `block_table` rows may be padded with any value: only the pages a block's positions
        reach are read.

    `is_valid` refuses what is measured to go silently wrong: a `num_accepted` past the
    request's own block (the source row then belongs to the next request -- the anchor is a
    real position, so nothing raises), a `query_start_loc` that decreases or runs past
    `target_positions` (an out-of-range read; the probe is unsafe and is not run), and a
    non-dense `block_table` row stride (the kernel walks the table at the row width it is
    given). An empty batch returns empty tensors before launching.

    Measured on sm_103 (GB300) against `reference`, every field of all six cells bitwise
    equal: 6 rows at one request, 384 at 64, 12 across the page boundary, 24 at the
    model-length edge (one inactive), 384 at the engine's largest batch (64 requests after a
    512-token prefix, one of them taking none of its five drafts) over the caller's scaled and
    shifted page table, and 6 rows at the request that took none, its whole block inactive. A
    block built with the anchor one position early moves all 12 positions, a page table one
    page along moves all 12 slots, and a block-table row stride wider than the table moves 6
    of 12 slots.
    """

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
                f"({gens + 1},), one exclusive offset per request plus its end"
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
        """The same block in plain torch: integer arithmetic, so every field must be exact."""
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
        # The page gather is two-dimensional, so the block stays [gens, width] until then.
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
        assert block_table.stride(0) == block_table.shape[1], (
            f"block_table rows must be dense, stride {block_table.stride(0)} for a width of "
            f"{block_table.shape[1]}: the kernel walks the table at that width"
        )
        assert max_model_len >= block_width, (
            f"max_model_len {max_model_len} is shorter than the {block_width}-row block"
        )
        if gens == 0:
            return
        # The source row is `start + num_accepted`: past `block_width - 1` it leaves the
        # request's own block and reads the next request's first position -- a real position,
        # so the anchor and every slot after it are plausible and wrong.
        assert int(num_accepted.min()) >= 0 and int(num_accepted.max()) <= block_width - 1, (
            f"num_accepted must be in [0, {block_width - 1}] (accepted drafts, not rows); got "
            f"[{int(num_accepted.min())}, {int(num_accepted.max())}], which moves the source "
            f"row out of the request's block"
        )
        starts = query_start_loc[: gens + 1]
        assert bool((starts[1:] >= starts[:-1]).all()), (
            "query_start_loc must be non-decreasing: a shorter request would read another's rows"
        )
        assert int(starts[-1]) <= target_positions.numel(), (
            f"query_start_loc ends at {int(starts[-1])} past the {target_positions.numel()} "
            f"positions of the batch it indexes"
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
    """One (request, draft layer) per program: compact the layer's picks, append the block.

    Row ``j`` of the block sees the request's valid picks in their own order, then the block's
    slots ``0 .. j`` -- its own row last, so the block is causal inside itself. The three
    writes to a row cover disjoint column ranges (``[0, n)``, ``[n, n + j + 1)`` and
    ``[n + j + 1, OUT)``), which is what keeps them race-free without a fill pass.
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

    ``history`` int32 ``[layers, states, topk]`` holds each draft layer's target-layer picks --
    pool rows, ``-1`` where there is no pick -- and ``state_rows`` int64 ``[gens]`` says which
    of its ``states`` rows belongs to each request. ``block_slots`` int32 ``[gens, width]`` is
    ``DflyBlockPrepare``'s ``slots``; ``active`` bool ``[gens]`` masks the padded requests.

    ``packed`` int32 ``[layers, gens * width, out]`` and ``lengths`` int32 ``[layers, gens *
    width]`` are the caller's buffers, written in place: row ``(layer, g * width + j)`` gets
    the request's valid picks compacted to the front, then ``block_slots[g, :j + 1]``, then
    ``-1``; ``lengths`` is that row's count, and 0 for an inactive request (whose whole row is
    ``-1``). ``out`` must be at least ``topk + width`` wide; the buffer may be wider (vLLM
    rounds it to 128), and the tail past each count is ``-1`` either way.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The drafter's selection at the checkpoint's values: five draft layers (target layers
    # 25/29/33/37/41), up to 2048 picks each, the six-row block appended. `states` is the
    # request-state array's row count -- a pool axis, not the batch: the small cells run a
    # batch of one against eight state rows. The drafter calls this once per layer, on that
    # layer's slice, so the last cell runs `layers=1` with `states` at the engine's per-step
    # token budget and `out_width` at the buffer the caller allocates (`dims.index_width`).
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
    Called by the target's drafter once per step, before its five draft layers run: layer `i`
    reads `packed[i]`/`lengths[i]` as `SparseMlaDecode`'s `packed`/`lengths`, so the block's
    rows attend over the target sparse layer's own latent pool and the drafter keeps no KV
    cache. `history` is the target layer's pick buffer for the anchor's source row, stacked
    over the five draft layers (target layers 25, 29, 33, 37, 41 at checkpoint ids).

    Pool rows, not positions: SGLang's `build_draft_topk` is the reference that writes slots
    (`verify_slots`) rather than vLLM's positions, and this target's sparse attention indexes
    the flat latent pool by row. A pick is therefore compacted when it is `>= 0` and there is
    no `< anchor` filter -- the checkpoint's `dfly_context_local_window` is null, so the
    selection is exactly the picks plus the block, and every pick is a past token.

    `lengths` is per layer: each layer's list is compacted by its own count. SGLang writes one
    count from layer 0, which holds only because this checkpoint's indexer keeps the same
    number of picks per sparse layer (`min(kv_length, 2048)`); a layer whose picks came out
    shorter would be read past its end there.

    Preconditions the callable cannot check (stated, not pinned by a test):
      - `history` values are rows of the same pool `block_slots` indexes. A picks buffer from
        another layer or another request is in range and selects real, wrong rows.
      - `state_rows` names the request's row in the *picks state*, which the target's indexer
        writes each step for the anchor's source row; it is not the batch row.
      - the block's slots must be distinct from the picks: a duplicate row is attended twice,
        which changes the softmax weights without any visible inconsistency.

    A `packed` narrower than `topk + width` is refused loudly (`ValueError`) before the launch:
    the block store would run past each row's count and the sparse attention's `min(length, W)`
    would silently drop the block. `is_valid` refuses what is measured to go silently wrong
    instead: a `-1` block slot on an active request (a pad row inside the counted region, read
    as a real latent row by a caller that only bounds the ids by the pool), a `history` whose
    last dimension is strided (the kernel reads it dense, so a `[..., ::2]` view selects
    adjacent columns), and `state_rows` outside the picks state (an out-of-range read; the
    probe is unsafe and is not run). An empty batch returns after the buffers are left as they
    are.

    Measured on sm_103 (GB300): every element of `packed` and `lengths` bitwise equal to
    `reference` in all six cells -- 5 x 6 rows at one request, 5 x 384 at 64 requests, the
    37-pick compaction, the mixed-activity rows, the empty-pick rows (counts 1..6) and the
    drafter's own per-layer slice: one layer, 64 requests, the state at the engine's 8192-row
    per-step budget and the caller's 2176-wide packed. Of the
    246480 `packed` entries the control cells write, a selection that appends all six block
    slots to every row moves 300 (and every count), one that leaves the picks uncompacted
    moves 840, and one read from a strided pick list moves 130860.
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
                f"topk + width = {topk + width} columns for the picks and the block"
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
        # Compact: the valid picks in their own order at the front, everything else -1. The
        # invalid entries scatter into a spare column so they never land inside [0, n).
        tmp = torch.full(
            (layers, gens, width, topk + 1), PAD_SLOT_ID, dtype=torch.int32, device=device
        )
        dest = torch.where(valid, order, torch.full_like(order, topk))
        # The index is `topk` wide against the `topk + 1` buffer: the spare column only ever
        # receives the invalid picks, which scatter there instead of inside `[0, n)`.
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
                # `counts` of an inactive request may be anything; clamp so the scatter stays
                # inside the row (the whole row is overwritten with -1 below anyway).
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
        assert history.dtype is torch.int32 and history.stride(-1) == 1, (
            "history must be int32 with a dense last dimension: the kernel reads each row's "
            "2048 columns at stride 1, so a strided view selects the wrong picks"
        )
        assert state_rows.dtype is torch.int64 and state_rows.shape == (gens,)
        assert block_slots.dtype is torch.int32 and block_slots.is_contiguous()
        assert active.dtype is torch.bool and active.shape == (gens,)
        assert packed.dtype is torch.int32 and packed.stride(-1) == 1
        assert lengths.dtype is torch.int32 and lengths.shape == (layers, gens * width)
        assert lengths.stride(0) == gens * width, (
            "lengths must be dense: the kernel writes each layer's counts at its layer stride"
        )
        if gens == 0:
            return
        live = active.nonzero().flatten()
        if live.numel() == 0:
            return
        rows = state_rows[live]
        assert int(rows.min()) >= 0 and int(rows.max()) < states, (
            f"state_rows must index the {states} rows of the picks state; got "
            f"[{int(rows.min())}, {int(rows.max())}]"
        )
        slots = block_slots[live]
        assert int(slots.min()) >= 0, (
            f"an active request has a block slot of {int(slots.min())}: a pad row inside the "
            f"counted region is read as a real latent row"
        )


dfly_block_indices = DflyBlockIndices()
