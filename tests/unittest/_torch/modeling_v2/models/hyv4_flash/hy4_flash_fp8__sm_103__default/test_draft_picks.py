# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFly2's picks state: the rows the target publishes and what `DflyBlockIndices` reads back.

A state row is a row of a target step: every generation row and each context request's last row.
"""

import torch
from _target import core, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.cache_manager import Hyv4FlashCacheManager
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    Hyv4FlashCore,
    _picks_rows,
    _recent_rows,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    PAD_SLOT_ID,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    dfly_block_indices as indices,
)
from tensorrt_llm._utils import get_sm_version

assert torch.cuda.is_available(), "the picks state is device memory"
assert get_sm_version() == 103, (
    f"the hyv4_flash target's entries are certified on sm_103, not {get_sm_version()}"
)

DEV = "cuda"
TOKENS_PER_BLOCK = 32
#: Long enough that a selection is the indexer's full top-k.
LONG = 2100
#: Above every pool row a test's pages produce, so a block slot is never a pick.
SLOT_BASE = 1 << 25


def _manager(num_layers: int, rows: int):
    """The real `draft_picks` accessor over a stand-in manager's fields."""
    manager = type("Manager", (), {})()
    manager.max_num_tokens = rows
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager._draft_picks = None
    manager.draft_picks = Hyv4FlashCacheManager.draft_picks.__get__(manager)
    manager.get_primary_pool_page_index_params = lambda layer: (num_layers, layer)
    return manager


def _step(ctx_lens, gen_lens, manager, kv_lens=None):
    """One target step's metadata: the row map and one disjoint run of pages per request."""
    row_request = [i for i, n in enumerate(ctx_lens) for _ in range(n)]
    row_request += [len(ctx_lens) + j for j, n in enumerate(gen_lens) for _ in range(n)]
    step = type("Step", (), {})()
    step.req_idx_per_token = torch.tensor(row_request, dtype=torch.int32, device=DEV)
    step.num_tokens = len(row_request)
    step.num_contexts = len(ctx_lens)
    step.num_ctx_tokens = sum(ctx_lens)
    lengths = list(ctx_lens) + list(gen_lens) if kv_lens is None else list(kv_lens)
    pages = [-(n // -TOKENS_PER_BLOCK) + 2 for n in lengths]
    table = torch.zeros(len(pages), max(pages), dtype=torch.int32, device=DEV)
    base = 0
    for request, count in enumerate(pages):
        table[request, :count] = torch.arange(base, base + count, dtype=torch.int32, device=DEV)
        base += count
    step.block_table = table
    step.kv_cache_manager = manager
    return step


def _capture(dims_, manager, md, positions, rows_arg):
    """Fill the state as one forward does: the decode path's selection, as pool rows, at the step's rows."""
    state = manager.draft_picks(
        dims_.draft_source_layers, dims_.index_topk, rows=rows_arg, device=torch.device(DEV)
    )
    rows = _picks_rows(md)
    stub = type("Core", (), {})()
    stub._pool_row = lambda md, layer: layer
    pool_rows = Hyv4FlashCore._pool_rows.__get__(stub)
    for layer in dims_.draft_source_layers:
        selection = _recent_rows(positions.to(torch.int32), dims_.index_topk)
        state.write(layer, pool_rows(md, layer, selection), rows)
    return state, rows


def _drive(history, state_rows, block_slots, active):
    """Run `DflyBlockIndices` on a captured history against its reference and its layout invariants."""
    layers, _, topk = history.shape
    gens, width = block_slots.shape
    out_width = topk + width
    packed = torch.full(
        (layers, gens * width, out_width), PAD_SLOT_ID, dtype=torch.int32, device=DEV
    )
    lengths = torch.zeros((layers, gens * width), dtype=torch.int32, device=DEV)
    args = dict(history=history, state_rows=state_rows, block_slots=block_slots, active=active)
    indices(**args, packed=packed, lengths=lengths)
    want_packed = torch.full_like(packed, PAD_SLOT_ID)
    want_lengths = torch.zeros_like(lengths)
    indices.reference(**args, packed=want_packed, lengths=want_lengths)
    torch.testing.assert_close(packed, want_packed)
    torch.testing.assert_close(lengths, want_lengths)

    # Each row: its request's valid picks compacted to the front, the block's causal prefix, padding.
    counts = (history[:, state_rows.long()] >= 0).sum(-1)
    rows = packed.reshape(layers, gens, width, out_width)
    counts_out = lengths.reshape(layers, gens, width)
    want_counts = torch.where(
        active[None, :, None],
        counts[..., None] + torch.arange(width, device=DEV) + 1,
        torch.zeros_like(counts[..., None]),
    )
    torch.testing.assert_close(counts_out.long(), want_counts)
    live = active.nonzero().flatten()
    counted = torch.arange(out_width, device=DEV)[None, None, None, :] < counts_out[..., None]
    got, inside = rows[:, live], counted[:, live]
    assert bool((got[inside] >= 0).all()), "a counted slot holds a real pool row"
    assert bool((got[~inside] == PAD_SLOT_ID).all()), "past the count a row is padding"
    for r in range(width):
        start = (counts_out[:, live, r] - (r + 1)).clamp_min(0)
        tail = rows[:, live, r].gather(
            -1, start[..., None] + torch.arange(r + 1, device=DEV)[None, None, :]
        )
        want = block_slots[None, live, : r + 1].expand(layers, live.numel(), r + 1)
        assert bool((tail == want).all()), f"row {r}'s block prefix"


def test_the_row_rule_is_every_generation_row_and_each_context_tail():
    manager = _manager(num_layers=42, rows=64)
    assert _picks_rows(_step([3, 2], [6], manager)).tolist() == [2, 4] + list(range(5, 11))
    assert _picks_rows(_step([4, 1], [], manager)).tolist() == [3, 4]
    assert _picks_rows(_step([], [1, 1, 1], manager)).tolist() == [0, 1, 2]
    assert _picks_rows(_step([], [6, 6], manager)).tolist() == list(range(12))
    assert _picks_rows(_step([3], [1], manager)).tolist() == [2, 3]


@requires_checkpoint
def test_a_verify_blocks_rows_are_each_the_anchors_source():
    """64 requests x 6 rows: any block row is readable as the anchor's source, and is its own request's."""
    gens, width = 64, BLOCK_WIDTH
    states = gens * width
    d = dims(core())
    manager = _manager(d.num_layers, rows=states)
    md = _step([], [width] * gens, manager, kv_lens=[LONG] * gens)
    # Same positions in every request; their pages differ, so the pool rows differ.
    positions = torch.tensor(
        [LONG + j for _ in range(gens) for j in range(width)], dtype=torch.int64, device=DEV
    )
    state, rows = _capture(d, manager, md, positions, rows_arg=states)
    assert rows.tolist() == list(range(states))
    source_rows = torch.arange(gens, device=DEV) * width + torch.arange(gens, device=DEV) % width
    slots = (SLOT_BASE + torch.arange(states, device=DEV)).reshape(gens, width).to(torch.int32)
    _drive(
        state.history,
        source_rows.to(torch.int64),
        slots,
        torch.ones(gens, dtype=torch.bool, device=DEV),
    )
    picks = state.history[0].index_select(0, source_rows)
    assert bool((picks >= 0).any(1).all()), "every request's source row carries picks"
    assert not bool((picks[0] == picks[1]).all()), "two requests' rows hold different pool rows"
