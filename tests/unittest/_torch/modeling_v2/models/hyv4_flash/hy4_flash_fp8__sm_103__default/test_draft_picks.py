# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFly2's picks state: what the target publishes and what the drafter reads.

The five draft layers attend over their target sparse layer's top-2048 picks of
the anchor's source token, as pool rows. The core computes that selection for
every row of every step anyway, so the state is a side effect of it:
`cache_manager.DraftPicks`, written from `Hyv4FlashCore._layer_forward` through
the manager both the target and a sharing draft model reach.

What a state row is is the whole contract: a row of a target *step*, not a batch
row and not a request slot. The writer fills every generation row (which of a
verify block's rows the sampler accepted is known only after the step) and each
context request's last row (the prompt tail a first block is built from), so the
drafter names the row its anchor's source token occupied in the last step the
target ran -- `DflyBlockPrepare`'s `source_rows`.

These tests drive `DflyBlockIndices` (the consumer, `ops/spec/dfly_block_prepare.py`)
with a *captured* history: the buffer a real step's row rule and pool-row
conversion filled, at the cells both entries certify and at the real verify
envelope (64 requests x 6 rows). The negative controls are the rows and layers
the writer must not fill.
"""

import torch

# The checkpoint stand-in this directory's tests share.
from _target import core, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.cache_manager import (
    NO_PICK,
    DraftPicks,
    Hyv4FlashCacheManager,
)
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
#: A request long enough that its selection is the indexer's full top-k, the
#: state's worst case (`DflyBlockIndices`'s `valid: 2048`).
LONG = 2100
#: Above every pool row a test's pages produce, so a block slot is never a pick
#: (`DflyBlockIndices` requires the block's slots to be distinct from the picks).
SLOT_BASE = 1 << 25


# --------------------------------------------------------------------------- the step


def _manager(num_layers: int, rows: int):
    """The manager the forward and the drafter both reach, as far as this state is concerned.

    `draft_picks` is the real accessor (allocation, idempotence, geometry check)
    over a stand-in's own fields: the per-step token budget it sizes the state
    from, and the primary pool's page scale and layer offsets, the real DSA
    manager's `(num_local_layers, local_layer_idx)`.
    """
    manager = type("Manager", (), {})()
    manager.max_num_tokens = rows
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager._draft_picks = None
    manager.draft_picks = Hyv4FlashCacheManager.draft_picks.__get__(manager)
    manager.get_primary_pool_page_index_params = lambda layer: (num_layers, layer)
    return manager


def _step(ctx_lens, gen_lens, manager, kv_lens=None):
    """One target step's metadata, as the row rule and the pool conversion read it.

    `req_idx_per_token` is the step's row map: the context requests' rows first,
    request by request, then one row per decoding request (here a whole verify
    block per request). `num_tokens`, `num_contexts` and `num_ctx_tokens` are
    the host ints the forward reads. `block_table` is the step's page table: one
    disjoint run of pages per request, from page 0 up, as long as the request's
    own KV length (`kv_lens`, defaulting to what this step runs for it), so a
    position resolves through the pages of the request that owns it.
    """
    row_request = [i for i, n in enumerate(ctx_lens) for _ in range(n)]
    row_request += [len(ctx_lens) + j for j, n in enumerate(gen_lens) for _ in range(n)]
    step = type("Step", (), {})()
    step.req_idx_per_token = torch.tensor(row_request, dtype=torch.int32, device=DEV)
    step.num_tokens = len(row_request)
    step.num_contexts = len(ctx_lens)
    step.num_ctx_tokens = sum(ctx_lens)
    lengths = list(ctx_lens) + list(gen_lens) if kv_lens is None else list(kv_lens)
    assert len(lengths) == len(ctx_lens) + len(gen_lens)
    pages = [-(n // -TOKENS_PER_BLOCK) + 2 for n in lengths]
    table = torch.zeros(len(pages), max(pages), dtype=torch.int32, device=DEV)
    base = 0
    for request, count in enumerate(pages):
        table[request, :count] = torch.arange(base, base + count, dtype=torch.int32, device=DEV)
        base += count
    step.block_table = table
    step.kv_cache_manager = manager
    return step


def _positions(ctx_lens, gen_positions) -> torch.Tensor:
    """`[T]` int64: each row's own key position, as the step's `position_ids` carry it."""
    values = [p for n in ctx_lens for p in range(n)]
    values += list(gen_positions)
    return torch.tensor(values, dtype=torch.int64, device=DEV)


def _convert(dims_, md, layer: int, selection: torch.Tensor) -> torch.Tensor:
    """The decode path's own conversion, `Hyv4FlashCore._pool_rows`, over a stand-in core.

    `_pool_rows` reads only `_pool_row` off the core, and the page scale and this
    layer's offset off the manager; both are the real manager's own values here
    (`_manager`), and the op it calls is the target's entry -- a request-local
    position resolved through the step's page table, an unresolvable one left at
    -1.
    """
    stub = type("Core", (), {})()
    stub._pool_row = lambda md, layer: layer
    return Hyv4FlashCore._pool_rows.__get__(stub)(md, layer, selection)


def _capture(dims_, manager, md, positions, rows_arg=None):
    """Fill the state the way one forward does; return it, the rows, and what each layer published.

    Per draft-source layer: the step's selection (`_recent_rows`, the layout the
    decode path hands the sparse entry), converted to pool rows by the decode
    path, then written at the step's own rows. `published[layer]` is the
    converted selection -- the values the state must hold after the step.
    """
    state = manager.draft_picks(
        dims_.draft_source_layers,
        dims_.index_topk,
        rows=dims_.max_num_tokens if rows_arg is None else rows_arg,
        device=torch.device(DEV),
    )
    rows = _picks_rows(md)
    published = {}
    for layer in dims_.draft_source_layers:
        selection = _recent_rows(positions.to(torch.int32), dims_.index_topk)
        packed = _convert(dims_, md, layer, selection)
        published[layer] = packed
        state.write(layer, packed, rows)
    return state, rows, published


def _dims():
    """The checkpoint's own geometry, read the way the target reads it."""
    return dims(core())


# --------------------------------------------------------------------------- the row rule


def test_the_row_rule_is_every_generation_row_and_each_context_tail():
    """A step publishes its generation rows and its context requests' tails, nothing else."""
    manager = _manager(num_layers=42, rows=64)
    # A mixed step: request 0's three new prompt tokens, request 1's two, then a
    # verify block of six rows for request 2.
    assert _picks_rows(_step([3, 2], [6], manager)).tolist() == [2, 4] + list(range(5, 11))
    # A context chunk publishes only the prompt tail of each request.
    assert _picks_rows(_step([4, 1], [], manager)).tolist() == [3, 4]
    # A pure decode step publishes every row it runs, one per request.
    assert _picks_rows(_step([], [1, 1, 1], manager)).tolist() == [0, 1, 2]
    # And a verify-only step publishes the whole block of every request.
    assert _picks_rows(_step([], [6, 6], manager)).tolist() == list(range(12))
    # A context row that is not its request's tail is never published: the rule
    # reads the row map, not the row's place in the step.
    assert _picks_rows(_step([3], [1], manager)).tolist() == [2, 3]


# --------------------------------------------------------------------------- the buffer


@requires_checkpoint
def test_the_buffer_holds_the_decode_paths_own_selection():
    """Every published row carries that layer's own pool rows; the rest stays padding."""
    d = _dims()
    manager = _manager(d.num_layers, rows=16)
    # Two context requests (5 and 2 prompt rows) and a verify block for a third.
    md = _step([5, 2], [6], manager)
    positions = _positions([5, 2], range(12, 18))
    state, rows, published = _capture(d, manager, md, positions, rows_arg=16)

    assert state.layers == d.draft_source_layers, "one slot per draft-source layer, in order"
    assert state.history.shape == (len(d.draft_source_layers), 16, d.index_topk)
    assert state.history.dtype is torch.int32, "the drafter reads int32 pool rows"
    assert state.topk == d.index_topk

    written = state.history.index_select(1, rows)
    for k, layer in enumerate(d.draft_source_layers):
        want = published[layer].index_select(0, rows)
        torch.testing.assert_close(written[k], want)
        assert int((written[k] >= 0).sum()) == int((want >= 0).sum()) > 0, (
            f"layer {layer}'s own selection published nothing, or its padding was filled"
        )
    # The layer axis matters: each layer's pool offset shifts every pick.
    assert not torch.equal(state.history[0], state.history[1])
    # The rows the rule did not publish are padding in every layer, and they are
    # exactly the context rows that are not their request's tail.
    unpublished = torch.tensor(
        [r for r in range(md.num_tokens) if r not in set(rows.tolist())], device=DEV
    )
    assert unpublished.tolist() == [0, 1, 2, 3, 5]
    padding = state.history.index_select(1, unpublished)
    assert bool((padding == NO_PICK).all()), "a row the rule did not publish is still padding"


@requires_checkpoint
def test_a_layer_outside_the_draft_source_list_publishes_nothing():
    """The write is the draft-source layers' alone: another sparse layer leaves the state alone."""
    d = _dims()
    manager = _manager(d.num_layers, rows=8)
    md = _step([], [1, 1], manager)
    state, rows, _ = _capture(d, manager, md, _positions([], [3, 7]), rows_arg=8)
    before = state.history.clone()
    other = next(i for i in d.sparse_layers if i not in d.draft_source_layers)
    state.write(
        other, torch.full((md.num_tokens, d.index_topk), 5, dtype=torch.int32, device=DEV), rows
    )
    assert state.slot(other) is None
    torch.testing.assert_close(state.history, before)


@requires_checkpoint
def test_the_state_is_allocated_once_and_refuses_another_geometry():
    """Both ends reach the same buffer: the second call returns it, a different one is refused."""
    d = _dims()
    manager = _manager(d.num_layers, rows=64)
    first = manager.draft_picks(d.draft_source_layers, d.index_topk, device=torch.device(DEV))
    assert (
        manager.draft_picks(d.draft_source_layers, d.index_topk, device=torch.device(DEV)) is first
    )
    assert first.history.shape == (5, 64, d.index_topk), "sized from the manager's token budget"
    assert bool((first.history == NO_PICK).all()), "a fresh state holds no pick"
    try:
        manager.draft_picks(d.draft_source_layers, d.index_topk // 2, device=torch.device(DEV))
    except ValueError as refused:
        assert "already holds layers" in str(refused), str(refused)
    else:
        raise AssertionError("a second geometry was served by the first state's buffer")
    alone = DraftPicks((25,), 2, 4, torch.device(DEV))
    assert alone.history.shape == (1, 2, 4) and alone.history.dtype is torch.int32
    assert alone.slot(25) == 0 and alone.slot(29) is None


# --------------------------------------------------------------------------- the drafter


def _drive(history, state_rows, block_slots, active):
    """Drive `DflyBlockIndices` on a captured history, against its own mirror and invariants."""
    layers, states, topk = history.shape
    gens, width = block_slots.shape
    out_width = topk + width
    packed = torch.full(
        (layers, gens * width, out_width), PAD_SLOT_ID, dtype=torch.int32, device=DEV
    )
    lengths = torch.zeros((layers, gens * width), dtype=torch.int32, device=DEV)
    indices(
        history=history,
        state_rows=state_rows,
        block_slots=block_slots,
        active=active,
        packed=packed,
        lengths=lengths,
    )
    want_packed = torch.full_like(packed, PAD_SLOT_ID)
    want_lengths = torch.zeros_like(lengths)
    indices.reference(
        history=history,
        state_rows=state_rows,
        block_slots=block_slots,
        active=active,
        packed=want_packed,
        lengths=want_lengths,
    )
    torch.testing.assert_close(packed, want_packed)
    torch.testing.assert_close(lengths, want_lengths)

    # Independent of the mirror: each row holds its own request's valid picks
    # compacted to the front, then the block's causal prefix, then padding.
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
        if not bool((tail == want).all()):
            raise AssertionError(
                f"row {r}'s block prefix: got {tail.flatten().tolist()}, "
                f"want {want.flatten().tolist()}"
            )
    return packed, lengths


def _slots(gens: int, width: int = BLOCK_WIDTH) -> torch.Tensor:
    """`[gens, width]` block slots: real rows, distinct from every pick (above the pool)."""
    return (SLOT_BASE + torch.arange(gens * width, device=DEV)).reshape(gens, width).to(torch.int32)


def _all(gens: int) -> torch.Tensor:
    return torch.ones(gens, dtype=torch.bool, device=DEV)


@requires_checkpoint
def test_the_drafter_reads_captured_picks_at_one_request():
    """The entry's small cell (5 layers, 1 request, 8 state rows, full rows) on a captured history."""
    d = _dims()
    manager = _manager(d.num_layers, rows=8)
    # One context request of two rows and one verify block of six: eight state rows.
    md = _step([2], [BLOCK_WIDTH], manager, kv_lens=[2, LONG])
    positions = _positions([2], range(LONG, LONG + BLOCK_WIDTH))
    state, rows, published = _capture(d, manager, md, positions, rows_arg=8)
    assert rows.tolist() == [1] + list(range(2, 8)), "the step's own row rule"
    # The anchor's source is whichever of the block's rows the sampler accepted;
    # the first and the last are the extremes.
    for source_row in (7, 2):
        _drive(
            state.history,
            torch.tensor([source_row], dtype=torch.int64, device=DEV),
            _slots(1),
            _all(1),
        )
    # The picks the entry read are the ones the target published for those rows:
    # past top-k tokens the selection is the indexer's full 2048, in pool rows.
    layer = d.draft_source_layers[0]
    assert int((published[layer][7] >= 0).sum()) == d.index_topk
    assert int((published[layer][2] >= 0).sum()) == d.index_topk
    assert bool((state.history[:, 7] >= 0).all()) and bool((state.history[:, 2] >= 0).all())


@requires_checkpoint
def test_the_drafter_reads_captured_picks_at_sixty_four_requests():
    """The entry's large cell (5 layers, 64 requests, 64 state rows, full rows)."""
    gens = 64
    d = _dims()
    manager = _manager(d.num_layers, rows=gens)
    md = _step([], [1] * gens, manager, kv_lens=[LONG] * gens)
    positions = _positions([], range(LONG, LONG + gens))
    state, rows, published = _capture(d, manager, md, positions, rows_arg=gens)
    assert rows.tolist() == list(range(gens)), "one generation row per request"
    assert int((published[d.draft_source_layers[-1]][rows] >= 0).sum()) == gens * d.index_topk, (
        "every request's row is the indexer's full selection"
    )
    _drive(
        state.history, torch.arange(gens, dtype=torch.int64, device=DEV), _slots(gens), _all(gens)
    )


@requires_checkpoint
def test_a_verify_blocks_rows_are_each_the_anchors_source():
    """The real envelope: 64 requests x 6 rows, every block row readable as the anchor's source.

    A verify step's rows are its block's, and which of them the sampler accepted
    is known only after the step -- so the state holds all six rows per request
    and the drafter names the accepted one. What a request reads must be its own
    picks: neighbouring requests' rows hold different pool rows.
    """
    gens, width = 64, BLOCK_WIDTH
    states = gens * width
    d = _dims()
    manager = _manager(d.num_layers, rows=states)
    md = _step([], [width] * gens, manager, kv_lens=[LONG] * gens)
    # Every request runs the same block at the same positions; their pages differ,
    # so the same position is a different pool row in each request.
    positions = _positions([], [LONG + j for _ in range(gens) for j in range(width)])
    state, rows, _ = _capture(d, manager, md, positions, rows_arg=states)
    assert rows.tolist() == list(range(states)), "every block row of every request is published"
    source_rows = torch.arange(gens, device=DEV) * width + torch.arange(gens, device=DEV) % width
    _drive(state.history, source_rows.to(torch.int64), _slots(gens), _all(gens))
    picks = state.history[0].index_select(0, source_rows)
    assert bool((picks >= 0).any(1).all()), "every request's source row carries picks"
    assert not bool((picks[0] == picks[1]).all()), "two requests' rows hold different pool rows"
