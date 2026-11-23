# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's DFly2 draft-block prepare entries.

Drives every cell both entries certify -- the six-row block of 1, 2, 4 and 64 draft requests
(the 64-request cells at the engine's largest batch, one request accepting none of its drafts,
the page table in the caller's per-layer form), and the per-layer selection of 1, 4, 8 and 64
requests at 2048 picks (the drafter's own call reads one layer's slice of the state, which the
engine sizes by its per-step token budget) -- through the entries' guards and against their
own plain-torch mirrors. Both entries compute integers, so a cell is judged by exact equality
of every output field, not by a band; the negative controls below show that gate rejects the
plausible wrong variants (an anchor one position early, a page early, a non-causal block, an
uncompacted pick list), and the structural checks judge the selection independently of the
mirror.

The environment is the checkpoint's: mask token 120009, block width 6, 2048 picks per target
layer, 32-row pool pages, the five draft layers inside the 12-layer sparse group, and a picks
state whose row count is the request pool's, not the batch's.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    MASK_TOKEN_ID,
    MROPE_PLANES,
    PAD_SLOT_ID,
    PAGE_ROWS,
    TARGET_TOPK,
    DraftBlock,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    dfly_block_indices as indices,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    dfly_block_prepare as prepare,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "dfly_block_prepare requires a CUDA device"
assert get_sm_version() == 103, (
    f"the hyv4_flash target's entries are certified on sm_103, not {get_sm_version()}"
)

PREPARE_CELLS = prepare.CELLS
INDICES_CELLS = indices.CELLS

DEV = "cuda"
VOCAB = 120832


# --------------------------------------------------------------------------- environments


def _prepare_args(spec: dict, seed: int) -> dict:
    """One block-prepare cell: a target batch of `prefix` context tokens then one six-row
    block per draft request, and a page table per request into a shared pool.

    The requests' page ranges are disjoint; with raw page ids (the default) the first request's
    starts at page 0, so a cell drives page 0 as a real page (vLLM pads it; TRT-LLM's manager
    has no null page). `page_scale`/`layer_offset` put the table in the caller's form instead --
    `md.block_table[num_contexts:] * scale + layer_offset`, the page index of one draft layer's
    slab in the coalesced pool -- at the values the manager's
    `get_primary_pool_page_index_params` gives: the sparse group's size 12 and that layer's own
    offset (7..11 for the draft layers 25/29/33/37/41).
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    gens = spec["gens"]
    width = spec.get("block_width", BLOCK_WIDTH)
    prefix, offset = spec.get("prefix", 0), spec.get("pos_offset", 0)
    pages, block_size = spec.get("pages", 256), spec.get("block_size", PAGE_ROWS)
    tokens = prefix + gens * width
    positions = torch.arange(tokens, device=DEV, dtype=torch.int64) + offset
    query_start_loc = torch.tensor(
        [prefix + i * width for i in range(gens + 1)], device=DEV, dtype=torch.int64
    )
    accepted = spec["accepted"]
    if accepted == "all":
        num_accepted = torch.full((gens,), width - 1, device=DEV, dtype=torch.int32)
    elif accepted == "mixed":
        num_accepted = torch.randint(0, width, (gens,), generator=g, device=DEV, dtype=torch.int32)
    elif accepted == "mixed-with-zero":
        # The call site's `num_drafts = (accepted - 1).clamp_min(0)`: a request that rejected
        # every draft hands the block a 0. Mixed counts, with that 0 guaranteed, not sampled.
        num_accepted = torch.randint(1, width, (gens,), generator=g, device=DEV, dtype=torch.int32)
        num_accepted[gens // 2] = 0
    elif accepted == "zero":
        num_accepted = torch.zeros(gens, device=DEV, dtype=torch.int32)
    else:
        raise ValueError(accepted)
    bonus = torch.randint(0, VOCAB, (gens,), generator=g, device=DEV, dtype=torch.int64)
    used = (offset + tokens + block_size - 1) // block_size + 1
    assert used <= pages, (used, pages)
    page_scale, layer_offset = spec.get("page_scale", 1), spec.get("layer_offset", 0)
    table = torch.zeros(gens, pages, dtype=torch.int32, device=DEV)
    page_ids = torch.arange(used, dtype=torch.int32, device=DEV)
    for i in range(gens):
        table[i, :used] = (page_ids + i * used) * page_scale + layer_offset
    return dict(
        bonus=bonus,
        num_accepted=num_accepted,
        query_start_loc=query_start_loc,
        target_positions=positions,
        block_table=table,
        max_model_len=spec.get("max_model_len", 5120),
        mask_token_id=MASK_TOKEN_ID,
        block_width=width,
        block_size=block_size,
    )


def _indices_args(spec: dict, seed: int) -> dict:
    """One selection cell: distinct pool rows as each layer's picks, a request-state array
    wider than the batch, and the block's slots above every pick so a wrong append shows."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    layers, gens, states = spec["layers"], spec["gens"], spec["states"]
    width = spec.get("block_width", BLOCK_WIDTH)
    valid = spec["valid"]
    ranks = torch.rand(layers, states, TARGET_TOPK, generator=g, device=DEV).argsort(-1)
    history = torch.full((layers, states, TARGET_TOPK), PAD_SLOT_ID, dtype=torch.int32, device=DEV)
    if valid > 0:
        history[..., :valid] = ranks[..., :valid].to(torch.int32)
    state_rows = torch.randperm(states, generator=g, device=DEV)[:gens].to(torch.int64)
    pool = 1 << 18
    block_slots = (pool + torch.arange(gens * width, device=DEV)).reshape(gens, width)
    block_slots = block_slots.to(torch.int32)
    if spec["active"] == "all":
        active = torch.ones(gens, dtype=torch.bool, device=DEV)
    elif spec["active"] == "mixed":
        active = torch.arange(gens, device=DEV) % 2 == 0
    else:
        raise ValueError(spec["active"])
    # A padded batch's inactive rows carry pad slots, as DflyBlockPrepare writes them.
    block_slots[~active] = PAD_SLOT_ID
    out_width = spec.get("out_width", TARGET_TOPK + width)
    return dict(
        history=history,
        state_rows=state_rows,
        block_slots=block_slots,
        active=active,
        packed=torch.empty(layers, gens * width, out_width, dtype=torch.int32, device=DEV),
        lengths=torch.empty(layers, gens * width, dtype=torch.int32, device=DEV),
    )


# --------------------------------------------------------------------------- comparisons


def _compare_block(got: DraftBlock, want: DraftBlock) -> None:
    """Every field of the block bitwise equal: integer arithmetic, no rounding to allow for."""
    for name in DraftBlock._fields:
        a, b = getattr(got, name), getattr(want, name)
        assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
        differing = int((a != b).sum())
        assert differing == 0, f"{name} differs in {differing} of {a.numel()} elements"


def _check_block_structure(out: DraftBlock, args: dict) -> None:
    """The block's own invariants, independent of the mirror."""
    gens, width = args["bonus"].shape[0], args["block_width"]
    active = out.active
    j = torch.arange(width, device=DEV)
    ids = out.block_ids.reshape(gens, width)
    positions = out.positions.reshape(gens, width)
    slots = out.slots.reshape(gens, width)
    anchors = out.anchors
    if bool(active.any()):
        assert torch.equal(positions[active], anchors[active, None] + j[None, :])
        assert torch.equal(ids[active][:, 0], args["bonus"][active])
        assert torch.equal(ids[active][:, 1:], torch.full_like(ids[active][:, 1:], MASK_TOKEN_ID))
        assert torch.equal(out.seq_lens[active], (anchors[active] + width).to(torch.int32))
        # Every active row's slot is a real pool row: its page times the page size plus its
        # offset inside the page, and it is the reference's value for that page table.
        assert int(slots[active].min()) >= 0
        assert torch.equal(out.mrope_positions, positions.reshape(1, -1).expand(MROPE_PLANES, -1))
    if bool((~active).any()):
        assert torch.equal(ids[~active], torch.full_like(ids[~active], MASK_TOKEN_ID))
        assert torch.equal(slots[~active], torch.full_like(slots[~active], PAD_SLOT_ID))
        assert torch.equal(out.seq_lens[~active], torch.full_like(out.seq_lens[~active], width))


def _check_selection_structure(packed: torch.Tensor, lengths: torch.Tensor, args: dict) -> None:
    """Each row: the request's valid picks compacted, then the block's causal prefix, then -1.

    Independent of the mirror: the counts are `n + j + 1`, the last `j + 1` entries are the
    block's slots `0 .. j`, nothing inside a count is a pad row, and everything past it is.
    """
    layers, gens = packed.shape[0], args["block_slots"].shape[0]
    width = args["block_slots"].shape[1]
    picks = args["history"][:, args["state_rows"].long()]
    counts = (picks >= 0).sum(-1)  # [L, G]
    rows = packed.reshape(layers, gens, width, -1)
    counts_out = lengths.reshape(layers, gens, width)
    active = args["active"]
    j = torch.arange(width, device=DEV)
    want_counts = torch.where(
        active[None, :, None], counts[..., None] + j + 1, torch.zeros_like(counts[..., None])
    )
    assert torch.equal(counts_out, want_counts)
    inside = torch.arange(packed.shape[2], device=DEV)[None, None, None, :] < counts_out[..., None]
    assert not bool((rows == PAD_SLOT_ID)[inside].any()), "a pad row inside a counted selection"
    live = args["active"]
    for row in range(width):
        start = (counts_out[:, live, row] - (row + 1)).clamp_min(0)
        got = rows[:, live, row].gather(
            -1, start[..., None] + torch.arange(row + 1, device=DEV)[None, None, :]
        )
        want = args["block_slots"][None, live, : row + 1].expand(layers, int(live.sum()), row + 1)
        assert torch.equal(got, want), f"row {row}'s causal prefix is not the block's slots"
    assert bool((rows[~inside] == PAD_SLOT_ID).all()), "a selection's tail is not all pad"


def _compare_selection(args: dict, ref_args: dict) -> None:
    differing = int((args["packed"] != ref_args["packed"]).sum())
    assert differing == 0, f"packed differs in {differing} of {args['packed'].numel()} elements"
    differing = int((args["lengths"] != ref_args["lengths"]).sum())
    assert differing == 0, f"lengths differs in {differing} elements"


# --------------------------------------------------------------------------- cells


@pytest.mark.parametrize("cell", PREPARE_CELLS, ids=[c.why[:44] for c in PREPARE_CELLS])
def test_block_prepare_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture
    certifying(prepare)
    args = _prepare_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = prepare(**args)
    ref = prepare.reference(**args)
    band = sum(int((getattr(out, n) != getattr(ref, n)).sum()) for n in DraftBlock._fields)
    _compare_block(out, ref)
    _check_block_structure(out, args)
    print(
        f"MEASURED {cell.why[:44]!r}: band {band} differing of "
        f"{sum(getattr(out, n).numel() for n in DraftBlock._fields)} field elements, "
        f"{int(out.active.sum())}/{int(out.active.numel())} requests active, "
        f"{int(out.slots.numel())} slots"
    )


@pytest.mark.parametrize("cell", INDICES_CELLS, ids=[c.why[:44] for c in INDICES_CELLS])
def test_block_indices_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture
    certifying(indices)
    args = _indices_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = _indices_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    indices(**args)
    indices.reference(**ref)
    band = int((args["packed"] != ref["packed"]).sum()) + int(
        (args["lengths"] != ref["lengths"]).sum()
    )
    _compare_selection(args, ref)
    _check_selection_structure(args["packed"], args["lengths"], args)
    print(
        f"MEASURED {cell.why[:44]!r}: band {band} differing of {args['packed'].numel()} "
        f"packed entries + {args['lengths'].numel()} counts, "
        f"{int((args['lengths'] > 0).sum())}/{args['lengths'].numel()} rows live"
    )


def test_page_zero_is_a_real_page() -> None:
    """vLLM's allocator reserves page 0 as a null block and its slot kernel pads every slot
    whose page is 0; TRT-LLM's manager has no null page. The first request's block sits in
    page 0 here and its slots are the page's offsets, not pads."""
    args = _prepare_args(dict(gens=1, prefix=0, pos_offset=3, accepted="all", pages=8), seed=5)
    assert int(args["block_table"][0, 0]) == 0
    out = prepare(**args)
    assert bool(out.active[0])
    assert torch.equal(
        out.slots, torch.arange(9, 9 + BLOCK_WIDTH, device=DEV, dtype=torch.int32)
    ), "page 0's slots are its own offsets, not pads"
    _compare_block(out, prepare.reference(**args))


# --------------------------------------------------------------------------- controls


def test_prepare_gate_rejects_an_anchor_one_position_early() -> None:
    """The gate is exact equality, so a block built one position early must not pass it."""
    spec = dict(gens=2, prefix=0, pos_offset=24, accepted="all", pages=8)
    args = _prepare_args(spec, seed=6)
    out = prepare(**args)
    early = prepare(**_prepare_args({**spec, "pos_offset": 23}, seed=6))
    differing = int((out.positions != early.positions).sum())
    print(
        f"MEASURED an anchor one position early moves {differing} of {out.positions.numel()} rows"
    )
    assert differing > 0
    with pytest.raises(AssertionError, match="differs"):
        _compare_block(out, early)


def test_prepare_gate_rejects_the_next_page() -> None:
    """A page table one page along keeps every shape and moves every slot by one page."""
    spec = dict(gens=2, prefix=0, pos_offset=24, accepted="all", pages=8)
    args = _prepare_args(spec, seed=7)
    out = prepare(**args)
    shifted = dict(args)
    shifted["block_table"] = args["block_table"] + 1
    other = prepare(**shifted)
    differing = int((out.slots != other.slots).sum())
    print(f"MEASURED a page table one page along moves {differing} of {out.slots.numel()} slots")
    assert differing > 0
    with pytest.raises(AssertionError, match="differs"):
        _compare_block(out, other)


def test_selection_gate_rejects_a_non_causal_block() -> None:
    """Every block row appending all six slots -- the causal bound dropped -- keeps the shapes
    and moves the counts and the appended rows."""
    spec = dict(layers=5, gens=4, states=16, valid=64, active="all")
    args = _indices_args(spec, seed=8)
    indices(**args)
    layers, gens, width = spec["layers"], spec["gens"], BLOCK_WIDTH
    picks = args["history"][:, args["state_rows"].long()]
    counts = (picks >= 0).sum(-1)
    wrong = torch.full_like(args["packed"], PAD_SLOT_ID)
    wrong_counts = torch.zeros_like(args["lengths"])
    for layer in range(layers):
        for g in range(gens):
            live = picks[layer, g][picks[layer, g] >= 0]
            for j in range(width):
                row = g * width + j
                wrong[layer, row, : counts[layer, g]] = live
                wrong[layer, row, counts[layer, g] : counts[layer, g] + width] = args[
                    "block_slots"
                ][g]
                wrong_counts[layer, row] = counts[layer, g] + width
    differing = int((args["packed"] != wrong).sum())
    print(f"MEASURED a non-causal block moves {differing} of {args['packed'].numel()} entries")
    assert differing > 0
    with pytest.raises(AssertionError, match="differs"):
        _compare_selection(args, dict(packed=wrong, lengths=wrong_counts))


def test_selection_gate_rejects_an_uncompacted_pick_list() -> None:
    """Leaving the picks where the indexer wrote them (holes and all) and appending the block
    after the buffer's full width: a plausible off-by-a-compaction that keeps every count."""
    spec = dict(layers=5, gens=4, states=16, valid=64, active="all")
    args = _indices_args(spec, seed=9)
    indices(**args)
    layers, gens, width = spec["layers"], spec["gens"], BLOCK_WIDTH
    picks = args["history"][:, args["state_rows"].long()]
    wrong = torch.full_like(args["packed"], PAD_SLOT_ID)
    wrong_counts = torch.zeros_like(args["lengths"])
    for layer in range(layers):
        for g in range(gens):
            for j in range(width):
                row = g * width + j
                wrong[layer, row, :TARGET_TOPK] = picks[layer, g]
                wrong[layer, row, TARGET_TOPK : TARGET_TOPK + j + 1] = args["block_slots"][g][
                    : j + 1
                ]
                wrong_counts[layer, row] = TARGET_TOPK + j + 1
    differing = int((args["packed"] != wrong).sum())
    print(f"MEASURED an uncompacted list moves {differing} of {args['packed'].numel()} entries")
    assert differing > 0
    with pytest.raises(AssertionError, match="differs"):
        _compare_selection(args, dict(packed=wrong, lengths=wrong_counts))


# --------------------------------------------------------------------------- guards


def test_prepare_guard_refuses_an_accepted_count_past_the_block() -> None:
    """`source = start + num_accepted` with `num_accepted == block_width` leaves the request's
    own block and lands on the next request's first row: a real position, so the anchor and
    every slot after it are plausible and wrong."""
    args = _prepare_args(dict(gens=2, prefix=0, pos_offset=0, accepted="all", pages=8), seed=10)
    args["num_accepted"] = torch.full((2,), BLOCK_WIDTH, dtype=torch.int32, device=DEV)
    out = prepare(**args)
    next_start = int(args["query_start_loc"][1])
    assert int(out.anchors[0]) == int(args["target_positions"][next_start]) + 1, (
        "the unguarded call was supposed to read the next request's row"
    )
    with pytest.raises(AssertionError, match="num_accepted"):
        with validating(prepare):
            prepare(**args)


def test_prepare_guard_refuses_a_decreasing_query_start_loc() -> None:
    args = _prepare_args(dict(gens=3, prefix=0, pos_offset=0, accepted="all", pages=8), seed=11)
    args["query_start_loc"] = torch.tensor([0, 12, 6, 18], dtype=torch.int64, device=DEV)
    with pytest.raises(AssertionError, match="non-decreasing"):
        with validating(prepare):
            prepare(**args)


def test_prepare_guard_refuses_a_start_location_past_the_positions() -> None:
    """An out-of-range read; the unguarded probe is unsafe and is not run here."""
    args = _prepare_args(dict(gens=2, prefix=0, pos_offset=0, accepted="all", pages=8), seed=12)
    args["query_start_loc"] = torch.tensor([0, 6, 19], dtype=torch.int64, device=DEV)
    with pytest.raises(AssertionError, match="past the"):
        with validating(prepare):
            prepare(**args)


def test_prepare_guard_refuses_a_padded_block_table() -> None:
    """A `[gens, pages]` slice of a wider table keeps every shape; the kernel walks it at the
    width it is given and reads the wrong pages."""
    args = _prepare_args(dict(gens=2, prefix=0, pos_offset=0, accepted="all", pages=8), seed=13)
    wide = torch.arange(32, dtype=torch.int32, device=DEV).reshape(2, 16)
    args["block_table"] = wide[:, :8]
    assert args["block_table"].stride(0) == 16 and args["block_table"].shape == (2, 8)
    got = prepare(**args)
    want = prepare.reference(**{**args, "block_table": args["block_table"].contiguous()})
    differing = int((got.slots != want.slots).sum())
    print(f"MEASURED a padded block table moves {differing} of {got.slots.numel()} slots")
    assert differing > 0, "the unguarded padded table was supposed to be wrong"
    with pytest.raises(AssertionError, match="dense"):
        with validating(prepare):
            prepare(**args)


def test_narrow_packed_buffer_is_refused_loudly() -> None:
    """A `packed` with no room for the block would run each row's count past its width; the
    callable raises instead of writing past the row."""
    args = _indices_args(
        dict(layers=5, gens=4, states=16, valid=TARGET_TOPK, active="all", out_width=TARGET_TOPK),
        seed=14,
    )
    with pytest.raises(ValueError, match=r"topk \+ width"):
        indices(**args)


def test_indices_guard_refuses_a_pad_slot_on_an_active_request() -> None:
    """A `-1` block slot lands inside the counted region: a caller that only bounds the ids by
    the pool reads a pad row as a latent row."""
    args = _indices_args(dict(layers=5, gens=4, states=16, valid=64, active="all"), seed=15)
    args["block_slots"][1, BLOCK_WIDTH - 1] = PAD_SLOT_ID
    indices(**args)
    rows = args["packed"].reshape(5, 4, BLOCK_WIDTH, -1)
    counts = args["lengths"].reshape(5, 4, BLOCK_WIDTH)
    row = BLOCK_WIDTH - 1
    assert int(rows[0, 1, row, counts[0, 1, row] - 1]) == PAD_SLOT_ID, (
        "the unguarded call was supposed to put a pad row inside the count"
    )
    with pytest.raises(AssertionError, match="pad row"):
        with validating(indices):
            indices(**args)


def test_indices_guard_refuses_a_strided_history() -> None:
    """The kernel reads each pick row dense; a `[..., ::2]` view selects adjacent columns.

    The interleaved filler is pool row 0, not `PAD_SLOT_ID`: a pad filler would be dropped by
    the compaction and the strided read would produce the same selection by accident, which
    measures nothing. A real row survives the compaction and moves every count."""
    args = _indices_args(dict(layers=5, gens=4, states=16, valid=64, active="all"), seed=16)
    wide = torch.zeros((5, 16, 2 * TARGET_TOPK), dtype=torch.int32, device=DEV)
    wide[..., ::2] = args["history"]
    strided = wide[..., ::2]
    assert strided.stride(-1) == 2 and torch.equal(strided, args["history"])
    args["history"] = strided
    indices(**args)
    ref = _indices_args(dict(layers=5, gens=4, states=16, valid=64, active="all"), seed=16)
    indices.reference(**ref)
    differing = int((args["packed"] != ref["packed"]).sum())
    print(f"MEASURED a strided pick list moves {differing} of {args['packed'].numel()} entries")
    assert differing > 0, "the unguarded strided read was supposed to be wrong"
    with pytest.raises(AssertionError, match="dense last dimension"):
        with validating(indices):
            indices(**args)


def test_indices_guard_refuses_a_state_row_past_the_state() -> None:
    """An out-of-range read of the picks state; the unguarded probe is unsafe and is not run."""
    args = _indices_args(dict(layers=5, gens=4, states=16, valid=64, active="all"), seed=17)
    args["state_rows"][0] = 16
    with pytest.raises(AssertionError, match="picks state"):
        with validating(indices):
            indices(**args)


# --------------------------------------------------------------------------- capture


def test_entries_are_cuda_graph_capturable() -> None:
    """The drafter runs under CUDA graphs: capture and replay must reproduce the eager result."""
    p_args = _prepare_args(
        dict(gens=6, prefix=64, pos_offset=0, accepted="mixed", pages=16), seed=18
    )
    i_args = _indices_args(
        dict(layers=5, gens=6, states=8, valid=TARGET_TOPK, active="all"), seed=19
    )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            prepare(**p_args)
            indices(**i_args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_block = prepare(**p_args)
        indices(**i_args)
        graph_packed = i_args["packed"].clone()
        graph_lengths = i_args["lengths"].clone()

    fresh = _prepare_args(
        dict(gens=6, prefix=64, pos_offset=0, accepted="mixed", pages=16), seed=20
    )
    p_args["bonus"].copy_(fresh["bonus"])
    p_args["num_accepted"].copy_(fresh["num_accepted"])
    graph.replay()
    torch.cuda.synchronize()

    eager_block = prepare(**p_args)
    _compare_block(graph_block, eager_block)
    eager_ref = _indices_args(
        dict(layers=5, gens=6, states=8, valid=TARGET_TOPK, active="all"), seed=19
    )
    indices.reference(**eager_ref)
    _compare_selection(dict(packed=graph_packed, lengths=graph_lengths), eager_ref)
