# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused settle-and-pack kernel must reproduce the original two-kernel
sequence byte for byte.

The reference legs are the pre-fusion kernels: the tie-settlement copy
kept in this file (its production original was deleted once the fused
kernel became the only launched settle path) and the move-source packing
kernel the module still ships for the draft flow: every case launches them on one set of buffers
and the fused kernel on an identically initialized set, then requires
``torch.equal`` on the kept ordinals, the dense move sources, and the SWA
move sources -- including the buffer regions neither path overwrites (rows
shorter than the keep count leave stale entries behind, and the packing
forwards those stale entries the same way in both paths).
"""

import pytest
import torch
import triton
import triton.language as tl
from conftest import encode_block_offsets as _encode_block_offsets

from tensorrt_llm._torch.kv_cache_compression.triattention.compaction import (
    BatchedKVCacheCompaction,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    _BatchedUnionKeepSetSelector,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    _settle_ties_and_pack_compaction_sources_kernel,
)


@triton.jit
def _pack_compaction_sources_kernel(
    selected_indices,
    valid_seq_lens,
    dense_offsets,
    dense_indices,
    swa_offsets,
    swa_indices,
    DENSE_TOTAL: tl.constexpr,
    SWA_TOTAL: tl.constexpr,
    SELECTION_ROWS: tl.constexpr,
    SELECTION_STRIDE: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    UNION: tl.constexpr,
    PER_LAYER: tl.constexpr,
    HAS_SWA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack selected decode ordinals and protected tails for the C++ updater."""
    request = tl.program_id(0)
    domain = tl.program_id(1)
    move = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)

    dense_begin = tl.load(dense_offsets + request)
    dense_end = tl.load(dense_offsets + request + 1)
    dense_count = dense_end - dense_begin
    seq_len = tl.load(valid_seq_lens + request)

    if UNION:
        selection_domain = 0
    else:
        selection_domain = domain
    # Selection rows carry decode-only kept ordinals (already absolute), so
    # rows are prompt-length independent and one cohort may mix prompt sizes.
    selection_row = request * SELECTION_ROWS + selection_domain
    selected = tl.load(
        selected_indices + selection_row.to(tl.int64) * SELECTION_STRIDE + move,
        mask=move < KEEP_COUNT,
        other=0,
    )
    dense_source = tl.where(move < KEEP_COUNT, selected, seq_len + move - KEEP_COUNT)
    dense_output = domain.to(tl.int64) * DENSE_TOTAL + dense_begin.to(tl.int64) + move
    tl.store(dense_indices + dense_output, dense_source, mask=move < dense_count)

    if HAS_SWA:
        # Per-layer selection has one dense domain per (layer, head). SWA uses
        # one shared source row per head, so only the first layer writes it.
        if PER_LAYER:
            write_swa = domain < NUM_KV_HEADS
        else:
            write_swa = move >= 0
        swa_begin = tl.load(swa_offsets + request)
        swa_end = tl.load(swa_offsets + request + 1)
        swa_count = swa_end - swa_begin
        head = domain % NUM_KV_HEADS
        swa_output = head.to(tl.int64) * SWA_TOTAL + swa_begin.to(tl.int64) + move
        swa_source = seq_len - SWA_WINDOW + move
        tl.store(
            swa_indices + swa_output,
            swa_source,
            mask=write_swa & (move < swa_count),
        )


# --------------------------------------------------------------------------- #
# Fused finalize: settle the top-k ties and pack the move indices in one      #
# launch.                                                                     #
# --------------------------------------------------------------------------- #


@triton.jit
def _settle_ties_after_topk_kernel(
    scores,
    seq_lens,
    prompt_offsets,
    provisional_indices,
    output_indices,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Resolve boundary ties and emit increasing physical token indices.

    Pre-fusion standalone kept verbatim as the fused kernel's bit-equality
    reference; the production module ships only the fused launch.
    """
    row = tl.program_id(0)
    row_scores = scores + row * WIDTH
    row_selected = provisional_indices + row * KEEP_COUNT
    row_output = output_indices + row * OUTPUT_WIDTH
    # Scores are decode-relative; this row's pinned prompt length rebases the
    # emitted ordinals to absolute positions (per row, so one launch may mix
    # prompt lengths).
    prompt_len = tl.load(prompt_offsets + row)

    threshold = float("inf")
    for start in tl.static_range(0, KEEP_COUNT, BLOCK):
        selected_offset = start + tl.arange(0, BLOCK)
        selected_mask = selected_offset < KEEP_COUNT
        token_index = tl.load(
            row_selected + selected_offset,
            mask=selected_mask,
            other=0,
        )
        selected_score = tl.load(
            row_scores + token_index,
            mask=selected_mask,
            other=float("inf"),
        ).to(tl.float32)
        threshold = tl.minimum(threshold, tl.min(selected_score, axis=0))

    seq_len = tl.load(seq_lens + row)
    greater_count = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        greater_count += tl.sum((valid & (score > threshold)).to(tl.int32))

    tie_quota = KEEP_COUNT - greater_count
    output_count = 0
    ties_seen = 0
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        valid = (token_index < WIDTH) & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        greater = valid & (score > threshold)
        tied = valid & (score == threshold)
        tied_i32 = tied.to(tl.int32)
        tie_rank = ties_seen + tl.cumsum(tied_i32, axis=0) - tied_i32
        selected = greater | (tied & (tie_rank < tie_quota))
        selected_i32 = selected.to(tl.int32)
        write_offset = output_count + tl.cumsum(selected_i32, axis=0) - selected_i32
        tl.store(
            row_output + write_offset,
            token_index + prompt_len,
            mask=selected,
        )
        output_count += tl.sum(selected_i32)
        ties_seen += tl.sum(tied_i32)


_BLOCK = 256
_NUM_WARPS = 4


def _selection_rows_for(eviction_mode: str, num_layers: int, num_kv_heads: int) -> int:
    if eviction_mode == "union":
        return 1
    if eviction_mode == "per_head":
        return num_kv_heads
    return num_layers * num_kv_heads


def _staged_offsets(counts, device):
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


@pytest.mark.parametrize("has_swa", [False, True])
@pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
@pytest.mark.parametrize(
    "width,keep_count",
    [
        # Small and ragged: rows shorter than the keep count, empty rows.
        (21, 5),
        # More than one 256-lane block along both the settle and move axes.
        (350, 300),
    ],
)
def test_fused_settle_pack_matches_two_kernel_sequence(eviction_mode, has_swa, width, keep_count):
    device = torch.device("cuda", torch.cuda.current_device())
    request_count, num_layers, num_kv_heads = 3, 2, 2
    union = eviction_mode == "union"
    per_layer = eviction_mode == "per_layer_perhead"
    selection_rows = _selection_rows_for(eviction_mode, num_layers, num_kv_heads)
    rows_total = request_count * selection_rows
    packed_rows = num_layers * num_kv_heads if per_layer else num_kv_heads

    # Move geometry: the last request is a padded row that moves nothing.
    protected_tails = [2, 0, 0]
    tail_capacity = max(protected_tails)
    dense_counts = [keep_count + protected_tails[0], keep_count + protected_tails[1], 0]
    swa_window = 6
    swa_counts = [swa_window + protected_tails[0], swa_window + protected_tails[1], 0]
    move_capacity = keep_count + tail_capacity
    if has_swa:
        move_capacity = max(move_capacity, swa_window + tail_capacity)
    dense_total = request_count * (keep_count + tail_capacity)
    swa_total = request_count * (swa_window + tail_capacity)
    valid_seq_lens = torch.tensor([10, 8, 0], dtype=torch.int32, device=device)
    dense_offsets = _staged_offsets(dense_counts, device)
    swa_offsets = _staged_offsets(swa_counts, device)

    for seed in range(5):
        generator = torch.Generator(device=device).manual_seed(seed)
        # Heavily tied integer scores force the tie-quota emission path.
        scores = torch.randint(
            -2, 3, (rows_total, width), generator=generator, dtype=torch.int32, device=device
        ).to(torch.float32)
        # Ragged rows: empty, shorter than the keep count (stale output
        # entries survive), and full width.
        row_lengths = torch.tensor(
            [[0, keep_count - 2, width - 4, width][row % 4] for row in range(rows_total)],
            dtype=torch.int32,
            device=device,
        )
        row_prompt_offsets = torch.tensor(
            [3 * (row % 3) for row in range(rows_total)], dtype=torch.int32, device=device
        )
        # Stand-in for the CuTE top-k: in-range indices covering the top
        # scores of each row with arbitrary tie breaking.
        masked = scores.clone()
        for row in range(rows_total):
            masked[row, int(row_lengths[row]) :] = float("-inf")
        provisional = torch.topk(masked, keep_count, dim=1).indices.to(torch.int32).contiguous()

        # Identical stale garbage on both sides so untouched regions must
        # match too.
        output_stale = torch.randint(
            -(2**30), 2**30, (rows_total, keep_count), dtype=torch.int32, device=device
        )
        dense_stale = torch.randint(
            -(2**30), 2**30, (packed_rows, dense_total), dtype=torch.int32, device=device
        )
        swa_stale = torch.randint(
            -(2**30), 2**30, (num_kv_heads, swa_total), dtype=torch.int32, device=device
        )

        output_reference = output_stale.clone()
        dense_reference = dense_stale.clone()
        swa_reference = swa_stale.clone()
        _settle_ties_after_topk_kernel[(rows_total,)](
            scores,
            row_lengths,
            row_prompt_offsets,
            provisional,
            output_reference,
            WIDTH=width,
            KEEP_COUNT=keep_count,
            OUTPUT_WIDTH=keep_count,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        swa_offsets_arg = swa_offsets if has_swa else dense_offsets
        swa_reference_arg = swa_reference if has_swa else dense_reference
        _pack_compaction_sources_kernel[
            (request_count, packed_rows, (move_capacity + _BLOCK - 1) // _BLOCK)
        ](
            output_reference,
            valid_seq_lens,
            dense_offsets,
            dense_reference,
            swa_offsets_arg,
            swa_reference_arg,
            DENSE_TOTAL=dense_total,
            SWA_TOTAL=swa_total if has_swa else 0,
            SELECTION_ROWS=selection_rows,
            SELECTION_STRIDE=keep_count,
            KEEP_COUNT=keep_count,
            NUM_KV_HEADS=num_kv_heads,
            SWA_WINDOW=swa_window if has_swa else 0,
            UNION=union,
            PER_LAYER=per_layer,
            HAS_SWA=has_swa,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )

        output_fused = output_stale.clone()
        dense_fused = dense_stale.clone()
        swa_fused = swa_stale.clone()
        swa_fused_arg = swa_fused if has_swa else dense_fused
        _settle_ties_and_pack_compaction_sources_kernel[(request_count, selection_rows)](
            scores,
            row_lengths,
            row_prompt_offsets,
            provisional,
            output_fused,
            valid_seq_lens,
            dense_offsets,
            dense_fused,
            swa_offsets_arg,
            swa_fused_arg,
            WIDTH=width,
            KEEP_COUNT=keep_count,
            OUTPUT_WIDTH=keep_count,
            SELECTION_ROWS=selection_rows,
            DENSE_TOTAL=dense_total,
            SWA_TOTAL=swa_total if has_swa else 0,
            MOVE_CAPACITY=move_capacity,
            NUM_KV_HEADS=num_kv_heads,
            SWA_WINDOW=swa_window if has_swa else 0,
            UNION=union,
            PER_LAYER=per_layer,
            HAS_SWA=has_swa,
            HAS_SETTLE=True,
            HAS_PACK=True,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        torch.cuda.synchronize(device)

        assert torch.equal(output_fused, output_reference), f"kept ordinals differ (seed {seed})"
        assert torch.equal(dense_fused, dense_reference), f"dense moves differ (seed {seed})"
        assert torch.equal(swa_fused, swa_reference), f"SWA moves differ (seed {seed})"


def test_fused_kernel_without_pack_matches_standalone_settle():
    """``HAS_PACK=False`` must leave exactly the standalone settle kernel."""
    device = torch.device("cuda", torch.cuda.current_device())
    rows_total, width, keep_count = 6, 33, 7
    generator = torch.Generator(device=device).manual_seed(11)
    scores = torch.randint(
        -2, 3, (rows_total, width), generator=generator, dtype=torch.int32, device=device
    ).to(torch.float32)
    row_lengths = torch.tensor([0, 3, 9, 17, 33, 33], dtype=torch.int32, device=device)
    row_prompt_offsets = torch.tensor([5, 0, 2, 0, 1, 4], dtype=torch.int32, device=device)
    masked = scores.clone()
    for row in range(rows_total):
        masked[row, int(row_lengths[row]) :] = float("-inf")
    provisional = torch.topk(masked, keep_count, dim=1).indices.to(torch.int32).contiguous()
    output_stale = torch.randint(
        -(2**30), 2**30, (rows_total, keep_count), dtype=torch.int32, device=device
    )

    output_reference = output_stale.clone()
    _settle_ties_after_topk_kernel[(rows_total,)](
        scores,
        row_lengths,
        row_prompt_offsets,
        provisional,
        output_reference,
        WIDTH=width,
        KEEP_COUNT=keep_count,
        OUTPUT_WIDTH=keep_count,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )

    output_fused = output_stale.clone()
    placeholder = row_lengths
    _settle_ties_and_pack_compaction_sources_kernel[(rows_total, 1)](
        scores,
        row_lengths,
        row_prompt_offsets,
        provisional,
        output_fused,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        WIDTH=width,
        KEEP_COUNT=keep_count,
        OUTPUT_WIDTH=keep_count,
        SELECTION_ROWS=1,
        DENSE_TOTAL=0,
        SWA_TOTAL=0,
        MOVE_CAPACITY=0,
        NUM_KV_HEADS=1,
        SWA_WINDOW=0,
        UNION=False,
        PER_LAYER=False,
        HAS_SWA=False,
        HAS_SETTLE=True,
        HAS_PACK=False,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    torch.cuda.synchronize(device)

    assert torch.equal(output_fused, output_reference)


def test_settle_handles_topk_sentinel_padding():
    """Rows shorter than KEEP_COUNT arrive -1-padded and must settle inertly.

    The production top-k pads a row shorter than KEEP_COUNT with -1
    sentinels (a zero-width padded row is all sentinels). The settle's
    threshold gather must skip those lanes -- never touching the score byte
    before the row -- while emitting exactly the real ordinals; the output
    slots past a short row's length stay untouched (rows that short move
    nothing downstream). Full rows must keep byte-identical behavior.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    rows_total, width, keep_count = 4, 33, 7
    generator = torch.Generator(device=device).manual_seed(23)
    scores = torch.randint(
        -2, 3, (rows_total, width), generator=generator, dtype=torch.int32, device=device
    ).to(torch.float32)
    row_lengths = torch.tensor([0, 3, 7, 33], dtype=torch.int32, device=device)
    row_prompt_offsets = torch.tensor([5, 1, 2, 0], dtype=torch.int32, device=device)

    # Provisional rows exactly as the production top-k emits them: rows with
    # length <= KEEP_COUNT carry [0..length) then -1 sentinels; longer rows
    # carry a dense top-k.
    provisional = torch.full((rows_total, keep_count), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(row_lengths.tolist()):
        if length <= keep_count:
            provisional[row, :length] = torch.arange(length, dtype=torch.int32, device=device)
        else:
            masked = scores[row].clone()
            masked[length:] = float("-inf")
            provisional[row] = torch.topk(masked, keep_count).indices.to(torch.int32)

    stale = 0x5EED
    output = torch.full((rows_total, keep_count), stale, dtype=torch.int32, device=device)
    placeholder = row_lengths
    _settle_ties_and_pack_compaction_sources_kernel[(rows_total, 1)](
        scores,
        row_lengths,
        row_prompt_offsets,
        provisional,
        output,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        WIDTH=width,
        KEEP_COUNT=keep_count,
        OUTPUT_WIDTH=keep_count,
        SELECTION_ROWS=1,
        DENSE_TOTAL=0,
        SWA_TOTAL=0,
        MOVE_CAPACITY=0,
        NUM_KV_HEADS=1,
        SWA_WINDOW=0,
        UNION=False,
        PER_LAYER=False,
        HAS_SWA=False,
        HAS_SETTLE=True,
        HAS_PACK=False,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    torch.cuda.synchronize(device)

    for row, length in enumerate(row_lengths.tolist()):
        prompt = int(row_prompt_offsets[row])
        emitted = min(length, keep_count)
        if length > keep_count:
            # Reference keep set: score-descending with lowest-index ties,
            # emitted as ascending absolute ordinals.
            order = sorted(range(length), key=lambda i: (-float(scores[row, i]), i))
            expected = sorted(order[:keep_count])
        else:
            expected = list(range(length))
        expected_row = torch.tensor(
            [ordinal + prompt for ordinal in expected], dtype=torch.int32, device=device
        )
        assert torch.equal(output[row, :emitted], expected_row), f"row {row}"
        assert (output[row, emitted:] == stale).all(), f"row {row} tail"


def test_pack_handoff_disables_compaction_dense_pack_and_selector_validates_buffers():
    """The handoff exports the live move buffers, drops the compaction-time
    dense pack launch, and the selector only accepts a packing that reads its
    own keep buffer."""
    device = torch.device("cuda", torch.cuda.current_device())
    request_count, num_kv_heads, keep_count, width = 2, 2, 4, 16
    # BatchedKVCacheCompaction admits only bf16 pools in the compact op's
    # supported geometry (32/128-token pages, head_dim 64/128).
    tokens_per_block, head_dim = 32, 64
    pools = [
        torch.zeros(
            6, 2, num_kv_heads, tokens_per_block, head_dim, dtype=torch.bfloat16, device=device
        )
        for _ in range(2)
    ]
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)

    selector = _BatchedUnionKeepSetSelector(
        rows=3,
        width=width,
        keep_count=keep_count,
        dtype=torch.float32,
        device=device,
        max_requests=request_count,
    )

    def build_compaction(kept_token_ordinals):
        return BatchedKVCacheCompaction(
            eviction_mode="union",
            layer_pools=pools,
            dense_layers=[0, 1],
            swa_layers=[],
            layer_group_representative={0: 0, 1: 1},
            layer_pool_keys=[("dense", 0), ("dense", 0)],
            kept_token_ordinals=kept_token_ordinals,
            valid_sequence_lengths=torch.full(
                (request_count,), 10, dtype=torch.int32, device=device
            ),
            kv_block_offsets=_encode_block_offsets(page_tables.unsqueeze(0)),
            page_table_slots={0: 0, 1: 0},
            request_count=request_count,
            prompt_offsets=torch.zeros(request_count, dtype=torch.int32, device=device),
            decode_keep_count=keep_count,
            swa_window=None,
            protected_tail_capacity=1,
        )

    compaction = build_compaction(selector.keep)
    assert compaction.target_dense_compaction.move_index_pack is not None

    pack_arguments = compaction.hand_move_source_pack_to_selection()
    assert compaction.target_dense_compaction.move_index_pack is None
    assert len(compaction.cache_compactions) == 1
    assert compaction.cache_compactions[0] is compaction.target_dense_compaction
    assert pack_arguments.dense_indices is compaction.target_dense_compaction.move_source_indices
    assert pack_arguments.dense_offsets is compaction.target_dense_compaction.move_source_offsets

    selector.fuse_move_source_pack(pack_arguments)
    assert selector._move_source_pack is pack_arguments

    # A packing built over any other keep buffer must be rejected: the fused
    # kernel reads back the ordinals it just wrote.
    foreign = build_compaction(torch.zeros_like(selector.keep))
    with pytest.raises(ValueError, match="keep buffer"):
        selector.fuse_move_source_pack(foreign.hand_move_source_pack_to_selection())
