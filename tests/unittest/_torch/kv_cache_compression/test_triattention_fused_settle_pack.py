# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The settle and pack kernels vs pure-torch integer oracles:
``_settle_ties_kernel`` (threshold recovery with sentinel-skip,
strictly-greater count, lowest-index tie quota, ascending prompt-rebased
emission) and ``_pack_move_sources_kernel`` (dense/SWA packing, fed
pre-settled rows). All outputs are integers, so comparisons are
``torch.equal`` including stale regions."""

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.compaction import _pack_move_sources_kernel
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    _settle_ties_kernel,
)

# Both kernels share the settle geometry parameter grid.
_WIDTH_KEEP_CASES = [
    # Small and ragged: rows shorter than the keep count, empty rows.
    (21, 5),
    # More than one 256-lane block along both the settle and move axes.
    (350, 300),
]


def _settle_oracle(scores, row_lengths, row_prompt_offsets, provisional, output, keep_count):
    """Settle in place: threshold = min over non-sentinel provisional
    lanes; keep strictly-greater, fill quota with lowest-index ties, emit
    ascending rebased by prompt; entries past the emitted count stay."""
    rows_total, width = scores.shape
    for row in range(rows_total):
        lanes = [int(i) for i in provisional[row, :keep_count] if int(i) >= 0]
        threshold = min((float(scores[row, i]) for i in lanes), default=float("inf"))
        length = min(width, int(row_lengths[row]))
        row_scores = scores[row, :length].tolist()
        greater = [i for i, s in enumerate(row_scores) if s > threshold]
        ties = [i for i, s in enumerate(row_scores) if s == threshold]
        quota = max(0, keep_count - len(greater))
        selected = sorted(greater + ties[:quota])
        if selected:
            prompt = int(row_prompt_offsets[row])
            output[row, : len(selected)] = torch.tensor(
                [i + prompt for i in selected], dtype=output.dtype, device=output.device
            )


def _pack_oracle(
    settled,
    valid_seq_lens,
    dense_offsets,
    dense_out,
    swa_offsets,
    swa_out,
    *,
    selection_rows,
    keep_count,
    num_kv_heads,
    swa_window,
    union,
    per_layer,
    has_swa,
):
    """Pack in place: dense rows forward settled content verbatim (stale
    included) then append the tail ``seq_len + move - keep_count``; SWA
    rows write latest-window ordinals once per KV head."""
    request_count = int(valid_seq_lens.shape[0])
    packed_rows = int(dense_out.shape[0])
    dense_total = int(dense_out.shape[1])
    swa_total = int(swa_out.shape[1])
    for request in range(request_count):
        seq_len = int(valid_seq_lens[request])
        dense_begin = int(dense_offsets[request])
        dense_count = int(dense_offsets[request + 1]) - dense_begin
        for domain in range(packed_rows):
            selection_domain = 0 if union else domain
            settled_row = settled[request * selection_rows + selection_domain]
            for move in range(dense_count):
                value = int(settled_row[move]) if move < keep_count else seq_len + move - keep_count
                dense_out.view(-1)[domain * dense_total + dense_begin + move] = value
            if has_swa and (domain < num_kv_heads if per_layer else True):
                swa_begin = int(swa_offsets[request])
                swa_count = int(swa_offsets[request + 1]) - swa_begin
                head = domain % num_kv_heads
                for move in range(swa_count):
                    swa_out.view(-1)[head * swa_total + swa_begin + move] = (
                        seq_len - swa_window + move
                    )


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


def _make_settle_inputs(request_count, selection_rows, width, keep_count, seed, device):
    """One seeded settle problem: heavily tied scores, ragged rows, a
    top-k stand-in, and per-request prompt rebases."""
    rows_total = request_count * selection_rows
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
    # Per-request prompt lengths (the kernel indexes them by request).
    prompt_offsets = torch.tensor(
        [3 * (request % 3) for request in range(request_count)], dtype=torch.int32, device=device
    )
    # Stand-in for the CuTE top-k: in-range indices covering the top
    # scores of each row with arbitrary tie breaking.
    masked = scores.clone()
    for row in range(rows_total):
        masked[row, int(row_lengths[row]) :] = float("-inf")
    provisional = torch.topk(masked, keep_count, dim=1).indices.to(torch.int32).contiguous()
    return scores, row_lengths, prompt_offsets, provisional


@pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
@pytest.mark.parametrize("width,keep_count", _WIDTH_KEEP_CASES)
def test_settle_matches_torch_oracle(eviction_mode, width, keep_count):
    device = torch.device("cuda", torch.cuda.current_device())
    request_count, num_layers, num_kv_heads = 3, 2, 2
    selection_rows = _selection_rows_for(eviction_mode, num_layers, num_kv_heads)
    rows_total = request_count * selection_rows

    for seed in range(5):
        scores, row_lengths, prompt_offsets, provisional = _make_settle_inputs(
            request_count, selection_rows, width, keep_count, seed, device
        )
        row_prompt_offsets = prompt_offsets.repeat_interleave(selection_rows)
        # Identical stale garbage on both sides so untouched regions must
        # match too.
        output_stale = torch.randint(
            -(2**30), 2**30, (rows_total, keep_count), dtype=torch.int32, device=device
        )
        output_reference = output_stale.clone()
        _settle_oracle(
            scores, row_lengths, row_prompt_offsets, provisional, output_reference, keep_count
        )

        output_actual = output_stale.clone()
        _settle_ties_kernel[(request_count, selection_rows)](
            scores,
            row_lengths,
            prompt_offsets,
            provisional,
            output_actual,
            WIDTH=width,
            KEEP_COUNT=keep_count,
            SELECTION_ROWS=selection_rows,
        )
        torch.cuda.synchronize(device)

        assert torch.equal(output_actual, output_reference), f"kept ordinals differ (seed {seed})"


@pytest.mark.parametrize("has_swa", [False, True])
@pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
@pytest.mark.parametrize("width,keep_count", _WIDTH_KEEP_CASES)
def test_pack_matches_torch_oracle_on_pre_settled_rows(eviction_mode, has_swa, width, keep_count):
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
        scores, row_lengths, prompt_offsets, provisional = _make_settle_inputs(
            request_count, selection_rows, width, keep_count, seed, device
        )
        row_prompt_offsets = prompt_offsets.repeat_interleave(selection_rows)
        # Pre-settled decision rows straight from the settle oracle: short
        # rows keep stale garbage past their emitted count, which the pack
        # must forward verbatim.
        settled = torch.randint(
            -(2**30), 2**30, (rows_total, keep_count), dtype=torch.int32, device=device
        )
        _settle_oracle(scores, row_lengths, row_prompt_offsets, provisional, settled, keep_count)

        # Identical stale garbage on both sides so untouched regions must
        # match too.
        dense_stale = torch.randint(
            -(2**30), 2**30, (packed_rows, dense_total), dtype=torch.int32, device=device
        )
        swa_stale = torch.randint(
            -(2**30), 2**30, (num_kv_heads, swa_total), dtype=torch.int32, device=device
        )
        dense_reference = dense_stale.clone()
        swa_reference = swa_stale.clone()
        _pack_oracle(
            settled,
            valid_seq_lens,
            dense_offsets,
            dense_reference,
            swa_offsets if has_swa else dense_offsets,
            swa_reference if has_swa else dense_reference,
            selection_rows=selection_rows,
            keep_count=keep_count,
            num_kv_heads=num_kv_heads,
            swa_window=swa_window if has_swa else 0,
            union=union,
            per_layer=per_layer,
            has_swa=has_swa,
        )

        dense_actual = dense_stale.clone()
        swa_actual = swa_stale.clone()
        _pack_move_sources_kernel[(request_count, selection_rows)](
            settled,
            valid_seq_lens,
            dense_offsets,
            dense_actual,
            swa_offsets if has_swa else None,
            swa_actual if has_swa else None,
            KEEP_COUNT=keep_count,
            DECISION_ROWS=selection_rows,
            MOVE_CAPACITY=move_capacity,
            NUM_KV_HEADS=num_kv_heads,
            PER_LAYER=per_layer,
            DENSE_TOTAL=dense_total,
            SWA_TOTAL=swa_total if has_swa else 0,
            SWA_WINDOW=swa_window if has_swa else 0,
        )
        torch.cuda.synchronize(device)

        assert torch.equal(dense_actual, dense_reference), f"dense moves differ (seed {seed})"
        assert torch.equal(swa_actual, swa_reference), f"SWA moves differ (seed {seed})"


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
    _settle_ties_kernel[(rows_total, 1)](
        scores,
        row_lengths,
        row_prompt_offsets,
        provisional,
        output,
        WIDTH=width,
        KEEP_COUNT=keep_count,
        SELECTION_ROWS=1,
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
