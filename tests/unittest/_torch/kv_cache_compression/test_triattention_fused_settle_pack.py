# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TriAttention settle kernel vs a pure-torch integer oracle."""

import pytest
import torch

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import settle_ties

# Settle geometry rows: the width/keep axes flip the WIDTH and KEEP_COUNT
# static_range trip counts across the 256-lane BLOCK.
_WIDTH_KEEP_CASES = [
    # Small and ragged: rows shorter than the keep count, empty rows.
    (21, 5),
    # More than one 256-lane block along both the settle and move axes.
    (350, 300),
]


def _settle_oracle(scores, row_lengths, row_prompt_offsets, provisional, output, keep_count):
    """Settle tied provisional scores deterministically into ascending ordinals."""
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


def _selection_rows_for(eviction_mode: str, num_layers: int, num_kv_heads: int) -> int:
    if eviction_mode == "union":
        return 1
    if eviction_mode == "per_head":
        return num_kv_heads
    return num_layers * num_kv_heads


def _make_settle_inputs(request_count, selection_rows, width, keep_count, seed, device):
    """Build a seeded, tied, ragged settle problem with prompt rebasing."""
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


# SELECTION_ROWS is stride/grid arithmetic only (no static branch): one
# single-row and one multi-row mode pin every settle path.
@pytest.mark.parametrize("eviction_mode", ["union", "per_layer_perhead"])
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
        settle_ties(
            scores,
            row_lengths,
            prompt_offsets,
            provisional,
            output_actual,
            request_count=request_count,
            selection_rows_per_request=selection_rows,
        )
        torch.cuda.synchronize(device)

        assert torch.equal(output_actual, output_reference), f"kept ordinals differ (seed {seed})"


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
    settle_ties(
        scores,
        row_lengths,
        row_prompt_offsets,
        provisional,
        output,
        request_count=rows_total,
        selection_rows_per_request=1,
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
