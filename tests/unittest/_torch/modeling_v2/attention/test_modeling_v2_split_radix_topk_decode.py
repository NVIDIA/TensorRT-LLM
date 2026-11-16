# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 regression tests for the experimental DSpark partition/radix-merge selector."""

from unittest.mock import patch

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("split radix requires SM107 validation", allow_module_level=True)

import tensorrt_llm._torch.custom_ops  # noqa: E402,F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (  # noqa: E402
    TOP_K,
    blocks_per_row,
    build_split,
    split_radix_topk_decode,
    validate_split_radix_topk_lengths,
)


def _buffers(batch, width, next_n=4):
    rows = batch * next_n
    stride = (width + 255) // 256 * 256
    storage = torch.empty(rows * stride + 8, dtype=torch.float32, device="cuda")
    logits = storage[8:].as_strided((rows, width), (stride, 1))
    output_storage = torch.full((rows + 1, TOP_K), -7, dtype=torch.int32, device="cuda")
    output = output_storage[1:]
    aux_i = torch.full((rows, 10, TOP_K), -7, dtype=torch.int32, device="cuda")
    aux_l = torch.full((rows, 10, TOP_K), float("nan"), dtype=torch.float32, device="cuda")
    return logits, output, aux_i, aux_l


def _fill(logits, lengths, next_n, seed, kind):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits.fill_(1.0e9)
    for row in range(logits.shape[0]):
        end = lengths[row // next_n] - next_n + row % next_n + 1
        scores = torch.randperm(end, generator=generator, device="cuda").float()
        if kind == "ties":
            scores.remainder_(7)
        elif kind == "boundaries":
            for part in range(1, 10):
                boundary = end // 10 * part
                if boundary < end:
                    scores[max(0, boundary - 1) : boundary + 1] += end
        logits[row, :end] = scores


def _check(logits, lengths, output, next_n=4):
    prefix = output.as_strided((TOP_K,), (1,), storage_offset=0)
    assert bool((prefix == -7).all()), "the output view must not overwrite its leading guard"
    for row in range(logits.shape[0]):
        end = lengths[row // next_n] - next_n + row % next_n + 1
        indices = output[row]
        valid = indices[indices >= 0].long()
        assert bool(((indices == -1) | (indices >= 0)).all())
        expected_count = min(TOP_K, end)
        assert valid.numel() == expected_count
        assert bool((indices[:expected_count] >= 0).all()), "valid columns must form a prefix"
        assert bool((indices[expected_count:] == -1).all()), "padding must be a suffix"
        assert valid.unique().numel() == valid.numel()
        assert bool((valid < end).all())
        selected = logits[row, valid].sort().values
        reference = logits[row, :end].topk(min(TOP_K, end)).values.sort().values
        torch.testing.assert_close(selected, reference, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("seed,kind", [(0, "ordinary"), (17, "ties"), (42, "boundaries")])
@pytest.mark.parametrize("width", [16384, 65536])
def test_split_radix_causal_selection(batch, seed, kind, width):
    compiled = build_split(torch.cuda.current_device())
    logits, output, aux_i, aux_l = _buffers(batch, width)
    edge_lengths = [3, 4, 5, 2047, 2048, 2049, 20479, 20480, 20481, width]
    for offset in range(0, len(edge_lengths), batch):
        lengths = [min(width, edge_lengths[(offset + b) % len(edge_lengths)]) for b in range(batch)]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        _fill(logits, lengths, 4, seed, kind)
        validate_split_radix_topk_lengths(logits, lens, next_n=4)
        output.fill_(-7)
        aux_i.fill_(-7)
        aux_l.fill_(float("nan"))
        assert (
            split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled) is output
        )
        _check(logits, lengths, output)


@pytest.mark.parametrize("next_n,batch,width", [(1, 8, 65536), (4, 8, 65536), (4, 1, 1048584)])
def test_split_radix_protection(next_n, batch, width):
    compiled = build_split(torch.cuda.current_device(), next_n=next_n)
    logits, output, aux_i, aux_l = _buffers(batch, width, next_n)
    lengths = [width] * batch
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, next_n, 42, "boundaries")
    split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    _check(logits, lengths, output, next_n)


def test_split_radix_graph_dynamic_lengths():
    compiled = build_split(torch.cuda.current_device())
    width = 32776
    assert blocks_per_row(4, width, compiled.sm_count) == 10
    logits, output, aux_i, aux_l = _buffers(1, width)
    lens = torch.tensor([width], dtype=torch.int32, device="cuda")
    _fill(logits, [width], 4, 0, "ordinary")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    edges = [3, 4, 5, 2047, 2048, 2049, 4096, width - 1, width]
    for replay in range(100):
        lengths = [edges[replay % len(edges)]]
        _fill(logits, lengths, 4, replay, "ties" if replay % 2 else "ordinary")
        lens.fill_(lengths[0])
        output.fill_(-7)
        aux_i.fill_(-7)
        aux_l.fill_(float("nan"))
        graph.replay()
        _check(logits, lengths, output)


@pytest.mark.parametrize("seed,kind", [(0, "ordinary"), (42, "boundaries")])
def test_split_radix_partition_heads_within_vector(seed, kind):
    compiled = build_split(torch.cuda.current_device())
    logits, output, aux_i, aux_l = _buffers(1, 65536)
    for residue in range(8):
        # Every row's partitions span 4096 + residue columns, so their starts fall at
        # every offset inside a 256-bit vector and the aligned reads cover a neighbor's
        # columns on both sides.
        lengths = [10 * (4096 + residue) + 3]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        _fill(logits, lengths, 4, seed + residue, kind)
        output.fill_(-7)
        split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
        _check(logits, lengths, output)


@pytest.mark.parametrize("layout", ["unaligned_rows", "unaligned_width"])
def test_split_radix_unaligned_layout_uses_native(layout):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (
        indexer_topk_decode as native_topk,
    )

    compiled = build_split(torch.cuda.current_device())
    width = 65536 if layout == "unaligned_rows" else 65535
    _, output, aux_i, aux_l = _buffers(1, 65536)
    storage = torch.empty(4 * 65536 + 9, dtype=torch.float32, device="cuda")
    offset = 1 if layout == "unaligned_rows" else 8
    logits = storage[offset:].as_strided((4, width), (65536, 1))
    lengths = [width]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, 42, "boundaries")
    target = (
        "tensorrt_llm._torch._experimental.modeling_v2.catalog.attention."
        "cute_dsl_radix_topk_decode.indexer_topk_decode"
    )
    with patch(target, wraps=native_topk) as fallback:
        split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
        fallback.assert_called_once()
    _check(logits, lengths, output)


def test_split_radix_rejects_invalid_inputs():
    compiled = build_split(torch.cuda.current_device())
    for partitions in (4, 8):
        with pytest.raises(ValueError, match="exactly 10"):
            build_split(torch.cuda.current_device(), partitions=partitions)
    logits, output, aux_i, aux_l = _buffers(1, 16384)
    lens = torch.tensor([16384], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="float32"):
        split_radix_topk_decode(logits.half(), lens, output, aux_i, aux_l, compiled=compiled)
    with pytest.raises(ValueError, match="scratch"):
        split_radix_topk_decode(
            logits, lens, output, aux_i.flatten()[: 4 * 8 * TOP_K - 1], aux_l, compiled=compiled
        )
    for invalid in (2, 16385):
        lens.fill_(invalid)
        with pytest.raises(ValueError, match="seq_lens"):
            validate_split_radix_topk_lengths(logits, lens, next_n=4)


@pytest.mark.parametrize("length,seed", [(5, 0), (2049, 17), (65536, 42)])
def test_split_radix_native_nonfinite_semantics(length, seed):
    compiled = build_split(torch.cuda.current_device())
    logits, output, aux_i, aux_l = _buffers(1, 65536)
    lengths = [length]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    _fill(logits, lengths, 4, seed, "ordinary")
    patterns = torch.tensor(
        [0x7FC00000, 0x7F800000, -0x400000, -0x800000, -0x800001, -0x80000000],
        dtype=torch.int32,
        device="cuda",
    ).view(torch.float32)
    for row in range(4):
        end = length - 3 + row
        positions = torch.arange(0, end, 19, device="cuda")
        logits[row, positions] = patterns[torch.arange(positions.numel(), device="cuda") % 6]
    reference = torch.empty_like(output)
    torch.ops.trtllm.indexer_topk_decode(logits, lens, reference, 4, TOP_K, 1, aux_i, aux_l)
    aux_i.fill_(-7)
    aux_l.fill_(float("nan"))
    split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    for row in range(4):
        selected_keys = []
        for selection in (output[row], reference[row]):
            valid = selection[selection >= 0].long()
            assert valid.unique().numel() == valid.numel()
            assert bool((valid < length - 3 + row).all())
            bits = logits[row, valid].view(torch.int32).long() & 0xFFFFFFFF
            keys = torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)
            padding = torch.full(
                (int((selection == -1).sum()),), 0xFF7FFFFF, device="cuda", dtype=torch.int64
            )
            selected_keys.append(torch.cat((keys, padding)).sort().values)
        assert torch.equal(*selected_keys)


def test_split_radix_preserves_native_partition_fallback():
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (
        indexer_topk_decode as native_topk,
    )

    compiled = build_split(torch.cuda.current_device())
    assert blocks_per_row(4, 16384, compiled.sm_count) == 8
    logits, output, aux_i, aux_l = _buffers(1, 16384)
    lens = torch.tensor([20], dtype=torch.int32, device="cuda")
    _fill(logits, [20], 4, 42, "ordinary")
    logits[:, 1] = -torch.inf
    logits[:, 2] = -torch.finfo(torch.float32).max
    logits[:, 3] = torch.tensor([-0x3FFFFF], dtype=torch.int32, device="cuda").view(torch.float32)
    logits[:, 4] = torch.inf
    reference = torch.empty_like(output)
    native_topk(logits, lens, reference, 4, TOP_K, 1, aux_i, aux_l)
    target = (
        "tensorrt_llm._torch._experimental.modeling_v2.catalog.attention."
        "cute_dsl_radix_topk_decode.indexer_topk_decode"
    )
    with patch(target, wraps=native_topk) as fallback:
        split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
        fallback.assert_called_once()
    for row in range(4):
        selected = []
        for result in (reference[row], output[row]):
            col = result[result >= 0].long()
            bits = logits[row, col].view(torch.int32)
            pads = torch.full((TOP_K - col.numel(),), -0x800001, dtype=torch.int32, device="cuda")
            selected.append(torch.cat((bits, pads)).sort().values)
        assert torch.equal(*selected)


def test_split_radix_sentinel_tie_count_bounds():
    compiled = build_split(torch.cuda.current_device())
    logits, output, aux_i, aux_l = _buffers(1, 1048584)
    lens = torch.tensor([16384], dtype=torch.int32, device="cuda")
    patterns = torch.tensor(
        [0x7F7FFFFF, 0x7F800000, -0x800001, -0x3FFFFF, 0x7FC00001, 0x3F800000, -0x40800000],
        dtype=torch.int32,
        device="cuda",
    ).view(torch.float32)
    logits.fill_(1.0e9)
    for row in range(4):
        end = 16381 + row
        logits[row, :end] = -torch.inf
        positions = {0, end - 1, 31, 32, 33, 2047, 2048, 2049}
        for partitions in (4, 8, 10):
            for part in range(1, partitions):
                edge = end // partitions * part
                positions.update((edge - 1, edge, edge + 1))
        positions = sorted(positions)
        logits[row, positions] = patterns[torch.arange(len(positions), device="cuda") % 7]
    reference = torch.empty_like(output)
    torch.ops.trtllm.indexer_topk_decode(logits, lens, reference, 4, TOP_K, 1, aux_i, aux_l)
    split_radix_topk_decode(logits, lens, output, aux_i, aux_l, compiled=compiled)
    for row in range(4):
        end = 16381 + row
        selected = []
        for result in (reference[row], output[row]):
            col = result[result >= 0].long()
            assert col.unique().numel() == col.numel()
            assert bool((col < end).all())
            bits = logits[row, col].view(torch.int32)
            selected.append(bits[bits != -0x800001].sort().values)
        assert torch.equal(*selected), "only real -FLT_MAX versus padding may differ"
        strict_count = selected[0].numel()
        real_ties = int((logits[row, :end].view(torch.int32) == -0x800001).sum())
        pads = 10 * TOP_K - end
        tie_slots = TOP_K - strict_count
        minimum = strict_count + max(0, tie_slots - pads)
        maximum = strict_count + min(tie_slots, real_ties)
        count = int((output[row] >= 0).sum())
        assert minimum <= count <= maximum
        assert bool((output[row, :count] >= 0).all())
        assert bool((output[row, count:] == -1).all())
