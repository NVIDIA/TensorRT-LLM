# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 NVFP4-E index-K format: torch reference, Triton quantize-scatter and page dequantize."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.index_k_nvfp4 import (
    INDEX_K_NVFP4_PAGE_BYTES,
    INDEX_K_NVFP4_SCALE_OFFSET,
    dequantize_index_k_nvfp4_reference,
    index_k_nvfp4_dequant_pages,
    index_k_nvfp4_quant_scatter,
    quantize_index_k_nvfp4_reference,
)

skip_pre_ada = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs a GPU with FP8 E4M3 support",
)

PAGE_SIZE = 128

# Worked out by hand from the format contract: the leading values of a 16-dim
# group (the rest are zero), its scale byte and its leading 4-bit codes.
GOLDEN_GROUPS = [
    # amax / 6 = 1. Scales 1 and 2 both leave err 0.0625; the earlier one wins.
    ([6.0, 3.75], 0x38, [0x7, 0x6]),
    ([-6.0, -3.75], 0x38, [0xF, 0xE]),
    # Midpoints round to the even code; a value rounding to zero is +0.
    ([6.0, -0.25], 0x38, [0x7, 0x0]),
    ([6.0, 0.5, 0.5, 0.75], 0x38, [0x7, 0x1, 0x1, 0x2]),
    ([6.0, 1.25], 0x38, [0x7, 0x2]),
    ([6.0, 1.75], 0x38, [0x7, 0x4]),
    ([6.0, 0.5, 0.5, 0.5, 0.5, 2.5], 0x38, [0x7, 0x1, 0x1, 0x1, 0x1, 0x4]),
    ([6.0, 3.5], 0x38, [0x7, 0x6]),
    ([6.0, 5.0] + [0.5] * 14, 0x38, [0x7, 0x6] + [0x1] * 14),
    # amax / 6 = 2^-3.
    ([0.75], 0x20, [0x7]),
    # Lower clamp: 2^-6, the first candidate, ties with 1.5 * 2^-6.
    ([0.08984375], 0x08, [0x7]),
    # A nonzero group too small for any code keeps the minimum scale.
    ([1e-3], 0x08, [0x0]),
    # Upper clamp: every candidate is 48.
    ([1000.0, 100.0, -300.0], 0x64, [0x7, 0x4, 0xF]),
    # All-zero groups store scale 0, negative zeros included.
    ([0.0], 0x00, [0x0]),
    ([-0.0] * 16, 0x00, [0x0] * 16),
    # Candidate order (e0 = -1): 0.75 (d = 0, m = 1.5), 1.25 and 1.5 (d = +1) each
    # leave err 0.0625; the earliest in d-major order wins.
    ([4.75], 0x34, [0x7]),
    # Candidate window (e0 = -3): only 2^(e0 + 1) = 0.25 represents 1.0 exactly.
    ([1.0], 0x28, [0x6]),
    # e0 = floor(log2(512 / 6)) = 6, so the clamp leaves 32, 40 and 48: s = 40
    # leaves err 272^2 against 224^2 + 15 * 48^2 for s = 48.
    ([512.0] + [240.0] * 15, 0x62, [0x7] * 16),
    # Squared error: s = 1.5 leaves 0.203125 against 0.265625 for s = 1, although
    # their absolute errors tie at 0.625.
    ([6.0, 0.375, 2.5], 0x3C, [0x6, 0x0, 0x3]),
    # Summation order: s = 1 and s = 1.5 tie in exact arithmetic. Rounded in the
    # tree order, s = 1.5 ends one ulp lower; summed sequentially, by halves or
    # with shfl_xor 2 before shfl_xor 1, they tie and s = 1 would win.
    (
        [0.0, 0.75] + [0.0] * 6 + [187 * 2.0**-21, 0.0, 0.0, 6.0, 0.5, 115 * 2.0**-19],
        0x3C,
        [0x0, 0x1] + [0x0] * 9 + [0x6, 0x1, 0x0],
    ),
]


def _golden_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GOLDEN_GROUPS packed eight to a row: BF16 rows, expected data and scale bytes."""
    num_groups = -(-len(GOLDEN_GROUPS) // 8) * 8
    values = torch.zeros(num_groups, 16)
    codes = torch.zeros(num_groups, 16, dtype=torch.uint8)
    scales = torch.zeros(num_groups, dtype=torch.uint8)
    for group, (group_values, scale, group_codes) in enumerate(GOLDEN_GROUPS):
        values[group, : len(group_values)] = torch.tensor(group_values)
        codes[group, : len(group_codes)] = torch.tensor(group_codes, dtype=torch.uint8)
        scales[group] = scale
    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return values.reshape(-1, 128).to(torch.bfloat16), data.reshape(-1, 64), scales.reshape(-1, 8)


def _test_rows() -> torch.Tensor:
    """BF16 index-K rows: random at several scales plus the contract's corner cases."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    parts = [
        torch.randn(4096, 128, generator=generator, device="cuda") * scale
        for scale in (1e-3, 1e-2, 0.1, 1.0, 10.0, 300.0)
    ]
    # One random scale per 16-dim group, reaching both scale clamps.
    group_scales = 10.0 ** torch.empty(4096, 8, 1, device="cuda").uniform_(
        -4.0, 3.5, generator=generator
    )
    groups = torch.randn(4096, 8, 16, generator=generator, device="cuda") * group_scales
    parts.append(groups.reshape(4096, 128))
    # Quarter integers make candidates with equal errors (exact ties) common.
    parts.append(torch.randint(-24, 25, (4096, 128), generator=generator, device="cuda") * 0.25)
    # Values that several candidate scales represent exactly, an amax of 6 and a few tiny
    # values: candidates then tie up to rounding, and the winner depends on the order in
    # which the squared residuals are summed.
    grid = torch.tensor([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 3.0, 4.0, 4.5, 6.0], device="cuda")
    shape = (4096 * 8, 16)
    tied = grid[torch.randint(0, grid.numel(), shape, generator=generator, device="cuda")]
    mantissa = 1.0 + torch.randint(0, 128, shape, generator=generator, device="cuda") / 128.0
    exponent = torch.randint(9, 15, shape, generator=generator, device="cuda").float()
    is_tiny = torch.rand(shape, generator=generator, device="cuda") < 0.25
    tied = torch.where(is_tiny, mantissa * 2.0**-exponent, tied)
    amax_at = torch.randint(0, 16, (shape[0], 1), generator=generator, device="cuda")
    tied.scatter_(1, amax_at, 6.0)
    sign = torch.where(torch.rand(shape, generator=generator, device="cuda") < 0.5, -1.0, 1.0)
    parts.append((tied * sign).reshape(4096, 128))
    # Rounding midpoints of a scale that represents the group's amax exactly: ties to
    # even and exact division decide every code.
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    mantissas = torch.tensor([1.0, 1.25, 1.5], device="cuda")
    mantissa = mantissas[torch.randint(0, 3, (shape[0], 1), generator=generator, device="cuda")]
    exponent = torch.randint(-5, 5, (shape[0], 1), generator=generator, device="cuda").float()
    group_scale = mantissa * 2.0**exponent
    mid = midpoints[torch.randint(0, 7, shape, generator=generator, device="cuda")] * group_scale
    amax_at = torch.randint(0, 16, (shape[0], 1), generator=generator, device="cuda")
    mid.scatter_(1, amax_at, 6.0 * group_scale)
    sign = torch.where(torch.rand(shape, generator=generator, device="cuda") < 0.5, -1.0, 1.0)
    parts.append((mid * sign).reshape(4096, 128))
    corner = torch.zeros(3, 128, device="cuda")
    corner[1] = -0.0
    # Groups alternating between the lower and the upper scale clamp.
    corner[2] = torch.tensor([1e-4, -1000.0]).repeat_interleave(16).repeat(4)
    parts.append(corner)
    parts.append(_golden_rows()[0].float().cuda())
    return torch.cat(parts).to(torch.bfloat16)


def _page_rows(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Data [num_pages * 128, 64] and scales [num_pages * 128, 8] of every row of a page cache."""
    data = cache[:, :INDEX_K_NVFP4_SCALE_OFFSET].reshape(-1, 64)
    scales = cache[:, INDEX_K_NVFP4_SCALE_OFFSET:INDEX_K_NVFP4_PAGE_BYTES].reshape(-1, 8)
    return data, scales


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu_only), pytest.param("cuda", marks=skip_pre_ada)],
)
def test_reference_matches_hand_derived_groups(device: str) -> None:
    rows, data, scales = _golden_rows()
    got_data, got_scales = quantize_index_k_nvfp4_reference(rows.to(device))
    assert torch.equal(got_scales.cpu(), scales)
    assert torch.equal(got_data.cpu(), data)
    # FP32 input is rounded to BF16 first.
    x = torch.randn(256, 128, generator=torch.Generator().manual_seed(4)).to(device) * 3
    got = quantize_index_k_nvfp4_reference(x)
    expected = quantize_index_k_nvfp4_reference(x.to(torch.bfloat16))
    assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])


@skip_pre_ada
def test_quant_scatter_matches_reference_bitwise() -> None:
    rows = _test_rows()
    num_rows = rows.shape[0]
    num_pages = -(-num_rows // PAGE_SIZE)
    cache = torch.zeros(num_pages, INDEX_K_NVFP4_PAGE_BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.arange(num_rows, dtype=torch.int32, device="cuda")
    index_k_nvfp4_quant_scatter(rows, cache, slots)
    data, scales = _page_rows(cache)
    data, scales = data[:num_rows], scales[:num_rows]
    ref_data, ref_scales = quantize_index_k_nvfp4_reference(rows)
    assert torch.equal(scales, ref_scales), f"{int((scales != ref_scales).sum())} scales differ"
    assert torch.equal(data, ref_data), f"{int((data != ref_data).any(-1).sum())} rows differ"
    # The reference does not depend on the device it runs on.
    cpu_data, cpu_scales = quantize_index_k_nvfp4_reference(rows.cpu())
    assert torch.equal(cpu_scales, ref_scales.cpu())
    assert torch.equal(cpu_data, ref_data.cpu())


@skip_pre_ada
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_quant_scatter_slot_addressing(slot_dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1)
    num_pages = 5
    # The layer's pages sit inside wider pool rows, as when the pool coalesces
    # buffers; no byte outside the stored rows may change.
    pool = torch.randint(
        0,
        256,
        (num_pages, 3 * INDEX_K_NVFP4_PAGE_BYTES + 512),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    offset = INDEX_K_NVFP4_PAGE_BYTES
    cache = pool[:, offset : offset + INDEX_K_NVFP4_PAGE_BYTES]
    live = torch.randperm(num_pages * PAGE_SIZE, device="cuda", generator=generator)[:200]
    # Padding (-1), other negative slots and slots past the last page are skipped.
    num_slots = num_pages * PAGE_SIZE
    skipped = torch.tensor([-1, -2, -PAGE_SIZE, num_slots, num_slots + 7, 1 << 30], device="cuda")
    slots = torch.cat([live, skipped])
    slots = slots[torch.randperm(slots.shape[0], device="cuda", generator=generator)]
    slots = slots.to(slot_dtype)
    # Strided rows, as when index-K is a column slice of a wider projection.
    wide = torch.randn(slots.shape[0], 256, device="cuda", generator=generator) * 3
    idx_k = wide.to(torch.bfloat16)[:, 64:192]

    expected = pool.clone()
    keep = (slots >= 0) & (slots < num_slots)
    page = slots[keep].long() // PAGE_SIZE
    row = slots[keep].long() % PAGE_SIZE
    data, scales = quantize_index_k_nvfp4_reference(idx_k[keep])
    data_cols = offset + 64 * row[:, None] + torch.arange(64, device="cuda")
    scale_cols = (
        offset + INDEX_K_NVFP4_SCALE_OFFSET + 8 * row[:, None] + torch.arange(8, device="cuda")
    )
    expected[page[:, None], data_cols] = data
    expected[page[:, None], scale_cols] = scales

    index_k_nvfp4_quant_scatter(idx_k, cache, slots)
    assert torch.equal(pool, expected)


@skip_pre_ada
def test_quant_scatter_replays_in_cuda_graph() -> None:
    generator = torch.Generator(device="cuda").manual_seed(2)
    num_pages, num_tokens = 4, 64
    cache = torch.zeros(num_pages, INDEX_K_NVFP4_PAGE_BYTES, dtype=torch.uint8, device="cuda")
    idx_k = torch.randn(num_tokens, 128, device="cuda", generator=generator).to(torch.bfloat16)
    slots = torch.randperm(num_pages * PAGE_SIZE, device="cuda", generator=generator)[:num_tokens]
    slots = slots.to(torch.int32)
    # The first call compiles the kernel outside the capture.
    index_k_nvfp4_quant_scatter(idx_k, cache, slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_k_nvfp4_quant_scatter(idx_k, cache, slots)

    idx_k.copy_(torch.randn(num_tokens, 128, device="cuda", generator=generator))
    slots.copy_(
        torch.randperm(num_pages * PAGE_SIZE, device="cuda", generator=generator)[:num_tokens]
    )
    slots[-8:] = -1
    cache.zero_()
    graph.replay()
    expected = torch.zeros_like(cache)
    index_k_nvfp4_quant_scatter(idx_k, expected, slots)
    assert torch.equal(cache, expected)


@skip_pre_ada
def test_dequantized_values_are_e4m3_exact() -> None:
    # Every allowed scale, {1, 1.25, 1.5} x 2^e for e in [-6, 5], under every 4-bit code.
    allowed = torch.tensor([m * 2.0**e for e in range(-6, 6) for m in (1.0, 1.25, 1.5)])
    allowed = allowed.to(torch.float8_e4m3fn).view(torch.uint8)
    codes = torch.arange(16, dtype=torch.uint8).repeat(8)
    data = (codes[0::2] | (codes[1::2] << 4)).expand(allowed.shape[0], 64)
    values = dequantize_index_k_nvfp4_reference(data, allowed[:, None].expand(-1, 8))
    assert torch.equal(values.to(torch.float8_e4m3fn).float(), values)
    assert torch.equal(values.half().float(), values)

    # The quantizer stores only those scales, or 0 for an all-zero group.
    data, scales = quantize_index_k_nvfp4_reference(_test_rows())
    stored = torch.cat([allowed, torch.zeros(1, dtype=torch.uint8)]).cuda()
    assert bool(torch.isin(scales, stored).all())
    values = dequantize_index_k_nvfp4_reference(data, scales)
    assert torch.equal(values.to(torch.float8_e4m3fn).float(), values)


@skip_pre_ada
def test_dequant_pages_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(3)
    num_pages = 7
    pool = torch.zeros(num_pages, INDEX_K_NVFP4_PAGE_BYTES + 1024, dtype=torch.uint8, device="cuda")
    cache = pool[:, :INDEX_K_NVFP4_PAGE_BYTES]
    group_scales = 10.0 ** torch.empty(num_pages * PAGE_SIZE, 8, 1, device="cuda").uniform_(
        -3.0, 2.5, generator=generator
    )
    rows = torch.randn(num_pages * PAGE_SIZE, 8, 16, device="cuda", generator=generator)
    data, scales = quantize_index_k_nvfp4_reference((rows * group_scales).reshape(-1, 128))
    cache[:, :INDEX_K_NVFP4_SCALE_OFFSET] = data.reshape(num_pages, -1)
    cache[:, INDEX_K_NVFP4_SCALE_OFFSET:] = scales.reshape(num_pages, -1)
    cache[4] = 0  # a page no token has been written to
    # A context page list may repeat a page that two requests share; ids outside
    # the cache, such as -1 padding, give zero pages.
    page_ids = torch.tensor([3, 0, 6, 3, 4, 1, -1, num_pages], dtype=torch.int32, device="cuda")
    # Prefilled with E4M3 NaN, so an element the kernel skips fails the comparison.
    out = torch.full(
        (page_ids.shape[0], 1, PAGE_SIZE, 128), 0x7F, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)
    index_k_nvfp4_dequant_pages(cache, page_ids, out)

    data, scales = _page_rows(cache)
    in_cache = page_ids.clamp(0, num_pages - 1).long()
    expected = dequantize_index_k_nvfp4_reference(
        data.reshape(num_pages, PAGE_SIZE, 64)[in_cache],
        scales.reshape(num_pages, PAGE_SIZE, 8)[in_cache],
    )
    expected[(page_ids < 0) | (page_ids >= num_pages)] = 0
    assert torch.equal(out.float(), expected[:, None])


@skip_pre_ada
def test_rejects_malformed_inputs() -> None:
    cache = torch.zeros(2, INDEX_K_NVFP4_PAGE_BYTES, dtype=torch.uint8, device="cuda")
    idx_k = torch.zeros(4, 128, dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="BF16"):
        index_k_nvfp4_quant_scatter(idx_k.float(), cache, slots)
    with pytest.raises(ValueError, match="cache"):
        index_k_nvfp4_quant_scatter(idx_k, cache[:, :-16], slots)
    overlapping = cache.as_strided(cache.shape, (INDEX_K_NVFP4_PAGE_BYTES // 2, 1))
    with pytest.raises(ValueError, match="cache"):
        index_k_nvfp4_quant_scatter(idx_k, overlapping, slots)
    with pytest.raises(ValueError, match="slot per row"):
        index_k_nvfp4_quant_scatter(idx_k, cache, slots[:3])
    out = torch.empty(2, 1, PAGE_SIZE, 128, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(ValueError, match="out must be"):
        index_k_nvfp4_dequant_pages(cache, slots[:2], out[:1])
    with pytest.raises(ValueError, match="out must be"):
        index_k_nvfp4_dequant_pages(
            cache, slots[:2], out.as_strided(out.shape, (PAGE_SIZE, out.stride(1), 128, 1))
        )
