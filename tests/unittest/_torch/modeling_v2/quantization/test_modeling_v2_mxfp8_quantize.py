# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_quantize catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize as op,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mxfp8_quantize requires a CUDA device"


BLOCK = 32


def _build(spec, seed):
    """A cell's input; `data="block_scaled"` is `_activations` (block maxima over 2^-24..2^24), else randn."""
    if spec.get("data") == "block_scaled":
        m, k = spec["shape"]
        assert spec["dtype"] is torch.bfloat16, spec
        return _activations(m, k, seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


def _ref(x: torch.Tensor, alignment: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """`op.reference` with its scale bytes as the [M, padded_k / 32] matrix: the linear layout, reshaped.

    The swizzle checks below index into it per (row, block).
    """
    data, sf = op.reference(x, False, alignment)
    return data, sf.view(data.shape[0], -1)


def _swizzle_index(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """[rows, cols] scale coordinates -> flat offsets in the 128x4-swizzled buffer."""
    padded_cols = (cols + 3) // 4 * 4
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(cols, device=device).view(1, -1)
    return (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    x = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    swizzled, alignment = cell.spec["swizzled"], cell.spec["alignment"]
    with validating(op):
        data, scales = op(x, swizzled, alignment)
    ref_data, ref_scales = op.reference(x, swizzled, alignment)
    op.compare(data, ref_data)
    op.compare(scales, ref_scales)


def test_the_quantization_round_trips_within_the_block_bound() -> None:
    """Independent of the reference: dequantizing has to return the input.

    The cells compare two implementations of the same arithmetic, which would
    both be wrong together if the arithmetic itself were wrong. This bounds the
    error against the input instead: one power-of-two scale per 32 elements and
    e4m3's 3 mantissa bits give at worst 2^-3 relative on the block maximum.
    """
    x = torch.randn(64, 2880, dtype=torch.bfloat16, device="cuda")
    data, scales = op(x, False, 512)
    rows, blocks = 64, 3072 // 32
    scale = torch.exp2(scales.reshape(rows, blocks).float() - 127.0).unsqueeze(-1)
    deq = (data.float().view(rows, blocks, 32) * scale).view(rows, 3072)[:, :2880]
    block_amax = x.float().view(rows, 90, 32).abs().amax(-1, keepdim=True)
    err = (deq - x.float()).view(rows, 90, 32).abs().amax(-1, keepdim=True)
    assert torch.all(err <= block_amax * 2.0**-3 + 1e-6)


def test_guard_refuses_a_partial_block() -> None:
    """The kernel reads past the end of the last block rather than failing."""
    x = torch.randn(4, 2864, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="whole number of 32-element blocks"):
        with validating(op):
            op(x, False, 512)


def test_guard_refuses_an_alignment_that_is_not_a_block_multiple() -> None:
    """Pads to a width the scale count cannot describe."""
    x = torch.randn(4, 2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="multiple of the 32-element"):
        with validating(op):
            op(x, False, 48)


def test_guard_refuses_a_non_contiguous_input() -> None:
    """Read flat, so a strided view quantizes the wrong elements."""
    x = torch.randn(4, 5760, dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, False, 512)


# ─── DeepSeek-V4.1 core target: its eight activation widths, every row count ──
#
# Every dense FP8 linear of the core target quantizes its bf16 input with the
# default call, `mxfp8_quantize(x)` (swizzled, alignment 32), and hands both
# outputs unchanged to the MXFP8 GEMM. K: 5120 hidden (wqkv_a, shared w1||w3),
# 1280 q_lora (wq_b, indexer wq_b), the wo_b input 8 / tp * 1024 (2048 under
# attention TP4, 8192 under attention DP, 4096 under attention TP2, 1024 under
# attention TP8), 2304 shared w2, 6144 engram wkv. The row count is the rank's
# token count this step: any integer up to max_num_tokens (at most 16384 per
# rank). In layers 21-39 the decoder bounded replay (every member) runs decoder
# passes over the finishing requests' 128-row tails plus the generation rows,
# cut by the target into passes of at most the step capacity; the uncut row set
# reaches 32768. The linears there take every width but 6144. The sweep below runs
# every row count 1..32768 at all eight widths. All eight widths are multiples
# of 128, so these calls never pad scale columns; row padding (to 128) happens
# at every row count that is not a multiple of 128.

TARGET_K = (5120, 1280, 2048, 8192, 2304, 6144, 4096, 1024)
MAX_ROWS = 32768
# The core target's decode CUDA-graph batch sizes, and one prefill chunk seen in its dump.
GRAPH_ROWS = (1, 2, 4, 8, 16, 24, 32, 1935)


def _activations(m: int, k: int, seed: int) -> torch.Tensor:
    """bf16 N(0, 1) with each 32-element block scaled by 2^e, e uniform in -24..24.

    Block maxima then spread over about 49 scale bytes, so a scale byte read from
    the wrong (row, block) is almost never equal by chance.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda", dtype=torch.float32)
    e = torch.randint(-24, 25, (m, k // BLOCK, 1), generator=g, device="cuda").float()
    return (x.view(m, k // BLOCK, BLOCK) * torch.exp2(e)).view(m, k).to(torch.bfloat16)


@pytest.mark.parametrize("k", TARGET_K)
def test_v41_target_width_swizzled_every_row_count(k: int) -> None:
    """Every row count 1..MAX_ROWS (32768) at each target width, in the swizzled layout the GEMM consumes.

    The reference quantizes one MAX_ROWS-row input once. Every prefix x[:r] must
    then give the reference's first r rows of data, byte for byte; the
    reference's scale bytes at the 128x4 offsets (the offset of (m, b) does not
    depend on the row count; only the buffer length does); and 0x00 in every
    row-padding byte. The scale bytes are unsigned, so once the addressed bytes
    match, a buffer sum equal to theirs means every padding byte is zero.
    Outputs are poisoned before they are freed, so a byte the kernel skipped
    cannot inherit the previous call's correct value from the allocator.
    """
    cols = k // BLOCK
    x = _activations(MAX_ROWS, k, seed=k)
    ref_data, ref_sf = _ref(x)
    ref_bytes = ref_data.view(torch.uint8)
    ref_sf_cumsum = ref_sf.to(torch.int64).sum(dim=1).cumsum(0)
    idx = _swizzle_index(MAX_ROWS, cols, x.device)
    bad_data = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_sf = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_pad = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(1, MAX_ROWS + 1):
        data, sf = op(x[:r])
        assert data.shape == (r, k) and data.dtype == torch.float8_e4m3fn
        assert sf.shape == (_pad_up(r, 128) * cols,) and sf.dtype == torch.uint8
        bad_data += (data.view(torch.uint8) != ref_bytes[:r]).sum()
        bad_sf += (sf[idx[:r]] != ref_sf[:r]).sum()
        bad_pad += (sf.sum(dtype=torch.int64) - ref_sf_cumsum[r - 1]).abs()
        data.view(torch.uint8).fill_(0x7F)
        sf.fill_(0xFF)
        if r % 2048 == 0:
            # Each call's outputs are larger than any cached block, so the allocator
            # would otherwise keep a segment per size and fill the device.
            del data, sf
            torch.cuda.empty_cache()
    assert (bad_data.item(), bad_sf.item(), bad_pad.item()) == (0, 0, 0), (
        f"K={k}: {bad_data.item()} data bytes, {bad_sf.item()} scale bytes differ; "
        f"padding bytes sum to {bad_pad.item()}"
    )


@pytest.mark.parametrize("k", TARGET_K)
def test_v41_target_width_cuda_graph_replay(k: int) -> None:
    """Captured after one eager call at the same shape, as the engine's warmup does; replayed on new inputs.

    The graph's own output buffers are poisoned before every replay, so the
    check shows that replay rewrites every data byte and every scale byte,
    padding included.
    """
    cols = k // BLOCK
    for m in GRAPH_ROWS:
        x_static = _activations(m, k, seed=7 * m + k)
        op(x_static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            data, sf = op(x_static)
        idx = _swizzle_index(m, cols, x_static.device)
        for seed in (1, 2):
            x_new = _activations(m, k, seed=100000 * seed + m + k)
            x_static.copy_(x_new)
            data.view(torch.uint8).fill_(0x7F)
            sf.fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            ref_data, ref_sf = _ref(x_new)
            assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8)), (k, m)
            assert torch.equal(sf[idx], ref_sf), (k, m)
            assert sf.sum(dtype=torch.int64).item() == ref_sf.to(torch.int64).sum().item(), (k, m)


def test_v41_target_width_layout_controls() -> None:
    """Wrong scale layouts at the target widths: the byte-exact check sees each one, except one case at a single row.

    - Row-major (linear) offsets read from the swizzled buffer are visible at
      every row count, 1 included: block b of row 0 sits at
      (b % 4) + (b // 4) * 512, not at b.
    - Swapping the two row terms of the swizzle (row % 32 scaled by 4, and
      (row % 128) // 32 scaled by 16) is visible from 2 rows up. At one row
      both terms are zero, and the swap is invisible.
    """
    for k in TARGET_K:
        cols = k // BLOCK
        c = torch.arange(cols, device="cuda").view(1, -1)
        for m in (1, 7, 1935):
            x = _activations(m, k, seed=3 * m + k)
            _, sf = op(x)
            _, ref_sf = _ref(x)
            right = _swizzle_index(m, cols, x.device)
            assert torch.equal(sf[right], ref_sf)
            r = torch.arange(m, device="cuda").view(-1, 1)
            linear = r * cols + c
            swapped = (
                (c % 4)
                + (c // 4) * 512
                + (r % 32) * 4
                + ((r % 128) // 32) * 16
                + (r // 128) * (128 * cols)
            )
            n_linear = int((sf[linear] != ref_sf).sum().item())
            n_swapped = int((sf[swapped] != ref_sf).sum().item())
            print(
                f"[control K={k} M={m}] linear offsets: {n_linear} / {m * cols} scale bytes differ; "
                f"swapped row terms: {n_swapped} / {m * cols}"
            )
            assert n_linear > 0
            if m == 1:
                assert n_swapped == 0, "at one row the swapped terms are both zero"
            else:
                assert n_swapped > 0


# ─── Capture set: decode, DSpark target verify, DSpark drafter ─────────────
#
# The captured batch sizes G, per rank (no MXFP8 linear of the target runs on
# the dep4-gathered token set): every G 1..32, plus TRT-LLM's padded default
# list for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list for 256 (1..32, 64, 128, 256)
# is a subset. A captured step is generation-only: the trunk linears see
# T = G rows (next_n 1) or 6G (DSpark target verify, 1 + 5 drafts), and the
# drafter's dense linears, whose widths are four of the eight (5120, 1280, 8192,
# 2304), see its 5-token draft block, 5G; every width is captured at all three
# multiples here. The engine captures one graph per batch size into one shared memory
# pool, largest first, each graph holding every linear of the step, and replays
# whichever one the step's batch selects.

GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
VERIFY_GRAPH_ROWS = tuple(sorted({f * g for g in GSET for f in (1, 5, 6)}, reverse=True))


def test_v41_target_widths_cuda_graph_at_dspark_verify_rows() -> None:
    """One graph per captured row count (G, 5G, 6G over GSET: 114 counts, 1..1536) holding a call at each
    target width, in one shared pool, replayed interleaved.

    Each graph is captured after an eager call at each of its shapes (the
    engine's warm-up), largest row count first, all into one pool. The graphs
    are then replayed in an interleaved order over three rounds. Before each
    replay every static input gets new data and every output buffer is
    poisoned (data 0x7F, scales 0xFF). After it, each call's outputs must be
    bitwise equal to an eager call on the same data, whole scale buffer
    included, and to the reference: data bytes, scale bytes at the 128x4
    offsets, and zero in every row-padding byte. A replay that did not run, or
    ran on the previous round's inputs, fails the reference check: the
    previous round's data bytes differ from this round's on most bytes (the
    smallest differing fraction is printed and asserted).

    Control: every captured row count above 128 spans at least two 128-row
    scale groups. Reading its scale buffer without the row-group term of the
    swizzle (rows 128.. at the offsets of rows 0..) must differ from the
    reference.
    """
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for m in sorted(VERIFY_GRAPH_ROWS, reverse=True):
        xs = [_activations(m, k, seed=11 * m + k) for k in TARGET_K]
        for x in xs:
            op(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outs = [op(x) for x in xs]
        graphs[m] = (graph, xs, outs)
    torch.cuda.synchronize()
    order = VERIFY_GRAPH_ROWS[::2] + VERIFY_GRAPH_ROWS[1::2][::-1]
    prev = {}
    stale_min = 1.0
    group_min = {k: 1.0 for k in TARGET_K}
    for rnd in range(3):
        for m in order:
            graph, xs, outs = graphs[m]
            news = [_activations(m, k, seed=500000 + 1000 * rnd + 7 * m + k) for k in TARGET_K]
            for x, x_new in zip(xs, news):
                x.copy_(x_new)
            for data, sf in outs:
                data.view(torch.uint8).fill_(0x7F)
                sf.fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            for k, x_new, (data, sf) in zip(TARGET_K, news, outs):
                cols = k // BLOCK
                assert data.shape == (m, k) and sf.shape == (_pad_up(m, 128) * cols,), (k, m)
                eager_data, eager_sf = op(x_new)
                assert torch.equal(data.view(torch.uint8), eager_data.view(torch.uint8)), (
                    k,
                    m,
                    rnd,
                )
                assert torch.equal(sf, eager_sf), (k, m, rnd)
                ref_data, ref_sf = _ref(x_new)
                idx = _swizzle_index(m, cols, x_new.device)
                assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8)), (k, m, rnd)
                assert torch.equal(sf[idx], ref_sf), (k, m, rnd)
                assert sf.sum(dtype=torch.int64).item() == ref_sf.to(torch.int64).sum().item(), (
                    k,
                    m,
                    rnd,
                )
                # The previous round's reference (what a stale replay would have left) must differ.
                if (k, m) in prev:
                    frac = (prev[(k, m)] != ref_data.view(torch.uint8)).float().mean().item()
                    stale_min = min(stale_min, frac)
                    assert frac > 0.5, (k, m, rnd, frac)
                prev[(k, m)] = ref_data.view(torch.uint8).clone()
                if rnd == 0 and m > 128:
                    r = torch.arange(m, device="cuda").view(-1, 1)
                    c = torch.arange(cols, device="cuda").view(1, -1)
                    no_group = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
                    n_bad = int((sf[no_group] != ref_sf).sum().item())
                    group_min[k] = min(group_min[k], n_bad / ((m - 128) * cols))
                    assert n_bad > 0, (k, m)
    print(
        f"[capture] {len(VERIFY_GRAPH_ROWS)} row counts x {len(TARGET_K)} widths x 3 rounds: every replay bitwise = "
        "eager = "
        f"reference; smallest fraction of data bytes differing from the previous round: {stale_min:.4f}"
    )
    for k in TARGET_K:
        print(
            f"[control K={k}] no row-group term, rows > 128: at least {group_min[k]:.4f} of the scale bytes of "
            "rows 128.. differ"
        )


# ─── DeepSeek-V4.1 DSpark drafter: its two drafter-only calls ──────────────
#
# The drafter makes two quantize calls the core target does not, at G (the
# drafter's batch, 1..256) or 5G rows (1..1280):
#   - its MoE input, mxfp8_quantize(x, False, 512) on bf16 [5G, 5120]: the
#     LINEAR layout the MXFP4 MoE runner consumes (5120 is a multiple of 512,
#     so there is no column padding);
#   - main_proj's input, mxfp8_quantize(x) (swizzled, alignment 32) on bf16
#     [G or 5G, 15360] (and the prefill seeding's n <= 128 rows, eager).
# Both run here at every row count 1..1280 eagerly. Under capture both run at
# every row count 1..160 plus G and 5G over GSET (176 row counts, up to 1280).
DSPARK_ROWS = tuple(range(1, 1281))
DSPARK_GRAPH_ROWS = tuple(sorted(set(range(1, 161)) | {f * g for g in GSET for f in (1, 5)}))
DSPARK_CASES = ((5120, False, 512), (15360, True, 32))


def _check_layout(data, sf, ref_data, ref_sf, m, k, swizzled) -> None:
    cols = k // BLOCK
    assert data.shape == (m, k) and data.dtype == torch.float8_e4m3fn
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8)), (k, m)
    if swizzled:
        assert sf.shape == (_pad_up(m, 128) * _pad_up(cols, 4),), (k, m)
        assert torch.equal(sf[_swizzle_index(m, cols, sf.device)], ref_sf), (k, m)
        assert sf.sum(dtype=torch.int64).item() == ref_sf.to(torch.int64).sum().item(), (k, m)
    else:
        assert sf.shape == (m * cols,), (k, m)
        assert torch.equal(sf.view(m, cols), ref_sf), (k, m)


@pytest.mark.parametrize("k,swizzled,alignment", DSPARK_CASES)
def test_v41_dspark_every_row_count_1_to_1280(k: int, swizzled: bool, alignment: int) -> None:
    """Every row count 1..1280: each prefix x[:r] gives the reference's first r rows, byte for byte."""
    x = _activations(DSPARK_ROWS[-1], k, seed=k + 1)
    ref_data, ref_sf = _ref(x)
    for r in DSPARK_ROWS:
        data, sf = op(x[:r], swizzled, alignment)
        _check_layout(data, sf, ref_data[:r], ref_sf[:r], r, k, swizzled)
        data.view(torch.uint8).fill_(0x7F)
        sf.fill_(0xFF)


@pytest.mark.parametrize("k,swizzled,alignment", DSPARK_CASES)
def test_v41_dspark_cuda_graph_every_row_count(k: int, swizzled: bool, alignment: int) -> None:
    """Captured at every row count of DSPARK_GRAPH_ROWS (1..160, G and 5G over GSET) after one eager
    call at that shape, one shared pool, largest first; each graph replayed on two new inputs with its
    outputs poisoned. Every replay is bitwise equal to an eager call on the same data and to the
    reference."""
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for m in sorted(DSPARK_GRAPH_ROWS, reverse=True):
        x = _activations(m, k, seed=13 * m + k)
        op(x, swizzled, alignment)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = op(x, swizzled, alignment)
        graphs[m] = (graph, x, out)
    torch.cuda.synchronize()
    for rnd in range(2):
        for m in DSPARK_GRAPH_ROWS[rnd::2] + DSPARK_GRAPH_ROWS[1 - rnd :: 2]:
            graph, x, (data, sf) = graphs[m]
            x_new = _activations(m, k, seed=700000 + 1000 * rnd + 13 * m + k)
            x.copy_(x_new)
            data.view(torch.uint8).fill_(0x7F)
            sf.fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            eager_data, eager_sf = op(x_new, swizzled, alignment)
            assert torch.equal(data.view(torch.uint8), eager_data.view(torch.uint8)), (k, m, rnd)
            assert torch.equal(sf, eager_sf), (k, m, rnd)
            ref_data, ref_sf = _ref(x_new)
            _check_layout(data, sf, ref_data, ref_sf, m, k, swizzled)


def test_v41_dspark_moe_input_layout_control() -> None:
    """At 128 rows the swizzled scale buffer of the MoE input has exactly the linear buffer's byte
    count (128 * 160), so a size check cannot tell them apart; the byte check does."""
    k, m = 5120, 128
    x = _activations(m, k, seed=77)
    _, ref_sf = _ref(x)
    _, sf_lin = op(x, False, 512)
    _, sf_sw = op(x, True, 512)
    assert sf_lin.numel() == sf_sw.numel() == m * (k // BLOCK)
    assert torch.equal(sf_lin.view(m, k // BLOCK), ref_sf)
    n_bad = int((sf_sw.view(m, k // BLOCK) != ref_sf).sum().item())
    print(
        f"[control MoE input M={m}] swizzled buffer read as linear: {n_bad} / {m * (k // BLOCK)} scale bytes differ"
    )
    assert n_bad > 0
