# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_quantize catalog entry (NVFP4: sf_vec_size=16, e4m3 scales).

Every cell is gated the same way: scale bytes bit-exact, codes bit-exact away
from the rounding ties the entry declines to predict. A cell with `input_scale`
runs under the Mistral Large 4 NVFP4 target's static global scale, 1 / input_scale divided in fp32,
on an input drawn against the amax that scale was calibrated for; the rest keep
the canonical 448*6/amax of their own input. A cell with `rank_rows` is the dep4
target's gathered token set: one segment per rank, each rank's live rows then
zero rows up to T_max. A graph cell with `ranks` replays the gathered rows of
that many ranks, each rank's segment padded on its own.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "fp4_quantize requires a CUDA device"

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cuda"
)

#: e4m3 448, the largest finite scale byte; 0x7F is e4m3's NaN.
_SF_SATURATED = 0x7E

#: Rows per reference call. The reference's intermediates at 32768 x 6144 hold
#: ~16 GiB; a 131072-row cell is gated a slice at a time, each row against the
#: same rows of the op's one call.
_GATE_ROWS = 32768


def _generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _global_scale(input_scale):
    """The target's static global scale: 1 / input_scale, divided in fp32."""
    return 1.0 / torch.tensor(input_scale, dtype=torch.float32, device="cuda").reshape(-1)


def _calibrated(shape, gs, over, g):
    """randn rescaled so its largest magnitude is `over` x the calibrated 448*6/gs."""
    x = torch.randn(shape, generator=g, device="cuda")
    return (x * (over * 448.0 * 6.0 / gs.item() / x.abs().max())).to(torch.bfloat16)


def _build(spec, seed):
    g = _generator(seed)
    if "input_scale" in spec:
        gs = _global_scale(spec["input_scale"])
        x = _calibrated(spec["shape"], gs, spec["input_amax_over_calibrated"], g)
    else:
        x = torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])
        # The canonical NVFP4 activation global scale, 448*6/amax.
        gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    if "rank_rows" in spec:
        # The gathered token set: each rank's segment is T_max rows, its live
        # rows first and zero padding after them.
        live = spec["rank_rows"]
        t_max = spec["shape"][0] // len(live)
        assert max(live) == t_max and t_max * len(live) == spec["shape"][0], live
        for rank, n in enumerate(live):
            x[rank * t_max + n : (rank + 1) * t_max] = 0
    return x, gs, spec["vec"], spec["swizzled"]


def _every_block_max(spec, seed):
    """One 16-element block per positive finite bf16 value, that value its max,
    plus one all-zero block: 32640 blocks, 85 rows of 6144."""
    g = _generator(seed)
    rows, k = spec["shape"]
    bits = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16)
    peak = torch.cat([bits.view(torch.bfloat16).float(), torch.zeros(1)]).cuda()
    assert peak.numel() == rows * k // 16
    rest = peak.unsqueeze(-1) * (torch.rand(peak.numel(), 16, generator=g, device="cuda") * 2 - 1)
    blocks = rest.to(torch.bfloat16).float()
    where = torch.randint(0, 16, (peak.numel(),), generator=g, device="cuda")
    sign = torch.where(torch.rand(peak.numel(), generator=g, device="cuda") < 0.5, -1.0, 1.0)
    blocks[torch.arange(peak.numel(), device="cuda"), where] = peak * sign
    x = blocks.to(torch.bfloat16).reshape(rows, k)
    assert torch.equal(x.float().abs().reshape(-1, 16).amax(-1), peak)
    return x, peak


def _unpack(packed):
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


def _gate(x, gs, vec, swizzled, packed, sf):
    """Codes bit-exact away from the rounding ties the entry declines to predict.

    `near_tie` is the entry's own statement about where the kernel's approximate
    reciprocals decide the direction. Everything outside it has to match; the
    count inside it is asserted small, so the window cannot quietly grow into a
    licence for the kernel to be wrong. A NaN product is 0 x inf: an exact zero
    in a block whose scale byte underflowed to 0, which dequantizes to 0 whatever
    its codes -- asserted to occur nowhere else, and not compared.
    """
    ref_packed, ref_sf = op.reference(x, gs, vec, False, swizzled)
    assert torch.equal(sf, ref_sf), "block scales are exact; a mismatch is a defect"
    assert not ((sf & 0x7F) == 0x7F).any(), "a scale byte is e4m3 NaN"

    scaled = op.scaled(x, gs, vec)
    tie = op.near_tie(scaled)
    nan = scaled.isnan()
    if nan.any():
        assert not swizzled, "only the linear layout keeps scale bytes in block order"
        nan_blocks = nan.reshape(-1, vec).any(-1)
        assert (ref_sf[nan_blocks] == 0).all(), "a NaN product outside an underflowed block"
    codes, ref_codes = _unpack(packed), _unpack(ref_packed)
    compared = ~tie & ~nan
    assert torch.equal(codes[compared], ref_codes[compared]), "codes differ away from a tie"
    # Measured across the canonical cells: 0.13% to 0.24%. Higher than a random
    # landing would give by some three orders of magnitude, because most of
    # these are *exact* midpoint hits rather than near misses -- a bf16 input
    # times an e4m3 scale lands on a grid coarse enough to hit midpoints often.
    # Under Mistral Large 4's static scales: 0.003% (dep4's zero-padded 131072
    # rows), 0.004% (32768 rows) to 0.24% (a one-row graph). The bound is 4x the worst measured, tight enough that a
    # window quietly widening into a licence for the kernel to be wrong would
    # trip it.
    assert tie.sum().item() <= 0.01 * tie.numel(), (
        f"{tie.sum().item()} of {tie.numel()} elements sit in the tie window; "
        "a window this wide is no longer a statement about rounding"
    )
    return ref_sf


def _gate_rows(x, gs, vec, swizzled, packed, sf):
    """`_gate` over `_GATE_ROWS`-row slices of one call's output (linear scales)."""
    rows = x.shape[0]
    if rows <= _GATE_ROWS:
        return _gate(x, gs, vec, swizzled, packed, sf)
    assert not swizzled, "only the linear layout keeps scale bytes in row order"
    blocks = x.shape[-1] // vec
    for lo in range(0, rows, _GATE_ROWS):
        hi = min(lo + _GATE_ROWS, rows)
        _gate(x[lo:hi], gs, vec, swizzled, packed[lo:hi], sf[lo * blocks : hi * blocks])
    return sf


def _padded_replays(b, ranks):
    """(live row indices, zero tail) for each padded replay of a b-row graph.

    One rank: live rows 1..b-1 lead and the dummy rows trail. Four or eight
    ranks: the gathered rows are one b/ranks segment per rank, each its own
    live rows then padding, every live count 0..b/ranks met at every rank (an
    idle rank is all padding); the padding alternates between dummy rows and
    zero rows. Eight ranks also replay a single live row in the whole batch,
    at each rank in turn, the other seven ranks all padding: at 8 rows, the
    batch an engine replays when one rank decodes a lone request beside
    seven idle ranks.
    """
    if ranks == 1:
        return [(torch.arange(r, device="cuda"), False) for r in range(1, b)]
    seg = b // ranks
    out = []
    for r in range(seg + 1):
        pattern = (r, seg - r, seg if r % 2 else 0, r // 2)
        if ranks == 8:
            pattern += (seg - r // 2, 0 if r % 2 else seg, 0, min(r, 1))
        assert len(pattern) == ranks, ranks
        for shift in range(ranks):
            counts = pattern[shift:] + pattern[:shift]
            live = torch.cat(
                [i * seg + torch.arange(n, device="cuda") for i, n in enumerate(counts)]
            )
            out.append((live, (r + shift) % 2 == 1))
    if ranks == 8:
        for rank in range(ranks):
            out.append((torch.tensor([rank * seg], device="cuda"), rank % 2 == 1))
    return out


def _graphs(spec, gs, seed):
    """Each batch captured once, then replayed on new rows written into its input."""
    vec, swizzled, over = spec["vec"], spec["swizzled"], spec["input_amax_over_calibrated"]
    width = spec["shape"][-1]
    blocks = width // vec
    ranks = spec.get("ranks", 1)
    g = _generator(seed)
    side = torch.cuda.Stream()
    for b in spec["graph_rows"]:
        x = _calibrated((b, width), gs, over, g)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(x, gs, vec, False, swizzled)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed, sf = op(x, gs, vec, False, swizzled)
        for _ in range(2):
            x.copy_(_calibrated((b, width), gs, over, g))
            graph.replay()
            eager_packed, eager_sf = op(x.clone(), gs, vec, False, swizzled)
            assert torch.equal(packed, eager_packed) and torch.equal(sf, eager_sf), b
            _gate(x, gs, vec, swizzled, packed, sf)
        # A padded batch: some live rows, the rest whatever the buffer holds.
        for live, zero_tail in _padded_replays(b, ranks):
            x.copy_(_calibrated((b, width), gs, over, g))
            pad = torch.ones(b, dtype=torch.bool, device="cuda")
            pad[live] = False
            if zero_tail:
                x[pad] = 0
            else:
                garbage = torch.randn(b, width, generator=g, device="cuda") * 1e4
                x[pad] = garbage[pad].to(x.dtype)
            rows = x[live].clone()
            graph.replay()
            eager_packed, eager_sf = op(rows, gs, vec, False, swizzled)
            assert torch.equal(packed[live], eager_packed), (b, live.tolist())
            assert torch.equal(sf.view(b, blocks)[live].reshape(-1), eager_sf), (b, live.tolist())


def _sweep_counts(spec):
    """`sweep_counts` as given, or `sweep_rows` (lo, hi[, step]) inclusive."""
    if "sweep_counts" in spec:
        return spec["sweep_counts"]
    lo, hi, *step = spec["sweep_rows"]
    return range(lo, hi + 1, *step)


def _sweep(x, gs, vec, swizzled, counts, packed, sf):
    """Every row count in `counts` as its own call: each bitwise the leading rows.

    Compared as int64 words, verdicts kept on the device and read once; on a
    mismatch the first failing count is found again and its bytes counted.
    """
    blocks = x.shape[-1] // vec
    full_codes, full_sf = packed.view(torch.int64), sf.view(torch.int64)

    def differs_at(m):
        p, s = op(x[:m], gs, vec, False, swizzled)
        return (p.view(torch.int64) != full_codes[:m]).any() | (
            s.view(torch.int64) != full_sf[: m * blocks // 8]
        ).any()

    differs = torch.zeros((), dtype=torch.bool, device="cuda")
    for m in counts:
        differs |= differs_at(m)
    if differs.item():
        m = next(m for m in counts if differs_at(m).item())
        p, s = op(x[:m], gs, vec, False, swizzled)
        bad = (p != packed[:m]).sum().item() + (s != sf[: m * blocks]).sum().item()
        raise AssertionError(f"{m} rows: {bad} bytes differ from the leading rows")


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31

    if "block_max" in spec:
        x, peak = _every_block_max(spec, seed)
        for input_scale in spec["input_scale"]:
            gs = _global_scale(input_scale)
            packed, sf = op(x, gs, spec["vec"], False, spec["swizzled"])
            ref_sf = _gate(x, gs, spec["vec"], spec["swizzled"], packed, sf)
            # Both ends of the e4m3 range are reached at every layer's scale.
            assert ((ref_sf == 0) & (peak != 0)).any(), input_scale
            assert (ref_sf == _SF_SATURATED).any(), input_scale
        return

    x, gs, vec, swizzled = _build(spec, seed)
    packed, sf = op(x, gs, vec, False, swizzled)
    ref_sf = _gate_rows(x, gs, vec, swizzled, packed, sf)
    if spec.get("input_amax_over_calibrated", 1.0) > 1.0:
        # Measured at twice the calibrated amax, layer 3: 7.6% of blocks at
        # 32768 rows, 5.4% at dep4's 131072 (a larger draw's own max sets it).
        saturated = (ref_sf == _SF_SATURATED).float().mean().item()
        assert saturated >= 0.01, f"only {saturated:.4%} of blocks saturate"
    if "rank_rows" in spec:
        # Zero padding rows quantize to zero codes under a zero scale byte.
        t_max = spec["shape"][0] // len(spec["rank_rows"])
        blocks = x.shape[-1] // vec
        for rank, n in enumerate(spec["rank_rows"]):
            lo, hi = rank * t_max + n, (rank + 1) * t_max
            assert not packed[lo:hi].any() and not sf[lo * blocks : hi * blocks].any(), rank
    if "graph_rows" in spec:
        _graphs(spec, gs, seed)
    if "sweep_rows" in spec or "sweep_counts" in spec:
        _sweep(x, gs, vec, swizzled, _sweep_counts(spec), packed, sf)


def test_the_quantization_round_trips() -> None:
    """Independent of the reference: dequantizing has to return the input.

    The cells compare two implementations of one arithmetic, which would both be
    wrong together if the arithmetic were. This bounds the error against the
    input instead -- e2m1 carries one mantissa bit, so at worst half a step of
    the block's own scale.
    """
    x = torch.randn(64, 7168, dtype=torch.bfloat16, device="cuda")
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    packed, sf = op(x, gs, 16, False, False)

    codes = _unpack(packed)
    value = _E2M1_VALUES[(codes & 7).long()]
    value = torch.where((codes & 8).bool(), -value, value)
    scale = sf.view(64, 7168 // 16).view(torch.float8_e4m3fn).float()
    deq = value * scale.repeat_interleave(16, dim=-1) / gs.item()

    block_amax = x.float().view(64, 7168 // 16, 16).abs().amax(-1, keepdim=True)
    err = (deq - x.float()).view(64, 7168 // 16, 16).abs().amax(-1, keepdim=True)
    assert torch.all(err <= block_amax * 0.5 + 1e-6)


def test_the_two_layouts_differ_at_equal_length() -> None:
    """At a multiple of 128 rows the linear and swizzled buffers are the same
    length, so a consumer handed the wrong one accepts it. The linear cells'
    exact scale gate is what tells them apart -- this shows it can."""
    x, gs, _, _ = _build(dict(shape=(128, 2560), dtype=torch.bfloat16, vec=16, swizzled=False), 0)
    _, linear = op(x, gs, 16, False, False)
    _, swizzled = op(x, gs, 16, False, True)
    assert linear.numel() == swizzled.numel() == 128 * 2560 // 16
    assert not torch.equal(linear, swizzled)
    assert torch.equal(linear, op.reference(x, gs, 16, False, False)[1])


def test_guard_refuses_a_per_token_global_scale() -> None:
    """Element 0 is applied to every row and the rest are ignored."""
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(x, torch.ones(8, device="cuda"), 16, False, True)


def test_guard_refuses_a_partial_block() -> None:
    x = torch.randn(8, 120, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="whole number of 16-element"):
        with validating(op):
            op(x, torch.ones(1, device="cuda"), 16, False, True)


def test_guard_refuses_a_non_contiguous_input() -> None:
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, torch.ones(1, device="cuda"), 16, False, True)
