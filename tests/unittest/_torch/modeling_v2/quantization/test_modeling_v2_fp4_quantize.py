# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_quantize catalog entry.

NVFP4 (sf_vec_size=16, e4m3 scales, both scale layouts) and MXFP4 (sf_vec_size=32, UE8M0
scales, linear layout: the DeepSeek-V4.1 index K / index Q quantizer). The DeepSeek-V4.1
section below keeps the sweeps behind the V4.1 cells: every row count, CUDA-graph capture of
the target's calls, and the MoE input's chunk views.
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize as op,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "fp4_quantize requires a CUDA device"

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cuda"
)


def _static_g1(calib):
    """A modelopt-style static activation scale, fixed from a calibration batch.

    modelopt stores `input_scale = amax / (448 * 6)`; the op takes its reciprocal.
    Calibrating on a *different* batch than the one quantized is what serving does:
    `g` is then not `448*6/amax` of the tensor in hand.
    """
    input_scale = calib.abs().max().float() / (448.0 * 6.0)
    return (1.0 / input_scale).reshape(1).contiguous()


def _build(spec, seed):
    """Cell spec -> the op's five arguments.

    Optional keys, absent from the deepseek-r1 cells: `ue8m0` (default False);
    `gs` -- "amax" (default: 448*6/amax of the input), "static" (`_static_g1` of a
    separate [4096, K] batch, DeepSeek-V4.1's routed-expert scale), or None (MXFP4);
    `block_mag` -- each 32-block of the input times its own 2^j, j in -20..20.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(spec["shape"], generator=g, device="cuda")
    if spec.get("block_mag", False):
        j = torch.randint(-20, 21, (x.numel() // 32, 1), generator=g, device="cuda")
        x = (x.view(-1, 32) * torch.exp2(j.float())).view(spec["shape"])
    x = x.to(spec["dtype"])
    kind = spec.get("gs", "amax")
    if kind == "amax":
        # The canonical NVFP4 activation global scale, 448*6/amax.
        gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    elif kind == "static":
        calib = torch.randn(4096, x.shape[-1], generator=g, device="cuda").to(spec["dtype"])
        gs = _static_g1(calib)
    else:
        assert kind is None, kind
        gs = None
    return x, gs, spec["vec"], spec.get("ue8m0", False), spec["swizzled"]


def _unpack(packed):
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Codes bit-exact away from the rounding ties the entry declines to predict.

    `near_tie` is the entry's own statement about where the kernel's approximate
    reciprocals decide the direction. Everything outside it has to match; the
    count inside it is asserted small, so the window cannot quietly grow into a
    licence for the kernel to be wrong.

    MXFP4 cells have no tie window: the scale is a power of two, so `x * 2^-e` is
    exact and a midpoint stays exactly a midpoint. Their data is bit-exact, ties
    included.
    """
    x, gs, vec, ue8m0, swizzled = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    with validating(op):
        packed, sf = op(x, gs, vec, ue8m0, swizzled)
    ref_packed, ref_sf = op.reference(x, gs, vec, ue8m0, swizzled)

    assert torch.equal(sf, ref_sf), "block scales are exact; a mismatch is a defect"
    if ue8m0:
        assert packed.shape == (*x.shape[:-1], x.shape[-1] // 2)
        assert torch.equal(packed, ref_packed), "MXFP4 data is bit-exact, ties included"
        return

    tie = op.near_tie(op.scaled(x, gs, vec))
    codes, ref_codes = _unpack(packed), _unpack(ref_packed)
    assert torch.equal(codes[~tie], ref_codes[~tie]), "codes differ away from a rounding tie"
    # Measured across these cells: 0.13% to 0.24%. Higher than a random landing
    # would give by some three orders of magnitude, because most of these are
    # *exact* midpoint hits rather than near misses -- a bf16 input times an
    # e4m3 scale lands on a grid coarse enough to hit midpoints often. The bound
    # is 4x the worst measured, tight enough that a window quietly widening into
    # a licence for the kernel to be wrong would trip it. The deepseek-v4.1 cells'
    # static scale, calibrated on another batch, gave 0.05% to 0.29% over the
    # sweeps in the DeepSeek-V4.1 section below.
    assert tie.sum().item() <= 0.01 * tie.numel(), (
        f"{tie.sum().item()} of {tie.numel()} elements sit in the tie window; "
        "a window this wide is no longer a statement about rounding"
    )


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


# ─── DeepSeek-V4.1 ────────────────────────────────────────────────────────────
# The V4.1 target quantizes, per step: index K `[n, 128]` and index Q `[n, 32, 128]`
# in MXFP4 mode (`op(x, None, 32, True, False)`, the checkpoint's `fp4_act_quant(x,
# 32)`), and the routed-expert input `[n, 5120]` in NVFP4 mode, linear layout, with the
# layer's static FC1 activation scale `g1 = 1 / input_scale` as a `[1]` fp32 device
# tensor. Decode steps run inside CUDA graphs. The cells cover the ends of each range;
# the tests below sweep it.

DEV = torch.device("cuda")
NV_VEC = 16
MX_VEC = 32
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 32
V41_HIDDEN = 5120
# The reference floors the block max at 6 * 2^-126, so its smallest scale is 2^-126
# (UE8M0 byte 0x01). The op has no floor: a block whose max is at or below 3 * 2^-126
# (so that max * fp32(1/6) <= 2^-127) gets scale byte 0x00 and is written as signed
# zeros. Every block with a larger max matches the reference bit for bit.
MX_REF_AMAX_FLOOR = 6 * 2.0**-126
MX_OP_ZERO_AMAX = 3 * 2.0**-126
_INV6_F32 = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=DEV)
# e2m1 midpoints, and the even code each one rounds to under round-to-nearest-even
_MX_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MX_MID_CODES = (0, 2, 2, 4, 4, 6, 6)
_MX_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _mx(x):
    return op(x, None, MX_VEC, True, False)


def _mx_ref(x):
    return op.reference(x, None, MX_VEC, True, False)


def _mx_wrong(x, ties="even", scale="ceil"):
    """Deliberately wrong MXFP4 variants, for the controls: (packed [M, K/2], sf [M, K/32]).

    `ties="zero"` rounds e2m1 ties toward zero instead of to even; `scale="floor"` takes
    2^floor instead of 2^ceil for the block scale. With the defaults it is the reference.
    """
    k = x.shape[-1]
    m = x.numel() // k
    xf = x.reshape(m, k // MX_VEC, MX_VEC).float()
    bits = (xf.abs().amax(-1).clamp_min(MX_REF_AMAX_FLOOR) * _INV6_F32).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127
    if scale == "ceil":
        e = e + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s = ((e + 127) << 23).view(torch.float32)
    a = (xf / s.unsqueeze(-1)).clamp(-6.0, 6.0).abs()

    def rnd(v):
        return torch.round(v) if ties == "even" else torch.ceil(v - 0.5)

    r = torch.where(a < 2, rnd(a * 2) / 2, torch.where(a < 4, rnd(a), rnd(a / 2) * 2))
    code = torch.where(r < 2, r * 2, torch.where(r < 4, r + 2, r / 2 + 4)).to(torch.uint8)
    codes = (code | (torch.signbit(xf).to(torch.uint8) << 3)).reshape(m, k)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), (e + 127).to(torch.uint8)


def _block_mag_rows(n, gen=None):
    """[n, 128] bf16 with a different power-of-two magnitude per 32-block (2^-20..2^20)."""
    mag = torch.exp2(torch.randint(-20, 21, (n, 4, 1), device=DEV, generator=gen).float())
    x = torch.randn(n, 4, MX_VEC, device=DEV, generator=gen) * mag
    return x.to(torch.bfloat16).view(n, INDEX_HEAD_DIM)


def _assert_data(got, ref, scaled):
    """NVFP4 data: bit-exact away from e2m1 rounding ties; one adjacent code at a tie.

    Returns `(near_tie_count, mismatch_count, total)`. The two counts are different
    quantities: the first is how many elements sit inside the tie window at all (the
    check's blind spot), the second how many of those the kernel resolved the other way.
    """
    g = _unpack(got)
    r = _unpack(ref)
    near_tie = op.near_tie(scaled)
    assert torch.equal(g[~near_tie], r[~near_tie]), "codes differ away from a rounding tie"
    diff = (g != r) & near_tie
    if diff.any():
        # at a tie the kernel may pick either neighbour, never anything else
        assert torch.equal(g[diff] >> 3, r[diff] >> 3), "sign flipped at a tie"
        delta = (g[diff] & 7).int() - (r[diff] & 7).int()
        assert int(delta.abs().max()) == 1, "non-adjacent e2m1 code at a tie"
    return int(near_tie.sum()), int(diff.sum()), diff.numel()


REF_CHUNK_ELEMS = 16 * 1024 * 1024


def _assert_call_matches_ref(x, data, sf_linear, g1):
    """Row-chunked check of one NVFP4 linear-layout call against `op.reference`.

    Scale bytes exact, data as `_assert_data`. Returns the summed counts.
    """
    rows, k = x.shape
    cols = k // NV_VEC
    step = max(1, REF_CHUNK_ELEMS // k)
    n_tie = n_miss = n_all = 0
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        ref_data, ref_sf = op.reference(x[lo:hi], g1, NV_VEC, False, False)
        assert torch.equal(sf_linear.view(rows, cols)[lo:hi], ref_sf.view(hi - lo, cols))
        tie, miss, total = _assert_data(data[lo:hi], ref_data, op.scaled(x[lo:hi], g1, NV_VEC))
        n_tie, n_miss, n_all = n_tie + tie, n_miss + miss, n_all + total
    return n_tie, n_miss, n_all


def _launched_kernels(call):
    """Names of the CUDA kernels one warm call launches."""
    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    ]


def test_v41_mxfp4_every_bf16_block_amax() -> None:
    """Every positive finite bf16 value as a 32-block max: the UE8M0 rule at every boundary.

    Subnormals included. The max lane rotates through all 32 positions and alternates
    sign, and the other 31 lanes are random in [-max, max]. Five all-zero blocks with
    different sign patterns close the set.
    """
    torch.manual_seed(20)
    pos_bits = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16)  # 0x0001..0x7F7F
    amax = pos_bits.view(torch.bfloat16).float().to(DEV)
    n = amax.numel()
    idx = torch.arange(n, device=DEV)
    u = torch.rand(n, MX_VEC, device=DEV) * 2 - 1
    blocks = (u * amax[:, None]).to(torch.bfloat16).float()
    blocks[idx, idx % MX_VEC] = torch.where(idx % 2 == 0, amax, -amax)
    neg0 = -torch.zeros(MX_VEC, device=DEV)
    pos0 = torch.zeros(MX_VEC, device=DEV)
    lane = torch.arange(MX_VEC, device=DEV)
    zeros = torch.stack(
        [
            pos0,
            neg0,
            torch.where(lane % 2 == 0, neg0, pos0),
            torch.where(lane % 2 == 1, neg0, pos0),
            torch.where(lane < 16, neg0, pos0),
        ]
    )
    x = torch.cat([blocks, zeros]).to(torch.bfloat16).view(-1, INDEX_HEAD_DIM)  # 8161 rows
    blk_amax = x.float().view(-1, MX_VEC).abs().amax(-1)
    assert torch.equal(blk_amax[:n], amax)

    with validating(op):
        data, sf = _mx(x)
    rows = x.shape[0]
    assert data.shape == (rows, INDEX_HEAD_DIM // 2) and sf.shape == (rows * 4,)
    ref_data, ref_sf = _mx_ref(x)
    got_c = _unpack(data).view(-1, MX_VEC)
    ref_c = _unpack(ref_data).view(-1, MX_VEC)

    # bit-exact wherever the block max is above 3 * 2^-126
    live = blk_amax > MX_OP_ZERO_AMAX
    assert torch.equal(sf[live], ref_sf[live])
    assert torch.equal(got_c[live], ref_c[live])
    # the bf16 range reaches every UE8M0 scale from 2^-126 to 2^126
    assert int(sf[live].min()) == 1 and int(sf[live].max()) == 253
    # each boundary: max = 6 * 2^e is scale 2^e exactly, the next bf16 up is 2^(e+1)
    e = torch.arange(-126, 126, device=DEV)
    at = (6 * torch.exp2(e.double())).to(torch.bfloat16).view(torch.int16).long() - 1
    assert torch.equal(sf[at], (e + 127).to(torch.uint8))
    assert torch.equal(sf[at + 1], (e + 128).to(torch.uint8))

    # At or below 3 * 2^-126 the op writes scale byte 0x00 where the reference's floor
    # gives 0x01, and the block comes out as signed zeros: bf16 bit patterns
    # 0x0001..0x0140, plus the five all-zero blocks.
    dead = ~live
    assert int(dead.sum()) == 0x140 + 5, int(dead.sum())
    sign_only = torch.signbit(x.float().view(-1, MX_VEC)).to(torch.uint8) << 3
    assert (sf[dead] == 0).all() and (ref_sf[dead] == 1).all()
    assert torch.equal(got_c[dead], sign_only[dead])
    # The reference also rounds such a block to signed zeros while max <= 2^-128.
    # Above that it keeps nonzero codes, so there the two value sets differ.
    tiny = dead & (blk_amax <= 2.0**-128)
    assert torch.equal(ref_c[tiny], sign_only[tiny])
    lost = dead & (blk_amax > 2.0**-128)
    assert int(lost.sum()) == 0x140 - 0x20, int(lost.sum())
    assert ((ref_c[lost] & 7) != 0).any(-1).all()

    # Deliberately wrong control: 2^floor instead of 2^ceil. The op disagrees with it
    # on every live block whose (floored) max / 6 is not a power of two: 31940 of the
    # 32319. The 379 it agrees on are the 252 maxima 1.5 * 2^j and the 128 maxima
    # floored to 6 * 2^-126 (one of them in both sets).
    _, fl_sf = _mx_wrong(x, scale="floor")
    ratio = (blk_amax.clamp_min(MX_REF_AMAX_FLOOR) * _INV6_F32).view(torch.int32)
    pow2 = (ratio & 0x7FFFFF) == 0
    moved = fl_sf.view(-1) != sf
    assert torch.equal(moved[live], ~pow2[live])
    assert int(moved[live].sum()) == 31940, int(moved[live].sum())


def test_v41_mxfp4_e2m1_midpoints_every_scale() -> None:
    """Every e2m1 midpoint and grid point, both signs, both nibbles, at every UE8M0 scale.

    For each exponent e in [-126, 126] one block holds max 6 * 2^e (3.5 * 2^126 at the
    top, where 6 * 2^126 is not a finite bf16), which fixes the scale at exactly 2^e.
    The block also holds +-{midpoints, grid points} * 2^e, and a copy rolled by one
    lane puts every value in the other nibble.
    """
    blocks, exps = [], []
    for e in range(-126, 127):
        top = 6.0 if e <= 125 else 3.5
        mags = [top] + [v for v in _MX_MIDS + _MX_GRID if v <= top]
        lanes = [v for m in mags for v in (m, -m)]
        lanes += [0.0] * (MX_VEC - len(lanes))
        for block in (lanes, lanes[-1:] + lanes[:-1]):
            blocks.append([v * 2.0**e for v in block])
            exps.append(e)
    # pad to a whole number of 128-wide rows with the e = 0 pair
    blocks += blocks[252:254]
    exps += exps[252:254]
    assert exps[252] == 0 and len(blocks) % 4 == 0
    xs = torch.tensor(blocks, dtype=torch.float64)
    x = xs.to(torch.bfloat16)
    assert torch.equal(x.double(), xs), "midpoint inputs must be exact in bf16"
    x = x.to(DEV).view(-1, INDEX_HEAD_DIM)
    exps = torch.tensor(exps, device=DEV)

    with validating(op):
        data, sf = _mx(x)
    ref_data, ref_sf = _mx_ref(x)
    assert torch.equal(sf, (exps + 127).to(torch.uint8))
    assert torch.equal(sf, ref_sf)
    assert torch.equal(data, ref_data)

    got = _unpack(data).view(-1, MX_VEC)
    scaled = x.double().view(-1, MX_VEC) / torch.exp2(exps.double()).unsqueeze(-1)  # exact
    for mid, code in zip(_MX_MIDS, _MX_MID_CODES):
        at = scaled.abs() == mid
        assert int(at.sum()) >= 4 * 252, (mid, int(at.sum()))
        want = code | (torch.signbit(scaled[at]).to(torch.uint8) << 3)
        assert torch.equal(got[at], want), mid

    # Deliberately wrong control: ties toward zero. It differs from the op on exactly
    # the midpoints that round-to-nearest-even takes away from zero (0.75, 1.75, 3.5),
    # and nowhere else.
    tz_data, _ = _mx_wrong(x, ties="zero")
    tz = _unpack(tz_data).view(-1, MX_VEC)
    away = (scaled.abs() == 0.75) | (scaled.abs() == 1.75) | (scaled.abs() == 3.5)
    assert torch.equal(tz != got, away)


MX_ROWS = (1, 2, 3, 7, 31, 32, 33, 127, 128, 129, 1023, 1024, 4096, 8192, 65536, 262143, 262144)


def _mx_check_call(x):
    """One MXFP4 call: bit-exact vs the reference, in the bytes indexer_k_cache_scatter takes."""
    with validating(op):
        data, sf = _mx(x)
    rows = x.numel() // INDEX_HEAD_DIM
    assert data.shape == (*x.shape[:-1], INDEX_HEAD_DIM // 2) and data.dtype == torch.uint8
    assert sf.shape == (rows * 4,) and sf.dtype == torch.uint8
    assert data.is_contiguous() and sf.is_contiguous()
    ref_data, ref_sf = _mx_ref(x)
    assert torch.equal(data, ref_data)
    assert torch.equal(sf, ref_sf)
    # The scatter takes k_data = data as is and k_scale = sf.view(int32).view(T, 1).
    # That is a free, 4-byte-aligned view. Word t holds block g's exponent in its byte g
    # (little-endian).
    word = sf.view(torch.int32).view(rows, 1)
    assert word.data_ptr() == sf.data_ptr() and sf.data_ptr() % 4 == 0
    r = ref_sf.view(rows, 4).to(torch.int64)
    want = r[:, 0] | (r[:, 1] << 8) | (r[:, 2] << 16) | (r[:, 3] << 24)
    assert torch.equal(word.view(-1).to(torch.int64) & 0xFFFFFFFF, want)


def test_v41_mxfp4_index_kq_rows_bit_exact() -> None:
    """Index K rows 1..262144, index Q as [8192, 32, 128], and random bf16 bit patterns."""
    for t in MX_ROWS:
        torch.manual_seed(1000 + t)
        _mx_check_call(_block_mag_rows(t))
    # index Q as the model forms it: 8192 tokens x 32 heads = 262144 rows, rank 3
    torch.manual_seed(2000)
    q = torch.randn(8192, INDEX_N_HEADS, INDEX_HEAD_DIM, device=DEV).to(torch.bfloat16)
    _mx_check_call(q)
    # Random finite bf16 bit patterns: blocks spanning the whole exponent range,
    # subnormal lanes included.
    torch.manual_seed(2001)
    bits = torch.randint(-32768, 32768, (65536, INDEX_HEAD_DIM), dtype=torch.int32, device=DEV)
    xb = bits.to(torch.int16).view(torch.bfloat16)
    xb = torch.where(torch.isfinite(xb), xb, torch.zeros_like(xb))
    assert (xb.float().view(-1, MX_VEC).abs().amax(-1) > MX_OP_ZERO_AMAX).all()
    _mx_check_call(xb)


def test_v41_mxfp4_ignores_global_scale_one_kernel() -> None:
    """MXFP4 mode never reads global_scale, and one call is one 32-block UE8M0 kernel launch."""
    torch.manual_seed(21)
    x = torch.randn(64, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    base = _mx(x)
    for g in (1.0, 2.0, 1e-3, 448.0 * 6.0 / x.abs().max().item()):
        gs = torch.tensor([g], dtype=torch.float32, device=DEV)
        got = op(x, gs, MX_VEC, True, False)
        assert torch.equal(got[0], base[0]) and torch.equal(got[1], base[1]), g
    for t in (1, 262144):
        xx = torch.randn(t, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        launched = _launched_kernels(lambda: _mx(xx))
        assert len(launched) == 1, (t, launched)
        assert "quantize_with_block_size" in launched[0], launched[0]
        assert "__nv_bfloat16, 32, true" in launched[0], launched[0]


# Every decode row count 1..32, dep4's gathered MoE rows (4 ranks x 16 / 24 / 32), and
# prefill chunks up to the MoE chunk size 8192.
V41_ROWS = tuple(range(1, 33)) + (64, 96, 128, 1023, 1024, 2048, 4096, 8192)
V41_GRAPH_ROWS = tuple(range(1, 33)) + (64, 96, 128, 1024, 8192)


def test_v41_moe_input_k5120_every_row_count() -> None:
    """NVFP4 at the V4.1 hidden width, linear and swizzled, rows 1 through 8192.

    `g` is a static `1 / input_scale` calibrated on a separate batch (the serving
    regime), so it is not the canonical `448*6/amax` of the tensor quantized; the row
    count and the global scale are the only things that change between calls. The
    swizzled buffer is checked against the reference's swizzle (offsets and zero
    padding), and its data against the linear call's.
    """
    cols = V41_HIDDEN // NV_VEC  # 320, a multiple of 4: row padding only when swizzled
    torch.manual_seed(5120)
    g1 = _static_g1(torch.randn(4096, V41_HIDDEN, device=DEV).to(torch.bfloat16))
    n_tie = n_miss = n_all = 0
    for t in V41_ROWS:
        x = torch.randn(t, V41_HIDDEN, dtype=torch.bfloat16, device=DEV)
        with validating(op):
            data, sf = op(x, g1, NV_VEC, False, False)
        assert data.shape == (t, V41_HIDDEN // 2) and sf.shape == (t * cols,)
        tie, miss, total = _assert_call_matches_ref(x, data, sf, g1)
        n_tie, n_miss, n_all = n_tie + tie, n_miss + miss, n_all + total
        with validating(op):
            data_sw, sf_sw = op(x, g1, NV_VEC, False, True)
        assert torch.equal(data_sw, data), "data must not depend on the layout"
        assert sf_sw.shape == (-(-t // 128) * 128 * cols,)
        assert torch.equal(sf_sw, op.reference(x, g1, NV_VEC, False, True)[1])
    # Ceilings on the check's blind spot; correctness is the bit-exactness above.
    # Measured over these 88 M elements: 0.20 % near-ties, 0 disagreements.
    assert n_miss / n_all < 0.005 and n_tie / n_all < 0.02, (n_tie, n_miss, n_all)


def test_v41_cuda_graph_replay_is_bitwise_eager() -> None:
    """The V4.1 target's three calls captured in one CUDA graph, replayed on new data.

    Per row count: index K `[n, 128]` and index Q `[n, 32, 128]` (MXFP4, linear) and
    the MoE input `[n, 5120]` (NVFP4, linear, static `[1]` global scale) are captured
    once, after one warm-up call on a side stream (what torch.cuda.graph callers do).
    Each replay then quantizes whatever the static input buffers hold: three different
    data sets, each bitwise equal to an eager call on the same values and matched
    against the reference (MXFP4 bit-exact, NVFP4 bit-exact outside the near-tie
    window). The global scale is read from device memory at replay: scaling it in
    place changes the replayed bytes exactly as it changes an eager call's.
    """
    torch.manual_seed(4141)
    g1 = _static_g1(torch.randn(4096, V41_HIDDEN, device=DEV).to(torch.bfloat16))
    for n in V41_GRAPH_ROWS:
        k_in = torch.empty(n, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        q_in = torch.empty(n, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEV)
        h_in = torch.empty(n, V41_HIDDEN, dtype=torch.bfloat16, device=DEV)

        def fill(seed: int) -> None:
            gen = torch.Generator(device=DEV).manual_seed(seed)
            k_in.copy_(_block_mag_rows(n, gen))
            q_in.copy_(torch.randn(n, INDEX_N_HEADS, INDEX_HEAD_DIM, device=DEV, generator=gen))
            h_in.copy_(torch.randn(n, V41_HIDDEN, device=DEV, generator=gen))

        def body():
            kd, ks = _mx(k_in)
            qd, qs = _mx(q_in)
            hd, hs = op(h_in, g1, NV_VEC, False, False)
            return kd, ks, qd, qs, hd, hs

        fill(10 * n)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = body()
        for seed in (10 * n, 10 * n + 1, 10 * n + 2):
            fill(seed)
            graph.replay()
            torch.cuda.synchronize()
            eager = body()
            for got, want in zip(outs, eager):
                assert torch.equal(got, want), (n, seed)
            kd, ks, qd, qs, hd, hs = outs
            ref_d, ref_s = _mx_ref(k_in)
            assert torch.equal(kd, ref_d) and torch.equal(ks, ref_s), (n, seed)
            ref_d, ref_s = _mx_ref(q_in)
            assert torch.equal(qd, ref_d) and torch.equal(qs, ref_s), (n, seed)
            _assert_call_matches_ref(h_in, hd, hs, g1)
        old = g1.clone()
        before = outs[5].clone()
        g1.mul_(1.37)
        graph.replay()
        torch.cuda.synchronize()
        eager = body()
        g1.copy_(old)
        assert torch.equal(outs[4], eager[4]) and torch.equal(outs[5], eager[5]), n
        assert not torch.equal(outs[5], before), (
            "the in-place scale change did not reach the replay"
        )
        del graph, outs


# ─── CUDA-graph capture over Gset: the target's whole step, DSpark off and on ──
# Graph batches: Gset, every G an engine of the envelope captures. That is every integer
# 1..32 (padding off at max_batch_size up to 32, and the head of the padded list) plus
# TRT-LLM's padded default list at max_batch_size 256 (1, 2, 4, 8, 16..128 step 8, 192,
# 256): 46 values. The padding-off list at 256 (1..32, 64, 128, 256) is a subset. A
# captured step is generation-only with next_n = 1 (DSpark off) or 6 (DSpark on: 1 + 5
# drafts), so it holds T = next_n * G token rows, and the target quantizes:
#   - index K of the ratio-2 KV sources at n = G * ceil(next_n / 2)
#     (`num_total_compressed_tokens[2]`): G or 3G;
#   - index K of the ratio-1 KV source at n = T: G or 6G;
#   - index Q `[T, 32, 128]` in every index layer: G or 6G;
#   - the MoE input `[T, 5120]` on tep4 / tp4: G or 6G; on dep4 the rows the 4-rank
#     all-gather hands the router, `[4T, 5120]`: 4G or 24G, at most 6144 <= 8192, so
#     one chunk.
# All five calls of one step are captured together, as in the target.
_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
assert len(_GSET) == 46
DEP4_RANKS = 4


def _step_rows(next_n, G):
    """Row counts of one captured step's five calls: index K ratio 2, index K ratio 1,
    index Q, MoE input (tep4 / tp4), MoE input (dep4, gathered)."""
    t = next_n * G
    return G * -(-next_n // 2), t, t, t, DEP4_RANKS * t


@pytest.mark.parametrize("next_n", (1, 6), ids=("next_n1", "next_n6"))
def test_v41_cuda_graph_step_over_gset(next_n) -> None:
    """The five calls of one captured step at every graph batch G of Gset, replayed on new data.

    Captured the way the engine does it: one graph per G, all in one shared memory pool,
    largest G first, each after an eager call at its shapes on a side stream. Then three
    interleaved replay rounds over every G, each on new data written in place: every
    output bitwise equal to an eager call on the same values, MXFP4 bit-exact against
    the reference, NVFP4 bit-exact outside the near-tie window. Two controls show the
    harness is not blind: every replay rewrote every data buffer (none equals the
    previous round's bytes), and a reference built on the inputs rolled by one row
    matches none of the op's rows. The global scale is read from device memory at
    replay, for both MoE calls of every G.
    """
    torch.manual_seed(4242 + next_n)
    g1 = _static_g1(torch.randn(4096, V41_HIDDEN, device=DEV).to(torch.bfloat16))
    pool = torch.cuda.graph_pool_handle()

    def alloc(G):
        k2, k1, q, h, hg = _step_rows(next_n, G)
        bf = torch.bfloat16
        return (
            torch.empty(k2, INDEX_HEAD_DIM, dtype=bf, device=DEV),
            torch.empty(k1, INDEX_HEAD_DIM, dtype=bf, device=DEV),
            torch.empty(q, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=bf, device=DEV),
            torch.empty(h, V41_HIDDEN, dtype=bf, device=DEV),
            torch.empty(hg, V41_HIDDEN, dtype=bf, device=DEV),
        )

    def fill(ins, seed):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        for k_in in ins[:2]:
            k_in.copy_(_block_mag_rows(k_in.shape[0], gen))
        for x in ins[2:]:
            x.copy_(torch.randn(x.shape, device=DEV, generator=gen))

    def body(ins):
        outs = []
        for x in ins[:3]:
            outs += _mx(x)
        for x in ins[3:]:
            outs += op(x, g1, NV_VEC, False, False)
        return tuple(outs)

    steps = []
    for G in sorted(_GSET, reverse=True):
        ins = alloc(G)
        fill(ins, 1000 * G)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body(ins)  # the eager call at this step's shapes, before its capture
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outs = body(ins)
        steps.append((G, ins, graph, outs))
    steps.reverse()

    n_tie = n_miss = n_all = 0
    prev = {}
    for rnd in range(1, 4):
        for G, ins, graph, outs in steps:
            fill(ins, 1000 * G + rnd)
            graph.replay()
            torch.cuda.synchronize()
            eager = body(ins)
            for got, want in zip(outs, eager):
                assert torch.equal(got, want), (next_n, G, rnd)
            if G in prev:
                for i, old_data in enumerate(prev[G]):
                    assert not torch.equal(outs[2 * i], old_data), ("replay left stale data", G, i)
            prev[G] = tuple(o.clone() for o in outs[::2])
            for j, x in enumerate(ins[:3]):
                ref_d, ref_s = _mx_ref(x)
                assert torch.equal(outs[2 * j], ref_d) and torch.equal(outs[2 * j + 1], ref_s)
            for j, x in enumerate(ins[3:]):
                d, s = outs[6 + 2 * j], outs[7 + 2 * j]
                tie, miss, total = _assert_call_matches_ref(x, d, s, g1)
                n_tie, n_miss, n_all = n_tie + tie, n_miss + miss, n_all + total
    # Row-addressing control: a reference on the inputs rolled by one row matches none of
    # the op's rows (the checks above can see a row slip).
    for G, ins, graph, outs in steps:
        for j, x in enumerate(ins[:3]):
            rows = x.numel() // INDEX_HEAD_DIM
            if rows == 1:
                continue
            ref_d, _ = _mx_ref(torch.roll(x.view(rows, INDEX_HEAD_DIM), 1, 0))
            same = (outs[2 * j].view(rows, -1) == ref_d).all(-1)
            assert int(same.sum()) == 0, ("row-shift control matched", G, j, int(same.sum()))
        for j, x in enumerate(ins[3:]):
            if x.shape[0] == 1:
                continue
            ref_d, _ = op.reference(torch.roll(x, 1, 0), g1, NV_VEC, False, False)
            same = (outs[6 + 2 * j] == ref_d).all(-1)
            assert int(same.sum()) == 0, ("row-shift control matched", G, j, int(same.sum()))
    # The global scale is read from device memory at replay, for both MoE calls.
    old = g1.clone()
    g1.mul_(1.37)
    for G, ins, graph, outs in steps:
        before = (outs[7].clone(), outs[9].clone())
        graph.replay()
        torch.cuda.synchronize()
        eager = body(ins)
        for i in range(6, 10):
            assert torch.equal(outs[i], eager[i]), (G, i)
        assert not torch.equal(outs[7], before[0]) and not torch.equal(outs[9], before[1]), (
            "the in-place scale change did not reach the replay",
            G,
        )
    g1.copy_(old)
    # Measured: next_n 1, 152,371,200 NVFP4 elements, 81,261 near a tie, 12,532 resolved
    # the other way; next_n 6, 914,227,200 / 484,652 / 74,032.
    assert n_miss / n_all < 0.005 and n_tie / n_all < 0.02, (n_tie, n_miss, n_all)
    del steps, prev


# ─── The row axis, 1..N_max, for the target's three call forms ─────────────────
# Rows only size the launch grid of this op, so the row axis is certified as a range:
# (a) one call at N_max against the reference, (b) every row count r in 1..N_max bitwise
# equal to the first r rows of that call (data bytes and linear scale bytes), and (c) the
# same single compiled kernel launched at every probed row count. N_max per form:
#   - index K `[n, 128]`: n = T at the ratio-1 KV source, at most max_num_tokens
#     (16384); certified to 32768.
#   - index Q `[T, 32, 128]`: T in every index layer, including the re-index layers of
#     the decoder bounded-replay pass, whose rows (the finishing requests' 128-row
#     tails plus the generation rows) reach 128 x 256 = 32768 where the pass is not
#     sliced to max_num_tokens; certified to 32768.
#   - MoE input `[n, 5120]`: the target splits the MoE rows into chunks of at most 8192,
#     so no call sees more; certified to 8192. The chunks are row views at nonzero
#     offsets: test_v41_moe_input_chunk_views_of_gathered_rows.
V41_NMAX_MX = 32768
V41_NMAX_MOE = 8192
# At K = 128 the launcher's grid is min(rows, SMs x 128) = 19456 rows on 152 SMs; at
# K = 5120 it is min(rows, SMs x 4) = 608. Index Q presents 32 rows per token, so its grid
# saturates above 608 tokens; index K's above 19456.
_KERNEL_PROBE_ROWS = (
    1,
    2,
    7,
    31,
    128,
    129,
    600,
    607,
    608,
    609,
    1024,
    4097,
    8191,
    8192,
    16383,
    16384,
    16385,
    19455,
    19456,
    19457,
    32767,
    32768,
)


def test_v41_rows_1_to_nmax_prefix_bitwise() -> None:
    """Index K `[r, 128]` and index Q `[r, 32, 128]` (MXFP4) at every r in 1..32768, and
    the MoE input `[r, 5120]` (NVFP4, linear, static `[1]` g1) at every r in 1..8192.

    Each prefix call reads `x[:r]` of one N_max-row input (a contiguous view, as the
    target's inputs are) and must reproduce the N_max call's first r rows bit for bit:
    `data[:r]` and the first `r * scale_bytes_per_row` linear scale bytes. The N_max
    call itself is checked against the reference.
    """
    torch.manual_seed(820)
    n = V41_NMAX_MX
    k = _block_mag_rows(n)
    qmag = torch.exp2(torch.randint(-20, 21, (n, INDEX_N_HEADS, 4, 1), device=DEV).float())
    q = (torch.randn(n, INDEX_N_HEADS, 4, MX_VEC, device=DEV) * qmag).to(torch.bfloat16)
    q = q.view(n, INDEX_N_HEADS, INDEX_HEAD_DIM)
    del qmag
    h = torch.randn(V41_NMAX_MOE, V41_HIDDEN, device=DEV).to(torch.bfloat16)
    g1 = _static_g1(torch.randn(4096, V41_HIDDEN, device=DEV).to(torch.bfloat16))

    def nv(x):
        return op(x, g1, NV_VEC, False, False)

    # (name, N_max input, call, linear scale bytes per token row)
    forms = (
        ("index K", k, _mx, INDEX_HEAD_DIM // MX_VEC),
        ("index Q", q, _mx, INDEX_N_HEADS * INDEX_HEAD_DIM // MX_VEC),
        ("MoE input", h, nv, V41_HIDDEN // NV_VEC),
    )
    for name, x, call, sf_row in forms:
        n = x.shape[0]
        # (a) the N_max call against the reference
        with validating(op):
            fd, fs = call(x)
        assert fs.numel() == n * sf_row
        if call is _mx:
            ref_d, ref_s = _mx_ref(x)
            assert torch.equal(fd, ref_d) and torch.equal(fs, ref_s), name
            del ref_d, ref_s
        else:
            tie, miss, total = _assert_call_matches_ref(x, fd, fs, g1)
            assert miss / total < 0.005 and tie / total < 0.02, (tie, miss, total)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(5):
            start.record()
            call(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        # Every r is run only because the N_max call is this cheap; typical best-of-5 is
        # about 40 us (index K), 260 us (index Q) and 55 us (MoE input).
        assert min(times) < 1.0, (name, times)
        # (b) every prefix, one launch and one device-side compare each (no host sync)
        bad = torch.zeros(n, dtype=torch.bool, device=DEV)
        for r in range(1, n + 1):
            d, s = call(x[:r])
            bad[r - 1] = (d != fd[:r]).any() | (s != fs[: r * sf_row]).any()
        n_bad = int(bad.sum())
        assert n_bad == 0, (name, n_bad, bad.nonzero()[:8].view(-1).tolist())
        # Control: the compare sees a one-row slip at the edges and in the middle.
        for r in (1, 2, n // 2, n - 1):
            d, _ = call(x[:r])
            assert (d != fd[1 : r + 1]).any(), (name, r)
        # (c) one compiled variant: the same single kernel at every probed row count
        names = set()
        for r in (r for r in _KERNEL_PROBE_ROWS if r <= n):
            launched = _launched_kernels(lambda: call(x[:r]))
            assert len(launched) == 1 and "quantize_with_block_size" in launched[0], (
                name,
                r,
                launched,
            )
            names.add(launched[0])
        assert len(names) == 1, (name, names)
        del fd, fs, bad


# ─── The MoE input as the target hands it over: 8192-row chunks at nonzero offsets ─
# The target splits the MoE rows into chunks of at most 8192, and each chunk reaches
# this op as a contiguous row view of the whole `[rows, 5120]` input, at a storage
# offset of a multiple of 8192 rows. The rows are this rank's T (tep4 / tp4, up to
# 16384; the decoder pass, up to 32768) or, on dep4, the 4-rank all-gather's 4 x the
# per-rank row count (up to 65536). Totals: 65536 (the dep4 maximum), and the gathered
# counts 65512, 41804, 41764 and 38096 (4 x 16378 / 10451 / 10441 / 9524, eager steps of
# a batch-256 dep4 engine), whose last chunks of 8168, 844, 804 and 5328 rows the same
# engine shows as router row counts.
_MOE_CHUNK = 8192
_GATHERED_TOTALS = (65536, 65512, 41804, 41764, 38096)


def test_v41_moe_input_chunk_views_of_gathered_rows() -> None:
    """Every 8192-row chunk view of a 65536-row MoE input, and the observed last chunks.

    Each chunk is passed as the row view the target's split makes (contiguous, at a
    nonzero storage offset) and must give the same bytes as the same rows at offset 0
    (a contiguous copy), and match the NVFP4 reference (bit-exact outside the near-tie
    window) with the layer's static `[1]` g1, linear layout.
    """
    torch.manual_seed(6553)
    g1 = _static_g1(torch.randn(4096, V41_HIDDEN, device=DEV).to(torch.bfloat16))
    big = torch.randn(max(_GATHERED_TOTALS), V41_HIDDEN, device=DEV).to(torch.bfloat16)
    checked = set()
    n_tie = n_miss = n_all = 0
    for total in _GATHERED_TOTALS:
        full, rest = divmod(total, _MOE_CHUNK)
        sizes = [_MOE_CHUNK] * full + ([rest] if rest else [])
        for c, chunk in enumerate(torch.split(big[:total], sizes, 0)):
            off = c * _MOE_CHUNK
            assert chunk.is_contiguous() and chunk.storage_offset() == off * V41_HIDDEN
            with validating(op):
                d, s = op(chunk, g1, NV_VEC, False, False)
            d0, s0 = op(chunk.clone(), g1, NV_VEC, False, False)
            assert torch.equal(d, d0) and torch.equal(s, s0), (total, off)
            if (off, chunk.shape[0]) in checked:
                continue
            checked.add((off, chunk.shape[0]))
            tie, miss, tot = _assert_call_matches_ref(chunk, d, s, g1)
            n_tie, n_miss, n_all = n_tie + tie, n_miss + miss, n_all + tot
    assert len(checked) == 12, sorted(checked)
    # Measured: 413,081,600 elements, 1,177,519 near a tie, 396,757 resolved the other way.
    assert n_miss / n_all < 0.005 and n_tie / n_all < 0.02, (n_tie, n_miss, n_all)
