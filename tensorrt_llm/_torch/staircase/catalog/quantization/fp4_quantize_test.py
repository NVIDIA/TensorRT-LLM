# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_quantize catalog entry (NVFP4: sf_vec_size=16, e4m3 scales)."""

import torch
from torch.profiler import ProfilerActivity, profile

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .fp4_quantize import fp4_quantize

assert torch.cuda.is_available(), "fp4_quantize requires a CUDA device"

DEV = torch.device("cuda")
VEC = 16
E4M3_MAX = 448.0
E2M1_MAX = 6.0

# e2m1 code -> value. code = (exponent << 1) | mantissa, bit 3 is the sign.
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=DEV
)
# Midpoints between consecutive e2m1 magnitudes: a value landing exactly here is
# a rounding tie. `_TIE_UP[i]` is True when rounding *up* at midpoint i yields the
# even code (round-to-nearest-even keeps it) and False when it yields the odd one
# (round-to-nearest-even goes down instead).
E2M1_MIDPOINTS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=DEV
)
_TIE_UP = torch.tensor([False, True, False, True, False, True, False], dtype=torch.bool, device=DEV)

# The kernel builds its per-block output scale through two `rcp.approx.ftz.f32`
# reciprocals (~2^-23 relative error each), so a value sitting within a few fp32
# ulps of an e2m1 midpoint can round to either neighbour. Everything further away
# than this window must match the reference bit for bit. The largest deviation
# observed on this machine was 7.9e-8 relative, ~12x inside this window.
TIE_WINDOW = 2.0**-20


def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> 4-bit e2m1 codes (bit 3 = sign), round-to-nearest-even, saturating to +-6."""
    a = v.abs().unsqueeze(-1)
    gt = (a > E2M1_MIDPOINTS).sum(-1)  # strict: ties provisionally round down
    tie_up = ((a == E2M1_MIDPOINTS) & _TIE_UP).any(-1)  # ties whose even side is up
    code = (gt + tie_up).to(torch.uint8)
    neg = (v < 0) | ((v == 0) & torch.signbit(v))
    return code | (neg.to(torch.uint8) << 3)


def _ref(x: torch.Tensor, gs: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent native-torch reference: (packed data [M, K/2], sf bytes [M, K/16], scaled).

    Per 16-element block along the last dim (rows = product of the leading dims):
    `vecmax = max|x|`; the block scale is `e4m3(gs * vecmax / 6)` (6 = e2m1 max,
    e4m3 conversion saturating at 448); the data is the fp32 input times
    `gs / scale`, rounded to e2m1. `scaled` is that pre-rounding product, returned
    so the caller can identify rounding ties.
    """
    k = x.shape[-1]
    m = x.numel() // k
    xr = x.reshape(m, k)
    vecmax = xr.abs().reshape(m, k // VEC, VEC).amax(-1).float()
    # torch's fp32 -> e4m3 cast emits NaN above 464 while the kernel's cast
    # saturates; clamping first makes the two agree (448 < v <= 464 rounds to
    # 448 either way).
    sf = (gs * (vecmax / E2M1_MAX)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    out_scale = torch.where(vecmax != 0, gs / sf.float(), torch.zeros_like(vecmax))
    scaled = (xr.float().reshape(m, k // VEC, VEC) * out_scale.unsqueeze(-1)).reshape(m, k)
    codes = _e2m1_codes(scaled)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), sf.view(torch.uint8), scaled


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] packed bytes -> [M, K] e2m1 codes (element 2i in the low nibble)."""
    m, kh = packed.shape
    codes = torch.empty(m, kh * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


def _dequant(packed: torch.Tensor, sf: torch.Tensor, gs: float) -> torch.Tensor:
    """data * scale / global_scale -- the value an NVFP4 consumer reconstructs.

    `sf` must be the linear-layout scale buffer of the same call.
    """
    codes = _unpack(packed)
    v = E2M1_VALUES[(codes & 7).long()]
    v = torch.where((codes & 8).bool(), -v, v)
    rows, k = codes.shape
    scale = sf.view(rows, k // VEC).view(torch.float8_e4m3fn).float()
    return v * scale.repeat_interleave(VEC, dim=-1) / gs


def _near_tie(scaled: torch.Tensor) -> torch.Tensor:
    """Elements whose pre-rounding value sits within TIE_WINDOW (relative) of an e2m1 midpoint."""
    a = scaled.abs().unsqueeze(-1)
    return ((a - E2M1_MIDPOINTS).abs() <= TIE_WINDOW * a).any(-1)


def _assert_data(
    got: torch.Tensor, ref: torch.Tensor, scaled: torch.Tensor
) -> tuple[int, int, int]:
    """Bit-exact away from e2m1 rounding ties; one adjacent code at a tie.

    Returns `(near_tie_count, mismatch_count, total)`. The two counts are
    different quantities: the first is how many elements sit inside the tie
    window at all (the check's blind spot), the second how many of those the
    kernel actually resolved the other way.
    """
    g = _unpack(got)
    r = _unpack(ref)
    near_tie = _near_tie(scaled)
    torch.testing.assert_close(g[~near_tie], r[~near_tie])
    diff = (g != r) & near_tie
    if diff.any():
        # at a tie the kernel may pick either neighbour, never anything else
        assert torch.equal(g[diff] >> 3, r[diff] >> 3), "sign flipped at a tie"
        delta = (g[diff] & 7).int() - (r[diff] & 7).int()
        assert int(delta.abs().max()) == 1, "non-adjacent e2m1 code at a tie"
    return int(near_tie.sum()), int(diff.sum()), diff.numel()


def _swizzle_index(rows: int, cols: int) -> torch.Tensor:
    """[rows, cols] scale coordinates -> flat offsets in the 128x4-swizzled buffer."""
    padded_cols = _pad_up(cols, 4)
    r = torch.arange(rows, device=DEV).view(-1, 1)
    c = torch.arange(cols, device=DEV).view(1, -1)
    return (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _global_scale(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The canonical NVFP4 activation global scale, 448*6/amax, as a [1] fp32 tensor."""
    gs = (E4M3_MAX * E2M1_MAX / x.abs().max().float()).reshape(1)
    return gs, gs.item()


def test_linear_layout_bf16() -> None:
    """Linear scale order, DeepSeek-V3-Lite widths, decode- through prefill-sized rows.

    Rows straddle 1024 and widths straddle a multiple of 512 -- the two
    conditions a TMA high-throughput variant would key on. This build has no
    such variant (see test_r1_one_kernel_at_every_width); the coverage is kept
    because a future one would.
    """
    torch.manual_seed(0)
    worst = 0.0
    for t in (1, 2, 7, 64, 1023, 1024, 2048):
        for k in (2560, 3072, 12288):
            if t * k > 16 * 1024 * 1024:  # keeps the reference's [M, K, 7] temporaries bounded
                continue
            x = torch.randn(t, k, dtype=torch.bfloat16, device=DEV)
            gs, gsf = _global_scale(x)
            data, sf = fp4_quantize(x, gs, VEC, False, False)
            assert data.shape == (t, k // 2) and data.dtype == torch.uint8
            assert data.is_contiguous()
            assert sf.shape == (t * k // VEC,) and sf.dtype == torch.uint8
            ref_data, ref_sf, scaled = _ref(x, gsf)
            torch.testing.assert_close(sf.view(t, k // VEC), ref_sf)
            _, mismatch, total = _assert_data(data, ref_data, scaled)
            worst = max(worst, mismatch / total)
    # the kernel resolves a near-tie the other way from the exact-fp32 reference
    # on at most a fraction of a percent of elements; how many is set by the
    # global scale rather than by the shape
    assert worst < 0.005, worst


def test_swizzled_layout_bf16() -> None:
    """Swizzled scales carry the same bytes at the 128x4 offsets; the data is layout-independent."""
    torch.manual_seed(1)
    for t, k in (
        (1, 2560),
        (7, 2560),
        (129, 2560),
        (1024, 2560),
        (200, 3072),
        (3, 112),
    ):
        x = torch.randn(t, k, dtype=torch.bfloat16, device=DEV)
        gs, gsf = _global_scale(x)
        cols = k // VEC
        data_sw, sf_sw = fp4_quantize(x, gs, VEC, False, True)
        data_li, sf_li = fp4_quantize(x, gs, VEC, False, False)
        assert sf_sw.shape == (_pad_up(t, 128) * _pad_up(cols, 4),)
        assert torch.equal(data_sw, data_li), "data must not depend on the scale layout"
        ref_data, ref_sf, scaled = _ref(x, gsf)
        torch.testing.assert_close(sf_li.view(t, cols), ref_sf)
        _assert_data(data_sw, ref_data, scaled)
        idx = _swizzle_index(t, cols).reshape(-1)
        torch.testing.assert_close(sf_sw[idx].view(t, cols), ref_sf)
        # every offset not addressed by a real (row, col) pair is row/column padding
        rest = torch.ones_like(sf_sw, dtype=torch.bool)
        rest[idx] = False
        assert (sf_sw[rest] == 0).all(), "swizzled padding must be zero"


def test_fp16_input() -> None:
    torch.manual_seed(2)
    for t in (1, 512, 1024):
        x = torch.randn(t, 2560, dtype=torch.float16, device=DEV)
        gs, gsf = _global_scale(x)
        data, sf = fp4_quantize(x, gs, VEC, False, False)
        ref_data, ref_sf, scaled = _ref(x, gsf)
        assert data.shape == (t, 1280)
        torch.testing.assert_close(sf.view(t, 160), ref_sf)
        _assert_data(data, ref_data, scaled)


def test_3d_input_collapses_leading_dims() -> None:
    """data keeps the leading dims; the scale buffer is laid out for their product."""
    torch.manual_seed(3)
    x = torch.randn(2, 5, 2560, dtype=torch.bfloat16, device=DEV)
    gs, gsf = _global_scale(x)
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    assert data.shape == (2, 5, 1280)
    assert sf.shape == (10 * 160,)
    ref_data, ref_sf, scaled = _ref(x, gsf)
    torch.testing.assert_close(sf.view(10, 160), ref_sf)
    _assert_data(data.reshape(10, 1280), ref_data, scaled)
    data_sw, sf_sw = fp4_quantize(x, gs, VEC, False, True)
    assert sf_sw.shape == (128 * 160,)
    assert torch.equal(data_sw, data)


def test_e2m1_ties_round_to_nearest_even() -> None:
    """global_scale=1 with a block max of 6 makes the output scale exactly 1, exposing the tie rule."""
    gs = torch.ones(1, dtype=torch.float32, device=DEV)
    mids = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    # the even-code neighbour of each midpoint
    expect = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    x = torch.zeros(2, 16, dtype=torch.bfloat16, device=DEV)
    x[0, 0] = E2M1_MAX
    x[1, 0] = -E2M1_MAX
    for i, mid in enumerate(mids):
        x[0, 1 + i] = mid
        x[1, 1 + i] = -mid
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    assert sf.tolist() == [0x38, 0x38], sf.tolist()  # e4m3 0x38 == 1.0
    deq = _dequant(data, sf, 1.0)
    for i, want in enumerate(expect):
        assert deq[0, 1 + i].item() == want, (mids[i], deq[0, 1 + i].item())
        assert deq[1, 1 + i].item() == -want, (mids[i], deq[1, 1 + i].item())
    assert deq[0, 0].item() == E2M1_MAX and deq[1, 0].item() == -E2M1_MAX


def test_dequant_error_bound() -> None:
    """data * sf / global_scale reconstructs the input within the nvfp4 rounding bound."""
    torch.manual_seed(4)
    x = torch.randn(1024, 2560, dtype=torch.bfloat16, device=DEV) * 3.0
    gs, gsf = _global_scale(x)
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    deq = _dequant(data, sf, gsf)
    scale = sf.view(1024, 160).view(torch.float8_e4m3fn).float()
    scale = scale.repeat_interleave(VEC, dim=-1)
    # In the block-scaled domain s = x*gs/sf the e2m1 grid has spacing 0.5 below
    # 2, 1.0 on [2,4), 2.0 on [4,6] and saturates at 6, so the rounding error is
    # at most max(0.25, |s|/4). Dividing back by gs/sf gives the bound below;
    # |s|/4 / (gs/sf) is exactly |x|/4.
    bound = torch.maximum(0.25 * scale / gsf, x.float().abs() / 4)
    err = (deq - x.float()).abs()
    assert (err <= bound).all(), (err - bound).max().item()
    # hard gate: the quantization is the reference, not merely inside the bound
    ref_data, ref_sf, scaled = _ref(x, gsf)
    torch.testing.assert_close(sf.view(1024, 160), ref_sf)
    _, mismatch, total = _assert_data(data, ref_data, scaled)
    assert mismatch / total < 0.005, mismatch / total
    # a block whose scale rounds down lets its max element clip to 6 -- that is
    # the format, and it is inside the bound above
    assert (scaled.abs() > E2M1_MAX).any()


def test_scale_saturation() -> None:
    """A global_scale calibrated on a smaller amax saturates the scale and clips every lane."""
    x = torch.zeros(1, 16, dtype=torch.bfloat16, device=DEV)
    x[0, :4] = torch.tensor([1.0, -2.0, 0.5, 3.0])
    gs = torch.tensor([E4M3_MAX * E2M1_MAX / 0.001], dtype=torch.float32, device=DEV)
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    assert sf.tolist() == [0x7E]  # e4m3 0x7E == 448, the finite max
    # every nonzero lane clips to +-6 regardless of its magnitude
    codes = _unpack(data)
    assert codes[0, :4].tolist() == [7, 15, 7, 7]
    assert (codes[0, 4:] == 0).all()
    deq = _dequant(data, sf, gs.item())
    clipped = torch.full((4,), E2M1_MAX * E4M3_MAX / gs.item(), device=DEV)
    clipped[1] = -clipped[1]
    torch.testing.assert_close(deq[0, :4], clipped)
    assert (deq[0, 4:] == 0.0).all()


def test_scale_underflow() -> None:
    """A global_scale so small that the block scale rounds to 0 zeroes the whole block."""
    x = torch.zeros(1, 16, dtype=torch.bfloat16, device=DEV)
    x[0, 0] = 1.0
    x[0, 1] = -0.5
    # e4m3's smallest subnormal is 2^-9; gs*vecmax/6 = 2^-11 rounds to zero
    gs = torch.tensor([E2M1_MAX * 2.0**-11], dtype=torch.float32, device=DEV)
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    assert sf.tolist() == [0x00]
    # the reciprocal of the zero scale is +inf: every lane saturates to +6,
    # including the exact zeros (0 * inf = NaN, which the e2m1 cast saturates)
    codes = _unpack(data)
    assert codes[0, 0].item() == 7 and codes[0, 1].item() == 15
    assert (codes[0, 2:] == 7).all()
    # with a zero scale the consumer reads the block back as all zeros
    assert (_dequant(data, sf, gs.item()) == 0.0).all()
    # one binade higher the scale is the e4m3 minimum subnormal and the block is exact
    gs2 = torch.tensor([E2M1_MAX * 2.0**-9], dtype=torch.float32, device=DEV)
    data2, sf2 = fp4_quantize(x, gs2, VEC, False, False)
    assert sf2.tolist() == [0x01]
    deq2 = _dequant(data2, sf2, gs2.item())
    assert deq2[0, 0].item() == 1.0 and deq2[0, 1].item() == -0.5


def test_degenerate_global_scale() -> None:
    """A zero or negative global_scale is accepted and silently unusable."""
    x = torch.zeros(1, 16, dtype=torch.bfloat16, device=DEV)
    x[0, :4] = torch.tensor([1.0, -2.0, 0.5, 3.0])
    # g = 0: the block scale is 0 and every lane saturates, so a consumer
    # dividing by g reads NaN
    data, sf = fp4_quantize(x, torch.zeros(1, device=DEV), VEC, False, False)
    assert sf.tolist() == [0x00]
    assert (_unpack(data)[0] == 7).all()
    # g < 0: the scale byte carries e4m3's sign bit, which every consumer reads
    # as an unsigned UE4M3 magnitude; the data bytes are those of |g|
    data_n, sf_n = fp4_quantize(x, -torch.ones(1, device=DEV), VEC, False, False)
    data_p, sf_p = fp4_quantize(x, torch.ones(1, device=DEV), VEC, False, False)
    assert sf_n.tolist() == [0xB0] and sf_p.tolist() == [0x30]  # -0.5 and +0.5
    assert torch.equal(data_n, data_p)


def test_reciprocal_global_scale_is_silently_destructive() -> None:
    """A modelopt checkpoint stores amax/(448*6); passing it unreciprocated is accepted.

    The mistake costs a factor of `(448*6/amax)^2`, which for normed activations
    puts almost every block under the e4m3 scale floor -- but not all of them, so
    the result is neither an error nor an obviously dead tensor.
    """
    torch.manual_seed(12)
    x = torch.randn(64, 7168, dtype=torch.bfloat16, device=DEV)
    amax = x.abs().max().float()
    stored = (amax / (E4M3_MAX * E2M1_MAX)).reshape(1)  # what is on disk
    data, sf = fp4_quantize(x, stored, VEC, False, False)
    zero_scales = (sf == 0).float().mean().item()
    assert zero_scales > 0.9, zero_scales
    assert zero_scales < 1.0, zero_scales
    deq = _dequant(data, sf, stored.item())
    # the blocks that survive saturate: reconstructed magnitudes above the true
    # amax, next to blocks that read back as exact zeros
    assert deq.abs().max().item() > amax.item()
    assert (deq == 0).any()


def test_zero_block_and_signed_zero() -> None:
    """An all-zero block emits a zero scale and zero data; -0.0 keeps its sign bit."""
    z = torch.zeros(1, 32, dtype=torch.bfloat16, device=DEV)
    gs = torch.tensor([100.0], dtype=torch.float32, device=DEV)
    data, sf = fp4_quantize(z, gs, VEC, False, False)
    assert sf.tolist() == [0x00, 0x00]
    assert (data == 0).all()
    y = torch.zeros(1, 16, dtype=torch.bfloat16, device=DEV)
    y[0, 0] = 6.0
    y[0, 1] = -0.0
    y[0, 2] = 0.0
    codes = _unpack(fp4_quantize(y, torch.ones(1, device=DEV), VEC, False, False)[0])
    assert codes[0, 1].item() == 8, "negative zero must keep its sign bit"
    assert codes[0, 2].item() == 0


def test_nonfinite_inputs() -> None:
    """Non-finite lanes are accepted silently; both outcomes are destructive."""
    gs = torch.ones(1, dtype=torch.float32, device=DEV)
    # +inf drives the block max to inf, so the scale saturates at 448 and every
    # finite lane is scaled by 1/448 -- small lanes round to zero.
    x = torch.zeros(1, 16, dtype=torch.bfloat16, device=DEV)
    x[0, 0] = float("inf")
    x[0, 1] = 1.0
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    assert sf.tolist() == [0x7E]
    codes = _unpack(data)
    assert codes[0, 0].item() == 7, "inf saturates to +6"
    assert (codes[0, 1:] == 0).all()
    # NaN is excluded from the block max, so the scale comes from the finite
    # lanes; the NaN lane itself saturates to +6 and stays confined.
    y = torch.zeros(1, 16, dtype=torch.bfloat16, device=DEV)
    y[0, 0] = float("nan")
    y[0, 1] = 1.0
    data, sf = fp4_quantize(y, gs, VEC, False, False)
    assert sf.tolist() == [0x23]  # e4m3 nearest to 1/6
    codes = _unpack(data)
    assert codes[0, 0].item() == 7
    assert codes[0, 1].item() == 7  # 1.0 * (1/0.171875) = 5.82 -> 6
    assert (codes[0, 2:] == 0).all()


def test_zero_rows() -> None:
    """An empty batch is accepted, not rejected: both outputs come back empty."""
    gs = torch.ones(1, dtype=torch.float32, device=DEV)
    x = torch.randn(0, 2560, dtype=torch.bfloat16, device=DEV)
    for swizzled in (False, True):
        data, sf = fp4_quantize(x, gs, VEC, False, swizzled)
        assert data.shape == (0, 1280) and data.dtype == torch.uint8
        assert sf.shape == (0,) and sf.dtype == torch.uint8


def test_input_not_mutated_and_deterministic() -> None:
    torch.manual_seed(5)
    x = torch.randn(64, 2560, dtype=torch.bfloat16, device=DEV)
    gs, _ = _global_scale(x)
    before = x.clone()
    a = fp4_quantize(x, gs, VEC, False, True)
    b = fp4_quantize(x, gs, VEC, False, True)
    assert torch.equal(x, before)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_multi_element_global_scale() -> None:
    """The op silently uses global_scale[0] for every row; the wrapper rejects that shape."""
    torch.manual_seed(6)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEV)
    per_row = torch.tensor(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], dtype=torch.float32, device=DEV
    )
    # raw op: accepted, and identical to broadcasting the first element
    got = torch.ops.trtllm.fp4_quantize(x, per_row, VEC, False, False)[1]
    want = torch.ops.trtllm.fp4_quantize(x, per_row[:1], VEC, False, False)[1]
    assert torch.equal(got, want), "multi-element global_scale is not applied per row"
    try:
        fp4_quantize(x, per_row, VEC, False, False)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrapper must reject a multi-element global_scale")


# DeepSeek-R1 activation widths: model hidden (dense-MLP / shared-expert / MoE
# input), dense-MLP intermediate, shared-expert intermediate. All three are
# multiples of 512, the width half of the condition a TMA high-throughput
# variant would key on if this build had one.
R1_WIDTHS = (7168, 18432, 2048)
# The receipt's existing row range plus max_num_tokens = 8192, which is also the
# MoE path's chunk bound.
R1_ROWS = (1, 2, 7, 64, 1023, 1024, 2048, 8192)
# Row-chunk budget for the reference: its [rows, K, 7] midpoint comparison is
# the memory ceiling, and the reference is row-independent so chunking is exact.
REF_CHUNK_ELEMS = 16 * 1024 * 1024


def _assert_call_matches_ref(
    x: torch.Tensor, data: torch.Tensor, sf_linear: torch.Tensor, gsf: float
) -> tuple[int, int, int]:
    """Row-chunked bit-exact check of one linear-layout call.

    Returns `(near_tie_count, mismatch_count, total)` summed over the chunks.
    """
    rows, k = x.shape
    cols = k // VEC
    step = max(1, REF_CHUNK_ELEMS // k)
    n_tie = n_mis = n_all = 0
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        ref_data, ref_sf, scaled = _ref(x[lo:hi], gsf)
        torch.testing.assert_close(sf_linear.view(rows, cols)[lo:hi], ref_sf)
        tie, mismatch, total = _assert_data(data[lo:hi], ref_data, scaled)
        n_tie += tie
        n_mis += mismatch
        n_all += total
    return n_tie, n_mis, n_all


def test_r1_widths_both_layouts_bf16() -> None:
    """DeepSeek-R1 activation widths in both scale layouts, decode- through 8192-row prefill.

    7168 is the model hidden, taken swizzled by the dense NVFP4 GEMM and linear
    by the trtllm-gen MoE runner; 18432 is the dense-MLP intermediate and 2048
    the shared-expert intermediate, both taken swizzled. Every width is a
    multiple of 16 with `K/16` already a multiple of 4, so the swizzled buffer
    carries row padding but no column padding.
    """
    worst_tie = worst_mis = 0.0
    seed = 100
    for k in R1_WIDTHS:
        cols = k // VEC
        for t in R1_ROWS:
            torch.manual_seed(seed)
            seed += 1
            x = torch.randn(t, k, dtype=torch.bfloat16, device=DEV)
            gs, gsf = _global_scale(x)
            data_li, sf_li = fp4_quantize(x, gs, VEC, False, False)
            data_sw, sf_sw = fp4_quantize(x, gs, VEC, False, True)
            assert data_li.shape == (t, k // 2) and data_li.dtype == torch.uint8
            assert data_li.is_contiguous() and sf_li.is_contiguous()
            assert sf_li.shape == (t * cols,) and sf_li.dtype == torch.uint8
            assert sf_sw.shape == (_pad_up(t, 128) * _pad_up(cols, 4),)
            # the same bytes out of two separate calls: layout-independent data
            assert torch.equal(data_sw, data_li), "data must not depend on the layout"
            tie, mismatch, total = _assert_call_matches_ref(x, data_li, sf_li, gsf)
            worst_tie = max(worst_tie, tie / total)
            worst_mis = max(worst_mis, mismatch / total)
            idx = _swizzle_index(t, cols).reshape(-1)
            torch.testing.assert_close(sf_sw[idx], sf_li)
            rest = torch.ones_like(sf_sw, dtype=torch.bool)
            rest[idx] = False
            assert (sf_sw[rest] == 0).all(), "swizzled padding must be zero"
            del x, data_li, sf_li, data_sw, sf_sw, idx, rest
    # Ceilings on the check's blind spot, not correctness gates -- correctness is
    # the bit-exactness above. Both quantities are set by the global scale rather
    # than by the shape (see the next test), so they are ceilings with room, not
    # fits: the worst of these 24 shapes was 0.60 % near-tie / 0.14 % mismatch.
    assert worst_mis < 0.005, worst_mis
    assert worst_tie < 0.02, worst_tie


def test_r1_one_kernel_at_every_width() -> None:
    """One call launches exactly one CUDA kernel, and it is the same one at every shape.

    Rows (7 vs 8192) and width (multiple of 512 or not) do not select a
    different kernel in this build: there is no TMA high-throughput variant on
    this op's path to switch to.
    """
    torch.manual_seed(8)
    for t, k, dtype in (
        (7, 7168, torch.bfloat16),
        (1024, 7168, torch.bfloat16),
        (8192, 7168, torch.bfloat16),
        (8192, 18432, torch.bfloat16),
        (8192, 2048, torch.bfloat16),
        (1024, 112, torch.bfloat16),
        (8192, 7168, torch.float16),
    ):
        x = torch.randn(t, k, dtype=dtype, device=DEV)
        gs, _ = _global_scale(x)
        for swizzled in (False, True):
            fp4_quantize(x, gs, VEC, False, swizzled)  # warm up the launch path
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fp4_quantize(x, gs, VEC, False, swizzled)
                torch.cuda.synchronize()
            launched = [
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
            ]
            assert len(launched) == 1, (t, k, dtype, swizzled, launched)
            assert "quantize_with_block_size" in launched[0], launched[0]
            assert "tma" not in launched[0].lower(), launched[0]
        del x


def test_r1_reference_check_is_discriminating() -> None:
    """The tie carve-out tolerates exactly one adjacent code at a near-tie, nothing else.

    Runs the wrong variants through the same comparison the R1 sweep uses, so
    the sweep's clean result is a measurement rather than a blind spot.
    """
    torch.manual_seed(9)
    x = torch.randn(1024, 7168, dtype=torch.bfloat16, device=DEV)
    gs, gsf = _global_scale(x)
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    ref_data, ref_sf, scaled = _ref(x, gsf)
    near = _near_tie(scaled)
    # the correct result passes, and the carve-out is a small fraction of the tensor
    n_tie, _, total = _assert_data(data, ref_data, scaled)
    assert 0 < n_tie / total < 0.02, n_tie / total
    codes = _unpack(data)
    # perturb only elements the kernel already got right, and only where the
    # magnitude code leaves room to move two steps up
    agree = (codes == _unpack(ref_data)) & ((codes & 7) <= 5)
    inside = (near & agree).nonzero()
    outside = (~near & agree & ((codes & 7) > 0)).nonzero()
    assert len(inside) > 0 and len(outside) > 0

    def bumped(row: int, col: int, delta: int, flip_sign: bool = False) -> torch.Tensor:
        c = codes.clone()
        code = int(c[row, col])
        mag = (code & 7) + delta
        assert 0 <= mag <= 7, mag
        c[row, col] = mag | ((code & 8) ^ (8 if flip_sign else 0))
        return c[:, 0::2] | (c[:, 1::2] << 4)

    def raises(fn) -> bool:
        try:
            fn()
        except AssertionError:
            return True
        return False

    ro, co = (int(v) for v in outside[len(outside) // 2])
    ri, ci = (int(v) for v in inside[len(inside) // 2])
    assert raises(lambda: _assert_data(bumped(ro, co, 1), ref_data, scaled)), (
        "one code away from a tie must be caught"
    )
    assert raises(lambda: _assert_data(bumped(ri, ci, 2), ref_data, scaled)), (
        "two codes at a tie must be caught"
    )
    assert raises(lambda: _assert_data(bumped(ri, ci, 0, flip_sign=True), ref_data, scaled)), (
        "a sign flip at a tie must be caught"
    )
    # the documented blind spot: one adjacent code at a near-tie is accepted,
    # and shows up only as one extra mismatch
    _, mis_before, _ = _assert_data(data, ref_data, scaled)
    _, mis_after, _ = _assert_data(bumped(ri, ci, 1), ref_data, scaled)
    assert mis_after == mis_before + 1, (mis_before, mis_after)
    # and the scale bytes carry no such carve-out: one code is caught
    bad_sf = sf.clone()
    bad_sf[0] = (int(bad_sf[0]) + 1) & 0xFF
    assert raises(lambda: torch.testing.assert_close(bad_sf.view(1024, 448), ref_sf)), (
        "one scale code must be caught"
    )
    # a swizzled buffer handed over as if it were linear is a different byte
    # string even when the two have the same length (rows a multiple of 128)
    _, sf_sw = fp4_quantize(x, gs, VEC, False, True)
    assert sf_sw.shape == sf.shape and not torch.equal(sf_sw, sf)


def test_r1_ties_track_the_global_scale_not_the_shape() -> None:
    """The near-tie population is a property of the global scale, not of M or K.

    One sample is reshaped to all three R1 widths. Scale blocks are 16
    *contiguous* elements, so the row width never moves a block boundary and the
    only thing a reshape can change is `amax`; holding that fixed, all three
    widths must agree on the exact count. Nudging `global_scale` off the
    canonical `448*6/amax` then removes the population altogether, because a
    near-tie needs `sf` to come out exactly `g*vecmax/6` -- which makes
    `out_scale` exactly `6/vecmax` and puts `6*x/vecmax` on an e2m1 midpoint.
    """
    # 129024 = lcm(7168, 18432, 2048), so one buffer reshapes to all three
    n = 129024 * 128
    torch.manual_seed(11)
    flat = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    gs, gsf = _global_scale(flat)
    counts = []
    for k in R1_WIDTHS:
        x = flat.view(n // k, k)
        data, sf = fp4_quantize(x, gs, VEC, False, False)
        tie, mismatch, total = _assert_call_matches_ref(x, data, sf, gsf)
        assert total == n
        counts.append((tie, mismatch))
        del data, sf
    assert counts[0] == counts[1] == counts[2], counts
    assert counts[0][0] > 0, counts
    # a global scale 0.1 % off the canonical one: no block keeps an exact sf, so
    # the kernel is bit-exact against the reference with no carve-out at all
    x = flat.view(n // 7168, 7168)
    off = gs * 1.001
    data, sf = fp4_quantize(x, off, VEC, False, False)
    tie, mismatch, _ = _assert_call_matches_ref(x, data, sf, off.item())
    assert (tie, mismatch) == (0, 0), (tie, mismatch)
    # a power-of-two multiple instead scales sf exactly and leaves out_scale
    # unchanged, so the population survives: it is g's mantissa that matters
    pow2 = gs * 2.0
    data, sf = fp4_quantize(x, pow2, VEC, False, False)
    tie, _, _ = _assert_call_matches_ref(x, data, sf, pow2.item())
    assert tie > 0, tie


def test_r1_dequant_error_bound() -> None:
    """The nvfp4 reconstruction bound holds at the widest R1 width.

    Evaluated in float64: every operand (e2m1 value, e4m3 scale, bf16 input,
    fp32 global scale) is exact there, so the comparison is the mathematical
    bound rather than a tolerance. In fp32 -- what a consumer actually computes
    -- an element sitting exactly on the bound lands up to half an fp32 ulp
    outside it; measured 3.0e-8 on this shape.
    """
    torch.manual_seed(10)
    x = torch.randn(1024, 18432, dtype=torch.bfloat16, device=DEV) * 3.0
    gs, gsf = _global_scale(x)
    data, sf = fp4_quantize(x, gs, VEC, False, False)
    codes = _unpack(data)
    v = E2M1_VALUES[(codes & 7).long()].double()
    v = torch.where((codes & 8).bool(), -v, v)
    scale = sf.view(1024, 1152).view(torch.float8_e4m3fn).double()
    scale = scale.repeat_interleave(VEC, dim=-1)
    # same bound as test_dequant_error_bound: the e2m1 subnormal half-step
    # carried back through the block scale, or the half-step of the coarsest
    # e2m1 binade (which also covers a block max clipping to 6)
    bound = torch.maximum(0.25 * scale / gsf, x.double().abs() / 4)
    err = (v * scale / gsf - x.double()).abs()
    assert (err <= bound).all(), (err - bound).max().item()


def test_rejects_unsupported() -> None:
    """Domains the contract declares unsupported must be rejected, not silently wrong."""
    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEV)
    gs = torch.ones(1, dtype=torch.float32, device=DEV)
    cases = {
        "no global_scale on the nvfp4 path": lambda: fp4_quantize(x, None, 16, False, True),
        "sf_vec_size 32 without ue8m0": lambda: fp4_quantize(x, gs, 32, False, True),
        "sf_vec_size 16 with ue8m0": lambda: fp4_quantize(x, gs, 16, True, True),
        "sf_vec_size 8": lambda: fp4_quantize(x, gs, 8, False, True),
        "sf_vec_size 0": lambda: fp4_quantize(x, gs, 0, False, True),
        "k not a multiple of 16": lambda: fp4_quantize(
            torch.randn(4, 24, dtype=torch.bfloat16, device=DEV), gs, 16, False, True
        ),
        "fp32 input": lambda: fp4_quantize(x.float(), gs, 16, False, True),
        "1-D input": lambda: fp4_quantize(x.reshape(-1), gs, 16, False, True),
        "fp16 global_scale": lambda: fp4_quantize(x, gs.half(), 16, False, True),
        "cpu global_scale": lambda: fp4_quantize(x, gs.cpu(), 16, False, True),
        "cpu input": lambda: fp4_quantize(x.cpu(), gs.cpu(), 16, False, True),
        "non-contiguous column slice": lambda: fp4_quantize(
            torch.randn(8, 512, dtype=torch.bfloat16, device=DEV)[:, :256],
            gs,
            16,
            False,
            True,
        ),
        "non-contiguous transpose": lambda: fp4_quantize(
            torch.randn(256, 8, dtype=torch.bfloat16, device=DEV).t(),
            gs,
            16,
            False,
            True,
        ),
    }
    for name, fn in cases.items():
        try:
            fn()
        except (RuntimeError, NotImplementedError):
            continue
        raise AssertionError(f"{name}: expected a raise, got a result")
