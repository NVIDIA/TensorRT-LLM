# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_quantize catalog entry."""

import torch

from tensorrt_llm._torch.staircase.catalog.quantization.mxfp8_quantize import mxfp8_quantize

assert torch.cuda.is_available(), "mxfp8_quantize requires a CUDA device"

E4M3_MAX = 448.0
BLOCK = 32


def _ref(x: torch.Tensor, padded_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent native-torch reference: (e4m3 data [M, padded_k], UE8M0 bytes [M, padded_k/32]).

    Per 32-element block along the last dim (rows = product of the leading dims):
    amax = max|x|; the block scale is the smallest power of two >= amax/448,
    encoded as the E8M0 byte `exp2_of_scale + 127`; the data is the fp32 input
    divided by that scale, rounded to e4m3. Columns beyond the input's K (up to
    `padded_k`) are treated as zero.

    Exact for every block with amax > 448 * 2^-127; below that the kernel takes
    a flush-to-zero path that `test_zero_and_denormal_blocks` pins directly.
    """
    k = x.shape[-1]
    m = x.numel() // k
    xp = torch.zeros(m, padded_k, dtype=torch.float32, device=x.device)
    xp[:, :k] = x.reshape(m, k).float()
    blk = xp.view(m, padded_k // BLOCK, BLOCK)
    amax = blk.abs().amax(dim=-1)
    # E8M0 of amax/448, rounded toward +inf: take the fp32 biased exponent and
    # bump it whenever any mantissa bit is set (i.e. the value is not already a
    # power of two). amax is exact in fp32 for bf16/fp16 inputs, so this is the
    # exact round-up.
    u = (amax / E4M3_MAX).view(torch.int32)
    byte = (((u >> 23) & 0xFF) + ((u & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
    scale = torch.exp2(byte.to(torch.float32) - 127.0).unsqueeze(-1)
    data = (blk / scale).view(m, padded_k).to(torch.float8_e4m3fn)
    return data, byte.to(torch.uint8)


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


def test_linear_layout_bf16() -> None:
    """Bit-exact match against the reference, decode- through prefill-sized rows."""
    torch.manual_seed(0)
    # gpt-oss-120b MoE hidden (2880) and its 512-padded width (3072)
    for t in (1, 2, 7, 64, 1024, 8192):
        for k, alignment in ((2880, 32), (2880, 512), (3072, 512), (3072, 128)):
            x = torch.randn(t, k, dtype=torch.bfloat16, device="cuda")
            data, sf = mxfp8_quantize(x, False, alignment)
            padded_k = _pad_up(k, alignment)
            ref_data, ref_sf = _ref(x, padded_k)
            assert data.shape == (t, padded_k), (t, k, alignment, data.shape)
            assert data.dtype == torch.float8_e4m3fn and data.is_contiguous()
            assert sf.shape == (t * padded_k // BLOCK,) and sf.dtype == torch.uint8
            assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
            assert torch.equal(sf.view(t, padded_k // BLOCK), ref_sf)


def test_swizzled_layout_bf16() -> None:
    """Swizzled scales carry the same bytes at the 128x4 offsets; padding is zero."""
    torch.manual_seed(1)
    for t, k, alignment in (
        (1, 2880, 512),
        (3, 128, 32),
        (129, 96, 32),
        (200, 3072, 512),
    ):
        x = torch.randn(t, k, dtype=torch.bfloat16, device="cuda")
        data_sw, sf_sw = mxfp8_quantize(x, True, alignment)
        padded_k = _pad_up(k, alignment)
        cols = padded_k // BLOCK
        ref_data, ref_sf = _ref(x, padded_k)
        assert sf_sw.shape == (_pad_up(t, 128) * _pad_up(cols, 4),)
        # data is layout-independent
        assert torch.equal(data_sw.view(torch.uint8), ref_data.view(torch.uint8))
        idx = _swizzle_index(t, cols, x.device).reshape(-1)
        assert torch.equal(sf_sw[idx].view(t, cols), ref_sf)
        # every offset not addressed by a real (row, col) is row/column padding: zero
        rest = torch.ones_like(sf_sw, dtype=torch.bool)
        rest[idx] = False
        assert (sf_sw[rest] == 0).all()


def test_alignment_padding() -> None:
    """alignment pads K with zeros: zero data bytes, zero scale bytes, valid part unchanged."""
    torch.manual_seed(2)
    x = torch.randn(64, 2880, dtype=torch.bfloat16, device="cuda")
    tight_data, tight_sf = mxfp8_quantize(x, False, 32)
    padded_data, padded_sf = mxfp8_quantize(x, False, 512)
    assert padded_data.shape == (64, 3072) and padded_sf.shape == (64 * 96,)
    assert torch.equal(
        padded_data[:, :2880].reshape(-1).view(torch.uint8),
        tight_data.reshape(-1).view(torch.uint8),
    )
    assert torch.equal(padded_sf.view(64, 96)[:, :90], tight_sf.view(64, 90))
    assert (padded_data[:, 2880:].view(torch.uint8) == 0).all()
    assert (padded_sf.view(64, 96)[:, 90:] == 0).all()


def test_fp16_input() -> None:
    torch.manual_seed(3)
    for t in (1, 512):
        x = torch.randn(t, 2880, dtype=torch.float16, device="cuda")
        data, sf = mxfp8_quantize(x, False, 512)
        ref_data, ref_sf = _ref(x, 3072)
        assert data.dtype == torch.float8_e4m3fn
        assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
        assert torch.equal(sf.view(t, 96), ref_sf)


def test_3d_input_collapses_leading_dims() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 5, 2880, dtype=torch.bfloat16, device="cuda")
    data, sf = mxfp8_quantize(x, False, 512)
    assert data.shape == (2, 5, 3072)
    assert sf.shape == (2 * 5 * 96,)
    ref_data, ref_sf = _ref(x, 3072)
    assert torch.equal(data.reshape(10, 3072).view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf.view(10, 96), ref_sf)
    # swizzled treats the collapsed rows as one 128-row-padded matrix
    _, sf_sw = mxfp8_quantize(x, True, 512)
    assert sf_sw.shape == (128 * 96,)


def test_dequant_error_bound() -> None:
    """data * 2^(sf-127) reconstructs the input within the mxfp8 rounding bound."""
    torch.manual_seed(5)
    x = torch.randn(256, 3072, dtype=torch.bfloat16, device="cuda") * 3.0
    data, sf = mxfp8_quantize(x, False, 32)
    scale = torch.exp2(sf.view(256, 96).to(torch.float32) - 127.0)
    deq = (data.float().view(256, 96, BLOCK) * scale.unsqueeze(-1)).view(256, 3072)
    # e4m3 has a 3-bit mantissa: a normal value carries at most 2^-4 relative
    # rounding error; a subnormal one at most half of the 2^-9 subnormal step,
    # i.e. 2^-10 absolute in units of the block scale (per-block atol below).
    atol = (2**-10) * scale.repeat_interleave(BLOCK, dim=-1)
    err = (deq - x.float()).abs()
    assert (err <= 2**-4 * x.float().abs() + atol).all()
    # hard gate: the quantization is the exact reference, not merely close
    ref_data, ref_sf = _ref(x, 3072)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf.view(256, 96), ref_sf)


def test_power_of_two_amax_boundary() -> None:
    """amax exactly 448*2^k: the scale byte lands on 127+k, the max element on +-448."""
    for k in (-2, 0, 1, 3):
        x = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = 448.0 * 2.0**k
        x[0, 1] = -112.0 * 2.0**k
        data, sf = mxfp8_quantize(x, False, 32)
        assert sf.tolist() == [127 + k], (k, sf.tolist())
        assert data[0, 0].float().item() == 448.0
        assert data[0, 1].float().item() == -112.0


def test_extreme_magnitude_block() -> None:
    """The largest finite bf16 magnitude still quantizes exactly, no inf/NaN."""
    big = torch.finfo(torch.bfloat16).max
    x = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = big
    x[0, 1] = -big / 256.0
    data, sf = mxfp8_quantize(x, False, 32)
    ref_data, ref_sf = _ref(x, 32)
    # amax/448 = 2^119.19 -> scale 2^120 -> byte 247, well below the E8M0 max
    assert sf.tolist() == [247]
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf, ref_sf.reshape(-1))
    assert not torch.isnan(data.float()).any()


def test_nonfinite_inputs() -> None:
    """Non-finite lanes are pinned here because both are destructive and silent."""
    # +inf: the block scale saturates to the E8M0 finite max (2^127), whose
    # reciprocal flushes to zero -- the inf lane becomes NaN and every finite
    # lane in the same block is zeroed.
    x = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = float("inf")
    x[0, 1] = 1.0
    data, sf = mxfp8_quantize(x, False, 32)
    assert sf.tolist() == [254]
    assert torch.isnan(data[0, 0].float())
    assert (data[0, 1:].float() == 0.0).all()

    # NaN: excluded from the block max, so the scale comes from the finite
    # lanes; the NaN stays confined to its own lane.
    y = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
    y[0, 0] = float("nan")
    y[0, 1] = 1.0
    data, sf = mxfp8_quantize(y, False, 32)
    assert sf.tolist() == [119]  # amax = 1.0 -> scale 2^-8
    assert torch.isnan(data[0, 0].float())
    assert data[0, 1].float().item() == 256.0
    assert (data[0, 2:].float() == 0.0).all()


def test_zero_and_denormal_blocks() -> None:
    """Pins the two degenerate block-scale paths, including the NaN one."""
    # all-zero block: scale byte 0x00 and all-zero data bytes
    x = torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda")
    data, sf = mxfp8_quantize(x, False, 32)
    assert sf.tolist() == [0, 0]
    assert (data.view(torch.uint8) == 0).all()

    # amax = 2^-118 > 448*2^-127: still the normal path, exact reference match
    y = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
    y[0, 0] = 2.0**-118
    y[0, 1] = 2.0**-119
    data, sf = mxfp8_quantize(y, False, 32)
    ref_data, ref_sf = _ref(y, 32)
    assert sf.tolist() == [1]
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert not torch.isnan(data.float()).any()

    # amax = 2^-119 <= 448*2^-127: the block scale underflows to the E8M0 minimum
    # (byte 0x00 = 2^-127, a denormal fp32) and the kernel's flush-to-zero
    # reciprocal turns it into +inf -- nonzero lanes saturate to +-448, exact-zero
    # lanes become NaN. Wrong answers, no error raised.
    z = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
    z[0, 0] = 2.0**-119
    z[0, 1] = -(2.0**-120)
    data, sf = mxfp8_quantize(z, False, 32)
    assert sf.tolist() == [0]
    assert data[0, 0].float().item() == 448.0
    assert data[0, 1].float().item() == -448.0
    assert torch.isnan(data[0, 2:].float()).all()


def test_zero_rows() -> None:
    """An empty batch is accepted, not rejected: both outputs come back empty."""
    x = torch.randn(0, 2880, dtype=torch.bfloat16, device="cuda")
    data, sf = mxfp8_quantize(x, False, 512)
    assert data.shape == (0, 3072) and data.dtype == torch.float8_e4m3fn
    assert sf.shape == (0,) and sf.dtype == torch.uint8


def test_input_not_mutated() -> None:
    torch.manual_seed(6)
    x = torch.randn(32, 2880, dtype=torch.bfloat16, device="cuda")
    before = x.clone()
    mxfp8_quantize(x, False, 512)
    mxfp8_quantize(x, True, 32)
    assert torch.equal(x, before)


def test_rejects_unsupported() -> None:
    """Domains the contract declares unsupported must be rejected, not silently wrong."""
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    cases = {
        "k not a multiple of 32": lambda: mxfp8_quantize(
            torch.randn(4, 112, dtype=torch.bfloat16, device="cuda"), False, 32
        ),
        "alignment not a multiple of 32": lambda: mxfp8_quantize(x, False, 48),
        "alignment below the block size": lambda: mxfp8_quantize(x, False, 16),
        "fp32 input": lambda: mxfp8_quantize(x.float(), False, 32),
        "1-D input": lambda: mxfp8_quantize(x.reshape(-1), False, 32),
        "non-contiguous column slice": lambda: mxfp8_quantize(
            torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")[:, :128], False, 32
        ),
        "non-contiguous transpose": lambda: mxfp8_quantize(
            torch.randn(128, 8, dtype=torch.bfloat16, device="cuda").t(), False, 32
        ),
        "cpu input": lambda: mxfp8_quantize(torch.randn(4, 128, dtype=torch.bfloat16), False, 32),
    }
    for name, fn in cases.items():
        try:
            fn()
        except (RuntimeError, NotImplementedError):
            continue
        raise AssertionError(f"{name}: expected a raise, got a result")
