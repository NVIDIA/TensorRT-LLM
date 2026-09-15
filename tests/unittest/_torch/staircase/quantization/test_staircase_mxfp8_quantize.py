# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_quantize catalog entry."""

import subprocess
import sys

import pytest
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


# ── Preconditions, driven on this arch and version ───────────────────────────
#
# Each rejection asserts the EXACT message, not merely that something raised. A
# broad `except (RuntimeError, NotImplementedError)` passes whether the op
# rejected the domain the contract names or failed for an unrelated reason --
# and two of the cases below (`alignment=48` and `alignment=16`) turn out to
# share one message, so they are one check in the op rather than two.

LOUD_CASES = [
    ("k_not_multiple_of_32", "k must be divisible by SF_VEC_SIZE = 32"),
    ("alignment_not_multiple_of_32", "alignment must be divisible by SF_VEC_SIZE = 32"),
    ("alignment_below_block_size", "alignment must be divisible by SF_VEC_SIZE = 32"),
    ("fp32_input", "mxfp8_quantize only supports input tensor with dtypes fp16/bf16."),
    ("one_d_input", "Input should be >=2D tensor."),
    ("noncontiguous_column_slice", "self must be contiguous"),
    ("noncontiguous_transpose", "self must be contiguous"),
    ("cpu_input", "Could not run 'trtllm::mxfp8_quantize' with arguments from the 'CPU' backend"),
]


@pytest.mark.parametrize("case,message", LOUD_CASES)
def test_op_rejects_loudly(case: str, message: str) -> None:
    """Domains the op rejects itself, each with the message the contract quotes."""
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    if case == "k_not_multiple_of_32":
        args = (torch.randn(4, 112, dtype=torch.bfloat16, device="cuda"), False, 32)
    elif case == "alignment_not_multiple_of_32":
        args = (x, False, 48)
    elif case == "alignment_below_block_size":
        args = (x, False, 16)
    elif case == "fp32_input":
        args = (x.float(), False, 32)
    elif case == "one_d_input":
        args = (x.reshape(-1), False, 32)
    elif case == "noncontiguous_column_slice":
        args = (torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")[:, :128], False, 32)
    elif case == "noncontiguous_transpose":
        args = (torch.randn(128, 8, dtype=torch.bfloat16, device="cuda").t(), False, 32)
    else:  # cpu_input
        args = (torch.randn(4, 128, dtype=torch.bfloat16), False, 32)

    with pytest.raises((RuntimeError, NotImplementedError)) as excinfo:
        mxfp8_quantize(*args)
    assert message in str(excinfo.value), f"{case}: got {excinfo.value}"


def test_negative_alignment_truncates_silently() -> None:
    """A negative multiple of 32 passes the modulus check and TRUNCATES K.

    The op computes `padded_k = ((k + alignment - 1) / alignment) * alignment`
    in C, where the division truncates toward zero, so a negative alignment
    turns the round-up into a round-DOWN. Nothing raises, the surviving columns
    are bit-exact and the scale tensor is self-consistent, so the result looks
    entirely well-formed -- the tail of every row is simply gone. The op is
    driven directly here because the wrapper's guard rejects this.
    """
    torch.manual_seed(40)
    k = 2880
    x = torch.randn(8, k, dtype=torch.bfloat16, device="cuda")
    tight, _ = torch.ops.trtllm.mxfp8_quantize(x, False, 32)
    for alignment, padded_k in ((-32, 2816), (-64, 2752), (-512, 2048)):
        data, sf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment)
        assert data.shape == (8, padded_k), (alignment, data.shape)
        assert sf.numel() == 8 * padded_k // BLOCK
        assert torch.equal(
            data.view(torch.uint8), tight[:, :padded_k].reshape(data.shape).view(torch.uint8)
        ), "the surviving columns should be bit-exact -- only the tail is dropped"
    with pytest.raises(AssertionError, match="alignment must be positive"):
        mxfp8_quantize(x, False, -32)


def test_alignment_zero_returns_empty_instead_of_trapping() -> None:
    """`alignment=0` does NOT kill the process on this host; it returns an empty result.

    `0 % 32 == 0` passes the op's own check, and `mxFp8Quantize.cpp:60` then
    divides by it with no zero guard. On x86 that integer division traps with
    SIGFPE; on this aarch64 host SDIV returns 0 instead, so `padded_k` becomes
    0 and the call comes back with `[M, 0]` data and an empty scale buffer --
    the silent direction, and what earns the wrapper's guard.

    Driven in a CHILD process, because if the SIGFPE behaviour ever returns on
    this host it would take the whole test session with it; the child's exit
    status is the evidence either way.
    """
    child = (
        "import torch, tensorrt_llm._torch.custom_ops;"
        "x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16);"
        "d, s = torch.ops.trtllm.mxfp8_quantize(x, False, 0);"
        "assert tuple(d.shape) == (4, 0), d.shape;"
        "assert s.numel() == 0, s.numel();"
        "print('EMPTY_RESULT_NO_TRAP')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, (
        f"alignment=0 exited {proc.returncode} "
        f"({'SIGFPE' if proc.returncode == -8 else 'unexpected'}); "
        f"stderr tail: {proc.stderr.strip().splitlines()[-3:]}"
    )
    assert "EMPTY_RESULT_NO_TRAP" in proc.stdout, proc.stdout
    with pytest.raises(AssertionError, match="alignment must be positive"):
        mxfp8_quantize(torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"), False, 0)


# ── DeepSeek-V4.1-Flash column ────────────────────────────────────────────────
#
# This op feeds `gemm/mxfp8_mxfp8_gemm`, so its K set is that entry's activation
# widths -- the distinct K of this checkpoint's dense FP8 surfaces, read from the
# RAW checkpoint's safetensors headers (`[out, in]`), which is what the target's
# `weights.py` reads. All are multiples of 32, so the target calls
# `alignment=32` and `padded_k == K`: no padding on this path.
#
# K=8192 IS THE ONE AN EARLIER REVISION MISSED. It is `wo_b`'s input width:
# `layers.<L>.attn.wo_b.weight` is `[5120, 8192]` in the raw checkpoint. The
# reference implementation declares `wo_b` as a `RowParallelLinear`, which
# splits the REDUCTION dim, so each of its four ranks quantizes a 2048-wide
# activation and 8192 never appears. The staircase target replicates it
# (plan.md lines 30 and 79), so 8192 is the width it actually quantizes and
# 2048 is a shape it never calls.
V41_WIDTHS = [
    pytest.param(8192, id="wo_b_in_8192"),
    pytest.param(6144, id="engram_wkv_in_6144"),
    pytest.param(5120, id="wq_a_wkv_shared_w1w3_in_5120"),
    pytest.param(2304, id="shared_w2_in_2304"),
    pytest.param(1280, id="wq_b_indexer_in_1280"),
]

#: The reference implementation's row-parallel shard of `wo_b`'s input. Not a
#: target call; kept because the module-parity leg runs the reference at it.
V41_REF_ONLY_WIDTH = 2048

#: Row counts a served target reaches; a quantize runs at every one of them.
V41_ROWS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: A representative subset for the reference-only width.
V41_REF_ONLY_ROWS = [1, 129, 4096]


def _check_v41_width(k: int, m: int) -> None:
    """Bit-exact data and scale bytes, plus the swizzled length the GEMM requires."""
    gen = torch.Generator(device="cuda").manual_seed(k * 131 + m)
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)

    data_lin, sf_lin = mxfp8_quantize(x, False, 32)
    data_sw, sf_sw = mxfp8_quantize(x, True, 32)
    ref_data, ref_sf = _ref(x, k)
    cols = k // BLOCK

    assert data_lin.shape == (m, k) and data_lin.dtype == torch.float8_e4m3fn
    assert torch.equal(data_lin.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(data_sw.view(torch.uint8), data_lin.view(torch.uint8)), (
        "data must be byte-identical under both layouts"
    )
    assert torch.equal(sf_lin.view(m, cols), ref_sf)

    # The length the MXFP8 GEMM contract requires of act_scale.
    assert sf_sw.numel() == _pad_up(m, 128) * _pad_up(cols, 4)
    idx = _swizzle_index(m, cols, x.device)
    assert torch.equal(sf_sw[idx.flatten()].view(m, cols), ref_sf), (
        "the swizzled buffer must carry the same bytes at the 128x4 offsets"
    )


@pytest.mark.parametrize("k", V41_WIDTHS)
@pytest.mark.parametrize("m", V41_ROWS)
def test_v41_width_by_row_bucket(k: int, m: int) -> None:
    """TARGET: bit-exact at every V4.1 activation width and every engine row count.

    The full cross-product rather than a sample: the target quantizes before
    every dense projection, at whatever token count the engine is serving, so
    certifying the widths at one row count and the rows at one width would leave
    exactly the combinations in use uncovered.

    Both layouts are checked in one case because `data` is documented as
    byte-identical under either, and the swizzled scale buffer is checked at the
    exact length `gemm/mxfp8_mxfp8_gemm` states for its `act_scale` -- the two
    entries are certified separately, so this is the only place that agreement
    is pinned.
    """
    _check_v41_width(k, m)


@pytest.mark.parametrize("m", V41_REF_ONLY_ROWS)
def test_v41_reference_rank_shard_width(m: int) -> None:
    """REFERENCE-ONLY: `wo_b`'s input at the native implementation's row-parallel shard.

    Not a target width -- the target replicates `wo_b` and quantizes 8192. Kept
    because the module-parity leg runs the reference at 2048.
    """
    _check_v41_width(V41_REF_ONLY_WIDTH, m)


def test_v41_leading_dims_collapse_at_checkpoint_width() -> None:
    """A `[B, S, K]` activation collapses to `B*S` scale rows but keeps its data shape."""
    gen = torch.Generator(device="cuda").manual_seed(4141)
    x = torch.randn(3, 43, 5120, generator=gen, device="cuda", dtype=torch.bfloat16)
    data, sf = mxfp8_quantize(x, False, 32)
    assert data.shape == (3, 43, 5120)
    assert sf.shape == (3 * 43 * 5120 // BLOCK,)
    ref_data, ref_sf = _ref(x, 5120)
    assert torch.equal(data.reshape(129, 5120).view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf.view(129, 160), ref_sf)
