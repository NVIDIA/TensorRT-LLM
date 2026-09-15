# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_mxfp8_gemm catalog entry (MXFP8 x MXFP8 dense GEMM).

The expected value is built here from native torch only: e4m3 values and UE8M0
scale bytes are constructed directly, the 128x4 swizzle is written in torch, and
the reference is an fp32 matmul of the dequantized operands. Nothing on the
expected side comes from `mxfp8_quantize`, `block_scale_interleave` or the op
under test -- a reference that shared a helper with the implementation could not
separate "my reference is wrong" from "the op is wrong".

The swizzle is not separately asserted: if it were written wrongly the kernel
would read the wrong scales and every correctness case here would fail.

TOLERANCE. `rtol` is `torch.testing.assert_close`'s default for the output
dtype, unchanged. The absolute floor is NOT the default and this file does not
pretend otherwise: `atol=1e-5` is a floor for unit-scale tensors while these
outputs measure 155 to 438, so it is replaced by

    atol = max(one output-dtype step, sqrt(K) * 2**-24) * |want|.max()

see `_tolerance`. Two tests hold that substitution to evidence rather than
argument: `test_default_floor_is_too_tight_for_fp16` shows the unmodified floor
rejecting correct near-zero results, and
`test_fp32_output_needs_the_accumulation_term` shows the `sqrt(K)` term is
load-bearing and not padding. `test_wrong_scale_layout_discriminates` measures
how far a genuinely wrong answer sits outside the floor that replaced it.
"""

import contextlib
import math
from typing import NamedTuple

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.staircase.catalog.gemm.mxfp8_mxfp8_gemm import mxfp8_mxfp8_gemm

assert torch.cuda.is_available(), "mxfp8_mxfp8_gemm requires a CUDA device"

DEV = torch.device("cuda")
VEC = 32  # MXFP8 block size: one UE8M0 scale per 32 contiguous elements along K

#: `torch.testing.assert_close`'s documented default RELATIVE tolerance per
#: output dtype. These are used unchanged.
DEFAULT_RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}

#: `torch.testing.assert_close`'s documented default ABSOLUTE floor. Not used as
#: the gate -- it is the thing two tests below measure as too tight for tensors
#: of this magnitude.
DEFAULT_ATOL = 1e-5

#: One rounding step of the output dtype, relative.
OUT_STEP = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11, torch.float32: 2.0**-24}

#: Every distinct FP8-E4M3 (K, N) the TARGET calls, derived from the RAW
#: checkpoint's safetensors headers (weights are stored `[out, in]`, so
#: `(K, N) = (in, out)`) -- which is what the target's `weights.py` reads. The
#: shared experts are FP8 here, not FP4, so they ride on this entry too.
#:
#: THREE OF THESE WERE THE REFERENCE'S RANK SHARD, not the target's surface.
#: The reference implementation shards `wq_b` and the indexer's `wq_b` with
#: `ColumnParallelLinear` (output dim / 4) and `wo_b` with `RowParallelLinear`
#: (reduction dim / 4). The staircase target does none of that: plan.md line 30
#: fixes attention DP at dep4 and replicates attention and every dense
#: projection, lines 78-79 state the target's own `5120 -> 1280 -> 32768` Q LoRA
#: and `wo_b [5120, 8192]`, and line 82 requires all 32 index heads replicated.
#: So the target's widths are the raw header's:
#:     attn_wq_b     (1280, 32768)   was (1280,  8192)  <- reference shard
#:     attn_wo_b     (8192,  5120)   was (2048,  5120)  <- reference shard
#:     indexer_wq_b  (1280,  4096)   was (1280,  1024)  <- reference shard
V41_SURFACES = [
    (6144, 25600, "engram_wkv"),
    (5120, 2304, "shared_w1w3"),
    (2304, 5120, "shared_w2"),
    (8192, 5120, "attn_wo_b"),
    (1280, 32768, "attn_wq_b"),
    (5120, 1280, "attn_wq_a"),
    (5120, 512, "attn_wkv"),
    (1280, 4096, "indexer_wq_b"),
]

#: The reference implementation's per-rank shards of the three surfaces above.
#: NOT target calls; kept because the module-parity leg runs the reference at
#: them, so knowing the op is correct there removes one variable.
V41_REFERENCE_SHARDS = [
    (2048, 5120, "attn_wo_b_shard"),
    (1280, 8192, "attn_wq_b_shard"),
    (1280, 1024, "indexer_wq_b_shard"),
]

#: Row counts a served target reaches: single decode, captured decode batches,
#: prefill chunks, and counts that straddle the 128-row padding boundary of the
#: swizzled scale layout.
ROW_BUCKETS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: A representative subset for the reference-only shards; target surfaces get
#: the full sweep.
REF_ONLY_ROWS = [1, 129, 4096]

#: The full cross-product, because that is what the target calls: every dense
#: projection runs at every token count the engine serves. Certifying the
#: surfaces at one row count and the rows at one surface would leave exactly the
#: combinations the target uses uncertified.
SURFACE_X_ROWS = [
    pytest.param(k, n, m, id=f"{tag}_{k}x{n}_M{m}")
    for k, n, tag in V41_SURFACES
    for m in ROW_BUCKETS
]

REFERENCE_SHARD_X_ROWS = [
    pytest.param(k, n, m, id=f"{tag}_{k}x{n}_M{m}")
    for k, n, tag in V41_REFERENCE_SHARDS
    for m in REF_ONLY_ROWS
]

#: The one shape every single-shape claim in the contract is measured at.
REF_K, REF_N = 5120, 1280

#: The number of compiled tactics this build carries. Asserted rather than
#: trusted by `test_compiled_tactic_count_and_miss_sentinel`, because the tactic
#: sweep below is parametrized over it and a silent change in the count would
#: silently change what the receipt covers.
EXPECTED_NUM_TACTICS = 10

#: The cache-miss sentinel `getCachedTactic` returns for an unregistered shape.
TACTIC_CACHE_MISS = -2


@contextlib.contextmanager
def _true_fp32_matmul():
    """Make torch's fp32 matmul actually fp32 for the duration, then put it back.

    This file's expected value is an fp32 matmul of the dequantized operands,
    and torch 2.12 defaults `matmul.fp32_precision` to `tf32` on this hardware
    -- so without this the reference is a TF32 product and is the inaccurate
    side of the comparison.

    IT MUST RESTORE, and an earlier revision did not: it set
    `allow_tf32 = False` at MODULE SCOPE. pytest imports every selected test
    module during collection, so that assignment ran before any test did and
    left the whole process in `ieee` -- including the other three entry files in
    the combined receipt job, whose contracts say their receipts are taken under
    torch defaults. A receipt taken under a neighbour's leaked global is not the
    receipt the contract describes. `test_fp32_precision_is_restored` asserts
    the state this file leaves behind.
    """
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


def _ref_matmul(act_truth: torch.Tensor, wgt_truth: torch.Tensor) -> torch.Tensor:
    """The expected value: a true-fp32 product of the dequantized operands."""
    with _true_fp32_matmul():
        return act_truth @ wgt_truth.t()


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _scale_len(rows: int, k: int) -> int:
    """Length of the 128x4-swizzled UE8M0 scale buffer for a `[rows, k]` operand."""
    return _pad_up(rows, 128) * _pad_up(k // VEC, 4)


def _swizzle(sf_2d: torch.Tensor) -> torch.Tensor:
    """`[rows, blocks]` UE8M0 scale bytes -> the flat 128x4-swizzled buffer, in native torch.

    Offset of the scale of (row r, block c):
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
        + (r // 128) * 128 * pad_up(blocks, 4)
    Buffer length is `pad_up(rows, 128) * pad_up(blocks, 4)`.
    """
    rows, cols = sf_2d.shape
    padded_cols = _pad_up(cols, 4)
    r = torch.arange(rows, device=DEV).view(-1, 1)
    c = torch.arange(cols, device=DEV).view(1, -1)
    offset = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * 128 * padded_cols
    )
    out = torch.zeros(_pad_up(rows, 128) * padded_cols, dtype=torch.uint8, device=DEV)
    out[offset.flatten()] = sf_2d.flatten()
    return out


def _operand(rows: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an MXFP8 operand the way a quantizer does: values, UE8M0 scales, fp32 truth.

    The scale of each 32-wide block is DERIVED FROM THAT BLOCK'S AMAX -- the OCP
    microscaling rule, `2 ** ceil(log2(amax / e4m3_max))` -- so the block's
    values fill the e4m3 range. That coupling is the point, and it is written
    here in native torch rather than taken from `mxfp8_quantize`, so the
    expected value stays independent of the implementation.

    An earlier version drew each block's exponent uniformly at random,
    independent of its values. That produces operand pairs no quantizer can
    emit -- blocks whose scale has nothing to do with their magnitude -- and the
    resulting products span a far wider dynamic range than a real GEMM sees. It
    put 71 of 158 cases outside the default gate on one or two near-zero
    elements each. The fix was to build realistic operands, not to loosen the
    gate; the two floor tests below then demonstrate, on these realistic
    operands, exactly where the default absolute floor still does not fit.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    blocks = k // VEC
    x = torch.randn(rows, k, generator=gen, device=DEV, dtype=torch.float32)
    xb = x.unflatten(-1, (blocks, VEC))
    amax = xb.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)
    scale = torch.ldexp(torch.ones_like(exp), exp.to(torch.int32))
    values = (xb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    truth = values.float().mul(scale.unsqueeze(-1)).flatten(-2)
    exps = (exp.to(torch.int32) + 127).to(torch.uint8)
    return values.flatten(-2).contiguous(), exps, truth


def _run(m: int, k: int, n: int, seed: int = 0, out_dtype=torch.bfloat16, alpha: float = 1.0):
    act, act_exp, act_truth = _operand(m, k, seed)
    wgt, wgt_exp, wgt_truth = _operand(n, k, seed + 1)
    # Ordinary contiguous row-major storage, exactly what nn.Linear holds. The
    # op rejects anything else with "must be contiguous"; there is no
    # column-major obligation on the caller.
    assert act.is_contiguous() and wgt.is_contiguous()
    gs = torch.full((1,), alpha, device=DEV, dtype=torch.float32)
    got = mxfp8_mxfp8_gemm(act, _swizzle(act_exp), wgt, _swizzle(wgt_exp), gs, out_dtype)
    want = _ref_matmul(act_truth, wgt_truth) * alpha
    return got, want


def _tolerance(out_dtype: torch.dtype, k: int, scale: float) -> tuple[float, float]:
    """`(rtol, atol)`: the dtype's default rtol, and an absolute floor sized to the data.

    `rtol` is `torch.testing.assert_close`'s default for the output dtype,
    unchanged. Only the absolute floor moves, and only because the default one
    does not describe these tensors: `atol=1e-5` is a floor for unit-scale
    results, while these outputs measure 155 to 438. Two named terms, each
    dominating exactly where it should:

      * `OUT_STEP`  one rounding step of the output dtype, which dominates for
                    bf16 (2**-8) and fp16 (2**-11);
      * `sqrt(K) * 2**-24`  fp32 accumulation over K terms summed in a different
                    order from this reference, which dominates for fp32 output
                    and is invisible under the other two because their output
                    cast is coarser.

    Measured on sm_103 over all eight TARGET surfaces at M=64 and M=4096 in all
    three output dtypes -- 48 cases, with `allow_tf32` pinned off in the probe
    exactly as it is here, so these are the same numbers the entry contract
    quotes. Under this bound **0 elements lie outside in all 48**. Under the
    unmodified default floor the same runs put up to 126 (bf16), 2,536 (fp16)
    and 2,099,381 (fp32) near-zero elements outside, and an output-step-only
    floor still leaves up to 350,411 fp32 elements outside -- which is the
    `sqrt(K)` term doing the work. The rolled-scale control puts 29,940 to
    1.30e8 elements outside this bound at max-error ratios of 97.3x to 1.43e6x,
    so the floor is nowhere near the distance a wrong answer sits at.

    Re-measured after the surface list was corrected from the reference
    implementation's rank shards to the target's replicated widths: the three
    per-dtype maxima above are unchanged (they come from the 6144x25600 engram
    surface, which never shard), and the widest new surface, 8192x5120 at
    M=4096 fp32, is the one that moved -- 933,348 elements over the default
    floor and 198,421 over step-only, still 0 over the final bound.
    """
    accum = math.sqrt(k) * 2.0**-24
    return DEFAULT_RTOL[out_dtype], max(OUT_STEP[out_dtype], accum) * scale


def _assert_default_close(
    got: torch.Tensor, want: torch.Tensor, out_dtype: torch.dtype, k: int
) -> None:
    rtol, atol = _tolerance(out_dtype, k, want.abs().max().item())
    torch.testing.assert_close(got.float(), want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("k,n,m", SURFACE_X_ROWS)
def test_surface_by_row_bucket(k: int, n: int, m: int) -> None:
    """TARGET: every FP8 (K, N) the target has, at every row count the engine serves."""
    got, want = _run(m, k, n, seed=(k + n + m) % 4096)
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    _assert_default_close(got, want, torch.bfloat16, k)


@pytest.mark.parametrize("k,n,m", REFERENCE_SHARD_X_ROWS)
def test_reference_rank_shard_surface(k: int, n: int, m: int) -> None:
    """REFERENCE-ONLY: the native implementation's per-rank shards of three surfaces.

    Not target calls -- the target replicates all three. Kept because the
    module-parity leg runs the reference at these widths.
    """
    got, want = _run(m, k, n, seed=(k + n + m) % 4096)
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    _assert_default_close(got, want, torch.bfloat16, k)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_output_dtypes(out_dtype: torch.dtype) -> None:
    got, want = _run(64, REF_K, REF_N, seed=30, out_dtype=out_dtype)
    assert got.dtype == out_dtype
    _assert_default_close(got, want, out_dtype, REF_K)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_global_scale_is_alpha(alpha: float) -> None:
    """`global_scale` multiplies the result."""
    got, want = _run(64, REF_K, REF_N, seed=40, alpha=alpha)
    _assert_default_close(got, want, torch.bfloat16, REF_K)


def test_global_scale_extra_elements_are_ignored() -> None:
    """A two-element global_scale is accepted by the op; the wrapper refuses it.

    The op reads element 0 and ignores the rest, so a caller that meant the
    second element gets a silently wrong answer. That silence is what earns the
    wrapper guard.
    """
    act, act_exp, _ = _operand(64, REF_K, 50)
    wgt, wgt_exp, _ = _operand(REF_N, REF_K, 51)
    a_sf, w_sf = _swizzle(act_exp), _swizzle(wgt_exp)
    one = torch.ones(1, device=DEV, dtype=torch.float32)
    two = torch.tensor([1.0, 7.0], device=DEV, dtype=torch.float32)

    baseline = torch.ops.trtllm.mxfp8_mxfp8_gemm(act, a_sf, wgt, w_sf, one, torch.bfloat16)
    sloppy = torch.ops.trtllm.mxfp8_mxfp8_gemm(act, a_sf, wgt, w_sf, two, torch.bfloat16)
    assert torch.equal(baseline, sloppy), "element 0 should be used and the rest ignored"

    with pytest.raises(AssertionError, match="exactly one element"):
        mxfp8_mxfp8_gemm(act, a_sf, wgt, w_sf, two, torch.bfloat16)


# ── scale-buffer length: both buffers, both truncations, and the padding trap ──
#
# `M=128` is not an arbitrary row count for these cases, it is the only kind
# that can see the act buffer's last byte. The swizzle packs rows in groups of
# 128, so at M=64 the buffer's tail belongs to padding rows 64..127 and dropping
# it cannot change any real row's result. `act_short_at_M64_is_harmless` keeps
# that on the record so nobody concludes from it that short act buffers are
# safe in general.
SCALE_BUFFER_CASES = [
    pytest.param("act_scale", 128, "one_byte_short", True, id="act_one_byte_short_M128"),
    pytest.param("act_scale", 64, "one_byte_short", False, id="act_short_at_M64_is_harmless"),
    pytest.param("act_scale", 128, "tenth", True, id="act_tenth_size_M128"),
    pytest.param("act_scale", 128, "oversized", False, id="act_oversized_M128"),
    pytest.param("weight_scale", 128, "one_byte_short", True, id="weight_one_byte_short"),
    pytest.param("weight_scale", 128, "tenth", True, id="weight_tenth_size"),
    pytest.param("weight_scale", 128, "oversized", False, id="weight_oversized"),
]


@pytest.mark.parametrize("which,m,mode,corrupts", SCALE_BUFFER_CASES)
def test_scale_buffer_length_is_unchecked(
    which: str, m: int, mode: str, corrupts: bool, capfd
) -> None:
    """The op reads both scale buffers by computed offset and never checks their length.

    There is no input-length validation at all: the buffer is read past its end
    and the result is whatever those bytes happened to be. What the caller gets
    is therefore not one behaviour but three -- a silently wrong result, an
    `inf`/`NaN` result, or an asynchronous device fault -- and none of them is a
    validation error raised by the op. That is what earns the wrapper's `>=`
    guard, and it is asserted here for BOTH buffers rather than argued from one.

    These cases pin the read to mapped memory (see below), so what they assert
    is the MILDEST of the three: the call returns, and it returns something
    wrong. The other two are the probe's job, because a test cannot assert on
    them without ending the session.

    The truncated buffer is a view into a larger arena whose tail is
    deterministically `0x00` (UE8M0 exponent 2**-127), not a bare `.clone()`.
    Three reasons, and all three have bitten this entry:

      * slicing the CORRECT buffer hands the op the correct bytes past the end,
        so the kernel reads valid data and the probe reports, wrongly, that a
        short buffer is harmless;
      * a fresh `.clone()` one byte short still lands inside the caching
        allocator's 512-byte rounding, so what it reads is stale allocator
        memory -- a real measurement, but the probe has seen the same call
        return 1.50e+01 and 5.37e+22 away from correct, so the verdict is
        reproducible and the number is not;
      * a fresh `.clone()` at a TENTH of the size leaves the allocation
        entirely, and that has been observed BOTH as `inf` with 4,096 NaNs and
        as `CUDA error: an illegal memory access was encountered`, which kills
        the CUDA context for the rest of the process. Which one a caller gets is
        not deterministic. The probe drives that variant in a child process; a
        test that did it here would take every case after it down. The arena
        keeps the same out-of-bounds read inside mapped memory.

    An oversized buffer is safe and must stay accepted: its tail is ignored.
    """
    k, n = REF_K, REF_N
    act, act_exp, _ = _operand(m, k, 70)
    wgt, wgt_exp, _ = _operand(n, k, 71)
    a_sf, w_sf = _swizzle(act_exp), _swizzle(wgt_exp)
    one = torch.ones(1, device=DEV, dtype=torch.float32)
    good = torch.ops.trtllm.mxfp8_mxfp8_gemm(act, a_sf, wgt, w_sf, one, torch.bfloat16)

    full = a_sf if which == "act_scale" else w_sf
    assert full.numel() == _scale_len(m if which == "act_scale" else n, k)
    if mode == "oversized":
        buf = torch.cat([full, full[:4096]]).clone()
    else:
        keep = full.numel() - 1 if mode == "one_byte_short" else full.numel() // 10
        arena = torch.zeros(full.numel() + 4096, dtype=torch.uint8, device=DEV)
        arena[:keep] = full[:keep]
        buf = arena[:keep]

    if which == "act_scale":
        args = (act, buf, wgt, w_sf, one, torch.bfloat16)
    else:
        args = (act, a_sf, wgt, buf, one, torch.bfloat16)

    # The op does not inspect the length, so with the read kept inside the arena
    # this returns rather than reporting anything. `synchronize()` is where a
    # device-side fault would surface, so it is called before the result is read.
    got = torch.ops.trtllm.mxfp8_mxfp8_gemm(*args)
    torch.cuda.synchronize()
    diff = (got.float() - good.float()).abs()
    n_nan = int(got.isnan().sum().item())
    with capfd.disabled():
        print(
            f"\n    {which} {mode} at M={m}: buffer {buf.numel()} of {full.numel()} bytes, "
            f"no length check, returned without reporting anything; "
            f"max_abs_diff_vs_correct="
            f"{torch.nan_to_num(diff, nan=float('inf')).max().item():.4e} nan={n_nan}"
        )

    if corrupts:
        assert not torch.equal(good, got), (
            f"a {mode} {which} buffer changed nothing; the op did not read past the end here, "
            f"so this case does not demonstrate the danger the wrapper guards"
        )
    else:
        assert torch.equal(good, got), (
            f"a {mode} {which} buffer changed the result; expected it to be inert"
        )

    if mode == "oversized":
        assert torch.equal(good, mxfp8_mxfp8_gemm(*args)), (
            "an oversized scale buffer must be accepted by the wrapper unchanged"
        )
    else:
        # Every short buffer is rejected, including the M=64 one the op happens
        # to survive: the guard is on the length, not on whether this particular
        # geometry got away with it.
        with pytest.raises(AssertionError, match=f"{which} holds"):
            mxfp8_mxfp8_gemm(*args)


def test_zero_rows_raises_through_the_wrapper() -> None:
    """A zero-row call fails loudly, so the wrapper does not guard it.

    This is why there is no `M >= 1` assert: the op already rejects it, and a
    wrapper guard on a loud domain only changes which error the caller sees. The
    caller obligation is real -- a dep4 rank with no logical rows must branch
    around this call -- but it is a documented precondition, not a guard.
    """
    act, act_exp, _ = _operand(1, REF_K, 60)
    wgt, wgt_exp, _ = _operand(512, REF_K, 61)
    with pytest.raises(RuntimeError, match="Failed to run cutlass MXFP8xMXFP8 gemm"):
        mxfp8_mxfp8_gemm(
            act[:0],
            _swizzle(act_exp),
            wgt,
            _swizzle(wgt_exp),
            torch.ones(1, device=DEV, dtype=torch.float32),
            torch.bfloat16,
        )


@pytest.mark.parametrize(
    "case,message",
    [
        ("act_dtype", "act dtype is BFloat16, while Float8_e4m3fn is expected"),
        ("weight_dtype", "weight dtype is BFloat16, while Float8_e4m3fn is expected"),
        ("scale_dtype", "actScale dtype is Char, while Byte is expected"),
        ("alpha_dtype", "globalScale dtype is BFloat16, while Float is expected"),
        ("act_noncontig", "act must be contiguous"),
        ("weight_noncontig", "weight must be contiguous"),
        ("act_cpu", "act must be a CUDA tensor"),
        ("act_rank3", r"act must be a 2D tensor \[M, K\]"),
        ("weight_rank1", r"weight must be a 2D tensor \[N, K\]"),
        ("k_mismatch", "act and weight K dims must match"),
        ("k_not_32", "must be divisible by MXFP8 block size 32"),
        ("n_not_32", r"N \(33\) must be divisible by 32"),
        ("bad_out_dtype", "out_dtype must be one of fp16/bf16/fp32"),
    ],
)
def test_op_rejects_loudly(case: str, message: str) -> None:
    """Every rejection the contract quotes to a caller, observed through the op.

    The contract promises these instead of guarding them, so the promise is what
    has to be tested. A `TORCH_CHECK` read out of the source and never driven is
    exactly the claim that turns out to be wrong.
    """
    k, n, m = REF_K, REF_N, 64
    act, act_exp, _ = _operand(m, k, 90)
    wgt, wgt_exp, _ = _operand(n, k, 91)
    a_sf, w_sf = _swizzle(act_exp), _swizzle(wgt_exp)
    one = torch.ones(1, device=DEV, dtype=torch.float32)
    args = [act, a_sf, wgt, w_sf, one, torch.bfloat16]

    if case == "act_dtype":
        args[0] = act.float().bfloat16()
    elif case == "weight_dtype":
        args[2] = wgt.float().bfloat16()
    elif case == "scale_dtype":
        args[1] = a_sf.view(torch.int8)
    elif case == "alpha_dtype":
        args[4] = one.bfloat16()
    elif case == "act_noncontig":
        args[0] = act.t().contiguous().t()
    elif case == "weight_noncontig":
        args[2] = wgt.t().contiguous().t()
    elif case == "act_cpu":
        args[0] = act.cpu()
    elif case == "act_rank3":
        args[0] = act.view(1, m, k)
    elif case == "weight_rank1":
        args[2] = wgt.flatten()
    elif case == "k_mismatch":
        short, short_exp, _ = _operand(m, 2048, 92)
        args[0], args[1] = short, _swizzle(short_exp)
    elif case == "k_not_32":
        args[0] = act[:, :33].contiguous()
        args[2] = wgt[:, :33].contiguous()
    elif case == "n_not_32":
        args[2] = wgt[:33].contiguous()
    elif case == "bad_out_dtype":
        args[5] = torch.int32
    else:  # pragma: no cover - the parametrize list is the whole domain
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(RuntimeError, match=message):
        torch.ops.trtllm.mxfp8_mxfp8_gemm(*args)


def test_wrong_scale_layout_discriminates(capfd) -> None:
    """The gate is tight enough to see a scale error, and by how much.

    Rolling the scale bytes along the block axis pairs every 32-wide block with
    its neighbour's exponent. Shape and values are unchanged, so only the
    numerical result can catch it. The three numbers -- correct error, gate,
    wrong error -- are printed and asserted rather than left implicit, because a
    gate with no measured wrong-variant distance is a guess.

    `capfd`, not `capsys`: this repo's conftest already holds `capfd`, and
    requesting `capsys` alongside it makes pytest ERROR AT SETUP with "cannot
    use capsys and capfd at the same time" -- which skipped this test and the
    tolerance ones entirely while the summary still read "158 passed". A test
    that cannot run records no receipt.
    """
    m, k, n = 64, REF_K, REF_N
    act, act_exp, act_truth = _operand(m, k, 80)
    wgt, wgt_exp, wgt_truth = _operand(n, k, 81)
    one = torch.ones(1, device=DEV, dtype=torch.float32)
    want = _ref_matmul(act_truth, wgt_truth)
    rtol, atol = _tolerance(torch.bfloat16, k, want.abs().max().item())

    got = mxfp8_mxfp8_gemm(act, _swizzle(act_exp), wgt, _swizzle(wgt_exp), one, torch.bfloat16)
    _assert_default_close(got, want, torch.bfloat16, k)

    rolled = mxfp8_mxfp8_gemm(
        act, _swizzle(act_exp), wgt, _swizzle(wgt_exp.roll(1, dims=-1)), one, torch.bfloat16
    )
    bound = atol + rtol * want.abs()
    correct = (got.float() - want).abs()
    wrong = (rolled.float() - want).abs()
    over_correct = int((correct > bound).sum().item())
    over_wrong = int((wrong > bound).sum().item())
    ratio = wrong.max().item() / max(correct.max().item(), 1e-30)
    with capfd.disabled():
        print(
            f"\n    discrimination at K={k} N={n} M={m}: "
            f"correct max_abs={correct.max().item():.4e} ({over_correct} elements over the "
            f"gate) | wrong max_abs={wrong.max().item():.4e} ({over_wrong} of "
            f"{wrong.numel()} over) | wrong/correct={ratio:.1f}x"
        )
    assert over_correct == 0, f"the correct result put {over_correct} elements over the gate"
    assert atol > 0.0
    assert over_wrong > wrong.numel() // 2, (
        f"the rolled-scale control put only {over_wrong} of {wrong.numel()} elements over the "
        f"gate; the gate cannot see a scale-association error"
    )
    assert ratio > 100.0, f"wrong/correct margin is only {ratio:.1f}x"


# ── why the absolute floor moved: the two cases that measure it ───────────────
#
# Both run the SAME realistic operands the correctness cases use, at the shape
# the contract's single-shape rows are measured at, with the probe's own seed so
# the numbers in the contract and the numbers here are the same numbers.
TOL_M = 64
TOL_SEED = (REF_K + REF_N + TOL_M) % 4096


class _FloorCounts(NamedTuple):
    scale: float
    max_abs: float
    over_default: int
    over_step_only: int
    over_final: int
    worst_default_ratio: float


def _floor_counts(out_dtype: torch.dtype) -> _FloorCounts:
    """Elements outside each candidate floor, on one real surface at one real row count."""
    got, want = _run(TOL_M, REF_K, REF_N, seed=TOL_SEED, out_dtype=out_dtype)
    diff = (got.float() - want).abs()
    scale = want.abs().max().item()
    rtol = DEFAULT_RTOL[out_dtype]
    rel = rtol * want.abs()
    step_only = OUT_STEP[out_dtype] * scale
    final = max(OUT_STEP[out_dtype], math.sqrt(REF_K) * 2.0**-24) * scale
    over_default = diff > (DEFAULT_ATOL + rel)
    n_default = int(over_default.sum().item())
    # How far from zero, relative to the tensor's own scale, the worst element
    # the default floor rejects actually sits.
    worst_ratio = (want.abs() * over_default).max().item() / scale if n_default else 0.0
    return _FloorCounts(
        scale=scale,
        max_abs=diff.max().item(),
        over_default=n_default,
        over_step_only=int((diff > (step_only + rel)).sum().item()),
        over_final=int((diff > (final + rel)).sum().item()),
        worst_default_ratio=worst_ratio,
    )


def test_default_floor_is_too_tight_for_fp16(capfd) -> None:
    """`atol=1e-5` rejects correct near-zero results, and this is that happening.

    This is the evidence for replacing the default absolute floor, and it is a
    deterministic case on realistic operands rather than an argument. The
    elements it rejects are results of a cancelling sum -- near zero against a
    tensor whose scale is in the hundreds -- where the fp32 accumulation error
    is an absolute quantity the relative term cannot cover and 1e-5 is far below
    it. Under the scale-aware floor nothing is outside.
    """
    c = _floor_counts(torch.float16)
    with capfd.disabled():
        print(
            f"\n    fp16 at K={REF_K} N={REF_N} M={TOL_M}: scale={c.scale:.4e} "
            f"max_abs={c.max_abs:.4e} | over UNMODIFIED default floor: {c.over_default} "
            f"| over scale-aware floor: {c.over_final} | worst rejected element sits at "
            f"{c.worst_default_ratio:.2e} of the tensor scale"
        )
    assert c.over_default > 0, (
        "the unmodified default floor rejected nothing here, so this case no longer "
        "demonstrates why the floor was replaced"
    )
    assert c.worst_default_ratio < 0.05, (
        f"an element the default floor rejects sits at {c.worst_default_ratio:.2e} of the "
        f"tensor scale, so this is not the near-zero cancellation regime"
    )
    assert c.over_final == 0, f"{c.over_final} elements outside the scale-aware floor"


def test_fp32_output_needs_the_accumulation_term(capfd) -> None:
    """The `sqrt(K) * 2**-24` term is load-bearing, not padding.

    With fp32 output there is no output cast to round the accumulation
    difference away, so the floor has to carry it. Measured here: the unmodified
    default floor and an output-step-only floor both reject correct elements,
    and only the full `max(step, sqrt(K) * 2**-24)` bound covers them. That
    ordering is the whole argument for the second term, so it is asserted.
    """
    c = _floor_counts(torch.float32)
    accum_bound = math.sqrt(REF_K) * 2.0**-24 * c.scale
    with capfd.disabled():
        print(
            f"\n    fp32 at K={REF_K} N={REF_N} M={TOL_M}: scale={c.scale:.4e} "
            f"max_abs={c.max_abs:.4e} | over default floor: {c.over_default} "
            f"| over output-step-only floor: {c.over_step_only} "
            f"| over max(step, sqrt(K)*2**-24) floor: {c.over_final} "
            f"(accumulation bound {accum_bound:.4e})"
        )
    assert c.over_default > 0, "the default floor rejected nothing; nothing to justify"
    assert c.over_step_only > 0, (
        "an output-step-only floor covered every element, so the sqrt(K) accumulation "
        "term is not what this case says it is"
    )
    assert c.over_final == 0, f"{c.over_final} elements outside the scale-aware floor"
    assert c.max_abs <= accum_bound, (
        f"fp32 max_abs {c.max_abs:.4e} exceeds the accumulation bound {accum_bound:.4e}; "
        f"that is no longer an accumulation-order difference"
    )


# ── the serving tactic cache ──────────────────────────────────────────────────
#
# `mxfp8_mxfp8_gemm` calls the implementation with `useTacticCache=true`, so it
# consults a process-global cache that `MXFP8GemmRunner.register_tactic`
# populates -- empty in a test process, warm in a served one whose autotuner has
# run. That difference is the thing a contract for this op is most likely to get
# wrong, so it is driven here through the real runner class rather than argued.
TACTIC_M = 128


class _TacticFixture(NamedTuple):
    act: torch.Tensor
    act_sf: torch.Tensor
    wgt: torch.Tensor
    wgt_sf: torch.Tensor
    alpha: torch.Tensor
    cold: torch.Tensor
    want: torch.Tensor


@pytest.fixture(scope="module")
def tactics() -> _TacticFixture:
    """Operands plus the cold-cache result every tactic case is compared against."""
    act, act_exp, act_truth = _operand(TACTIC_M, REF_K, 100)
    wgt, wgt_exp, wgt_truth = _operand(REF_N, REF_K, 101)
    alpha = torch.ones(1, device=DEV, dtype=torch.float32)
    runner = torch.classes.trtllm.MXFP8GemmRunner(torch.bfloat16)
    runner.clear_tactic_cache()
    act_sf, wgt_sf = _swizzle(act_exp), _swizzle(wgt_exp)
    cold = mxfp8_mxfp8_gemm(act, act_sf, wgt, wgt_sf, alpha, torch.bfloat16)
    return _TacticFixture(
        act=act,
        act_sf=act_sf,
        wgt=wgt,
        wgt_sf=wgt_sf,
        alpha=alpha,
        cold=cold,
        want=_ref_matmul(act_truth, wgt_truth),
    )


@pytest.fixture
def empty_tactic_cache():
    """A runner whose cache is empty on entry and left empty on exit.

    The cache is process-global, so a registration that outlived its test would
    change which kernel every later case in this file runs. Cleared on both
    sides rather than trusted.
    """
    runner = torch.classes.trtllm.MXFP8GemmRunner(torch.bfloat16)
    runner.clear_tactic_cache()
    try:
        yield runner
    finally:
        runner.clear_tactic_cache()


def test_compiled_tactic_count_and_miss_sentinel(empty_tactic_cache, tactics) -> None:
    """The tactic sweep below is parametrized over this count, so it is asserted.

    Also pins the cache-miss sentinel the contract quotes, and that calling the
    plain op does NOT populate the cache -- only `register_tactic` does, which is
    why a test process stays cold unless it asks not to.
    """
    runner = empty_tactic_cache
    assert runner.get_num_configs() == EXPECTED_NUM_TACTICS, (
        f"this build carries {runner.get_num_configs()} compiled tactics, not "
        f"{EXPECTED_NUM_TACTICS}; the certified tactic sweep no longer covers them all"
    )
    assert runner.get_cached_tactic(TACTIC_M, REF_N, REF_K) == TACTIC_CACHE_MISS
    mxfp8_mxfp8_gemm(
        tactics.act, tactics.act_sf, tactics.wgt, tactics.wgt_sf, tactics.alpha, torch.bfloat16
    )
    assert runner.get_cached_tactic(TACTIC_M, REF_N, REF_K) == TACTIC_CACHE_MISS, (
        "calling the op populated the tactic cache; a test process is then not cold"
    )


@pytest.mark.parametrize("idx", [-1, *range(EXPECTED_NUM_TACTICS)])
def test_every_tactic_matches_the_cold_path(idx: int, empty_tactic_cache, tactics) -> None:
    """Every compiled tactic, plus the -1 generic fallback, against the cold-cache answer.

    A served process runs whichever tactic its autotuner picked; a test process
    runs the default. This asserts the two cannot disagree at this shape -- both
    bit-identically equal to each other and inside the gate against the
    independent fp32 reference.
    """
    got = empty_tactic_cache.run_gemm(
        tactics.act, tactics.act_sf, tactics.wgt, tactics.wgt_sf, tactics.alpha, idx
    )
    assert torch.equal(got, tactics.cold), (
        f"tactic {idx} differs from the cold-cache result by "
        f"{(got.float() - tactics.cold.float()).abs().max().item():.4e}"
    )
    _assert_default_close(got, tactics.want, torch.bfloat16, REF_K)


@pytest.mark.parametrize("idx", [0, EXPECTED_NUM_TACTICS - 1])
def test_warm_cache_leaves_the_result_bit_identical(idx: int, empty_tactic_cache, tactics) -> None:
    """The served path: register a tactic, then call the op the target calls.

    This is the case a test process never reaches by accident and the case a
    served process is always in. Registering changes which kernel runs; it must
    not change the answer.
    """
    runner = empty_tactic_cache
    runner.register_tactic(TACTIC_M, REF_N, REF_K, idx)
    assert runner.get_cached_tactic(TACTIC_M, REF_N, REF_K) == idx
    warm = mxfp8_mxfp8_gemm(
        tactics.act, tactics.act_sf, tactics.wgt, tactics.wgt_sf, tactics.alpha, torch.bfloat16
    )
    assert torch.equal(warm, tactics.cold), (
        f"with tactic {idx} registered the op returned a result "
        f"{(warm.float() - tactics.cold.float()).abs().max().item():.4e} from the cold one"
    )


# ── the process-global state this file touches ───────────────────────────────


def _fp32_state() -> tuple:
    return (
        torch.backends.cuda.matmul.allow_tf32,
        getattr(torch.backends.cuda.matmul, "fp32_precision", None),
    )


def test_fp32_precision_is_restored() -> None:
    """This file must leave `matmul.fp32_precision` exactly as it found it.

    The regression this pins is a real one, found in review. An earlier revision
    set `allow_tf32 = False` at MODULE SCOPE; pytest imports every selected
    module during collection, so the assignment ran before any test did and left
    the whole process in `ieee`. In the combined receipt job that silently moved
    the other three entry files -- `bmm_out`, `cublas_mm` and `mxfp8_quantize`
    all state that their receipts are taken under torch defaults, and they were
    not. A receipt taken under a neighbour's leaked global is not the receipt its
    contract describes.

    Three assertions, because "it restores" and "it is not leaking right now"
    are different claims:
      1. the context actually forces ieee while it is open, so it is doing its
         job rather than being a no-op;
      2. it restores the exact prior state on exit -- round trip;
      3. the state visible to this test is NOT ieee, which is what a surviving
         module-scope leak from this file would look like.
    """
    before = _fp32_state()
    with _true_fp32_matmul():
        inside = _fp32_state()
    after = _fp32_state()

    assert inside[0] is False, "the context did not disable tf32; it is a no-op"
    if inside[1] is not None:
        assert inside[1] == "ieee", f"the context left fp32_precision at {inside[1]}"
    assert after == before, f"the context did not restore: {before} -> {after}"

    # A representative reference call must not leak either.
    _run(64, REF_K, REF_N, seed=99)
    assert _fp32_state() == before, f"a reference matmul leaked state: {before} -> {_fp32_state()}"

    assert before[0] is True and (before[1] is None or before[1] == "tf32"), (
        f"this process is already in {before}, not the torch default. Either this file still "
        f"mutates fp32 precision at module scope, or a neighbouring test file in the same "
        f"session does -- and every receipt taken in that session is then taken under a "
        f"state its contract does not describe"
    )
