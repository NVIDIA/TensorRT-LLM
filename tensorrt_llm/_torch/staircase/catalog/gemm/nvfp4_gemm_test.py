# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_gemm catalog entry (NVFP4 x NVFP4 dense GEMM)."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.autotuner import AutoTuner, autotune

from .nvfp4_gemm import nvfp4_gemm

assert torch.cuda.is_available(), "nvfp4_gemm requires a CUDA device"
# The reference matmul must be true fp32, never tf32.
torch.backends.cuda.matmul.allow_tf32 = False

DEV = torch.device("cuda")
VEC = 16  # NVFP4 block size: one e4m3 scale per 16 contiguous elements along K

# e2m1 code -> value. code = (exponent << 1) | mantissa, bit 3 is the sign.
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=DEV
)
# e4m3 scale bytes: 0x30..0x40 are the positive finite values 0.5 .. 2.0, whose
# products with e2m1 data stay exactly representable in fp32 (see
# test_wide_dynamic_range for the opposite regime).
SF_LO, SF_HI = 0x30, 0x41


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _swizzle(sf_2d: torch.Tensor, pad_fill: int = 0) -> torch.Tensor:
    """[rows, cols] e4m3 scale bytes -> the flat 128x4-swizzled buffer, in native torch.

    Offset of the scale of (row r, block c), the layout every NVFP4 GEMM operand
    scale uses:
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
        + (r // 128) * 128 * pad_up(cols, 4)
    Buffer length is `pad_up(rows, 128) * pad_up(cols, 4)`; `pad_fill` is written
    to every offset no real (r, c) addresses.
    """
    rows, cols = sf_2d.shape
    padded_cols = _pad_up(cols, 4)
    r = torch.arange(rows, device=DEV).view(-1, 1)
    c = torch.arange(cols, device=DEV).view(1, -1)
    idx = (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    out = torch.full((_pad_up(rows, 128) * padded_cols,), pad_fill, dtype=torch.uint8, device=DEV)
    out[idx.flatten()] = sf_2d.flatten()
    return out


def _operand(
    rows: int, k: int, seed: int, sf_lo: int = SF_LO, sf_hi: int = SF_HI
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random NVFP4 operand: (packed data, linear scales, swizzled scales, fp32 values).

    The operand is built byte-first — random e2m1 codes and random positive
    finite e4m3 scale bytes — and the exact real-valued matrix it encodes is
    derived from those bytes in native torch. Nothing here depends on how a
    quantizer would have produced them.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 16, (rows, k), generator=g, device=DEV, dtype=torch.uint8)
    codes = torch.where(codes == 8, torch.zeros_like(codes), codes)  # drop -0.0
    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    sf = torch.randint(sf_lo, sf_hi, (rows, k // VEC), generator=g, device=DEV, dtype=torch.uint8)
    values = E2M1_VALUES[(codes & 7).long()]
    values = torch.where((codes & 8).bool(), -values, values)
    values *= sf.view(torch.float8_e4m3fn).float().repeat_interleave(VEC, dim=-1)
    return data, sf, _swizzle(sf), values


def _reference(
    a_values: torch.Tensor,
    b_values: torch.Tensor,
    alpha: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """alpha * (A @ B.T) (+ bias), fp32 — the kernel accumulates in fp32."""
    out = alpha * (a_values @ b_values.T)
    if bias is not None:
        out = out + bias.float()
    return out


def _alpha(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32, device=DEV)


ALPHA = 0.03125


def test_target_dense_linears() -> None:
    """The four NVFP4 dense linears of DeepSeek-V3-Lite, decode through prefill rows.

    (K, N) in nn.Linear [out, in] orientation: layer-0 MLP gate_up [24576, 2560]
    and down [2560, 12288], shared-expert gate_up [6144, 2560] and down
    [2560, 3072].
    """
    alpha = _alpha(ALPHA)
    shapes = [
        (2560, 24576, [1, 2, 8, 64, 1024, 4096]),
        (12288, 2560, [1, 8, 64, 1024]),
        (2560, 6144, [1, 8, 64, 1024]),
        (3072, 2560, [1, 8, 64, 1024]),
    ]
    for k, n, rows in shapes:
        b_data, _, b_sf, b_values = _operand(n, k, seed=1000 + n)
        for m in rows:
            a_data, _, a_sf, a_values = _operand(m, k, seed=m * 31 + k)
            out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
            assert out.shape == (m, n) and out.dtype == torch.bfloat16
            ref = _reference(a_values, b_values, ALPHA)
            torch.testing.assert_close(out, ref.to(torch.bfloat16))
            del a_data, a_sf, a_values, out, ref
        del b_data, b_sf, b_values
        torch.cuda.empty_cache()


def test_row_counts_around_swizzle_boundaries() -> None:
    """M is unconstrained: the activation scale buffer pads rows up to a multiple of 128."""
    alpha = _alpha(ALPHA)
    k, n = 2560, 512
    b_data, _, b_sf, b_values = _operand(n, k, seed=7)
    for m in [1, 3, 7, 9, 127, 128, 129, 1000]:
        a_data, _, a_sf, a_values = _operand(m, k, seed=m)
        assert a_sf.numel() == _pad_up(m, 128) * (k // VEC)
        out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
        torch.testing.assert_close(out, _reference(a_values, b_values, ALPHA).to(torch.bfloat16))


def test_output_dtypes() -> None:
    """bf16 / fp16 / fp32 outputs of the same call, each at its own dtype tolerance."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 8
    a_data, _, a_sf, a_values = _operand(m, k, seed=11)
    b_data, _, b_sf, b_values = _operand(n, k, seed=12)
    ref = _reference(a_values, b_values, ALPHA)
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, dtype)
        assert out.dtype == dtype
        torch.testing.assert_close(out, ref.to(dtype))


def test_alpha_scales_the_accumulator() -> None:
    """alpha multiplies the whole fp32 accumulator, before the output cast."""
    k, n, m = 2560, 512, 8
    a_data, _, a_sf, a_values = _operand(m, k, seed=13)
    b_data, _, b_sf, b_values = _operand(n, k, seed=14)
    for value in [1.0, 0.5, ALPHA, 1e-4]:
        out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, _alpha(value), torch.float32)
        torch.testing.assert_close(out, _reference(a_values, b_values, value))


def test_bias_fused() -> None:
    """The optional per-column bias is added after alpha, in the output dtype."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 64
    a_data, _, a_sf, a_values = _operand(m, k, seed=15)
    b_data, _, b_sf, b_values = _operand(n, k, seed=16)
    torch.manual_seed(0)
    bias_bf16 = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16, bias=bias_bf16)
    torch.testing.assert_close(
        out, _reference(a_values, b_values, ALPHA, bias_bf16).to(torch.bfloat16)
    )
    bias_fp32 = torch.randn(n, dtype=torch.float32, device=DEV)
    out32 = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.float32, bias=bias_fp32)
    torch.testing.assert_close(out32, _reference(a_values, b_values, ALPHA, bias_fp32))


def test_weight_scale_must_be_swizzled() -> None:
    """The weight scale buffer is 128x4-swizzled, exactly like the activation one.

    Pins the load-time obligation for a checkpoint that stores `weight_scale`
    row-major linear: the buffer the GEMM needs is `block_scale_interleave` of
    that linear tensor, which this test also shows equals the native-torch
    swizzle above byte for byte. Feeding the linear buffer instead is accepted
    and silently wrong.
    """
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 16
    a_data, _, a_sf, a_values = _operand(m, k, seed=17)
    b_data, b_sf_linear, b_sf, b_values = _operand(n, k, seed=18)
    ref = _reference(a_values, b_values, ALPHA)

    interleaved = torch.ops.trtllm.block_scale_interleave(b_sf_linear)
    assert torch.equal(interleaved.flatten(), b_sf)
    torch.testing.assert_close(
        nvfp4_gemm(a_data, b_data, a_sf, interleaved.flatten(), alpha, torch.bfloat16),
        ref.to(torch.bfloat16),
    )

    # Same buffer size, linear content: in bounds, accepted, wrong.
    linear_padded = torch.zeros_like(b_sf)
    linear_padded[: n * (k // VEC)] = b_sf_linear.flatten()
    wrong = nvfp4_gemm(a_data, b_data, a_sf, linear_padded, alpha, torch.bfloat16)
    assert (wrong.float() - ref).abs().max() > 0.1 * ref.abs().max()


def test_weight_rows_not_multiple_of_128() -> None:
    """N need only be a multiple of 32; the weight scale buffer still pads rows to 128."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 160, 8
    a_data, _, a_sf, a_values = _operand(m, k, seed=19)
    b_data, _, b_sf, b_values = _operand(n, k, seed=20)
    assert b_sf.numel() == 256 * (k // VEC)
    out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
    torch.testing.assert_close(out, _reference(a_values, b_values, ALPHA).to(torch.bfloat16))


def test_scale_padding_bytes_are_ignored() -> None:
    """Whatever fills the row/column padding of a scale buffer never reaches the result."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 7
    a_data, a_sf_linear, _, a_values = _operand(m, k, seed=21)
    b_data, b_sf_linear, _, b_values = _operand(n, k, seed=22)
    zero_pad = (_swizzle(a_sf_linear, 0x00), _swizzle(b_sf_linear, 0x00))
    # 0x7E is e4m3's largest finite value (448); 0x7F would be NaN.
    loud_pad = (_swizzle(a_sf_linear, 0x7E), _swizzle(b_sf_linear, 0x7E))
    out_zero = nvfp4_gemm(a_data, b_data, *zero_pad, alpha, torch.bfloat16)
    out_loud = nvfp4_gemm(a_data, b_data, *loud_pad, alpha, torch.bfloat16)
    assert torch.equal(out_zero, out_loud)
    torch.testing.assert_close(out_zero, _reference(a_values, b_values, ALPHA).to(torch.bfloat16))


def test_backends() -> None:
    """Every selectable backend computes the same GEMM; forcing one is a valid call."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 8  # m <= 8 so the cuda_core backend is legal too
    a_data, _, a_sf, a_values = _operand(m, k, seed=23)
    b_data, _, b_sf, b_values = _operand(n, k, seed=24)
    ref = _reference(a_values, b_values, ALPHA).to(torch.bfloat16)
    for backends in [
        "cutlass,cublaslt,cuda_core",
        "cutlass",
        "cublaslt",
        "cuda_core",
        "cutedsl",  # JIT-compiled on first use
    ]:
        out = nvfp4_gemm(
            a_data,
            b_data,
            a_sf,
            b_sf,
            alpha,
            torch.bfloat16,
            allowed_backends=backends,
        )
        torch.testing.assert_close(out, ref, msg=lambda s, b=backends: f"{b}: {s}")


def test_autotuned_selection() -> None:
    """A tuner-selected tactic computes the same GEMM as the untuned fallback."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 6144, 8
    a_data, _, a_sf, a_values = _operand(m, k, seed=25)
    b_data, _, b_sf, b_values = _operand(n, k, seed=26)
    ref = _reference(a_values, b_values, ALPHA).to(torch.bfloat16)
    cache = AutoTuner.get().profiling_cache
    cache.clear()
    try:
        untuned = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
        torch.testing.assert_close(untuned, ref)
        with autotune():
            nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
        assert len(cache.cache) > 0, "autotune recorded no tactic"
        tuned = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
        torch.testing.assert_close(tuned, ref)
    finally:
        cache.clear()  # leave the process-global tuner state as we found it


def test_wide_dynamic_range() -> None:
    """Block scales spanning 2^-5 .. 2^5, where the fp32 accumulation actually rounds.

    With the narrow scale range used elsewhere every product and partial sum is
    exactly representable, so the kernel matches an fp32 reference bit for bit.
    Here it cannot, and a fixed tolerance would be a fitted number: the check is
    the textbook recursive-summation bound instead, computed elementwise from
    the operands.
    """
    alpha = _alpha(ALPHA)
    k, n, m = 12288, 512, 32
    a_data, _, a_sf, a_values = _operand(m, k, seed=27, sf_lo=0x10, sf_hi=0x61)
    b_data, _, b_sf, b_values = _operand(n, k, seed=28, sf_lo=0x10, sf_hi=0x61)
    ref64 = ALPHA * (a_values.double() @ b_values.double().T)
    out32 = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.float32)
    out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
    # fp32 summation of K terms deviates from the exact sum by at most
    # K * 2^-24 * sum|a_i b_i| (recursive-summation bound), and the bf16 cast
    # adds at most half a bf16 ulp (2^-9 relative). Both are bounds computed
    # from the operands, not fitted tolerances.
    conditioning = ALPHA * (a_values.abs().double() @ b_values.abs().double().T)
    summation_bound = k * 2.0**-24 * conditioning
    assert not torch.equal(out32.double(), ref64), "this regime should not be exact"
    assert ((out32.double() - ref64).abs() <= summation_bound).all()
    assert ((out.double() - ref64).abs() <= summation_bound + 2.0**-9 * ref64.abs()).all()


def test_inputs_untouched_and_deterministic() -> None:
    """The op writes only its freshly allocated output, and repeats bit for bit."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 64
    a_data, _, a_sf, _ = _operand(m, k, seed=29)
    b_data, _, b_sf, _ = _operand(n, k, seed=30)
    snapshots = [t.clone() for t in (a_data, b_data, a_sf, b_sf, alpha)]
    first = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
    second = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
    for before, after in zip(snapshots, (a_data, b_data, a_sf, b_sf, alpha)):
        assert torch.equal(before, after), "an input was modified"
    assert torch.equal(first, second), "not bitwise deterministic"
    assert first.is_contiguous() and first.data_ptr() != second.data_ptr()


# The four NVFP4 dense linears of DeepSeek-R1-0528, in nn.Linear [out, in]
# orientation. Every K and N here exceeds the largest value the shapes above
# reach: K = 18432 (vs 12288) and N = 36864 (vs 24576), and the K = 18432
# weight-scale buffer is pad_up(N, 128) * pad_up(K/16, 4) with K/16 = 1152
# columns -- the widest block-scale buffer in this file.
R1_DENSE_LINEARS = [
    (7168, 36864),  # dense-MLP gate_up (layers 0-2), N = 2 * 18432
    (18432, 7168),  # dense-MLP down
    (7168, 4096),  # shared-expert gate_up, N = 2 * 2048
    (2048, 7168),  # shared-expert down
]
# Decode rows through the serving prefill cap: the dense path is not chunked,
# so M reaches max_num_tokens = 8192 in a single call.
R1_ROWS = [1, 2, 8, 64, 1024, 4096, 8192]


def _gate_rejects(out: torch.Tensor, ref_bf16: torch.Tensor) -> bool:
    """True when the default-tolerance comparison the tests above use fails."""
    try:
        torch.testing.assert_close(out, ref_bf16)
    except AssertionError:
        return True
    return False


def test_r1_dense_linears() -> None:
    """The four NVFP4 dense linears of DeepSeek-R1-0528, decode through max_num_tokens."""
    alpha = _alpha(ALPHA)
    for k, n in R1_DENSE_LINEARS:
        b_data, _, b_sf, b_values = _operand(n, k, seed=2000 + k + n)
        assert b_sf.numel() == _pad_up(n, 128) * _pad_up(k // VEC, 4)
        for m in R1_ROWS:
            a_data, _, a_sf, a_values = _operand(m, k, seed=2100 + m + k)
            assert a_sf.numel() == _pad_up(m, 128) * _pad_up(k // VEC, 4)
            out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
            assert out.shape == (m, n) and out.dtype == torch.bfloat16
            ref = _reference(a_values, b_values, ALPHA)
            torch.testing.assert_close(out, ref.to(torch.bfloat16))
            del a_data, a_sf, a_values, out, ref
        del b_data, b_sf, b_values
        torch.cuda.empty_cache()


def test_r1_backends_agree() -> None:
    """Every backend computes the same GEMM at the R1 shapes too.

    Backend selection is shape-sensitive (cuda_core is only offered for M <= 8,
    and the tactic space differs per (K, N)), so agreement is re-established at
    each new shape rather than inherited. In this scale band every product and
    partial sum is exactly representable in fp32 -- checked directly at K =
    18432 by test_r1_wide_scale_range -- so the backends agree *bitwise*, which
    is a far sharper check than the tolerance one.
    """
    alpha = _alpha(ALPHA)
    for k, n in R1_DENSE_LINEARS:
        b_data, _, b_sf, b_values = _operand(n, k, seed=2200 + k + n)
        for m in [8, 8192]:  # 8: the only band where cuda_core is selectable
            a_data, _, a_sf, a_values = _operand(m, k, seed=2300 + m + k)
            ref = _reference(a_values, b_values, ALPHA).to(torch.bfloat16)
            backends = ["cutlass", "cublaslt", "cutedsl"]
            if m <= 8:
                backends.append("cuda_core")
            outs = []
            for backend in backends:
                out = nvfp4_gemm(
                    a_data,
                    b_data,
                    a_sf,
                    b_sf,
                    alpha,
                    torch.bfloat16,
                    allowed_backends=backend,
                )
                torch.testing.assert_close(
                    out, ref, msg=lambda s, b=backend: f"{b} K={k} N={n} M={m}: {s}"
                )
                outs.append(out)
            for backend, out in zip(backends[1:], outs[1:]):
                assert torch.equal(out, outs[0]), (
                    f"{backend} differs from cutlass at K={k} N={n} M={m}"
                )
            del a_data, a_sf, a_values, ref, outs
            torch.cuda.empty_cache()
        del b_data, b_sf, b_values
        torch.cuda.empty_cache()


def test_r1_autotuned_selection() -> None:
    """The tuned path at the R1 shapes, which is the one a serving run executes.

    One tuning call at M = 8192 fills all 14 power-of-2 buckets for that
    (K, N), and every cached winner carries a profiled tactic id rather than
    the fallback marker -- so warm really is a different execution from cold.
    Each M is then run cold, tuned, and run again warm, and the two results
    compared **bitwise**: a tactic that changed the result by a single bit
    would fail here even though it would sail through the tolerance gate. The M
    values land in four different buckets, two of them non-powers of 2.
    """
    alpha = _alpha(ALPHA)
    rows = [1, 3, 8, 4097, 8192]
    cache = AutoTuner.get().profiling_cache
    try:
        for k, n in R1_DENSE_LINEARS:
            b_data, _, b_sf, b_values = _operand(n, k, seed=2400 + k + n)
            cache.clear()
            cold = {}
            for m in rows:
                a_data, _, a_sf, a_values = _operand(m, k, seed=2600 + m + k)
                cold[m] = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
                ref = _reference(a_values, b_values, ALPHA)
                torch.testing.assert_close(cold[m], ref.to(torch.bfloat16))
                del a_data, a_sf, a_values, ref

            a_data, _, a_sf, _ = _operand(8192, k, seed=2500 + k)
            with autotune():
                nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
            del a_data, a_sf
            assert len(cache.cache) == 14, (
                f"K={k} N={n}: expected buckets 1..8192, got {len(cache.cache)}"
            )
            for value in cache.cache.values():
                backend, sub_tactic = value[1]
                assert backend in {"cutlass", "cublaslt", "cuda_core"}, backend
                assert sub_tactic >= 0, f"{backend} winner is the fallback marker"

            for m in rows:
                a_data, _, a_sf, a_values = _operand(m, k, seed=2600 + m + k)
                tuned = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
                ref = _reference(a_values, b_values, ALPHA)
                torch.testing.assert_close(tuned, ref.to(torch.bfloat16))
                assert torch.equal(tuned, cold[m]), (
                    f"the tuned tactic changed the bits at K={k} N={n} M={m}"
                )
                del a_data, a_sf, a_values, tuned, ref
            del b_data, b_sf, b_values, cold
            torch.cuda.empty_cache()
    finally:
        cache.clear()  # leave the process-global tuner state as we found it


def test_r1_wrong_variants_are_visible() -> None:
    """Control: the comparisons above can see wrongness at the R1 widths.

    A clean sweep is only worth what the harness could have caught. Three wrong
    variants at the two extreme R1 shapes, each measured against the same
    default-tolerance gate the correctness tests use:
      - the weight scale fed linear (unswizzled) instead of 128x4-swizzled;
      - one e4m3 weight-scale byte moved by one code;
      - one e2m1 weight nibble moved by one code.
    The last two are the smallest perturbation either operand admits, and at
    M = 8192 they still fail the gate -- so a tactic that changed results by
    one code anywhere could not pass unnoticed.
    """
    alpha = _alpha(ALPHA)
    for k, n, m in [(18432, 7168, 8192), (7168, 36864, 8192)]:
        a_data, _, a_sf, a_values = _operand(m, k, seed=2700 + k)
        b_data, b_sf_linear, b_sf, b_values = _operand(n, k, seed=2800 + k + n)
        ref = _reference(a_values, b_values, ALPHA)
        ref_bf16 = ref.to(torch.bfloat16)

        # Positive control: the op's own relayout of a row-major [N, K/16]
        # scale tensor is byte-identical to the native-torch swizzle at these
        # widths (K/16 = 1152 and 448), and gives the reference result.
        interleaved = torch.ops.trtllm.block_scale_interleave(b_sf_linear).flatten()
        assert torch.equal(interleaved, b_sf)
        base = nvfp4_gemm(a_data, b_data, a_sf, interleaved, alpha, torch.bfloat16)
        assert not _gate_rejects(base, ref_bf16)

        # Same buffer size, linear content: in bounds, accepted, wrong.
        linear_padded = torch.zeros_like(b_sf)
        linear_padded[: n * (k // VEC)] = b_sf_linear.flatten()
        wrong = nvfp4_gemm(a_data, b_data, a_sf, linear_padded, alpha, torch.bfloat16)
        assert (wrong.float() - ref).abs().max() > 0.1 * ref.abs().max()
        assert _gate_rejects(wrong, ref_bf16)
        del linear_padded, wrong

        # One e4m3 scale byte, one code up.
        b_sf_bumped = b_sf.clone()
        i = b_sf_bumped.numel() // 3
        b_sf_bumped[i] += 1
        one_byte = nvfp4_gemm(a_data, b_data, a_sf, b_sf_bumped, alpha, torch.bfloat16)
        assert _gate_rejects(one_byte, ref_bf16), "a one-code scale change is invisible"
        del b_sf_bumped, one_byte

        # One e2m1 data nibble, one code up.
        b_bumped = b_data.clone()
        flat = b_bumped.view(-1)
        j = flat.numel() // 3
        flat[j] = (flat[j] & 0xF0) | (((flat[j] & 0x0F) + 1) & 0x0F)
        one_nibble = nvfp4_gemm(a_data, b_bumped, a_sf, b_sf, alpha, torch.bfloat16)
        assert _gate_rejects(one_nibble, ref_bf16), "a one-code data change is invisible"

        del a_data, a_sf, a_values, b_data, b_sf, b_sf_linear, b_values
        del ref, ref_bf16, base, interleaved, b_bumped, flat, one_nibble
        torch.cuda.empty_cache()


def test_r1_wide_scale_range() -> None:
    """The fp32 accumulator at K = 18432, the largest K certified here.

    Narrow block scales (the band every other test uses) keep every product and
    partial sum exactly representable, so the fp32 output is bit-exact against
    an fp64 reference at this K too -- which is what licenses the bitwise
    cross-backend assertions above. Widening the scales to 2^-5..2^5 makes the
    sum round; the deviation is then checked against the recursive-summation
    bound computed from the operands, not against a fitted tolerance.
    """
    alpha = _alpha(ALPHA)
    k, n = 18432, 7168
    a_data, _, a_sf, a_values = _operand(64, k, seed=2900)
    b_data, _, b_sf, b_values = _operand(n, k, seed=2901)
    out32 = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.float32)
    assert torch.equal(out32.double(), ALPHA * (a_values.double() @ b_values.double().T))
    del a_data, a_sf, a_values, b_data, b_sf, b_values, out32
    torch.cuda.empty_cache()

    m = 32
    a_data, _, a_sf, a_values = _operand(m, k, seed=2902, sf_lo=0x10, sf_hi=0x61)
    b_data, _, b_sf, b_values = _operand(n, k, seed=2903, sf_lo=0x10, sf_hi=0x61)
    ref64 = ALPHA * (a_values.double() @ b_values.double().T)
    out32 = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.float32)
    out = nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.bfloat16)
    conditioning = ALPHA * (a_values.abs().double() @ b_values.abs().double().T)
    summation_bound = k * 2.0**-24 * conditioning
    assert not torch.equal(out32.double(), ref64), "this regime should not be exact"
    assert ((out32.double() - ref64).abs() <= summation_bound).all()
    assert ((out.double() - ref64).abs() <= summation_bound + 2.0**-9 * ref64.abs()).all()
    torch.cuda.empty_cache()


def test_rejected_domains() -> None:
    """Domains the op rejects loudly (backend-independent unless noted)."""
    alpha = _alpha(ALPHA)
    k, n, m = 256, 128, 8
    a_data, _, a_sf, _ = _operand(m, k, seed=31)
    b_data, _, b_sf, _ = _operand(n, k, seed=32)
    bf16 = torch.bfloat16

    def rejects(fn) -> None:
        try:
            fn()
        except (RuntimeError, ValueError):
            return
        raise AssertionError("expected the op to raise")

    # K and N alignment (16-byte operand lines): both must be multiples of 32.
    a48, _, a48_sf, _ = _operand(m, 48, seed=33)
    b48, _, b48_sf, _ = _operand(n, 48, seed=34)
    rejects(lambda: nvfp4_gemm(a48, b48, a48_sf, b48_sf, alpha, bf16))
    b_narrow, _, b_narrow_sf, _ = _operand(16, k, seed=35)
    rejects(lambda: nvfp4_gemm(a_data, b_narrow, a_sf, b_narrow_sf, alpha, bf16))

    # dtypes: data and scales are uint8 byte buffers, alpha is a CUDA fp32 scalar
    rejects(lambda: nvfp4_gemm(a_data.view(torch.int8), b_data, a_sf, b_sf, alpha, bf16))
    rejects(lambda: nvfp4_gemm(a_data, b_data.view(torch.int8), a_sf, b_sf, alpha, bf16))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf.view(torch.float8_e4m3fn), b_sf, alpha, bf16))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha.half(), bf16))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha.cpu(), bf16))
    rejects(lambda: nvfp4_gemm(a_data.cpu(), b_data, a_sf, b_sf, alpha, bf16))

    # rank and K agreement
    rejects(lambda: nvfp4_gemm(a_data.reshape(2, 4, k // 2), b_data, a_sf, b_sf, alpha, bf16))
    rejects(lambda: nvfp4_gemm(a_data, b_data[:, : k // 4].contiguous(), a_sf, b_sf, alpha, bf16))

    # zero rows
    rejects(lambda: nvfp4_gemm(a_data[:0], b_data, a_sf, b_sf, alpha, bf16))

    # bias must be 1-D [N] in the output dtype
    bias = torch.zeros(n, dtype=torch.bfloat16, device=DEV)
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, bias=bias.float()))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, bias=bias.view(1, n)))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, bias=bias[: n - 32]))

    # allowed_backends parsing, and backends this arch cannot run
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, 0, ""))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, 0, "cutlas"))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, 0, "marlin"))
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, torch.float16, 0, "cutedsl"))
    # the cuda_core kernel tops out at M = 16 (the tuner only ever picks it for
    # M <= 8; forcing it is what can reach the kernel's own limit)
    a_big, _, a_big_sf, _ = _operand(32, k, seed=36)
    rejects(lambda: nvfp4_gemm(a_big, b_data, a_big_sf, b_sf, alpha, bf16, 0, "cuda_core"))
    # userbuffers output without a userbuffers workspace
    rejects(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, alpha, bf16, 1))


def test_wrapper_guards_silent_domains() -> None:
    """Each wrapper assert stands where the op itself is silently wrong."""
    alpha = _alpha(ALPHA)
    k, n, m = 2560, 512, 8
    a_data, _, a_sf, a_values = _operand(m, k, seed=37)
    b_data, _, b_sf, b_values = _operand(n, k, seed=38)
    ref = _reference(a_values, b_values, ALPHA)

    # A non-contiguous view whose *values* equal the contiguous operand: any
    # deviation is the kernel ignoring the stride.
    a_nc = torch.cat([a_data, a_data], 1)[:, : k // 2]
    b_nc = torch.cat([b_data, b_data], 1)[:, : k // 2]
    a_sf_nc = torch.stack([a_sf, torch.zeros_like(a_sf)], 1).flatten()[::2]
    b_sf_nc = torch.stack([b_sf, torch.zeros_like(b_sf)], 1).flatten()[::2]
    assert torch.equal(a_nc, a_data) and not a_nc.is_contiguous()
    assert torch.equal(a_sf_nc, a_sf) and not a_sf_nc.is_contiguous()

    def silently_wrong(*args) -> None:
        out = torch.ops.trtllm.nvfp4_gemm(*args, alpha, torch.bfloat16, 0, "cublaslt", None, None)
        assert (out.float() - ref).abs().max() > 0.1 * ref.abs().max()

    def guarded(fn) -> None:
        try:
            fn()
        except AssertionError:
            return
        raise AssertionError("expected the wrapper to reject this")

    silently_wrong(a_nc, b_data, a_sf, b_sf)
    guarded(lambda: nvfp4_gemm(a_nc, b_data, a_sf, b_sf, alpha, torch.bfloat16))
    silently_wrong(a_data, b_nc, a_sf, b_sf)
    guarded(lambda: nvfp4_gemm(a_data, b_nc, a_sf, b_sf, alpha, torch.bfloat16))
    silently_wrong(a_data, b_data, a_sf_nc, b_sf)
    guarded(lambda: nvfp4_gemm(a_data, b_data, a_sf_nc, b_sf, alpha, torch.bfloat16))
    silently_wrong(a_data, b_data, a_sf, b_sf_nc)
    guarded(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf_nc, alpha, torch.bfloat16))

    # A multi-element alpha is accepted and every element past the first ignored.
    fat_alpha = torch.tensor([ALPHA, 99.0], dtype=torch.float32, device=DEV)
    out = torch.ops.trtllm.nvfp4_gemm(a_data, b_data, a_sf, b_sf, fat_alpha, torch.bfloat16)
    torch.testing.assert_close(out, ref.to(torch.bfloat16))
    guarded(lambda: nvfp4_gemm(a_data, b_data, a_sf, b_sf, fat_alpha, torch.bfloat16))
