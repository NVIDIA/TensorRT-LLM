# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry."""

import contextlib

import pytest
import torch

from tensorrt_llm._torch.staircase.catalog.gemm.cublas_mm import cublas_mm

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


@contextlib.contextmanager
def _true_fp32_matmul():
    """Make torch's fp32 matmul actually fp32 for the duration.

    torch 2.12 defaults ``matmul.fp32_precision`` to ``tf32`` and
    ``allow_tf32`` to True on this hardware, so a plain ``a.float() @
    b.float()`` is a *TF32* product -- ~1e-3 relative, which is 30x the error
    of the op being tested. Left alone the reference is the inaccurate side of
    the comparison and the entry fails against a correct kernel. Measured
    here: with TF32 off the op is bit-identical to torch and both sit 1.9e-5
    from a float64 product; with TF32 on the reference alone moves by 0.035.
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


def _ref_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    """fp32-accumulated reference: mat_a @ mat_b (+ bias), cast to out dtype."""
    with _true_fp32_matmul():
        ref = mat_a.float() @ mat_b.float()
    if bias is not None:
        ref = ref + bias.float()
    return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)


def _check(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> None:
    out = cublas_mm(mat_a, mat_b, bias, out_dtype)
    ref = _ref_mm(mat_a, mat_b, bias, out_dtype)
    expected_dtype = out_dtype if out_dtype is not None else mat_a.dtype
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype == expected_dtype
    # rtol: torch.testing defaults per output dtype. atol: kernel and reference
    # both accumulate in fp32 but in different summation orders; for K <= 4096
    # unit-variance inputs the order-dependent absolute noise is up to
    # ~K * 2^-24 ~= 2.4e-4, which dominates on near-zero outputs produced by
    # cancellation, so atol=1e-3 instead of the ~1e-5 defaults.
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
    torch.testing.assert_close(out, ref, rtol=rtol[expected_dtype], atol=1e-3)


def _make(m: int, k: int, n: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Build mat_a [M,K] row-major, mat_b [K,N] column-major, bias [N]."""
    mat_a = torch.randn(m, k, device="cuda").to(dtype)
    weight = torch.randn(n, k, device="cuda").to(dtype)  # linear weight [N, K]
    bias = torch.randn(n, device="cuda").to(dtype)
    return mat_a, weight.t(), bias


def test_bf16_no_bias() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens) and prefill-like (many tokens) shapes
    for m, k, n in [(1, 4096, 4096), (8, 4096, 11008), (2048, 4096, 4096)]:
        mat_a, mat_b, _ = _make(m, k, n, torch.bfloat16)
        _check(mat_a, mat_b)


def test_bf16_bias() -> None:
    torch.manual_seed(1)
    for m, k, n in [(1, 4096, 4096), (512, 2048, 6144)]:
        mat_a, mat_b, bias = _make(m, k, n, torch.bfloat16)
        _check(mat_a, mat_b, bias)


def test_bf16_out_fp32() -> None:
    # bias must match the output dtype (fp32 here), not the input dtype
    torch.manual_seed(2)
    mat_a, mat_b, _ = _make(16, 1024, 2048, torch.bfloat16)
    bias_fp32 = torch.randn(2048, device="cuda", dtype=torch.float32)
    _check(mat_a, mat_b, out_dtype=torch.float32)
    _check(mat_a, mat_b, bias_fp32, out_dtype=torch.float32)


def test_bf16_unaligned_shapes() -> None:
    # dims not multiples of typical tile/vector widths
    torch.manual_seed(3)
    for m, k, n in [(5, 100, 60), (7, 333, 129)]:
        mat_a, mat_b, bias = _make(m, k, n, torch.bfloat16)
        _check(mat_a, mat_b)
        _check(mat_a, mat_b, bias)


def test_fp16() -> None:
    torch.manual_seed(4)
    for m, k, n in [(1, 4096, 4096), (1024, 2048, 2048)]:
        mat_a, mat_b, bias = _make(m, k, n, torch.float16)
        _check(mat_a, mat_b)
        _check(mat_a, mat_b, bias)


def test_fp32() -> None:
    # no bias: the op silently ignores bias when inputs are fp32
    # (contract precondition; guarded by an assert in the wrapper)
    torch.manual_seed(5)
    for m, k, n in [(2, 1024, 1024), (256, 2048, 1024)]:
        mat_a, mat_b, _ = _make(m, k, n, torch.float32)
        _check(mat_a, mat_b)


def test_fp8_e4m3_to_bf16() -> None:
    # fp8 inputs require an explicit out_dtype; no scales are applied (alpha=1)
    torch.manual_seed(6)
    for m, k, n in [(1, 1024, 1024), (128, 1024, 512)]:
        mat_a = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        weight = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        _check(mat_a, weight.t(), out_dtype=torch.bfloat16)
        _check(mat_a, weight.t(), bias, out_dtype=torch.bfloat16)


# ── DeepSeek-V4.1-Flash column: the language head ─────────────────────────────
#
# Derived from the RAW checkpoint's safetensors header, which is what the
# target's `weights.py` reads: `head.weight` is `[129280, 5120]` BF16, untied
# from `embed.weight` (`tie_word_embeddings=false`).
#
# THE TARGET WIDTH IS 129280, NOT 32320. The reference implementation's
# `ParallelHead` holds `[vocab_size // world_size, dim]` = `[32320, 5120]` per
# rank and all-gathers the four shards. The staircase target does not: plan.md
# line 88 requires it to "Replicate and load independently" and "Produce
# complete local logits without a vocab collective", and line 30 puts the head
# among the replicated weights under attention DP. So the target's own surface
# is `[M, 5120] @ [5120, 129280]`, at every row bucket the engine serves. The
# 32320 shard stays below as explicitly reference-only coverage.
#
# BOTH DTYPES ARE CERTIFIED, and for a stated reason: the checkpoint stores the
# head in bf16 (plan.md line 35: "Replicate the BF16 embedding and untied BF16
# language head"), while the reference promotes the same weight to fp32 so its
# logits come out fp32 directly. The target's parity leg compares against the
# reference, so both dtypes are on the path.
HEAD_DIM = 5120
HEAD_N_FULL = 129280  # the target: complete local logits, no vocab collective
HEAD_N_SHARD = 32320  # the reference implementation's per-rank shard
HEAD_ROWS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: A representative subset for the reference-only shard; the target width gets
#: the full row sweep.
REF_ONLY_ROWS = [1, 129, 4096]


def _head_operands(m: int, n: int, dtype: torch.dtype, seed: int):
    """`mat_a` [M, 5120] row-major and the head weight `[N, 5120]` as a `.t()` view."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mat_a = torch.randn(m, HEAD_DIM, generator=gen, device="cuda", dtype=dtype)
    weight = torch.randn(n, HEAD_DIM, generator=gen, device="cuda", dtype=dtype)
    return mat_a, weight


@pytest.mark.parametrize("m", HEAD_ROWS)
def test_v41_head_replicated_bf16(m: int) -> None:
    """TARGET: complete `129280`-wide local logits from the checkpoint's bf16 head."""
    mat_a, weight = _head_operands(m, HEAD_N_FULL, torch.bfloat16, HEAD_N_FULL + m)
    _check(mat_a, weight.t())


@pytest.mark.parametrize("m", HEAD_ROWS)
def test_v41_head_replicated_fp32(m: int) -> None:
    """TARGET width in the reference's dtype: `ParallelHead` promotes the weight to fp32.

    Certified at the full width because the module-parity leg compares the
    target's logits against the reference's, and the reference computes in
    fp32. Measured bit-identical to the fp32 reference at every row bucket.
    """
    mat_a, weight = _head_operands(m, HEAD_N_FULL, torch.float32, HEAD_N_FULL + m)
    _check(mat_a, weight.t())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("m", REF_ONLY_ROWS)
def test_v41_head_reference_rank_shard(m: int, dtype: torch.dtype) -> None:
    """REFERENCE-ONLY: the native implementation's `[32320, 5120]` per-rank shard.

    Not a target surface -- the target produces complete local logits. Kept
    because `N = 32320` is 64-aligned but NOT 128-aligned, which is the kind of
    `N` a gemm kernel can have an opinion about, and because the parity leg runs
    the reference at this width.
    """
    mat_a, weight = _head_operands(m, HEAD_N_SHARD, dtype, HEAD_N_SHARD + m)
    _check(mat_a, weight.t())


def test_v41_head_gate_sees_a_reordered_vocabulary() -> None:
    """The gate must catch a head whose weight rows are assembled in the wrong order.

    Reversing the vocabulary rows keeps the shape and every value identical, so
    only the numbers can catch it -- this is the shape of a real weight-loading
    bug, and it must be far outside the same bound the correct result passes.
    Driven at the target width.
    """
    m = 129
    mat_a, weight = _head_operands(m, HEAD_N_FULL, torch.bfloat16, 9090)
    out = cublas_mm(mat_a, weight.t())
    ref = _ref_mm(mat_a, weight.t(), None, None)
    wrong = cublas_mm(mat_a, torch.flip(weight, dims=(0,)).t())

    bound = 1e-3 + 1.6e-2 * ref.float().abs()
    over_correct = int(((out.float() - ref.float()).abs() > bound).sum().item())
    over_wrong = int(((wrong.float() - ref.float()).abs() > bound).sum().item())
    assert over_correct == 0, f"the correct result put {over_correct} elements over the gate"
    assert over_wrong > ref.numel() * 0.9, (
        f"a reversed vocabulary put only {over_wrong} of {ref.numel()} elements over the "
        f"gate; the gate cannot see a head weight assembled in the wrong row order"
    )


# ── Preconditions, driven on this arch and version ───────────────────────────
#
# Every claim in the contract's Preconditions section is asserted here, on
# sm_103 under the trtllm version the receipt records, with the exact message
# quoted. A previous revision of this contract attributed all of it to
# trtllm 1.3.0rc21 on sm_100 -- a path no receipt covers.

LOUD_CASES = [
    ("mat_b_row_major", "Expected mat_b.strides()[0] == 1 to be true"),
    ("three_d_mat_a", "Expected mat_a.dim() == 2 && mat_b.dim() == 2 to be true"),
    ("mixed_input_dtypes", "CUBLAS_STATUS_NOT_SUPPORTED"),
    ("fp8_without_out_dtype", "CUBLAS_STATUS_NOT_SUPPORTED"),
    ("bf16_to_fp16_out", "CUBLAS_STATUS_NOT_SUPPORTED"),
    ("cpu_operands", "Could not run 'trtllm::cublas_mm' with arguments from the 'CPU' backend"),
    ("k_mismatch", "mat_a.sizes()[1] == mat_b.sizes()[0]"),
]


@pytest.mark.parametrize("case,message", LOUD_CASES)
def test_op_rejects_loudly(case: str, message: str) -> None:
    """Domains the op rejects itself, so the wrapper must not repeat them."""
    torch.manual_seed(30)
    m, k, n = 64, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    if case == "mat_b_row_major":
        args = (a, w.t().contiguous(), None, None, 0, None)
    elif case == "three_d_mat_a":
        args = (a.unsqueeze(0), w.t(), None, None, 0, None)
    elif case == "mixed_input_dtypes":
        args = (a, w.to(torch.float16).t(), None, None, 0, None)
    elif case == "fp8_without_out_dtype":
        args = (a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t(), None, None, 0, None)
    elif case == "bf16_to_fp16_out":
        args = (a, w.t(), None, torch.float16, 0, None)
    elif case == "cpu_operands":
        args = (a.cpu(), w.t().cpu(), None, None, 0, None)
    else:  # k_mismatch
        args = (a, torch.randn(2 * k, n, device="cuda", dtype=torch.bfloat16), None, None, 0, None)

    with pytest.raises((RuntimeError, NotImplementedError)) as excinfo:
        torch.ops.trtllm.cublas_mm(*args)
    assert message in str(excinfo.value), f"{case}: got {excinfo.value}"


def test_mat_a_row_strides_are_ignored_and_the_wrapper_guards_it() -> None:
    """A row-strided `mat_a` is accepted and silently wrong. Not catchable, hence a guard."""
    torch.manual_seed(31)
    m, k, n = 64, 256, 128
    wide = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    strided = wide[:, :k]
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    got = torch.ops.trtllm.cublas_mm(strided, w.t(), None, None, 0, None)
    ref = _ref_mm(strided, w.t(), None, None)
    bound = 1e-3 + 1.6e-2 * ref.float().abs()
    over = int(((got.float() - ref.float()).abs() > bound).sum().item())
    assert over > ref.numel() * 0.5, (
        f"only {over} of {ref.numel()} elements moved; the row-stride hazard this guard exists "
        f"for is not being exercised"
    )
    with pytest.raises(AssertionError, match="mat_a must be dense row-major"):
        cublas_mm(strided, w.t())


def test_mat_b_column_stride_is_unchecked_and_the_wrapper_guards_it() -> None:
    """`stride(1) != K` is accepted and silently wrong: the op only checks `stride(0) == 1`."""
    torch.manual_seed(32)
    m, k, n = 64, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    wide_w = torch.randn(n, 2 * k, device="cuda", dtype=torch.bfloat16)
    w_strided = wide_w[:, :k]
    mat_b = w_strided.t()
    assert mat_b.stride(0) == 1 and mat_b.stride(1) != k, "this case must keep stride(0) == 1"
    got = torch.ops.trtllm.cublas_mm(a, mat_b, None, None, 0, None)
    ref = _ref_mm(a, mat_b, None, None)
    bound = 1e-3 + 1.6e-2 * ref.float().abs()
    over = int(((got.float() - ref.float()).abs() > bound).sum().item())
    assert over > ref.numel() * 0.5, f"only {over} of {ref.numel()} elements moved"
    with pytest.raises(AssertionError, match="mat_b must be dense column-major"):
        cublas_mm(a, mat_b)


def test_bias_shape_is_unchecked_and_the_wrapper_guards_it() -> None:
    """A bias of the wrong length is accepted, not rejected."""
    torch.manual_seed(33)
    m, k, n = 64, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    short = torch.randn(n // 2, device="cuda", dtype=torch.bfloat16)
    got = torch.ops.trtllm.cublas_mm(a, w.t(), short, None, 0, None)
    assert got.shape == (m, n), "a short bias was accepted and produced a full-width result"
    with pytest.raises(AssertionError, match=r"bias must be a contiguous \[N\] tensor"):
        cublas_mm(a, w.t(), short)


def test_bias_dtype_must_equal_the_output_dtype() -> None:
    """An fp32 bias with bf16 output is accepted and silently wrong."""
    torch.manual_seed(34)
    m, k, n = 64, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    got = torch.ops.trtllm.cublas_mm(a, w.t(), bias.float(), None, 0, None)
    want = _ref_mm(a, w.t(), bias, None)
    bound = 1e-3 + 1.6e-2 * want.float().abs()
    over = int(((got.float() - want.float()).abs() > bound).sum().item())
    assert over > 0, "an fp32 bias with bf16 output was expected to produce a wrong result"
    with pytest.raises(AssertionError, match="bias dtype must equal the output dtype"):
        cublas_mm(a, w.t(), bias.float())


def test_bias_is_silently_ignored_with_fp32_inputs() -> None:
    """With fp32 inputs the bias is accepted and dropped -- exactly, not approximately."""
    torch.manual_seed(35)
    m, k, n = 64, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    got = torch.ops.trtllm.cublas_mm(a, w.t(), bias, None, 0, None)
    unbiased = _ref_mm(a, w.t(), None, None)
    assert torch.equal(got, unbiased), "the bias was not silently dropped; this claim has changed"
    with pytest.raises(AssertionError, match="bias is silently ignored for fp32 inputs"):
        cublas_mm(a, w.t(), bias)


def test_meta_registration_accepts_nd_mat_a() -> None:
    """The fake (meta) kernel accepts N-D `mat_a` where the real one raises.

    Pinned because it means shape inference under `torch.compile` can diverge
    from eager behaviour for N-D inputs: the trace succeeds and the runtime
    call does not.
    """
    meta_a = torch.empty(2, 64, 256, device="meta", dtype=torch.bfloat16)
    meta_b = torch.empty(256, 128, device="meta", dtype=torch.bfloat16)
    out = torch.ops.trtllm.cublas_mm(meta_a, meta_b, None, None, 0, None)
    assert out.shape == (2, 64, 128), out.shape
