# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out catalog entry.

THE REFERENCE IS NOT `torch.bmm`. The op's body is exactly
`torch.bmm(a, b, out=out)`, so an expected value built with `torch.bmm` is built
from the op under test: it cannot separate "my reference is wrong" from "the op
is wrong", and a bug inside `torch.bmm` would be invisible because both sides
carry it. `_ref` therefore loops over the batch calling `torch.mm` per slice in
fp32 -- a different op and a different kernel -- and
`test_the_reference_itself_is_independent` pins that loop against a construction
that launches no GEMM at all (broadcast multiply plus a sum reduction), so the
independence is measured rather than asserted.
"""

import warnings

import pytest
import torch

from tensorrt_llm._torch.staircase.catalog.gemm.bmm_out import bmm_out

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"

#: `torch.testing.assert_close`'s default relative tolerance per output dtype,
#: used unchanged. Only the absolute floor moves; see `_check`.
RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
ATOL = 1e-3


def _ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 per-batch `torch.mm` -- deliberately NOT `torch.bmm`, the op under test."""
    return torch.stack([torch.mm(a[i].float(), b[i].float()) for i in range(a.shape[0])])


def _check(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    ref = _ref(a, b)
    bmm_out(a, b, out)
    # rtol: torch.testing defaults per output dtype. atol: kernel and reference
    # both accumulate in fp32 but in different summation orders; the
    # order-dependent absolute noise is up to ~K * 2^-24, which dominates on
    # near-zero outputs produced by cancellation, so atol=1e-3 instead of the
    # ~1e-5 defaults. Measured at this checkpoint's K=4096 in
    # `test_v41_output_lora_group_projection`: 0 elements outside at every row
    # bucket, against 1..2,341 outside at the unmodified 1e-5.
    torch.testing.assert_close(out.float(), ref.to(torch.float32), rtol=RTOL[out.dtype], atol=ATOL)


def _make(
    batch: int, m: int, k: int, n: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn(batch, m, k, device="cuda").to(dtype)
    b = torch.randn(batch, k, n, device="cuda").to(dtype)
    out = torch.empty(batch, m, n, device="cuda", dtype=dtype)
    return a, b, out


def test_the_reference_itself_is_independent() -> None:
    """`_ref`'s per-batch `mm` must agree with a construction that launches no GEMM.

    This is what makes `_ref` usable as an expected value: it is checked against
    an elementwise broadcast multiply plus a sum reduction, which shares no
    kernel with either `torch.mm` or the op under test. Small shapes only --
    the mul+sum intermediate is `[B, M, K, N]`.
    """
    gen = torch.Generator(device="cuda").manual_seed(11)
    a = torch.randn(3, 5, 64, generator=gen, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(3, 64, 7, generator=gen, device="cuda", dtype=torch.bfloat16)
    mulsum = (a.float().unsqueeze(-1) * b.float().unsqueeze(1)).sum(dim=2)
    torch.testing.assert_close(_ref(a, b), mulsum, rtol=1.3e-6, atol=1e-4)


def test_bf16() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens per batch entry) and prefill-like (many) shapes
    for batch, m, k, n in [(64, 1, 512, 128), (16, 8, 576, 512), (8, 2048, 512, 128)]:
        _check(*_make(batch, m, k, n, torch.bfloat16))


def test_bf16_noncontiguous_views() -> None:
    # The op exists so `out` may be a strided view (e.g. a transposed slice of
    # a [tokens, groups, rank] buffer, as MLA uses it); `a` and `b` may be
    # strided views too. Exercise all three.
    torch.manual_seed(1)
    batch, m, k, n = 16, 32, 256, 64
    a_wide = torch.randn(batch, m, 2 * k, device="cuda").to(torch.bfloat16)
    a = a_wide[:, :, :k]  # row-strided view
    b = (
        torch.randn(batch, n, k, device="cuda").to(torch.bfloat16).transpose(1, 2)
    )  # transposed view
    out_buf = torch.empty(m, batch, n, device="cuda", dtype=torch.bfloat16)
    out = out_buf.transpose(0, 1)  # non-contiguous out
    _check(a, b, out)
    # writes landed in the aliased buffer, not a reallocation
    torch.testing.assert_close(
        out_buf.transpose(0, 1).float(), _ref(a, b), rtol=RTOL[torch.bfloat16], atol=ATOL
    )


def test_bf16_unaligned_shapes() -> None:
    # dims not multiples of typical tile/vector widths
    torch.manual_seed(2)
    for batch, m, k, n in [(3, 5, 100, 60), (7, 13, 333, 129)]:
        _check(*_make(batch, m, k, n, torch.bfloat16))


def test_fp16() -> None:
    torch.manual_seed(3)
    for batch, m, k, n in [(64, 1, 512, 128), (8, 1024, 512, 256)]:
        _check(*_make(batch, m, k, n, torch.float16))


def test_fp32() -> None:
    torch.manual_seed(4)
    for batch, m, k, n in [(32, 2, 256, 128), (4, 1024, 512, 256)]:
        _check(*_make(batch, m, k, n, torch.float32))


# ── DeepSeek-V4.1-Flash column: the grouped output LoRA ───────────────────────
#
# Derived from the RAW checkpoint's safetensors header, which is what the
# target's `weights.py` reads: `layers.<L>.attn.wo_a.weight` is `[8192, 4096]`,
# i.e. `[o_groups * o_lora_rank, n_heads * head_dim // o_groups]` =
# `[8 * 1024, 64 * 512 / 8]`. Viewed as `[o_groups, o_lora_rank, 4096]` and
# applied with `einsum("bsgd,grd->bsgr")`, that is a batched matmul of
# BATCH 8, `[M, 4096] @ [4096, 1024]`, with `b` a TRANSPOSE VIEW of the stored
# `[8, 1024, 4096]` weight -- never a contiguous copy.
#
# BATCH 8, NOT 2. The reference implementation declares `wo_a` as a
# `ColumnParallelLinear`, so at world_size 4 it holds `[2048, 4096]` per rank
# and `n_local_groups = o_groups // world_size = 2`. The staircase target does
# not shard it: plan.md line 30 fixes attention DP at dep4 and replicates
# attention and every dense projection, and plan.md line 79 states the target's
# own geometry as "eight group-local A projections followed by the full B
# projection". Certifying batch 2 certifies the reference's rank shard, which
# the target never calls; batch 2 is kept below only as reference-only coverage.
LORA_GROUPS = 8  # the target's batch: all eight groups are local under ADP
LORA_GROUPS_REF_SHARD = 2  # the reference's per-rank batch at world_size 4
LORA_IN = 4096
LORA_RANK = 1024
LORA_ROWS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: A representative subset for the reference-only shard, which no target call
#: reaches; the target batch gets the full row sweep.
REF_ONLY_ROWS = [1, 129, 4096]


def _lora_operands(batch: int, m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention output `[G, M, 4096]` and the stored `[G, 1024, 4096]` LoRA-A weight."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    o = torch.randn(batch, m, LORA_IN, generator=gen, device="cuda", dtype=torch.bfloat16)
    wo_a = torch.randn(
        batch, LORA_RANK, LORA_IN, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    return o, wo_a


@pytest.mark.parametrize("m", LORA_ROWS)
def test_v41_output_lora_group_projection(m: int) -> None:
    """TARGET: batch 8 x `[M,4096] @ [4096,1024]`, `b` a transpose view, every row bucket."""
    o, wo_a = _lora_operands(LORA_GROUPS, m, 4096 + m + LORA_GROUPS)
    b = wo_a.transpose(1, 2)
    assert not b.is_contiguous() and b.shape == (LORA_GROUPS, LORA_IN, LORA_RANK)
    out = torch.empty(LORA_GROUPS, m, LORA_RANK, device="cuda", dtype=torch.bfloat16)
    _check(o, b, out)


@pytest.mark.parametrize("m", REF_ONLY_ROWS)
def test_v41_output_lora_reference_rank_shard(m: int) -> None:
    """REFERENCE-ONLY: the native implementation's batch-2 per-rank shard.

    Not a target surface. It is kept because the module-parity leg compares
    against the reference implementation, and knowing the op is correct at the
    shard the reference runs removes one variable from that comparison.
    """
    o, wo_a = _lora_operands(LORA_GROUPS_REF_SHARD, m, 4096 + m + LORA_GROUPS_REF_SHARD)
    b = wo_a.transpose(1, 2)
    out = torch.empty(LORA_GROUPS_REF_SHARD, m, LORA_RANK, device="cuda", dtype=torch.bfloat16)
    _check(o, b, out)


def test_v41_output_lora_is_block_diagonal_over_groups() -> None:
    """Each group must project only its own heads, and the gate must see it if not.

    The LoRA-A is block diagonal over groups by construction -- group `g` reads
    only the 8 heads it owns. Rolling the eight groups by one keeps every shape
    and value identical and produces a perfectly well-formed result, so numbers
    are the only thing that can catch it. It must land far outside the same
    bound the correct result passes.
    """
    m = 129
    o, wo_a = _lora_operands(LORA_GROUPS, m, 7777)
    b = wo_a.transpose(1, 2)
    out = torch.empty(LORA_GROUPS, m, LORA_RANK, device="cuda", dtype=torch.bfloat16)
    bmm_out(o, b, out)
    ref = _ref(o, b)

    rolled = torch.empty_like(out)
    bmm_out(o, b.roll(1, dims=0), rolled)

    bound = ATOL + RTOL[torch.bfloat16] * ref.abs()
    over_correct = int(((out.float() - ref).abs() > bound).sum().item())
    over_rolled = int(((rolled.float() - ref).abs() > bound).sum().item())
    assert over_correct == 0, f"the correct result put {over_correct} elements over the gate"
    assert over_rolled > ref.numel() * 0.9, (
        f"rolling the eight groups' weights put only {over_rolled} of {ref.numel()} elements "
        f"over the gate; the gate cannot see a group-association error"
    )


# ── Preconditions, driven on this arch and version ───────────────────────────
#
# Every claim in the contract's Preconditions section is asserted here, on
# sm_103 under the trtllm version the receipt records, with the exact message
# quoted. The previous revision of this contract attributed all of it to
# trtllm 1.3.0rc21 / torch 2.11.0 on sm_100 -- a path no receipt covers.

LOUD_CASES = [
    ("two_d_operands", "batch1 must be a 3D tensor"),
    ("batch_mismatch", "Expected size for first two dimensions of batch2 tensor to be"),
    ("k_mismatch", "Expected size for first two dimensions of batch2 tensor to be"),
    ("out_dtype_mismatch", "Expected out tensor to have dtype"),
    ("fp8_operands", "\"baddbmm_cuda\" not implemented for 'Float8_e4m3fn'"),
    ("mixed_device", "Expected all tensors to be on the same device"),
]


def _loud_args(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Operands for one loud-rejection case, shared by the wrapper and raw-op tests."""
    torch.manual_seed(20)
    bsz, m, k, n = 4, 8, 64, 32
    a = torch.randn(bsz, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(bsz, k, n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(bsz, m, n, device="cuda", dtype=torch.bfloat16)

    if case == "two_d_operands":
        return (a[0], b[0], out[0])
    if case == "batch_mismatch":
        return (a, b[:2], out)
    if case == "k_mismatch":
        return (a, torch.randn(bsz, 2 * k, n, device="cuda", dtype=torch.bfloat16), out)
    if case == "out_dtype_mismatch":
        return (a, b, out.float())
    if case == "fp8_operands":
        return (a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), out.to(torch.float8_e4m3fn))
    assert case == "mixed_device", case
    return (a.cpu(), b, out)


@pytest.mark.parametrize("case,message", LOUD_CASES)
def test_wrapper_preserves_the_op_rejection(case: str, message: str) -> None:
    """The WRAPPER must surface the op's own message, not shadow it with its own error.

    The regression this pins was found in review: the shape guard read
    `b.shape[2]`, which on a 2-D `b` raises `IndexError: tuple index out of
    range` from inside the wrapper before the op is ever called. A caller then
    sees an error that names no operand instead of the documented
    `batch1 must be a 3D tensor`, and the contract's Preconditions table
    describes behaviour the wrapper made unreachable. Driving the wrapper --
    not the raw op -- is the whole point of this case.
    """
    with pytest.raises((RuntimeError, NotImplementedError)) as excinfo:
        bmm_out(*_loud_args(case))
    assert message in str(excinfo.value), (
        f"{case}: got {type(excinfo.value).__name__}: {excinfo.value}"
    )


@pytest.mark.parametrize("case,message", LOUD_CASES)
def test_op_rejects_loudly(case: str, message: str) -> None:
    """Domains the op rejects itself, so the wrapper must not repeat them."""
    with pytest.raises((RuntimeError, NotImplementedError)) as excinfo:
        torch.ops.trtllm.bmm_out(*_loud_args(case))
    assert message in str(excinfo.value), f"{case}: got {excinfo.value}"


MIXED_DTYPES = [
    (da, db, dout)
    for da in (torch.bfloat16, torch.float16, torch.float32)
    for db in (torch.bfloat16, torch.float16, torch.float32)
    if da is not db
    for dout in (da, db)
]


@pytest.mark.parametrize("da,db,dout", MIXED_DTYPES)
def test_mixed_input_dtypes_always_raise(
    da: torch.dtype, db: torch.dtype, dout: torch.dtype
) -> None:
    """All twelve mixed-dtype combinations raise: there is no promotion path.

    This is why the wrapper carries no dtype assert. The meta check demands
    `out` in `b.dtype` while the kernel demands `a.dtype`, so when
    `a.dtype != b.dtype` the two can never both be satisfied, whichever of the
    two the caller picks for `out`.
    """
    torch.manual_seed(21)
    bsz, m, k, n = 4, 8, 64, 32
    a = torch.randn(bsz, m, k, device="cuda").to(da)
    b = torch.randn(bsz, k, n, device="cuda").to(db)
    out = torch.empty(bsz, m, n, device="cuda", dtype=dout)
    with pytest.raises(RuntimeError) as excinfo:
        torch.ops.trtllm.bmm_out(a, b, out)
    text = str(excinfo.value)
    assert "Expected out tensor to have dtype" in text or "expected scalar type" in text, text


def test_all_cpu_is_accepted_and_correct() -> None:
    """CPU is not rejected: the op computes there. Only a MIXED device raises.

    The contract used to say all three tensors must be on the same CUDA device,
    which reads as a rejection the caller can rely on. It is not one -- an
    all-CPU call returns the right answer on the CPU, silently taking the
    forward off the GPU. Recorded as accepted, since that is what it does.
    """
    torch.manual_seed(22)
    bsz, m, k, n = 4, 8, 64, 32
    a = torch.randn(bsz, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(bsz, k, n, device="cuda", dtype=torch.bfloat16)
    out_cpu = torch.empty(bsz, m, n, dtype=torch.bfloat16)
    bmm_out(a.cpu(), b.cpu(), out_cpu)
    torch.testing.assert_close(
        out_cpu.cuda().float(), _ref(a, b), rtol=RTOL[torch.bfloat16], atol=ATOL
    )


def test_wrong_shaped_out_is_silently_resized_and_the_wrapper_guards_it() -> None:
    """The one domain the op does NOT reject, in all three forms it takes.

    A standalone `out` is REALLOCATED: its `data_ptr` moves, so any view that
    aliased the old storage is silently stale. A view into a larger arena is
    resized IN PLACE and overwrites the arena's surrounding layout. A 2-D `out`
    is silently reshaped to 3-D. None of the three raises; all emit only a
    deprecation warning, which is why the wrapper asserts the shape.
    """
    torch.manual_seed(23)
    bsz, m, k, n = 4, 8, 64, 32
    a = torch.randn(bsz, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(bsz, k, n, device="cuda", dtype=torch.bfloat16)
    ref = _ref(a, b)
    bound = ATOL + RTOL[torch.bfloat16] * ref.abs()

    # (1) a standalone half-width out: reallocated, warning only.
    small = torch.zeros(bsz, m, n // 2, device="cuda", dtype=torch.bfloat16)
    ptr_before = small.data_ptr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.ops.trtllm.bmm_out(a, b, small)
    assert small.shape == (bsz, m, n), "the op did not resize; this precondition has changed"
    assert small.data_ptr() != ptr_before, "a standalone wrong-shaped out was not reallocated"
    assert any("was resized" in str(w.message) for w in caught), (
        f"expected the resize deprecation warning, got {[str(w.message) for w in caught]}"
    )

    # (2) a half-width VIEW into an arena: resized in place, arena corrupted.
    arena = torch.zeros(bsz, m, 4 * n, device="cuda", dtype=torch.bfloat16)
    sub = arena[:, :, : n // 2]
    ptr_before = sub.data_ptr()
    sibling = arena[:, :, : n // 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.ops.trtllm.bmm_out(a, b, sub)
    assert sub.data_ptr() == ptr_before, "a view with room to grow should resize in place"
    over_arena = int(((arena[:, :, :n].float() - ref).abs() > bound).sum().item())
    assert over_arena > ref.numel() * 0.9, (
        f"only {over_arena} of {ref.numel()} arena elements moved; the collateral damage this "
        f"guard exists for is not being exercised"
    )
    assert sibling.shape == (bsz, m, n // 2), "the sibling view kept its own stale shape"

    # (3) a 2-D out is silently reshaped to 3-D.
    flat = torch.zeros(bsz * m, n, device="cuda", dtype=torch.bfloat16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.ops.trtllm.bmm_out(a, b, flat)
    assert flat.shape == (bsz, m, n), "a 2-D out was not silently reshaped"

    # (4) the wrapper rejects all three before the op sees them.
    for bad in (
        torch.zeros(bsz, m, n // 2, device="cuda", dtype=torch.bfloat16),
        torch.zeros(bsz * m, n, device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, bsz, m, n, device="cuda", dtype=torch.bfloat16),
    ):
        with pytest.raises(AssertionError, match="out must be"):
            bmm_out(a, b, bad)
