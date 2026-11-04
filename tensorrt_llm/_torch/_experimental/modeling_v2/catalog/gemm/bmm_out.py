# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul writing into a caller-owned output, via the trtllm bmm_out op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class BmmOut(OpWrapper):
    """`out[i] = a[i] @ b[i]` for every batch index, written into `out`, which
    is returned.

    The point of the op over `torch.bmm` is the caller-owned `out`: every
    shipped call site passes a transposed view of a buffer the rest of the
    forward already holds, so the product lands where it is needed with no
    copy. That is also the hazard -- see `is_valid`.

    Fusion boundary: the product only. The weight regroup and transposition,
    and the buffers `a` and `out` are views of, are the caller's.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # deepseek-r1's MLA uses the op twice per layer in opposite orientations:
    # absorbing the nope half of q into the latent space on the way in, and
    # projecting the latent attention output back out to v. Batch is the head
    # count (128, replicated under attention DP).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, q_nope -> latent: [H, 1, nope] @ [H, nope, kv_lora]",
            spec=dict(batch=128, m=1, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="decode, latent -> v: [H, 1, kv_lora] @ [H, kv_lora, v_dim]",
            spec=dict(batch=128, m=1, k=512, n=128, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="prefill token count, the same absorbing orientation",
            spec=dict(batch=128, m=2048, k=128, n=512, dtype=torch.bfloat16, transposed_out=True),
        ),
        Cell(
            why="contiguous out, to separate the op's own correctness from the "
            "transposed-view path the targets happen to use",
            spec=dict(batch=128, m=64, k=128, n=512, dtype=torch.bfloat16, transposed_out=False),
        ),
    )

    note = """
    What runs: `trtllm::bmm_out` is a `torch.library.custom_op` registered in
    `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` whose body is
    `torch.bmm(a, b, out=out)`, so one call is one cuBLAS strided-batched GEMM.
    At Mistral Large 4's views the profiler shows a single `nvjet_sm103_tst_*`
    kernel per call and no copy, at each of 34 values of m from 1 to 32768 and
    on the trailing 15 rows of 32768-row buffers: the product is written
    straight into the caller's view. cuBLAS picks the kernel by size -- over
    those 34 values it named 14, 13 and 15 distinct kernels for K-B RoPE, K-B
    NoPE and V-B, 5, 4 and 5 of them at the 13 values above 9216 -- and every
    m's rows are still bitwise the leading rows of the full-size run, which
    the test's 16-head sweep cells hold at 9216 and at 32768 rows. A transposed-view
    `b` and the same values made contiguous give bitwise the same product. At
    those three shapes and 32768 rows the product is within one bf16 rounding
    of an fp64 product. The cells run under torch's matmul defaults
    (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` True).

    At DEP4's 64 heads the profiler shows one `nvjet_sm103_tst_*` launch per
    call and no copy, at every m from 1 to 32768: 36, 42 and 67
    distinct kernels for K-B RoPE, K-B NoPE and V-B, in 770, 634 and 1936 runs
    of m with one kernel and grid, every change between m = 8 and 9 or
    between 16k and 16k + 1. Every m's rows are bitwise the leading
    rows of the 32768-row run: the `large` cells drive each m, and pre-merge
    drives every m to 16 and both edges of every 16-row window, which holds
    both edges of every run. A transposed-view `b` and the same values made
    contiguous give bitwise the same product here too. At these shapes and
    32768 rows the product is within one bf16 rounding of an fp64 product.

    At TEP8's and TP8's 8 heads the profiler shows one `nvjet_sm103_tst_*`
    launch per call and no copy, at every m from 1 to 32768: 58, 59 and 86
    distinct kernels for K-B RoPE, K-B NoPE and V-B, in 615, 658 and 1563 runs
    of m with one kernel and grid, every change between m = 8 and 9 or between
    16k and 16k + 1. Every m's rows are bitwise the leading rows of the
    32768-row run: the test's 8-head sweep cells drive each m in pre-merge. A
    transposed-view `b` and the same values made contiguous give bitwise the
    same product here too. At these shapes and 32768 rows the product is
    within one bf16 rounding of an fp64 product.

    On the fp8_ds_mla latent KV a tep8 or tp8 rank's K-B output is heads 0..7
    of the `[T, 64, 576]` fused query FlashMLA takes (the test's `_ML4_8_FQ64`): the same
    batch of 8, an output token stride of 64 x 576 = 36864 instead of 8 x 576,
    and heads 8..63 left as the step zeroed them. Those cells hold the product
    there at every m 1..32768, each m's rows bitwise the 32768-row run's (and
    at m 1..16 and both edges of every 16-row window on their own), in CUDA
    graphs at every batch 1..16 and on a mixed step's 15-row tail, with heads
    8..63 never written. The profiler shows one `nvjet_sm103_tst_*` launch
    per call and no copy at each of 40 values of m from 1 to 32768 and on the
    15-row tail, at every one of them the kernel cuBLAS picks for the compact
    8-head layout: 18 distinct for K-B RoPE, 14 for K-B NoPE (tep8's
    profile). At 32768 rows the product is within one bf16 rounding of an
    fp64 product here too.

    On the fp8_ds_mla latent KV a tep4 (or tp4) rank's K-B output is heads
    0..15 of the `[T, 64, 576]` fused query FlashMLA takes (the test's `_ML4_16_FQ64`): the
    same batch of 16, an output token stride of 64 x 576 = 36864 instead of
    16 x 576, and heads 16..63 left as the step zeroed them. Those cells hold
    the product there at every m 1..32768, each m's rows bitwise the
    32768-row run's, in CUDA graphs at every batch 1..16 and on a mixed
    step's 15-row tail, with heads 16..63 never written. The profiler shows
    one `nvjet_sm103_tst_*` launch per call and no copy at each of 40 values
    of m from 1 to 32768 and on the 15-row tail, at every one of them the
    kernel cuBLAS picks for the compact 16-head layout: 16 distinct for K-B
    RoPE, 14 for K-B NoPE (tep4's profile; tp4's, at 32 values of m and the
    tail, found the same at every one, 13 distinct and 11). At 32768 rows the
    product is within one bf16 rounding of an fp64 product here too.

    `out` is written, not accumulated into: its prior contents are ignored.
    m = 0 returns at once and writes nothing. Capturable: a captured call
    replays bitwise equal to an eager one. Metadata consumed: none.

    Refused loudly by the op itself, each kind held by the test: `a` or `b`
    not 3-D (`batch1 must be a 3D tensor`), a batch or K mismatch, and a mix of
    dtypes -- every pairing of bf16, fp16 and fp32 with `a` and `b` differing,
    under each `out` dtype, and the two `out` dtypes other than the operands'
    that were driven, bf16 into fp32 and fp16 into bf16 (`expected scalar type
    BFloat16 but found Half`, or `Expected out tensor to have dtype ...`).
    Refused by `is_valid`, beside the 3-D and single-dtype checks it restates:
    an `out` of the wrong shape, which the op resizes in place with only a
    deprecation warning. A standalone `out` is
    re-allocated, so a buffer the caller aliased with it never sees the
    product. A Mistral Large 4 K-B view one token short is re-strided contiguous
    over the fused query's storage: the product lands there scrambled for the
    buffer's layout, and over its rope halves.

    Not in any cell: fp16 or fp32 operands (accepted), a contiguous `a` or
    `out` (accepted), an `out` whose last dimension is not unit-stride, `a`
    overlapping `out`, batch counts other than 8, 16 and 64, a batch written
    into a wider head buffer other than 16 of 64 and 8 of 64, and m above
    32768 at 64 heads, at 16 of 64 or at 8 of 64 (four probes, below).

    Hard limits (sm_103). The NVFP4 target's m <= 32768 reaches at most
    1,207,959,552 elements into the fused query at 64 heads (56% of 2^31) --
    the 64-head fused query a tep4 or tp4 rank writes 16 heads of, and a tep8
    rank 8 heads of, is that size too, the last element written 1,207,931,839
    and 1,207,927,231 -- 301,989,888 at 16 and 150,994,944 at 8; the FP8
    target's m <= 9216 at most 84,934,656.
      row offset  m x token stride   first m with an offset at 2^31:
                  64 heads           K-B out (36864) 58,255 -- probed at m 58260
                                     on K-B RoPE and NoPE: passed, every row
                                     within the band, nothing written past n,
                                     calls on the rows from 58247, 58254, 58255
                                     and 58257 bitwise the full call's rows and
                                     nothing written ahead of them;
                                     V-B a (32768) 65,537 -- probed at m 65540:
                                     passed, every row within the band, calls
                                     on the rows from 65529, 65536 and 65537
                                     the same; K-B a and V-B out (8192)
                                     262,145, computed
                  16 heads           K-B out (9216) 233,017; V-B a (8192)
                                     262,145; K-B a and V-B out (2048)
                                     1,048,577, computed
                  8 heads            K-B out (4608) 466,034; V-B a (4096)
                                     524,289; K-B a and V-B out (1024)
                                     2,097,153, computed
                  16 of 64 heads     K-B out (36864) 58,255 -- probed at m 58260
                                     on K-B RoPE and NoPE, heads 0..15 of a
                                     64-head buffer (last element written
                                     2,147,668,927): passed, every row within
                                     the band, nothing written past n or into
                                     heads 16..63, calls on the rows from
                                     58247, 58254, 58255 and 58257 (storage
                                     offsets to 2,147,586,048) bitwise the
                                     full call's rows and nothing written
                                     ahead of them
                  8 of 64 heads      K-B out (36864) 58,255 -- probed at m 58260
                                     on K-B RoPE and NoPE, heads 0..7 of a
                                     64-head buffer (last element written
                                     2,147,664,319): passed, every row within
                                     the band, nothing written past n or into
                                     heads 8..63, calls on the rows from
                                     58247, 58254, 58255 and 58257 (storage
                                     offsets to 2,147,586,048) bitwise the
                                     full call's rows and nothing written
                                     ahead of them
      m           the row count      2^31, computed
    No index passes through fp32. The op touches no KV pool, only the step's
    activations, so the pool's size sets no limit on it.
    """

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.bmm_out(a, b, out)
        return out

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated.

        `out` is named to mirror `__call__`: the op takes a destination,
        the reference returns a value, and hiding the argument would hide that
        the op mutates one of its inputs.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, ~1e-3 relative, which would make the reference the inaccurate
        side of the comparison.
        """
        with true_fp32_matmul():
            return (a.float() @ b.float()).to(a.dtype)

    def is_valid(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # A wrong-shaped `out` is silently resized and re-allocated by the op,
        # detaching it from whatever buffer the caller aliased it with -- the
        # product then lands somewhere nobody reads, and the forward continues
        # with stale data rather than failing.
        assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3, "a, b, out must be 3D"
        assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
            "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
        )
        # Mixed input dtypes are type-promoted instead of rejected, changing the
        # required out dtype; the catalog exposes only the single-dtype form.
        assert a.dtype == b.dtype == out.dtype, "a, b, out must share one dtype"


bmm_out = BmmOut()
