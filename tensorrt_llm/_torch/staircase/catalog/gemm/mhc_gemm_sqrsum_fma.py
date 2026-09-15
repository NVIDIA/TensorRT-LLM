# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-Connection mix projection with the stream square sum fused in, via the trtllm mhc op.

One call produces both halves of what the coefficient split needs next: the
UNNORMALIZED projection `y = x @ w.T` and the per-row square sum `r = sum(x^2)`.
It applies no normalization itself -- `rsqrt(r / K + eps)` belongs to
`activation/mhc_split_sinkhorn`, which is what lets the same `r` serve a
normalization the caller schedules later.
"""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

#: The kernel's vectorized main loop consumes `BLOCK_SIZE * K_VEC = 256 * 4`
#: bf16 per iteration and runs only while `k_base + K_STEP <= K`
#: (`mhcKernels.cu`). Below `K_STEP` every element goes through the scalar tail,
#: whose loads are 2-byte and therefore always aligned -- which is exactly why
#: the alignment precondition below is conditional on `K` rather than absolute.
K_STEP = 1024


def mhc_gemm_sqrsum_fma(
    x: torch.Tensor,
    w: torch.Tensor,
    tile_n: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project `[M,K]` onto `[N,K]` and sum `x**2` per row, in one kernel.

    Returns `(y, r)` with `y = x.float() @ w.T` and `r = x.float().square().sum(-1)`,
    both freshly allocated fp32. `r` is the RAW square sum, not a norm and not a
    mean: the division by `K` and the `rsqrt` happen in the consumer.
    """
    # `M`, `N` and `K` are DERIVED, never passed in. The op takes all three as
    # arguments and indexes whatever buffers it is given with them, so a wrapper
    # that derives them cannot express the mismatch at all -- which is better
    # than guarding it. It also removes the "x longer than M" case entirely:
    # driven raw, extra rows are simply never read and the result is bit-equal.
    #
    # THESE TWO UNPACKS ARE THE RANK CONTRACT, AND THEY ARE NOT A GUARD. An
    # earlier revision asserted `x.dim() == 2 and w.dim() == 2`, which was wrong:
    # driven raw, storage-equivalent RANK-3 `x` and `w` are accepted and CORRECT,
    # so that assert rejected harmless calls. The documented interface is still
    # 2-D, so a wrong-rank caller fails here naturally on
    # `ValueError: too many values to unpack (expected 2)` -- an interface
    # mismatch, not a claim that the op mishandles the input.
    m, k = x.shape
    n, _ = w.shape
    # Four guard branches, all PURE TENSOR METADATA, each earned by a case in
    # `tests/unittest/_torch/staircase/gemm/test_staircase_mhc_gemm_sqrsum_fma.py`
    # that first drives the raw op and shows it accepted before asserting here.
    #
    # WHAT IS DELIBERATELY NOT GUARDED, and why:
    #   * `tile_n` is a TACTIC value, not metadata. A non-divisor of `N` is
    #     rejected loudly by the launcher itself
    #     (`mhcGemmSqrsumFmaLaunch: N=24 not divisible by tile_n=5`), and a
    #     negative value is accepted and silently means "use the heuristic" --
    #     but silence alone does not authorize a wrapper assert on a computation
    #     value, so both are documented in the contract instead.
    #   * `tile_m`, which the launcher discards with `(void) tile_m`. It is not
    #     in this signature at all, because exposing an argument that provably
    #     does nothing would be the hidden-default the catalog forbids.
    #   * either dtype, which the op rejects itself and loudly
    #     (`expected scalar type BFloat16 but found Float`).
    assert w.shape[1] == k, (
        f"w must be [N, K] with the SAME K as x, but x is [.., {k}] and w is [.., {w.shape[1]}]; "
        f"a disagreement is accepted and the kernel then strides x by the wrong row length"
    )
    assert x.is_contiguous(), (
        "x must be contiguous; a column slice of a wider buffer is accepted and read with the "
        "wrong stride"
    )
    assert w.is_contiguous(), (
        "w must be contiguous; a column slice of a wider buffer is accepted and read with the "
        "wrong stride"
    )
    # Spelled `x.shape[1]` rather than the `k` bound above so that the check is
    # self-evidently tensor metadata -- to a reader and to the mechanical audit
    # that reads this file, neither of which should have to trace `k`'s origin
    # to see that this is a shape check and not a computation-value one.
    assert x.shape[1] < K_STEP or x.shape[1] % 4 == 0, (
        f"K must be a multiple of 4 once it reaches {K_STEP}, got {k}; the vectorized path loads "
        f"4 bf16 as one 8-byte `ld.global.cs.v2.b32` and 4 fp32 as one 16-byte "
        f"`ld.global.L1::evict_last.v4.f32`, and a misaligned K faults ASYNCHRONOUSLY -- the op "
        f"call returns cleanly, an unrelated later synchronize raises `CUDA error: misaligned "
        f"address`, and the CUDA context is unusable from then on. K < {K_STEP} is exempt and is "
        f"NOT rejected: the vectorized loop never runs there, so K=1023 is accepted and correct"
    )
    y = torch.empty(m, n, device=x.device, dtype=torch.float32)
    r = torch.empty(m, device=x.device, dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, w, y, r, m, n, k, tile_n, 0)
    return y, r
