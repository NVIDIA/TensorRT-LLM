# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""(rule 1b, bitwise) moe_unpermute with an optional residual: the shared-expert output folded into the routed MoE's
fixed-order finalize. Per token t and hidden column c, in the top-k order of expanded_idx_to_permuted_idx exactly as
the C++ moe_unpermute_inplace (cute_dsl moeUnpermuteKernel) does it:

    acc = 0; for k in 0..top_k-1: if p = e2p[t, k] >= 0: acc = fma(float(rows[p, c]), float(scales[t, k]), acc)
    out = bf16_rn(acc)                       # = moe_unpermute's output
    out = bf16_rn(float(out) + float(res[t, c]))  # = shared_output.add_(routed_output)

so the output is bitwise moe_unpermute_inplace followed by the ATen bf16 add (``fma`` is the contraction nvcc applies
to the C++ ``acc += float(x) * float(s)``; ``contract=False`` is the separately rounded form, kept for the test)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpermute_residual_kernel(
    rows_ptr, out_ptr, e2p_ptr, scales_ptr, res_ptr, hidden, TOP_K: tl.constexpr, BLOCK: tl.constexpr,
    HAS_RES: tl.constexpr, CONTRACT: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(e2p_ptr + t * TOP_K + k)
        s = tl.load(scales_ptr + t * TOP_K + k).to(tl.float32)
        x = tl.load(rows_ptr + p.to(tl.int64) * hidden + cols, mask=mask & (p >= 0), other=0.0).to(
            tl.float32
        )
        if CONTRACT:
            upd = tl.fma(x, s, acc)
        else:
            upd = acc + x * s
        acc = tl.where(p >= 0, upd, acc)
    out = acc.to(tl.bfloat16)
    if HAS_RES:
        r = tl.load(res_ptr + t.to(tl.int64) * hidden + cols, mask=mask, other=0.0).to(tl.float32)
        out = (out.to(tl.float32) + r).to(tl.bfloat16)
    tl.store(out_ptr + t.to(tl.int64) * hidden + cols, out, mask=mask)


# (BLOCK, num_warps): tuned on SM107 at the M3 CTX shape (784698: 54.7 / 97.6 us at 8K / 16K local tokens vs 58.5 /
# 102.1 at (1024, 2), copy floor 49.3 / 94.0; bitwise in every config).
_CFG = (2048, 4)


def unpermute_residual(
    permuted_rows: torch.Tensor,
    output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    residual: torch.Tensor = None,
    contract: bool = True,
) -> torch.Tensor:
    """output [T, H] bf16 = moe_unpermute(permuted_rows [M, H], e2p [T * top_k], topk_scales [T, top_k]) (+ residual
    [T, H] bf16, rounded as the ATen add). Returns ``output``."""
    t, hidden = output.shape
    top_k = topk_scales.shape[1]
    assert permuted_rows.dtype == output.dtype == torch.bfloat16 and permuted_rows.is_contiguous()
    assert output.is_contiguous() and expanded_idx_to_permuted_idx.numel() == t * top_k
    if residual is not None:
        assert (
            residual.shape == output.shape
            and residual.dtype == torch.bfloat16
            and residual.is_contiguous()
        )
    block, warps = _CFG
    _unpermute_residual_kernel[(t, triton.cdiv(hidden, block))](
        permuted_rows, output, expanded_idx_to_permuted_idx, topk_scales.contiguous(),
        residual if residual is not None else output, hidden, TOP_K=top_k, BLOCK=block,
        HAS_RES=residual is not None, CONTRACT=contract, num_warps=warps,
    )  # fmt: skip
    return output
