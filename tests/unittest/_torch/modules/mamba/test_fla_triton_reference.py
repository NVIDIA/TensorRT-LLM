# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton FLA prefill kernels against reference formulas, at shapes with partial
tail rows and columns.

The Triton gated delta rule is the GDN prefill wherever FlashInfer does not cover
the GPU (SM80/86/89). Packed lengths are unaligned to the 16/64-token chunks, K and
V are not multiples of 64, and there are fewer key heads than value heads. Which
tiles are partial depends on the block sizes autotune picks; K above 128 and
equal-length batches are not covered here.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="BF16 Triton kernels require CUDA compute capability >= 8.0",
)

SEQLENS = (1, 17, 63, 65, 130)
NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 2, 4, 96, 48
NUM_SLOTS = 8
#: One pool slot per sequence, out of order; slots 2, 4 and 5 stay untouched.
SLOTS = (6, 0, 3, 7, 1)
#: The band of the qwen4_exp test_modeling_v2_flashinfer_chunk_gated_delta_rule.py:
#: the WY operands are rounded to bf16.
RTOL, ATOL_OUTPUT, ATOL_STATE = 1.6e-2, 2e-3, 2e-2


def _gdn_operands(device: str, dtype: torch.dtype, seed: int = 2026):
    """Operands in the domain the GDN prefill path produces (unit-norm
    q/k, negative log gates, sigmoid beta), drawn on the CPU and then moved."""
    gen = torch.Generator().manual_seed(seed)
    total = sum(SEQLENS)

    def unit(*shape):
        x = torch.randn(*shape, generator=gen)
        return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)

    q = unit(1, total, NUM_K_HEADS, HEAD_K_DIM).to(dtype)
    k = unit(1, total, NUM_K_HEADS, HEAD_K_DIM).to(dtype)
    v = torch.randn(1, total, NUM_V_HEADS, HEAD_V_DIM, generator=gen).to(dtype)
    g = -torch.nn.functional.softplus(torch.randn(1, total, NUM_V_HEADS, generator=gen) - 3)
    beta = torch.randn(1, total, NUM_V_HEADS, generator=gen).sigmoid()
    pool = torch.randn(NUM_SLOTS, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, generator=gen) * 0.1
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], dtype=torch.long)
    indices = torch.tensor(SLOTS, dtype=torch.int32)
    return [t.to(device) for t in (q, k, v, g, beta, pool, cu_seqlens, indices)]


def _fp64_reference(q, k, v, g, beta, pool, indices, cu_seqlens):
    """`(o, pool as the call leaves it)` from the token-at-a-time recurrence in
    fp64, rounded once: `o` to `q.dtype`, the states to the pool's dtype."""
    dt = torch.float64
    repeats = v.shape[2] // q.shape[2]
    scale = q.shape[-1] ** -0.5
    out = torch.zeros(v.shape, dtype=dt, device=q.device)
    ref_pool = pool.clone()
    bounds = cu_seqlens.tolist()
    for n, slot in enumerate(indices.tolist()):
        state = pool[slot].to(dt)
        for t in range(bounds[n], bounds[n + 1]):
            q_t = q[0, t].to(dt).repeat_interleave(repeats, dim=0) * scale
            k_t = k[0, t].to(dt).repeat_interleave(repeats, dim=0)
            state = state * g[0, t].to(dt).exp()[:, None, None]
            u = (v[0, t].to(dt) - (state * k_t[:, None, :]).sum(-1)) * beta[0, t].to(dt)[:, None]
            state = state + u[:, :, None] * k_t[:, None, :]
            out[0, t] = (state * q_t[:, None, :]).sum(-1)
        ref_pool[slot] = state.to(pool.dtype)
    return out.to(q.dtype), ref_pool


def check_chunk_gated_delta_rule(device: str, dtype: torch.dtype) -> None:
    """Packed varlen calls against the fp64 recurrence: dense states in and out,
    as the GDN mixer calls it, then an indexed pool updated in place."""
    q, k, v, g, beta, pool, cu_seqlens, indices = _gdn_operands(device, dtype)
    ref_o, ref_pool = _fp64_reference(q, k, v, g, beta, pool, indices, cu_seqlens)
    pool_before = pool.clone()
    # Every token and head has an output above atol, so a zeroed row cannot pass.
    assert ref_o.float().abs().amax(-1).min() > 2 * ATOL_OUTPUT

    o, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool.index_select(0, indices),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(o.float(), ref_o.float(), rtol=RTOL, atol=ATOL_OUTPUT)
    torch.testing.assert_close(
        final.to(pool.dtype), ref_pool.index_select(0, indices), rtol=RTOL, atol=ATOL_STATE
    )

    o, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        initial_state_indices=indices,
        inplace_indexed_state_update=True,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(o.float(), ref_o.float(), rtol=RTOL, atol=ATOL_OUTPUT)
    torch.testing.assert_close(pool, ref_pool, rtol=RTOL, atol=ATOL_STATE)
    untouched = [s for s in range(NUM_SLOTS) if s not in SLOTS]
    assert torch.equal(pool[untouched], pool_before[untouched])


def check_l2norm_fwd(device: str, dtype: torch.dtype, dim: int) -> None:
    """17 rows and a column tail: dim 96 (of 128) runs l2norm_fwd_kernel in
    16-row blocks, dim 600 (of 1024) runs the row-per-program l2norm_fwd_kernel1."""
    eps = 1e-6
    x = torch.randn(17, dim, generator=torch.Generator().manual_seed(17)).to(device, dtype)
    y = l2norm_fwd(x, eps=eps)
    xf = x.float()
    ref = xf / torch.sqrt(xf.pow(2).sum(-1, keepdim=True) + eps)
    torch.testing.assert_close(y, ref.to(dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_matches_fp64_reference(dtype: torch.dtype) -> None:
    check_chunk_gated_delta_rule("cuda", dtype)


@pytest.mark.parametrize("dim", [96, 600])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_l2norm_fwd_matches_formula(dtype: torch.dtype, dim: int) -> None:
    check_l2norm_fwd("cuda", dtype, dim)
