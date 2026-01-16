# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/wy_fast.py

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices
from tensorrt_llm._torch.modules.fla.solve_tril import solve_tril
from tensorrt_llm._torch.modules.fla.utils import check_shared_mem

from .utils import autotune_cache_kwargs


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False)
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A.to(b_kb.dtype), b_kb, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def prepare_wy_repr_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
    )
    return w, u, A


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k,
        v,
        beta,
        w,
        u,
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u
