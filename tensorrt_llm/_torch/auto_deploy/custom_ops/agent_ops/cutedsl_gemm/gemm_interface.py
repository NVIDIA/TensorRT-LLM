from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor

from .cute_dsl_utils import get_device_capacity
from .gemm import gemm as gemm_kernel
from .gemm_act import gemm_act as gemm_act_kernel


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int
    pingpong: bool
    max_swizzle_size: int = 8
    swap_ab: bool = False


def default_config(device: torch.device) -> GemmConfig:
    major, _ = get_device_capacity(device)
    if major != 9:
        raise ValueError("Only SM90 (H100) is supported")
    return GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)


def gemm_tuned(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    C: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    batch_idx_permute: Optional[Tensor] = None,
    add_to_output: bool = False,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if dynamic_scheduler:
        raise NotImplementedError("dynamic scheduler is not supported in the minimal kernel")
    if config is None:
        config = default_config(A.device)
    if config.swap_ab:
        raise NotImplementedError("swap_ab configurations are not supported")

    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    if gather_A:
        assert varlen, "gather_A requires variable-length sequences"
        assert config.cluster_n == 1, "gather_A requires cluster_n=1"
    if varlen_m:
        assert not config.swap_ab, "varlen_m is incompatible with swap_ab"

    if A.ndim == 2 and not varlen:
        A = A.unsqueeze(0)
    B = B.mT
    if B.ndim == 2 and not varlen_k:
        B = B.unsqueeze(0)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)
    if out.ndim == 2 and not varlen_m:
        out = out.unsqueeze(0)
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)

    batch_size = B.shape[0] if not varlen_k else cu_seqlens_k.shape[0] - 1
    if varlen_m:
        total_m = A_idx.shape[0] if A_idx is not None else A.shape[0]
        out_shape = (total_m, B.shape[-2])
    else:
        out_shape = (batch_size, A.shape[-2], B.shape[-2])
    assert out.shape == out_shape, (
        f"out shape mismatch: expected {out_shape}, got {tuple(out.shape)}"
    )

    gemm_kernel(
        A,
        B,
        out,
        C=C,
        tile_count_semaphore=None,
        tile_M=config.tile_m,
        tile_N=config.tile_n,
        cluster_M=config.cluster_m,
        cluster_N=config.cluster_n,
        pingpong=config.pingpong,
        persistent=True,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias,
        colvec_bias=None,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
    )


def gemm_act_tuned(
    A: Tensor,
    B: Tensor,
    preact_out: Optional[Tensor],
    postact_out: Tensor,
    C: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    activation: Literal[None, "relu_sq"] = None,
    cu_seqlens_m: Optional[Tensor] = None,
    A_idx: Optional[Tensor] = None,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if dynamic_scheduler:
        raise NotImplementedError("dynamic scheduler is not supported in the minimal kernel")
    if activation not in (None, "relu_sq"):
        raise ValueError(f"Unsupported activation {activation}")
    if config is None:
        config = default_config(A.device)
    if config.swap_ab:
        raise NotImplementedError("swap_ab configurations are not supported")

    varlen_m = cu_seqlens_m is not None
    if A.ndim == 2 and not varlen_m:
        A = A.unsqueeze(0)
    B = B.mT
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if C is not None and C.ndim == 2 and not varlen_m:
        C = C.unsqueeze(0)
    if preact_out is not None and preact_out.ndim == 2 and not varlen_m:
        D = preact_out.unsqueeze(0)
    else:
        D = preact_out
    if postact_out.ndim == 2 and not varlen_m:
        PostAct = postact_out.unsqueeze(0)
    else:
        PostAct = postact_out
    if bias is not None and bias.ndim == 1:
        bias = bias.unsqueeze(0)

    gemm_act_kernel(
        A,
        B,
        D,
        C,
        PostAct,
        tile_count_semaphore=None,
        activation=activation,
        tile_M=config.tile_m,
        tile_N=config.tile_n,
        cluster_M=config.cluster_m,
        cluster_N=config.cluster_n,
        pingpong=config.pingpong,
        persistent=True,
        max_swizzle_size=config.max_swizzle_size,
        rowvec_bias=bias,
        colvec_bias=None,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
