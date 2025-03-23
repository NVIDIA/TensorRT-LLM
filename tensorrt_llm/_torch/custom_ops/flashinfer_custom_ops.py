import os
import platform
import traceback
from typing import Optional

import torch

from ...logger import logger

IS_FLASHINFER_AVAIABLE = False


def get_env_enable_pdl():
    return os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"


ENABLE_PDL = get_env_enable_pdl()
if ENABLE_PDL:
    logger.info("PDL is enabled")

if platform.system() != "Windows":
    try:
        import flashinfer
        IS_FLASHINFER_AVAIABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )

if IS_FLASHINFER_AVAIABLE:
    from flashinfer.activation import silu_and_mul
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x, enable_pdl=ENABLE_PDL)

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1]

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return rmsnorm(input, weight, eps, enable_pdl=ENABLE_PDL)

    @flashinfer_rmsnorm.register_fake
    def rmsnorm_fake(input: torch.Tensor, weight: torch.Tensor,
                     eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        fused_add_rmsnorm(input, residual, weight, eps, enable_pdl=ENABLE_PDL)

    @torch.library.custom_op("trtllm::flashinfer_apply_rope_inplace",
                             mutates_args=("q", "k"))
    def flashinfer_apply_rope_inplace(
        q: torch.Tensor,
        k: torch.Tensor,
        indptr: torch.Tensor,
        offsets: torch.Tensor,
        rotary_dim: Optional[int] = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 1e4,
    ) -> None:
        flashinfer.apply_rope_inplace(q, k, indptr, offsets, rotary_dim,
                                      interleave, rope_scale, rope_theta)

    @flashinfer_apply_rope_inplace.register_fake
    def apply_rope_inplace_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        indptr: torch.Tensor,
        offsets: torch.Tensor,
        rotary_dim: Optional[int] = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 1e4,
    ):
        return
