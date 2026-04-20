import errno
import time

import torch

from ...logger import logger
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

_STALE_FILE_HANDLE_MAX_RETRIES = 5
_STALE_FILE_HANDLE_BASE_DELAY = 1.0


def _call_with_nfs_retry(fn, *args, **kwargs):
    """Retry a flashinfer call on NFS stale file handle errors.

    Flashinfer JIT-compiles kernels on first use and guards compilation with a
    file lock.  When multiple MPI workers race to compile the same module on an
    NFS filesystem, the lock file can become stale (ESTALE / errno 116).  A
    short retry with back-off lets the winning process finish compilation so
    that subsequent attempts find the cached artifact and succeed immediately.
    """
    for attempt in range(_STALE_FILE_HANDLE_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if e.errno == errno.ESTALE and attempt < _STALE_FILE_HANDLE_MAX_RETRIES - 1:
                delay = _STALE_FILE_HANDLE_BASE_DELAY * (attempt + 1)
                logger.warning(
                    "Flashinfer JIT hit NFS stale file handle (attempt "
                    "%d/%d), retrying in %.1fs ...", attempt + 1,
                    _STALE_FILE_HANDLE_MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            raise


if IS_FLASHINFER_AVAILABLE:
    from flashinfer.activation import gelu_tanh_and_mul, silu_and_mul
    from flashinfer.norm import (fused_add_rmsnorm, gemma_fused_add_rmsnorm,
                                 gemma_rmsnorm, rmsnorm)
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return _call_with_nfs_retry(silu_and_mul,
                                    x,
                                    enable_pdl=get_env_enable_pdl())

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    @torch.library.custom_op("trtllm::flashinfer_gelu_tanh_and_mul",
                             mutates_args=())
    def flashinfer_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
        return _call_with_nfs_retry(gelu_tanh_and_mul,
                                    x,
                                    enable_pdl=get_env_enable_pdl())

    @flashinfer_gelu_tanh_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return _call_with_nfs_retry(rmsnorm,
                                    input,
                                    weight,
                                    eps,
                                    enable_pdl=get_env_enable_pdl())

    @flashinfer_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_gemma_rmsnorm",
                             mutates_args=())
    def flashinfer_gemma_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                                 eps: float) -> torch.Tensor:
        return _call_with_nfs_retry(gemma_rmsnorm,
                                    input,
                                    weight,
                                    eps,
                                    enable_pdl=get_env_enable_pdl())

    @flashinfer_gemma_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        _call_with_nfs_retry(fused_add_rmsnorm,
                             input,
                             residual,
                             weight,
                             eps,
                             enable_pdl=get_env_enable_pdl())

    @torch.library.custom_op("trtllm::flashinfer_gemma_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_gemma_fused_add_rmsnorm(input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           eps: float) -> None:
        _call_with_nfs_retry(gemma_fused_add_rmsnorm,
                             input,
                             residual,
                             weight,
                             eps,
                             enable_pdl=get_env_enable_pdl())

    @torch.library.custom_op(
        "trtllm::flashinfer_apply_rope_with_cos_sin_cache_inplace",
        mutates_args=("query", "key"))
    def flashinfer_apply_rope_with_cos_sin_cache_inplace(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> None:
        _call_with_nfs_retry(
            apply_rope_with_cos_sin_cache_inplace,
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

    @flashinfer_apply_rope_with_cos_sin_cache_inplace.register_fake
    def _(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ):
        return
