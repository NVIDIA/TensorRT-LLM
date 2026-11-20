# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable

import torch

from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn_backend import (
    sol_attn_ineligible_reason,
)
from tensorrt_llm.logger import logger


@torch.compiler.disable
def run_sol_attn_bthd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    thresh_type: str,
    dense_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    reason = sol_attn_ineligible_reason(q)
    if reason is None and (k.shape != q.shape or v.shape != q.shape):
        reason = f"k/v shape must match q {tuple(q.shape)}"
    if reason is None and (k.dtype != q.dtype or v.dtype != q.dtype):
        reason = f"k/v dtype must match q {q.dtype}"
    if reason is not None:
        if os.environ.get("TRTLLM_SOL_ATTN_STRICT") == "1":
            raise RuntimeError(f"[sol-attn] cannot run the CuTe kernel: {reason}")
        logger.warning_once(
            f"[sol-attn] falling back to dense attention: {reason}. Set "
            "TRTLLM_SOL_ATTN_STRICT=1 to raise instead.",
            key=("sol_attn_ineligible", reason),
        )
        return dense_fn(q, k, v)

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn import sol_attn

    return sol_attn(q, k, v, tau=tau, thresh_type=thresh_type)
