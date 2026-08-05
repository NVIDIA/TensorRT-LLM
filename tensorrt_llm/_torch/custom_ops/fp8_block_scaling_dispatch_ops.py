# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-side registrations for the native FP8 block-scaling dispatcher."""

import torch

from tensorrt_llm import deep_gemm


def register_fp8_block_scaling_dispatch_ops() -> None:
    """Configure runtime identity and register fake-tensor behavior."""
    torch.ops.trtllm.fp8_block_scaling_gemm_configure_dispatch(deep_gemm.__version__)

    @torch.library.register_fake("trtllm::fp8_block_scaling_gemm")
    def _fake(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        tune_max_num_tokens: int = 4096,
    ) -> torch.Tensor:
        del a_scale, b_scale, tune_max_num_tokens
        return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)
