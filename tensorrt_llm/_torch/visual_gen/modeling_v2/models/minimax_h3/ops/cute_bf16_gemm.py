# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.visual_gen.runtime import get_sm_version


class CuteBf16Gemm(OpWrapper):
    """BF16 A[M,K] @ W[N,K].T with BF16 output."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell(why="Contiguous BF16 activations and weights", spec=dict(m=32, n=128, k=256)),)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = x.new_empty((x.shape[0], weight.shape[0]))
        if get_sm_version() == 107:
            torch.ops.trtllm.cute_dsl_bf16_gemm_rubin(x, weight, out)
        else:
            torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell(x, weight, out)
        return out

    def reference(self, x, weight):
        with true_fp32_matmul():
            return (x.float() @ weight.float().T).to(x.dtype)

    def is_valid(self, x, weight):
        assert x.dtype == weight.dtype == torch.bfloat16
        assert x.ndim == weight.ndim == 2 and x.shape[1] == weight.shape[1]
        assert x.is_contiguous() and weight.is_contiguous()


cute_bf16_gemm = CuteBf16Gemm()
