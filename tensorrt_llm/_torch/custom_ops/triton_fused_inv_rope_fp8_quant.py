# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
``trtllm::fused_inv_rope_fp8_quant_vllm_port`` — fake/meta registration.

The schema and CUDA implementation now live in
``cpp/tensorrt_llm/{kernels,thop}/inverseRopeFp8QuantKernel.{cu,h}`` and
``inverseRopeFp8QuantOp.cpp`` (registered via ``TORCH_LIBRARY_FRAGMENT(trtllm, ...)``).
This module only registers a Python-side fake/meta function so that
``torch.compile`` / dynamo can trace the op without running it on real
tensors. See ``tensorrt_llm/_torch/modules/attention.py:_deepseek_v4_o_proj``
for the production call site.

Historical note: this op was originally a Triton kernel ported from vLLM
(``deepseek_v4_ops/fused_inv_rope_fp8_quant``). The Triton implementation
was replaced by a hand-tuned CUDA kernel that covers DSv4 head_dim ∈
{128, 256, 384, 512} with software-pipelined load/compute interleaving;
see the .cu file for the design notes.
"""

from __future__ import annotations

import torch


def _tma_aligned_size(x: int, tma_align_size_in_elems: int = 4) -> int:
    """Match the BMM dequant consumer's m-dim stride: ``pad_up(m, 4)``."""
    return (x + tma_align_size_in_elems - 1) // tma_align_size_in_elems * tma_align_size_in_elems


def _fused_inv_rope_fp8_quant_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, head_dim = o.shape
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    tma_aligned_T = _tma_aligned_size(num_tokens, 4)
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_buf = torch.empty(
        (n_groups, num_scale_blocks, tma_aligned_T),
        dtype=torch.float32,
        device=o.device,
    )
    return fp8_buf, scale_buf


# The op schema + CUDA impl are registered from the C++ side
# (TORCH_LIBRARY_FRAGMENT in inverseRopeFp8QuantOp.cpp), so Python only
# attaches the fake.
torch.library.register_fake(
    "trtllm::fused_inv_rope_fp8_quant_vllm_port",
)(_fused_inv_rope_fp8_quant_fake)
