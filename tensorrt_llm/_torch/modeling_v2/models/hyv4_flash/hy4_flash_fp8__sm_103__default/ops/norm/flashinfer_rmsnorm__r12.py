# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm via flashinfer; local variant of the catalog flashinfer_rmsnorm entry with this target's cells."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class FlashinferRmsnorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * weight`, new tensor out."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: q_a_layernorm over the q_lora rank (q_a_proj 3072->1024)",
            spec=dict(shape=(1, 1024), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="64-row end: q_a_layernorm at the q_lora rank",
            spec=dict(shape=(64, 1024), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: q_a_layernorm over 5120 rows",
            spec=dict(shape=(5120, 1024), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: kv_a_layernorm over the kv_lora rank",
            spec=dict(shape=(1, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: kv_a_layernorm over 5120 rows",
            spec=dict(shape=(5120, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: post_proj_norm over 32 sparse heads x v_head 256",
            spec=dict(shape=(1, 8192), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: post_proj_norm at 8192",
            spec=dict(shape=(5120, 8192), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: post_proj_norm over 64 SWA heads x v_head 256",
            spec=dict(shape=(1, 16384), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: post_proj_norm at 16384, the widest row",
            spec=dict(shape=(5120, 16384), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: model.norm / combine_output_norm at hidden 3072",
            spec=dict(shape=(1, 3072), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: model.norm at hidden 3072",
            spec=dict(shape=(5120, 3072), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="engram: 16 per-head norms over 96 at a 5120-token chunk, the most rows",
            spec=dict(shape=(81920, 96), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="indexer: token_key_norm/key_norm over 5120 context rows",
            spec=dict(shape=(5120, 128), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="indexer: key_norm over the 5120/4 pooled groups",
            spec=dict(shape=(1280, 128), dtype=torch.bfloat16, eps=1e-6),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`). The CUDA-JIT path
    (`FLASHINFER_USE_CUDA_NORM=1`) is not certified.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_rmsnorm(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """fp32-accumulated; the kernel reduces in fp32 and rounds once."""
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        # The op takes fp8 and returns a plausible fp8 rmsnorm instead of raising.
        assert x.dtype is not torch.float8_e4m3fn, "dequantize float8_e4m3fn before normalizing"
        assert x.stride(-1) == 1, (
            f"the normalized dim must be contiguous; got stride {x.stride(-1)}"
        )


flashinfer_rmsnorm = FlashinferRmsnorm()
