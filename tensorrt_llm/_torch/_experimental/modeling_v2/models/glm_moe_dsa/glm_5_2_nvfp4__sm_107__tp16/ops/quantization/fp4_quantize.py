# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization for the GLM-5.2 routed-expert input on SM107."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize as _shared_fp4_quantize,
)

__all__ = ["fp4_quantize"]


class GlmFp4Quantize(OpWrapper):
    """Quantize BF16 activations to packed e2m1 data and e4m3 scale bytes.

    The certified GLM input is [8, 6144], with one e4m3 scale per 16 elements
    and linear scale layout. The callable preserves the shared op's arguments;
    its arithmetic reference and input guards are the shared NVFP4 contract.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="glm-5.2 verify step: 8 rows at hidden 6144, linear -- its routed-expert input",
            spec=dict(shape=(8, 6144), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
    )

    note = """
    The target uses sf_vec_size=16, sf_use_ue8m0=False and linear scale layout.
    Other quantization modes and layouts are not certified by this entry.
    """

    def __call__(
        self,
        input: torch.Tensor,
        global_scale: torch.Tensor | None,
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp4_quantize(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )

    def reference(
        self,
        input: torch.Tensor,
        global_scale: torch.Tensor | None,
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the shared NVFP4 arithmetic in the requested scale layout."""
        return _shared_fp4_quantize.reference(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )

    def is_valid(
        self,
        input: torch.Tensor,
        global_scale: torch.Tensor | None,
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> None:
        """Apply the shared guards for scalar scale, whole blocks and contiguity."""
        _shared_fp4_quantize.is_valid(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )


fp4_quantize = GlmFp4Quantize()
