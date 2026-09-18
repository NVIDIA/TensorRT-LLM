# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exported MegaMoE KernelClass entry points."""

from .kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel as BlackwellInferenceMegaMoE,
)
from .kernel_src.rubin.inference.local_mega.block_scaled_swap_ab_local_mega_moe_kernel import (
    BlockScaledSwapAbLocalMegaMoeKernel as RubinInferenceLocalMegaMoE,
)
from .kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel as RubinInferenceMegaMoE,
)
from .kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized import (
    BlockScaledSwapAbGenphaseMoeKernel as RubinInferenceGenphaseMegaMoE,
)

__all__ = [
    "BlackwellInferenceMegaMoE",
    "RubinInferenceGenphaseMegaMoE",
    "RubinInferenceLocalMegaMoE",
    "RubinInferenceMegaMoE",
]
