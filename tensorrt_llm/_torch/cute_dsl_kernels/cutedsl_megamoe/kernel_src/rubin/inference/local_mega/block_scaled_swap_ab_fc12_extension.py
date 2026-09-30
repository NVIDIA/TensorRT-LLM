# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local MegaMoE shim for the compatible Rubin MegaMoE FC12 extension."""

# <<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>
from ..mega.block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension, TensorRole

__all__ = ["BlockScaledSwapAbFc12Extension", "TensorRole"]
