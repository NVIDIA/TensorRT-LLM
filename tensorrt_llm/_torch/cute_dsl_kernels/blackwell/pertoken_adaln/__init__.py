# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fused per-token AdaLN modulate (CuTe DSL) for the WAN per-token
# (temb.ndim == 4) path. The ops take raw bf16 temb chunk views plus the
# [D] fp32 scale_shift_table rows and fuse the fp32 table+chunk add inline,
# avoiding materialized fp32 [B, S, D] modulator tensors.
#
# Custom ops registered under the ``trtllm::`` torch.library namespace.

from .pertoken_adaln import fused_pertoken_adaln, fused_pertoken_adaln_residual  # noqa: F401

__all__ = [
    "fused_pertoken_adaln",
    "fused_pertoken_adaln_residual",
]
