# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for Qwen-Image 2.1 normalization layers.

The canonical implementations live in ``transformer_qwen_image_21.py`` so the
checkpoint-facing module keeps the official Diffusers state-dict key layout.
This module exists only as a stable import surface for focused parity tests and
future refactors; it must not introduce a second copy of the layer logic.
"""

from .transformer_qwen_image_21 import QwenImage21RMSNorm, QwenImage21ZeroCenterRMSNorm

__all__ = ["QwenImage21RMSNorm", "QwenImage21ZeroCenterRMSNorm"]
