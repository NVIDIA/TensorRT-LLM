# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class AttentionTensorLayout(str, Enum):
    """Attention tensor axis order, with batch first and head dimension last."""

    NHD = "NHD"  # [B, S, H, D]
    HND = "HND"  # [B, H, S, D]
