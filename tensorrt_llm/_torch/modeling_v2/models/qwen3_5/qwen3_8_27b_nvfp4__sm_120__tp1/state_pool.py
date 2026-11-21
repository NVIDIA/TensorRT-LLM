# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage geometry for the target's int16 row codes and fp32 row scales."""

import torch

from tensorrt_llm._torch.pyexecutor.config_utils import SSMStatePoolGeometry


def q8_state_rows(v: int, k: int) -> int:
    """K-byte rows per head: int16 codes followed by one fp32 scale per value row."""
    assert (v * 4) % k == 0, f"V={v} fp32 scales must fill whole K={k} rows"
    return 2 * v + (v * 4) // k


def state_pool_geometry(v: int, k: int) -> SSMStatePoolGeometry:
    """Declare the byte tensor consumed by the target's recurrent kernels."""
    return SSMStatePoolGeometry(
        layout="gdn_int16_row_codes_fp32_scales",
        dtype=torch.int8,
        shape=(q8_state_rows(v, k), k),
    )
