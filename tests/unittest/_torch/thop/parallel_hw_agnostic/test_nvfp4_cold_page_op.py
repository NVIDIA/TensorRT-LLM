# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-only ABI checks for NVFP4 cold-page custom ops."""

import pytest
import torch

pytestmark = pytest.mark.cpu_only


def _encode(wide: torch.Tensor, integers: torch.Tensor, scales: torch.Tensor) -> None:
    torch.ops.trtllm.nvfp4_cold_page_encode(
        wide,
        integers,
        scales,
        1,
        2,
        16,
        0,
        0,
        0,
        0,
        0,
    )


def test_rejects_malformed_metadata_tensors() -> None:
    wide = torch.zeros((256, 6), dtype=torch.int64)
    integers = torch.zeros((256, 5), dtype=torch.int32)
    scales = torch.zeros((256, 4), dtype=torch.float32)

    invalid = (
        (wide.to(torch.int32), integers, scales, "wide metadata"),
        (torch.zeros((256, 12), dtype=torch.int64)[:, ::2], integers, scales, "wide metadata"),
        (wide, integers[:, :4], scales, "integer metadata"),
        (wide, integers, scales.to(torch.float64), "scale metadata"),
    )
    for bad_wide, bad_integers, bad_scales, message in invalid:
        with pytest.raises(RuntimeError, match=message):
            _encode(bad_wide, bad_integers, bad_scales)


def test_rejects_invalid_launch_scalars() -> None:
    wide = torch.zeros((256, 6), dtype=torch.int64)
    integers = torch.zeros((256, 5), dtype=torch.int32)
    scales = torch.zeros((256, 4), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="buffer count"):
        torch.ops.trtllm.nvfp4_cold_page_encode(wide, integers, scales, 0, 2, 16, 0, 0, 0, 0, 0)
