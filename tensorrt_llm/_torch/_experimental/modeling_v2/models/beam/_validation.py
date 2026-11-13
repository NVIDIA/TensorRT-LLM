# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side shape and device admission for directly constructed Beam components."""

import torch


def explicit_device(device: str | torch.device | None) -> torch.device:
    """Resolve an unindexed CUDA device against the current device once at construction."""
    result = torch.device("cuda" if device is None else device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def check_positions(positions: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> None:
    """Require int32/int64 position metadata of the exact shape without reading device values."""
    if (
        not isinstance(positions, torch.Tensor)
        or positions.dtype not in (torch.int32, torch.int64)
        or tuple(positions.shape) != shape
        or positions.device != device
    ):
        raise ValueError(f"position_ids must be an int32/int64 tensor of shape {shape} on {device}")


def metadata_values(values, count: int, name: str) -> list[int]:
    """Read a complete prefix of a host metadata array; never silently truncate it."""
    if len(values) < count:
        raise ValueError(f"{name} needs {count} entries, got {len(values)}")
    prefix = values[:count]
    return prefix.tolist() if hasattr(prefix, "tolist") else list(prefix)
