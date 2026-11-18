# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime global scales and their independently inferred argument kinds."""

from __future__ import annotations

import math
import numbers
import struct
from typing import TypeAlias, overload

import torch

GlobalScale: TypeAlias = float | torch.Tensor


def normalize_scale(value: object, name: str, device: torch.device) -> GlobalScale:
    """Validate a host FP32 value or a device scalar without reading device data.

    Tensor values must remain positive and finite on every launch and replay.
    Only their metadata is validated; reading their contents would synchronize
    the host and would prevent CUDA graph capture.
    """
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cuda"
            or value.device != device
            or value.dtype != torch.float32
            or value.numel() != 1
            or value.ndim > 1
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA float32 tensor of shape () or (1,) "
                f"on {device}"
            )
        return value.view(1) if value.ndim == 0 else value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a positive Python scalar or CUDA float32 tensor"
        )
    try:
        host_value = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive Python scalar") from error
    if not math.isfinite(host_value) or host_value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        rounded = struct.unpack("=f", struct.pack("=f", host_value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"{name} must be representable as a positive float32"
        ) from error
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError(f"{name} must be representable as a positive float32")
    return rounded


def resolve_scale(
    value: GlobalScale | None,
    planned: GlobalScale,
    name: str,
    device: torch.device,
) -> GlobalScale:
    """Resolve a run override while preserving the eagerly compiled signature."""
    if value is None:
        return planned
    resolved = normalize_scale(value, name, device)
    if isinstance(resolved, torch.Tensor) != isinstance(planned, torch.Tensor):
        raise TypeError(
            f"{name} must have the same scalar/tensor type as plan(); "
            "replan before changing scale argument types"
        )
    return resolved


@overload
def scale_types(first: GlobalScale, second: GlobalScale, /) -> tuple[bool, bool]: ...


@overload
def scale_types(
    first: GlobalScale, second: GlobalScale, third: GlobalScale, /
) -> tuple[bool, bool, bool]: ...


def scale_types(*values: GlobalScale) -> tuple[bool, ...]:
    """Return a cacheable ABI signature that contains no values or tensor owners."""
    return tuple(isinstance(value, torch.Tensor) for value in values)


def fake_scale(is_tensor: bool):
    """Describe one runtime FP32 argument for ahead-of-time specialization."""
    import cutlass
    import cutlass.cute as cute

    if is_tensor:
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), stride_order=(0,), assumed_align=4
        )
    return cutlass.Float32(1.0)
