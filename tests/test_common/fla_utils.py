# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed access to FLA reference kernels for parity tests."""

from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    import torch


def fused_recurrent_kda_reference(
    *args: Any, **kwargs: Any
) -> tuple["torch.Tensor", "torch.Tensor | None"]:
    """Call the KDA reference through FLA's union-typed ``input_guard`` wrapper."""
    from fla.ops.kda import fused_recurrent_kda

    reference = cast(
        Callable[..., tuple["torch.Tensor", "torch.Tensor | None"]], fused_recurrent_kda
    )
    return reference(*args, **kwargs)
