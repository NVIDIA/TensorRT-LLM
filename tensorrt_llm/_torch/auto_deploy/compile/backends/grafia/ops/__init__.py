# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grafia op-specific backend adapter helpers."""

from functools import cache

from .base import GrafiaOpLowering, build_op_lowering_map


@cache
def get_grafia_op_lowerings_by_target() -> dict[object, GrafiaOpLowering]:
    from .....custom_ops.normalization import rms_norm as _rms_norm_ops  # noqa: F401
    from .rmsnorm import RMSNORM_LOWERING

    return build_op_lowering_map(RMSNORM_LOWERING)


__all__ = [
    "GrafiaOpLowering",
    "build_op_lowering_map",
    "get_grafia_op_lowerings_by_target",
]
