# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tensor-parallel split direction of a projection.

``COLUMN`` splits the output features (weight dim 0) over the TP group, ``ROW`` the input
features (weight dim 1). The members are strings (``"column"`` / ``"row"``), so they compare
and hash equal to the engine's own ``tp_mode`` values (``LMHead.tp_mode``), which are the same
strings.
"""

import enum

__all__ = ["TensorParallelMode"]


class TensorParallelMode(str, enum.Enum):
    COLUMN = "column"
    ROW = "row"

    @classmethod
    def split_dim(cls, mode) -> int:
        """The weight dim ``mode`` splits: 1 for ``ROW``, else 0."""
        return 1 if mode == cls.ROW else 0

    @classmethod
    def flip(cls, mode) -> "TensorParallelMode":
        """The orthogonal direction (how a per-channel activation scale of a ``mode`` weight is
        split)."""
        return cls.ROW if mode == cls.COLUMN else cls.COLUMN
