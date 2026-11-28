# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum


class KeyPadding(enum.Enum):
    """The padded key rows an attention op can exclude.

    ``NONE``: the op attends every key row. ``PREFIX``: each sample's tokens come first and its
    padding follows them. ``MASK``: any boolean key mask.
    """

    NONE = "none"
    PREFIX = "prefix"
    MASK = "mask"
