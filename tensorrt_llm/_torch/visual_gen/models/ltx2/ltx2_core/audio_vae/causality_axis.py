# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from enum import Enum


class CausalityAxis(Enum):
    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"
