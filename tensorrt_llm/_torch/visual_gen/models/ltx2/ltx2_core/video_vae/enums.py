# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from enum import Enum


class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
