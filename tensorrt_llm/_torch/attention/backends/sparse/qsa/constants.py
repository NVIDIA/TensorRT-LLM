# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape and dtype contracts shared by QSA runtime components.

Model geometry remains in :mod:`params`; implementation-local tuning values
remain with their owning module.
"""

from typing import Final

import torch

from tensorrt_llm.bindings import DataType

QSA_POSITION_COORDINATE_AXES: Final = 3
QSA_INDEX_KV_HEADS: Final = 1
QSA_MAIN_KV_ROLES: Final = 2
QSA_KEY_ROLE_INDEX: Final = 0
QSA_VALUE_ROLE_INDEX: Final = 1
QSA_COS_SIN_CACHE_COMPONENTS: Final = 2
# Fused index kernels require index_head_dim == rotary_dim * this ratio.
QSA_INDEX_HEAD_TO_ROTARY_WIDTH_RATIO: Final = 2

# These side-cache dtypes are kernel contracts, not checkpoint weight dtypes.
QSA_INDEX_K_CACHE_DTYPE: Final = torch.bfloat16
QSA_POSITION_CACHE_DTYPE: Final = torch.int32
# The native QSA sparse path reads these main-cache formats directly. Other
# formats remain owned by the regular attention backend and its scale pages.
QSA_SPARSE_KV_CACHE_DTYPES: Final = frozenset((DataType.HALF, DataType.BF16, DataType.FP8))


__all__ = [
    "QSA_COS_SIN_CACHE_COMPONENTS",
    "QSA_INDEX_HEAD_TO_ROTARY_WIDTH_RATIO",
    "QSA_INDEX_K_CACHE_DTYPE",
    "QSA_INDEX_KV_HEADS",
    "QSA_KEY_ROLE_INDEX",
    "QSA_MAIN_KV_ROLES",
    "QSA_POSITION_CACHE_DTYPE",
    "QSA_POSITION_COORDINATE_AXES",
    "QSA_SPARSE_KV_CACHE_DTYPES",
    "QSA_VALUE_ROLE_INDEX",
]
