# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's tests run under its served profile (``ull16/generation``).

``features.select`` refuses once another module of the package is imported, and the entry tests
under ``ops/`` import package modules at collection. Selecting here, before any of them, keeps every
order valid. A process that imported the package earlier (another directory's tests) keeps what it
selected.
"""

import sys

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.common import (  # noqa: E501
    rubin_ull_features as features,
)

_PACKAGE = features._PACKAGE
if features.selected() is None and not any(
    name.startswith(_PACKAGE + ".") and name != features.__name__ for name in sys.modules
):
    features.select(features.ULL16_GENERATION)
