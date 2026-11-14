# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's shared native ops (``csrc/``), ``torch.ops.mv2_minimax_m3``.

``family_ops()`` opens ``libth_modeling_v2_minimax_m3.so`` on first use and returns its shared ops namespace; later
calls are a dictionary lookup. The lookup is deferred to the call so that importing a family module does not need the
library (CPU checks import family modules without it). It raises ``NoTarget`` when this installation does not carry
the family library and ``NativeLoadError`` when the library is broken (``_native.load_family_ops``).
"""

from typing import Any

from ..._native import load_family_ops


def family_ops() -> Any:
    """``torch.ops.mv2_minimax_m3``: the ops of the family's shared native code."""
    return load_family_ops(__name__)
