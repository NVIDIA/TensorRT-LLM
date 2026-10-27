# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Visual Generation Modules

This module provides modular neural network components for visual generation models.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attention.attention import Attention, QKVMode

# Compatibility re-export for callers of the retired location; the canonical
# home is ``visual_gen.attention.attention``. Resolved lazily because that
# module imports ``modules.rms_norm``, so an eager import here would form a cycle.
__all__ = [
    "Attention",
    "QKVMode",
]


def __getattr__(name):
    if name in __all__:
        try:
            from ..attention import attention
        except AttributeError as e:
            # An AttributeError escaping the module is a bug inside it, not a
            # missing name here. Raised as is, hasattr() and getattr() with a
            # default would hide it, and ``from ... import`` would run the
            # module twice (losing the cause unless the name is a submodule).
            raise ImportError(
                f"cannot import {name!r} from {__name__!r}: importing "
                f"tensorrt_llm._torch.visual_gen.attention.attention raised {e!r}",
                name=__name__,
            ) from e

        return getattr(attention, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
