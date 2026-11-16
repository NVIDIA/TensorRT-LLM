# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Compatibility shim for ``tensorrt_llm.serialization``.

The implementation moved to ``tensorrt_llm.llmapi.serialization`` in T01
(TRTLLM-14831) under Epic TRTLLM-14558.

This module re-exports the implementation unchanged and defines nothing of its
own, so the objects reachable here are the *same* objects as at the canonical
path: ``isinstance`` still matches and pre-migration pickles still load.

Removal is tracked by the removal ticket T25 (TRTLLM-14855) delivers; the compatibility window spans at least
one release (Epic decision D4 (b)).  Do not add definitions, do not rewrite
``__module__``, and do not replace the explicit list below with ``import *`` --
Epic §0.4 and §6.1 explain why each of those quietly breaks the backstop.
"""

import warnings

from tensorrt_llm.llmapi.serialization import (  # noqa: F401
    BASE_EXAMPLE_CLASSES,
    Unpickler,
    dump,
    dumps,
    load,
    loads,
    register_approved_class,
)

warnings.warn(
    "tensorrt_llm.serialization has moved to "
    "tensorrt_llm.llmapi.serialization. The old path still works during "
    "the compatibility window and will be removed by the removal ticket "
    "T25 (TRTLLM-14855) delivers.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BASE_EXAMPLE_CLASSES",
    "Unpickler",
    "dump",
    "dumps",
    "load",
    "loads",
    "register_approved_class",
]
