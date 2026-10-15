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
"""Field metadata shared by the Python classes that generate native structs.

This lives on its own because every schema class needs it, and those classes sit
in modules that already import one another.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# Namespaced so the native schema metadata cannot collide with other users of
# dataclasses.field(metadata=...).
CPP_METADATA_KEY = "fmha.cpp"


@dataclass(frozen=True)
class CppMetadata:
    """Controls a field's representation in the native FmhaParams struct.

    The generator reads the schema's source, not this object, so the value here is
    kept only for introspection. See cpp_metadata() for what ``dtype`` means.
    """

    dtype: object


def cpp_metadata(*, dtype: object, default: object = None) -> dataclasses.Field[object]:
    """Customize a dataclass field's representation in the native struct.

    The annotation gives the shape -- tensor, optional, set, scalar, or a named
    type -- and ``dtype`` gives its fixed element type as a ``torch`` dtype.

    The default is ``None`` unless explicitly supplied. Scalars annotated as
    bool, int, float, or their Optional forms do not need this helper: their
    native types are bool, std::int64_t, and double, respectively.

    A tensor without this helper is a native field with no generated getter,
    for example when its pointer type is supplied by C++ dtype dispatch or its
    view needs offsets. These getters are handwritten in attentionOp.h.
    Registered native structs and supported named types are recognized by their
    annotations and need no marker either. Defaults and default factories remain
    Python-owned.
    """
    return dataclasses.field(default=default, metadata={CPP_METADATA_KEY: CppMetadata(dtype)})
