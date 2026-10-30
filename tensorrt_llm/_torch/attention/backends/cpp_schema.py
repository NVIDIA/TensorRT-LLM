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

from dataclasses import dataclass, field
from typing import Any, Mapping

# Namespaced so the native schema metadata cannot collide with other users of
# dataclasses.field(metadata=...).
CPP_METADATA_KEY = "fmha.cpp"


@dataclass(frozen=True)
class CppMetadata:
    """Controls a field's representation in the native FmhaParams struct.

    The generator reads the schema's source; nothing at runtime consults this,
    the native reader is generated C++.
    """

    dtype: object = None


def cpp_metadata(*, dtype: object = None) -> Mapping[str, Any]:
    """Metadata for ``dataclasses.field`` customizing a field's native representation.

    Schema classes use it through ``cpp_field``; this helper only carries the native
    options and never touches the dataclass default.

    The annotation gives the shape -- tensor, optional, set, scalar, enum, or a
    nested schema class -- and ``dtype`` gives a tensor's fixed element type as a
    ``torch`` dtype, which produces a typed pointer accessor. A tensor without a
    dtype is a native field with no generated getter, for example when its pointer
    type is supplied by C++ dtype dispatch or its view needs offsets; those getters
    are handwritten in attentionOp.h. Scalars, enums, nested schema classes and
    plain tensors need no metadata at all. The native field keeps the Python name, and an
    ``Optional`` annotation is what makes it ``std::optional`` natively.
    """
    return {CPP_METADATA_KEY: CppMetadata(dtype)}


def cpp_field(*, default: Any = None, dtype: object = None) -> Any:
    """``dataclasses.field`` with native metadata attached.

    Use as ``x: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)``. ``default`` is the
    ordinary dataclass default and stays Python-owned; it is ``None`` because almost every
    native-visible field is an optional tensor, and a field with another default passes
    it explicitly.
    """
    return field(default=default, metadata=cpp_metadata(dtype=dtype))
