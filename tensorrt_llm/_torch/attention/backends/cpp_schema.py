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
_UNSET = object()


@dataclass(frozen=True)
class CppMetadata:
    """Marks a field as part of the native FmhaParams struct.

    The generator reads the schema's source, not this object, so the value here is
    kept only for introspection. See cpp_metadata() for what ``ctype`` means.
    """

    ctype: object = _UNSET


def cpp_metadata(*, default: object, ctype: object = _UNSET) -> dataclasses.Field[object]:
    """Mark a dataclass field as part of the native FmhaParams struct.

    The annotation gives the shape -- tensor, optional, set, scalar, or a named
    type -- and ``ctype`` gives the element type within it:

    * a ``torch`` dtype: a tensor's or a set's element type;
    * omitted: read the type off the annotation. On a tensor this means the element
      type is whatever dtype the op runs at, so the getter is templated;
    * ``None``: generate no getter. Use it where the native view differs from the
      tensor's dtype, and hand-write the getter in attentionOp.h.
    """
    return dataclasses.field(default=default, metadata={CPP_METADATA_KEY: CppMetadata(ctype)})
