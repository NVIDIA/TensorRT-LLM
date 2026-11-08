# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorrt_llm.bindings as _bindings


class KVCacheType(str, Enum):
    """Python enum wrapper for KVCacheType.

    This is a pure Python enum that mirrors the C++ KVCacheType enum exposed
    through pybind11.
    """
    CONTINUOUS = "continuous"
    PAGED = "paged"
    DISABLED = "disabled"

    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive string values to be converted to enum members."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None

    def to_cpp(self) -> '_bindings.KVCacheType':
        import tensorrt_llm.bindings as _bindings
        return getattr(_bindings.KVCacheType, self.name)

    @classmethod
    def from_cpp(cls, cpp_enum) -> 'KVCacheType':
        # C++ enum's __str__ returns "KVCacheType.PAGED", extract the name
        name = str(cpp_enum).split('.')[-1]
        return cls(name)
