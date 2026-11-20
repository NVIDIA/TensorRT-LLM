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
"""Source markers for scripts/check_mock_interfaces.py; no runtime validation.

Decorate named test doubles with ``@mock_target("module:ClassOrFunction")``.
Use ``verify_instance(value, "module:Class")`` after initializing a namespace or
an instance created with ``object.__new__``. Both markers preserve identity and
never import the target. The standalone checker reads them without running tests.
"""

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

__all__ = ["mock_target", "verify_instance"]


def mock_target(target: str, *, bound: bool = False) -> Callable[[_T], _T]:
    """Mark an interface; bound=True describes a replacement on an instance."""

    def identity(value: _T) -> _T:
        return value

    return identity


def verify_instance(
    value: _T,
    target: str,
    *,
    allow_extra: bool = False,
    members_only: bool = False,
    members: tuple[str, ...] | None = None,
) -> _T:
    """Check instance structure, direct members, or explicitly selected members."""
    return value
