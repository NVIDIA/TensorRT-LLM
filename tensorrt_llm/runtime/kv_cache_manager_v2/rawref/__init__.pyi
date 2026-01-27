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

from typing import Generic, Optional, TypeVar

T = TypeVar("T")

class ReferenceType(Generic[T]):
    """A mutable reference holder that stores an object ID.

    This class holds a reference to an object via its ID and allows
    dereferencing it. The reference can be invalidated.

    Like weakref.ref, but stores raw object IDs instead of proper weak references.

    Implements a singleton pattern: calling ref(obj) multiple times returns the
    same reference if obj.__rawref__ exists and is valid.
    """

    @property
    def is_valid(self) -> bool:
        """Check if the reference is still valid (read-only)."""
        ...

    def __init__(self, obj: T) -> None:
        """Initialize a ReferenceType with an object.

        If obj.__rawref__ exists and is valid, returns that instead.
        Otherwise creates a new reference and sets obj.__rawref__ to it.

        Args:
            obj: The object to reference.
        """
        ...

    def __call__(self) -> Optional[T]:
        """Dereference the object.

        Returns:
            The referenced object, or None if the reference is invalid.
        """
        ...

    def invalidate(self) -> None:
        """Invalidate the reference.

        After calling this method, __call__() will return None.
        This should be called from T.__del__ to invalidate the reference.
        """
        ...

# Alias 'ref' to 'ReferenceType' (like weakref.ref is an alias to weakref.ReferenceType)
ref = ReferenceType

# NULL is an invalid reference constant that can be used to initialize __rawref__
NULL: ReferenceType

# For type hints, you can use either:
#   r: ref[MyClass] = ref(obj)
# or:
#   r: ReferenceType[MyClass] = ReferenceType(obj)
# Both are equivalent.
