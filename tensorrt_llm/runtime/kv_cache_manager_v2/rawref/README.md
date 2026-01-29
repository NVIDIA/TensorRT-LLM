<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# rawref - Mutable Reference C Extension

A C extension that provides a mutable reference class similar to `weakref.ref` for holding weak-like references to Python objects.

## Features

- **`ref[T]`**: A generic reference class (like `weakref.ref`) that stores an object's ID
- **Singleton pattern**: `ref(obj)` returns the same reference if `obj.__rawref__` is valid
- **Dereferencing**: Call `r()` to get the object, or `None` if invalid
- **Invalidation**: Call `r.invalidate()` to mark the reference as invalid
- **NULL constant**: Use `NULL` to initialize `__rawref__` attributes
- **Type-safe**: Comes with `.pyi` stub file for proper type checking
- **API compatible with weakref**: Use `ref` for both object creation and type hints

## Building

From the `rawref` directory:

```bash
python setup.py build_ext --inplace
```

Or install it:

```bash
pip install -e .
```

## Usage

```python
from rawref import ref, NULL

class MyClass:
    # Class attribute: default value for __rawref__
    # Each instance will get its own __rawref__ instance attribute when ref() is called
    __rawref__ = NULL
    
    def __init__(self, value):
        self.value = value
    
    def __del__(self):
        # self.__rawref__ is an instance attribute (set by ref())
        # Invalidate the canonical reference when object is destroyed
        if self.__rawref__.is_valid:
            self.__rawref__.invalidate()

# Create an object and a reference to it (just like weakref.ref)
obj = MyClass(42)
r1 = ref(obj)

# The reference is automatically stored as an instance attribute obj.__rawref__
print(obj.__rawref__ is r1)  # True

# Singleton pattern: creating another ref returns the same one
r2 = ref(obj)
print(r1 is r2)  # True

# Dereference to get the object back
print(r1())  # <MyClass instance>
print(r1().value)  # 42

# Check validity
print(r1.is_valid)  # True

# After invalidation
r1.invalidate()
print(r1())  # None
print(r1.is_valid)  # False

# Creating a new ref after invalidation creates a new reference
r3 = ref(obj)
print(r1 is r3)  # False
print(r3.is_valid)  # True
```

## Type Hints

Like `weakref.ref`, you can use `ref` for both object creation and type hints:

```python
from rawref import ref, NULL

class MyClass:
    __rawref__ = NULL

# Create and type a reference
r: ref[MyClass] = ref(MyClass())

# Alternative: use ReferenceType directly
from rawref import ReferenceType
r: ReferenceType[MyClass] = ReferenceType(MyClass())
```

## Warning

This implementation uses raw object IDs (memory addresses) and attempts to dereference them. This is inherently unsafe and should be used with caution. The reference does not keep the object alive (unlike a strong reference), so care must be taken to ensure the object is not garbage collected while references exist.

## API

### Classes and Constants
- `ReferenceType`: The main reference class
- `ref`: Alias for `ReferenceType` (like `weakref.ref`)
- `NULL`: An invalid reference constant for initialization

### Creation
- `ref(obj)`: Create a reference to `obj`, or return existing valid reference from `obj.__rawref__`

### Properties
- `r.is_valid`: Check if the reference is still valid (read-only)

### Methods
- `r()`: Dereference to get the object, or `None` if invalid
- `r.invalidate()`: Mark the reference as invalid

## Singleton Pattern

The `ref()` function implements a singleton pattern:
1. When `ref(obj)` is called, it checks if `obj.__rawref__` (instance attribute) exists and is valid
2. If yes, it returns the existing reference
3. If no, it creates a new reference and sets `obj.__rawref__` as an instance attribute

**Note**: The class attribute `__rawref__ = NULL` is just a default value. When `ref(obj)` is called, it creates an **instance attribute** `obj.__rawref__` that shadows the class attribute. Each instance gets its own `__rawref__` instance attribute, ensuring each object has at most one canonical reference at a time.
