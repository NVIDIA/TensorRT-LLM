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

"""rawref - Mutable reference with singleton pattern.

This module provides a C extension for creating mutable references to Python
objects, similar to weakref.ref but with manual invalidation control and a
singleton pattern via __rawref__.

The main purpose is to work around the issue that mypyc does not support
weakref.

Main exports:
- ReferenceType: The reference class
- ref: Alias for ReferenceType (recommended, like weakref.ref)
- NULL: Invalid reference constant for initialization
"""

from ._rawref import NULL, ReferenceType, ref

__all__ = ["ReferenceType", "ref", "NULL"]

__version__ = "2.0.0"
