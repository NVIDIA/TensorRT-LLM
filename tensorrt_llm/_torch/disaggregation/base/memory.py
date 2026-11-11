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
"""The kinds of memory a cache backend can access."""

import enum

__all__ = ["MemoryKind"]


class MemoryKind(enum.Enum):
    """Memory a backend can access: HOST is CPU-addressable; DEVICE is the GPU holding the cache.

    Each region of memory is of one kind."""

    HOST = "host"
    DEVICE = "device"
