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
"""The cache-transfer contract.

What this package exports is the contract; what it does not is backend-internal and has to be
imported by file path.
"""

from .backend import (
    Attempt,
    CacheExtent,
    CacheKind,
    Cancelled,
    Chunk,
    Delivered,
    Failed,
    Fetches,
    Outcome,
    Publishes,
    TokenRange,
)

__all__ = [
    "Attempt",
    "CacheExtent",
    "CacheKind",
    "Cancelled",
    "Chunk",
    "Delivered",
    "Failed",
    "Fetches",
    "Outcome",
    "Publishes",
    "TokenRange",
]
