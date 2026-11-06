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
"""The KV cache transfer backend contract.

The package exports the contract, plus ``TokenRange`` and ``CacheKind``, which belong to native and
the page table. All of them come from ``backend.py`` except ``MemoryKind``, which ``memory.py``
defines. The package's other modules are not part of the contract and are imported by their own
module paths, such as ``base.transfer``.
"""

from .backend import (
    Attempt,
    Backend,
    CacheKind,
    Capabilities,
    Hint,
    InvalidSubmission,
    LayerGroupRows,
    Outcome,
    SlotRegion,
    SubmissionRejected,
    TokenRange,
    View,
    WakeFn,
)
from .memory import MemoryKind

__all__ = [
    "Attempt",
    "Backend",
    "CacheKind",
    "Capabilities",
    "Hint",
    "InvalidSubmission",
    "LayerGroupRows",
    "MemoryKind",
    "Outcome",
    "SlotRegion",
    "SubmissionRejected",
    "TokenRange",
    "View",
    "WakeFn",
]
