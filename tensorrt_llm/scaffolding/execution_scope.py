# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextvars
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ExecutionScope:
    """Identifies a point in the scaffolding execution tree.

    Every scaffolding request gets a root scope (``branch_path == ()``).
    Each ``ParallelProcess`` fork appends a zero-based branch index,
    producing a child scope.  The resulting ``scope_id`` is globally
    unique and can be used as a key for resource ownership (SSE
    connections, sandboxes, etc.).

    Examples::

        root = ExecutionScope("req-1")  # scope_id = "req-1"
        child0 = root.child(0)  # scope_id = "req-1:0"
        child1 = root.child(1)  # scope_id = "req-1:1"
        grand = child0.child(2)  # scope_id = "req-1:0.2"
    """

    request_id: str
    branch_path: Tuple[int, ...] = ()

    @property
    def scope_id(self) -> str:
        """Globally unique, hashable key for this scope."""
        if not self.branch_path:
            return self.request_id
        return f"{self.request_id}:{'.'.join(map(str, self.branch_path))}"

    def child(self, branch_index: int) -> "ExecutionScope":
        """Create a child scope for a parallel fork."""
        return ExecutionScope(
            request_id=self.request_id,
            branch_path=self.branch_path + (branch_index,),
        )

    @property
    def branch_path_list(self) -> List[int]:
        """Return branch_path as a mutable list (for JSON serialization)."""
        return list(self.branch_path)


current_scope: contextvars.ContextVar[Optional[ExecutionScope]] = contextvars.ContextVar(
    "current_scope", default=None
)
"""The execution scope of the currently running scaffolding code path.

Set automatically by :class:`ScaffoldingLlm` at request entry and at
each ``ParallelProcess`` fork.  Inherited by every
``asyncio.create_task`` in the processing chain.  Workers and
``TaskCollection`` implementations read this to obtain per-request
or per-branch identity for resource routing, tracing, and replay.
"""
