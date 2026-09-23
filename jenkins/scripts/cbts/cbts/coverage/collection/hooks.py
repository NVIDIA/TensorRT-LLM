# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The callbacks the guest bootstrap installs once CBTS coverage is active in this process.

Peers that need to reach the active tracker (the MPI pool patch, the pytest plugin) call
through ``active`` below, rather than reflecting by name on the guest-injected
``sitecustomize`` module.

A name-based lookup (``getattr(sitecustomize, "switch_test_context", None)``) would also
work, but a typo'd or renamed callback then fails silently as a no-op instead of a direct
attribute error caught by lint or a test import.
"""

from __future__ import annotations

from typing import Iterable, Optional, TypeAlias

# (process_uid, test, kind, reason); process_uid is None only when reported by whichever
# caller owns this tracker (see record_channel_taints below).
_TaintTuple: TypeAlias = tuple[Optional[str], str, str, str]


class Hooks:
    """The points in a test's lifecycle the guest bootstrap needs to react to.

    A test starting (``switch_test_context``) or finishing (``record_test_outcome``), the
    session ending (``flush_coverage``, ``record_channel_taints``), and a pool executor
    spawning workers for the current test (``note_expected_workers``). No-op by default;
    ``sitecustomize.py`` installs the real implementations as ``active`` once CBTS coverage
    is active in this process.
    """

    def switch_test_context(self, nodeid: Optional[str]) -> None:
        """Switch the active test context; each test's entered functions are recorded separately."""

    def record_test_outcome(self, nodeid: Optional[str], outcome: str) -> None:
        """Record a test's pytest outcome for the merge-side completeness signal (outer pytest only)."""

    def flush_coverage(self) -> Optional[str]:
        """Write this process's coverage now instead of waiting for the atexit save."""

    def record_channel_taints(self, taints: Iterable[_TaintTuple]) -> None:
        """Fold ``(process_uid, test, kind, reason)`` into this process's leaf database."""

    def note_expected_workers(self, nodeid: Optional[str], n: int) -> None:
        """Record that the coordinator spawned n subprocess pool workers for a test."""


active: Hooks = Hooks()
