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
"""Shared source fixtures for CBTS dataclass analysis and selection tests."""

import difflib

ADDED_DATACLASS = (
    "@dataclass(frozen=True)\n"
    "class Budget:\n"
    '    "A value produced by helper."\n'
    "    amount: int\n\n"
    "    @property\n"
    "    def doubled(self) -> int:\n"
    "        return self.amount * 2\n"
)


def added_dataclass_change(
    declaration: str, *, prefix: str = "from dataclasses import dataclass\n\n"
) -> tuple[str, str]:
    """Return a module and patch adding a dataclass consumed by helper and caller."""
    helper = "def helper():\n    return Budget(2).doubled\n\n"
    caller = "def caller():\n    return helper()\n"
    before = prefix + "def helper():\n    return None\n\n" + caller
    source = prefix + declaration + "\n" + helper + caller
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    return source, diff
