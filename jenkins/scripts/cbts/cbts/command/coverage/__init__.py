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
"""`cbts.command coverage ...` subgroup."""

from __future__ import annotations

import click

from cbts.command.coverage import pilot as _pilot
from cbts.command.coverage.collection import group as _collection_group
from cbts.command.coverage.selection import group as _selection_group


@click.group("coverage")
def group() -> None:
    """Coverage-related tools (Tier 2 selection, Layer C collection)."""


group.add_command(_pilot.main)
group.add_command(_selection_group)
group.add_command(_collection_group)
