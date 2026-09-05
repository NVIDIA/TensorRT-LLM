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
"""`cbts.command coverage collection ...` subgroup."""

from __future__ import annotations

import click

from cbts.command.coverage.collection import compact_touch_db as _compact_touch_db
from cbts.command.coverage.collection import pystart_report as _pystart_report


@click.group("collection")
def group() -> None:
    """Layer C (function-level) coverage collection tools."""


group.add_command(_pystart_report.main)
group.add_command(_compact_touch_db.main)
