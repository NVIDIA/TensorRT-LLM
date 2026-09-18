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
"""Pytest entry point: deselect the tests a target machine cannot run.

    pytest --collect-only -p qa_selection.plugin --machine=B200 --gpus=4 --ladder=1,4,8

Load with `-p` on the command line. No hardware is touched: every decision
is read from marks. The logic lives in `collection.py` and `report.py`;
this file is hooks only.
"""

from pathlib import Path
from typing import List

import pytest

from .collection import ResourceMarkers, Selection, SelectionOptions, SelectionRequest
from .report import SelectionOutput, TerminalSummary


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the selection options before pytest parses the command line."""
    SelectionOptions.add_to(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Declare the resource markers, and resolve the target before collection."""
    ResourceMarkers.declare(config)
    request = SelectionRequest.of(config)
    if request is not None:
        config.stash[SelectionRequest.STASH_KEY] = request


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Deselect what the target machine cannot run.

    A non-wrapper on purpose. `trylast` orders this against
    other non-wrappers, xdist and pytest-split among them.
    """
    request = config.stash.get(SelectionRequest.STASH_KEY, None)
    if request is None:
        return

    selection = Selection.of(request, items)
    config.stash[Selection.STASH_KEY] = selection

    kept, dropped = selection.partition(items)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


def pytest_collection_finish(session: pytest.Session) -> None:
    """Build the record and write the artifacts.

    After `modifyitems`, so every other plugin's deselection is already in,
    and before `--collect-only` prints.
    """
    config = session.config
    selection = config.stash.get(Selection.STASH_KEY, None)
    if selection is None:
        return
    config.stash[SelectionOutput.STASH_KEY] = SelectionOutput.of(
        selection, Path(str(config.rootpath))
    )


def pytest_terminal_summary(terminalreporter) -> None:
    """Render the human-readable block after pytest's own counts."""
    output = terminalreporter.config.stash.get(SelectionOutput.STASH_KEY, None)
    if output is not None:
        TerminalSummary.render(terminalreporter, output)
