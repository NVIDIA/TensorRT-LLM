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
"""The integration suite does not load the plugin: a QA caller passes `-p qa_selection.plugin`.

    cd tests/integration/defs
    pytest --collect-only --machine=B200    -> usage error, unrecognized arguments

The integration conftest needs the wheel, so its `pytest.ini` is read as text.
"""

import configparser
import shlex
from pathlib import Path
from typing import List


class IntegrationIni:
    """`tests/integration/defs/pytest.ini`, read without pytest."""

    PATH = Path(__file__).parents[2] / "integration" / "defs" / "pytest.ini"
    PLUGIN = "qa_selection.plugin"

    @classmethod
    def addopts(cls) -> List[str]:
        """The `addopts` value, split as a shell splits it."""
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(cls.PATH, encoding="utf-8")
        return shlex.split(parser["pytest"]["addopts"])


# The integration suite does not load the plugin / Selection options are unknown without `-p`
def test_ini_does_not_load_the_plugin():
    """No `addopts` token names the plugin, so `--machine` exists only after `-p`."""
    addopts = IntegrationIni.addopts()

    assert "--durations=0" in addopts
    assert IntegrationIni.PLUGIN not in addopts
