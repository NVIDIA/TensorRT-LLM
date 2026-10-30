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
"""TEMPORARY, for the draft pull request only: no Artifactory credential is reachable from tests.

For about three minutes, so that the check spans the checkpoint uploads that run beside the tests,
it checks every 15 seconds that none of the environment variables the pipelines bind for
Artifactory is set, and that no readable small file under the workspace, its temporary folder,
$HOME, or /tmp holds a netrc entry with a login for artifactory.nvidia.com or urm.nvidia.com. It
reports where it found one, never the secret.
"""

import os
import re
import time
from pathlib import Path

_DURATION_SECONDS = 180
_INTERVAL_SECONDS = 15
_CREDENTIAL_VARIABLES = ("ART_USER", "ART_PASS", "ARTIFACTORY_USER", "ARTIFACTORY_PASSWORD")
# A netrc entry with a real login and password; a printf template ("login %s") is not one.
_NETRC_ENTRY = re.compile(
    rb"machine\s+(?:artifactory|urm)\.nvidia\.com\s+login\s+(?!%s\b)\S+\s+password\s+(?!%s\b)\S+"
)
# Netrc files are small; larger files are not read.
_MAX_FILE_BYTES = 16 * 1024
_SKIPPED_DIRECTORIES = {".git", "__pycache__", "node_modules"}


def _roots() -> list[Path]:
    roots = [Path("/tmp")]
    workspace = os.environ.get("WORKSPACE")
    if workspace:
        roots += [Path(workspace), Path(f"{workspace}@tmp")]
    if os.environ.get("HOME"):
        roots.append(Path(os.environ["HOME"]))
    return [root for root in roots if root.is_dir()]


def _files_with_netrc_entries() -> list[str]:
    found = []
    seen = set()
    for root in _roots():
        for directory, subdirectories, files in os.walk(root, onerror=lambda error: None):
            subdirectories[:] = [
                name for name in subdirectories if name not in _SKIPPED_DIRECTORIES
            ]
            for name in files:
                path = Path(directory) / name
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    stat = path.stat()
                    if stat.st_size > _MAX_FILE_BYTES or (stat.st_dev, stat.st_ino) in seen:
                        continue
                    seen.add((stat.st_dev, stat.st_ino))
                    if _NETRC_ENTRY.search(path.read_bytes()):
                        found.append(str(path))
                except OSError:
                    continue
    return found


def test_no_artifactory_credentials_while_tests_run():
    deadline = time.monotonic() + _DURATION_SECONDS
    while True:
        variables = [name for name in _CREDENTIAL_VARIABLES if name in os.environ]
        assert not variables, f"Artifactory credential variables are set: {variables}"
        files = _files_with_netrc_entries()
        assert not files, f"netrc entries for Artifactory are readable in: {files}"
        if time.monotonic() >= deadline:
            return
        time.sleep(_INTERVAL_SECONDS)
