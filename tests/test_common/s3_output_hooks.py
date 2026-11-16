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
"""Register the S3 report transformer on pytest workers."""

import os

import pytest

from test_common import s3_output


def _is_xdist_worker(config: pytest.Config) -> bool:
    return bool(os.environ.get("PYTEST_XDIST_WORKER")) or hasattr(config, "workerinput")


def _is_xdist_controller(config: pytest.Config) -> bool:
    if _is_xdist_worker(config):
        return False
    namespace = getattr(config, "known_args_namespace", config.option)
    num_processes = getattr(namespace, "numprocesses", None)
    if num_processes in (None, 0, "0"):
        num_processes = getattr(config.option, "numprocesses", None)
    return num_processes not in (None, 0, "0")


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    if not _is_xdist_controller(config):
        s3_output.register_plugin(config)
