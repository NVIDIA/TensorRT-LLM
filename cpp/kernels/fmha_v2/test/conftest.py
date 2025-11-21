# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--dry-run",
                     "--dryrun",
                     action="store_true",
                     help="dry run")
    parser.addoption("--disable-rules",
                     "--disable_rules",
                     action="store_true",
                     help="disable filtering of test prompts")


@pytest.fixture
def dryrun(pytestconfig):
    return pytestconfig.getoption("dry_run")


@pytest.fixture
def disable_rules(pytestconfig):
    # - converted internally to _ in argparse
    return pytestconfig.getoption("disable_rules")


@pytest.fixture
def gpu_compute_cap():
    output = subprocess.check_output(
        ['nvidia-smi', "--query-gpu=compute_cap", "--format=csv"])
    csv_header, csv_value, *other_csv_values = output.splitlines()
    return int(float(csv_value) * 10)


# debug
@pytest.fixture
def mock_gpu_compute_cap():
    return 30
