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


def gpu_compute_cap_str():
    output = subprocess.check_output(
        ['nvidia-smi', "--query-gpu=compute_cap", "--format=csv"])
    csv_header, csv_value, *other_csv_values = output.splitlines()
    return 'sm' + str(int(float(csv_value) * 10))


def pytest_collection_modifyitems(config, items):
    for item in items:
        # print("[debug] " + item.nodeid)
        # skip tests that are not runnable on local machine
        if 'arch' in item.nodeid and gpu_compute_cap_str() not in item.nodeid:
            item.add_marker(pytest.mark.skip(reason='arch-dependent test'))
        # mark tests as unit when under 'unit'
        if 'unit' in item.nodeid:
            item.add_marker(pytest.mark.unit)
