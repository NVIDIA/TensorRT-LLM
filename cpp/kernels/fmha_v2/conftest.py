# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
