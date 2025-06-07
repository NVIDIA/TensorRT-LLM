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
