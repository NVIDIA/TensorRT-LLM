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
"""Tests for how ``full:<platform>/`` waive names are mapped to a test prefix."""

import sys
import types

import pytest
from integration.defs import test_list_parser

pytestmark = pytest.mark.cpu_only


def _parse(line, test_prefix, convert_unittest=True):
    names, _ = test_list_parser.parse_test_list_lines(
        "waives.txt", [line], test_prefix, convert_unittest=convert_unittest
    )
    return names[0]


@pytest.fixture
def sm_version(monkeypatch):
    """Stub the SM lookup that ``full:smXX/`` waives perform."""
    stub = types.ModuleType("integration.defs.conftest")
    monkeypatch.setitem(sys.modules, stub.__name__, stub)

    def set_version(version):
        stub.get_sm_version = lambda: version

    return set_version


@pytest.mark.parametrize(
    ("line", "test_prefix", "expected"),
    [
        (
            "accuracy/test_x.py::T::test[a] SKIP (r)",
            "DGX_H100-4_GPUs-PyTorch-1",
            "DGX_H100-4_GPUs-PyTorch-1/accuracy/test_x.py::T::test[a]",
        ),
        (
            "full:DGX_H100/accuracy/test_x.py::T::test[a] SKIP (r)",
            "DGX_H100-4_GPUs-PyTorch-1",
            "DGX_H100-4_GPUs-PyTorch-1/accuracy/test_x.py::T::test[a]",
        ),
        # The platform string inside the test name must be left untouched.
        (
            "full:CPU/unittest/test_x.py::test_x[CPU] SKIP (r)",
            "CPU-Generic-1",
            "CPU-Generic-1/test_unittests.py::test_unittests_v2[unittest/test_x.py::test_x[CPU]]",
        ),
        (
            "full:A10/accuracy/test_x.py::T::test[A100_ref] SKIP (r)",
            "A10-PyTorch-1",
            "A10-PyTorch-1/accuracy/test_x.py::T::test[A100_ref]",
        ),
        # A waive for another platform keeps its own prefix and cannot match.
        (
            "full:A100X/accuracy/test_x.py::T::test[A10] SKIP (r)",
            "A10-PyTorch-1",
            "A100X/accuracy/test_x.py::T::test[A10]",
        ),
        # Platforms whose names start with a letter of "full:" keep that letter.
        (
            "full:lab/test_x.py::t SKIP (r)",
            "lab-1",
            "lab-1/test_x.py::t",
        ),
    ],
)
def test_platform_prefix(line, test_prefix, expected):
    assert _parse(line, test_prefix) == expected


def test_platform_prefix_inner_unittest():
    name = _parse(
        "full:CPU/unittest/test_x.py::test_x[CPU] SKIP (r)",
        "CPU-Generic-1/unittest",
        convert_unittest=False,
    )
    assert name == "CPU-Generic-1/unittest/unittest/test_x.py::test_x[CPU]"


def test_sm_prefix_rewrites_leading_component_only(sm_version):
    sm_version(90)
    name = _parse("full:sm90/kernels/test_x.py::test_fp8_sm90 SKIP (r)", "H100_PCIe-1")
    assert name == "H100_PCIe-1/kernels/test_x.py::test_fp8_sm90"


def test_sm_prefix_other_version_does_not_match(sm_version):
    sm_version(100)
    name = _parse("full:sm90/kernels/test_x.py::test_fp8 SKIP (r)", "B200-1")
    assert name == "sm90/kernels/test_x.py::test_fp8"
