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
"""Mock suite: GPU demands. Collected by the run under test, never run here.

One test per rung of `1,4,8`, plus an 8-GPU demand stated in MPI ranks.

Touching `imported` on import is how a criterion sees whether collection was
reached at all: a usage error must be raised before any test module is
imported, and the file's absence is the only direct evidence of that.
"""

import pathlib

import pytest

pathlib.Path(__file__).with_name("imported").touch()


def test_unmarked():
    pass


@pytest.mark.skip_less_device(2)
def test_two_gpus():
    pass


@pytest.mark.skip_less_device(8)
def test_eight_gpus():
    pass


@pytest.mark.skip_less_mpi_world_size(8)
def test_eight_ranks():
    pass
