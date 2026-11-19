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

import pytest

import tensorrt_llm._utils as utils

pytestmark = pytest.mark.cpu_only


def test_mpi_allgather_without_multi_device(monkeypatch):
    monkeypatch.setattr(utils, "ENABLE_MULTI_DEVICE", False)

    assert utils.mpi_allgather(1.5) == [1.5]
    assert utils.mpi_allgather((1, 2)) == [(1, 2)]
