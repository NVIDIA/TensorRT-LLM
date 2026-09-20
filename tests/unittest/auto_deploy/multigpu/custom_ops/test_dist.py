# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for custom dist ops."""

import pytest
import torch
from _dist_test_utils import get_device_counts

from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job


def _run_all_reduce_test(rank, world_size):
    x = torch.ones(10, 10).to("cuda")
    y = torch.ops.auto_deploy.torch_dist_all_reduce(x, "AUTO")

    assert torch.equal(x * world_size, y)


def _run_all_gather_test(rank, world_size):
    x = torch.ones(10, 10).to("cuda")
    # Test torch backend (demollm mode with Python multiprocessing).
    y = torch.ops.auto_deploy.torch_dist_all_gather(x)

    assert torch.sum(y) == world_size * torch.sum(x)
    assert y.shape == (world_size * x.shape[0], *x.shape[1:])


@pytest.mark.parametrize("device_count", get_device_counts())
def test_all_reduce(device_count):
    spawn_multiprocess_job(job=_run_all_reduce_test, size=device_count)


@pytest.mark.parametrize("device_count", get_device_counts())
def test_all_gather(device_count):
    spawn_multiprocess_job(job=_run_all_gather_test, size=device_count)
