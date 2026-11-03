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
"""A failed MNNVL all-reduce workspace creation is recorded per mapping and not retried.

Every AllReduce module of a model asks for the workspace of its mapping. A failed
McastGPUBuffer construction does not free its fabric allocation, so retrying it in
each module used to leak device memory until the model build ran out of memory.
"""

from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.distributed import ops
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _FakeComm:
    def Split(self, color, key):
        return self

    def py2f(self):
        return 0

    def Barrier(self):
        pass

    def Get_size(self):
        return 1

    def allreduce(self, value):
        return value


@pytest.fixture(autouse=True)
def _clean_failures():
    saved = dict(ops.MNNVLAllReduce.failed_mnnvl_workspaces)
    ops.MNNVLAllReduce.failed_mnnvl_workspaces.clear()
    yield
    ops.MNNVLAllReduce.allreduce_mnnvl_pending_comms.clear()
    ops.MNNVLAllReduce.failed_mnnvl_workspaces.clear()
    ops.MNNVLAllReduce.failed_mnnvl_workspaces.update(saved)


def _mapping(rank):
    return Mapping(world_size=16, rank=rank, gpus_per_node=4, tp_size=16)


def test_failed_workspace_is_not_retried():
    mapping = _mapping(0)
    mcast = mock.Mock(side_effect=RuntimeError("fabric handle export failed"))
    with (
        mock.patch.object(ops, "McastGPUBuffer", mcast),
        mock.patch.object(ops, "mpi_comm", return_value=_FakeComm()),
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            ops.get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        for _ in range(3):
            with pytest.raises(RuntimeError, match="failed earlier"):
                ops.get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        # The module constructor reaches the same cached failure.
        with pytest.raises(RuntimeError, match="failed earlier"):
            ops.MNNVLAllReduce(mapping, torch.bfloat16)
    assert mcast.call_count == 1
    assert "fabric handle export failed" in ops.MNNVLAllReduce.failed_mnnvl_workspaces[mapping]
    assert mapping not in ops.MNNVLAllReduce.allreduce_mnnvl_workspaces


def test_failure_is_per_mapping():
    failed, other = _mapping(0), _mapping(1)
    mcast = mock.Mock(side_effect=RuntimeError("no IMEX channel"))
    with (
        mock.patch.object(ops, "McastGPUBuffer", mcast),
        mock.patch.object(ops, "mpi_comm", return_value=_FakeComm()),
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            ops.get_or_scale_allreduce_mnnvl_workspace(failed, torch.bfloat16)
        with pytest.raises(RuntimeError, match="construction failed"):
            ops.get_or_scale_allreduce_mnnvl_workspace(other, torch.bfloat16)
    assert mcast.call_count == 2
    assert set(ops.MNNVLAllReduce.failed_mnnvl_workspaces) == {failed, other}
