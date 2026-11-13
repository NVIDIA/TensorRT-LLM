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
"""The Mooncake store's node budget counts every GPU's rank and the job's cgroup limit."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import validation

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30


def _config(segment_gib):
    return SimpleNamespace(
        global_segment_size=segment_gib * GIB,
        stage_through_host=False,
        role=SimpleNamespace(loads=True, saves=True),
        staging_buffer_bytes=0,
    )


@pytest.fixture
def node(monkeypatch):
    """A node with 1,500 GiB available, two local ranks and no cgroup limit unless set."""
    state = SimpleNamespace(available=1500 * GIB, room=None)
    monkeypatch.setattr(validation, "_available_host_memory", lambda: state.available)
    monkeypatch.setattr(validation, "_cgroup_room", lambda: state.room)
    monkeypatch.setattr("tensorrt_llm._utils.local_mpi_size", lambda: 2)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    return state


def test_local_ranks_alone_when_slurm_names_no_gpus(node):
    # 2 x 600 GiB fits 1,500 GiB less the reserve.
    validation.validate_node_budget(_config(600))


def test_every_gpu_on_the_node_counts(node, monkeypatch):
    # Two co-located 2-rank servers: 4 x 400 GiB does not fit, although 2 x 400 GiB would.
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    with pytest.raises(ValueError, match="4 rank"):
        validation.validate_node_budget(_config(400))
    validation.validate_node_budget(_config(300))


def test_the_cgroup_limit_caps_the_host_figure(node, monkeypatch):
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
    node.room = 1280 * GIB
    # 4 x 315 GiB = 1,260 GiB fits the host but not the job's 1,280 GiB less the reserve.
    validation.validate_node_budget(_config(315), available_bytes=1500 * GIB)
    with pytest.raises(ValueError, match="cannot afford"):
        validation.validate_node_budget(_config(315))


def test_an_explicit_rank_count_is_kept(node, monkeypatch):
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    validation.validate_node_budget(_config(600), ranks_on_node=2)


@pytest.mark.parametrize("value", ["", "0", "gpu:4", "-1"])
def test_unreadable_gpu_counts_are_ignored(value, monkeypatch):
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", value)
    assert validation._gpus_on_node() is None
