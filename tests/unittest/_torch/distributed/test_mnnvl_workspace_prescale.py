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
"""MNNVL AllReduce workspace behaviour under torch.compile.

The lamport workspace is grown lazily by forward(); growing allocates multicast
memory over MPI/ProcessGroup and synchronizes the device, which cannot run
inside a dynamo-traced region (and a post-capture regrow would leave captured
graphs holding freed buffer pointers). These tests pin the two halves of the
contract without GPUs or a multi-rank job: the scaling path must refuse to run
under tracing, and prescale_workspace must request a size that covers every
call forward() can serve so the traced path stays a pure lookup.
"""

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as ops
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm.mapping import Mapping


@pytest.fixture
def tp4_mapping():
    mapping = Mapping(world_size=4, tp_size=4, rank=0)
    yield mapping
    MNNVLAllReduce.allreduce_mnnvl_workspaces.pop(mapping, None)


def _seed_workspace(mapping, buffer_size_bytes):
    MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping] = {
        "buffer_size_bytes": buffer_size_bytes,
    }


def test_scaling_refused_under_compile(tp4_mapping, monkeypatch):
    _seed_workspace(tp4_mapping, 1024)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="prescale_workspace"):
        ops.get_or_scale_allreduce_mnnvl_workspace(
            tp4_mapping, torch.bfloat16, buffer_size_bytes=2048
        )


def test_creation_refused_under_compile(tp4_mapping, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="prescale_workspace"):
        ops.get_or_scale_allreduce_mnnvl_workspace(tp4_mapping, torch.bfloat16)


def test_lookup_allowed_under_compile(tp4_mapping, monkeypatch):
    workspace = MNNVLAllReduce.allreduce_mnnvl_workspaces
    _seed_workspace(tp4_mapping, 1 << 20)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    result = ops.get_or_scale_allreduce_mnnvl_workspace(
        tp4_mapping, torch.bfloat16, buffer_size_bytes=1 << 20
    )
    assert result is workspace[tp4_mapping]


def _make_module(mapping, dtype):
    module = MNNVLAllReduce.__new__(MNNVLAllReduce)
    module.mapping = mapping
    module.dtype = dtype
    return module


def test_prescale_covers_every_forward_size(tp4_mapping, monkeypatch):
    requested = {}

    def fake_get_or_scale(mapping, dtype, buffer_size_bytes=None):
        requested["size"] = buffer_size_bytes

    monkeypatch.setattr(ops, "get_or_scale_allreduce_mnnvl_workspace", fake_get_or_scale)
    module = _make_module(tp4_mapping, torch.bfloat16)
    max_num_tokens, hidden_dim = 16384, 4096
    module.prescale_workspace(max_num_tokens, hidden_dim)

    # Every reachable forward() call must fit: sample the token range on both
    # sides of the one-shot/two-shot switch.
    for num_tokens in (1, 8, 32, 33, 128, 1024, 1025, 8192, max_num_tokens):
        needed = MNNVLAllReduce.get_required_workspace_size(
            num_tokens, hidden_dim, tp4_mapping.tp_size, torch.bfloat16
        )
        assert requested["size"] >= needed, num_tokens


def test_prescale_clamps_below_uint32(tp4_mapping, monkeypatch):
    requested = {}

    def fake_get_or_scale(mapping, dtype, buffer_size_bytes=None):
        requested["size"] = buffer_size_bytes

    monkeypatch.setattr(ops, "get_or_scale_allreduce_mnnvl_workspace", fake_get_or_scale)
    module = _make_module(tp4_mapping, torch.bfloat16)
    # Large enough that the unclamped two-shot size exceeds the uint32 limit
    # forward() enforces; prescale must not allocate beyond what forward() can
    # ever use.
    module.prescale_workspace(1 << 22, 1 << 14)
    assert requested["size"] < 2**32 - 1
    assert requested["size"] % (8 << 20) == 0
