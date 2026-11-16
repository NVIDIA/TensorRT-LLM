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

from types import SimpleNamespace

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
        # The lookup revalidates the multicast mapping before reusing a workspace.
        "handle": SimpleNamespace(is_mapped=lambda: True),
        "buffer_size_bytes": buffer_size_bytes,
        "captured": False,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for a native method handle")
def test_cached_lookup_traces_with_a_native_handle(tp4_mapping):
    """The cached lookup of a compiled forward must not call the workspace handle.

    The real handle is a nanobind object, which Dynamo cannot trace; a C++-bound method
    (``torch.cuda.Stream.query``) stands in for ``is_mapped`` so that the fast path of the public
    lookup is traced for real, not with a Python lambda.
    """
    stream = torch.cuda.Stream()
    MNNVLAllReduce.allreduce_mnnvl_workspaces[tp4_mapping] = {
        "handle": SimpleNamespace(is_mapped=stream.query),
        "buffer_size_bytes": 1 << 20,
        "captured": False,
    }

    def lookup(x):
        workspace = ops.get_or_scale_allreduce_mnnvl_workspace(
            tp4_mapping, torch.bfloat16, buffer_size_bytes=1 << 20
        )
        return x + workspace["buffer_size_bytes"]

    torch._dynamo.reset()
    out = torch.compile(lookup, backend="eager", fullgraph=True)(torch.zeros(2))
    assert out.tolist() == [float(1 << 20)] * 2
    # The compiled graph embeds the workspace buffers, so the workspace must not be replaced later.
    assert MNNVLAllReduce.allreduce_mnnvl_workspaces[tp4_mapping]["captured"] is True


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

    # Every reachable forward() call must fit, including both sides of the one-shot/two-shot
    # switch. The size is not monotonic across it, so check every count (pure arithmetic).
    for num_tokens in range(1, max_num_tokens + 1):
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
