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
"""DeviceMeshTopology reads under torch.compile.

Model forwards read mapping.tp_group / tp_rank inside torch.compile'd
regions. On the Ray topology these used to resolve through the
compiler-disabled DeviceMesh lookup on every access -- a hard Unsupported
under fullgraph=True (first hit: Qwen3Next MoE block passing mapping.tp_group
to trtllm::allocate_output). Group membership is fixed once the mesh exists,
so the topology now memoizes each dim on first eager access and later reads
trace as constants. These tests pin that contract with a fake mesh: memoize
once, trace after an eager warm, refuse (loudly) to resolve under tracing,
and keep Mapping picklable by stripping the ProcessGroup caches.
"""

import pickle

import pytest
import torch

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl


class _FakePg:
    def __init__(self, rank, ranks):
        self._rank = rank
        self._ranks = ranks

    def rank(self):
        return self._rank

    def size(self):
        return len(self._ranks)


class _FakeMeshDim:
    def __init__(self, pg):
        self._pg = pg

    def get_group(self):
        return self._pg


@pytest.fixture
def mapping(monkeypatch):
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    from tensorrt_llm.mapping import Mapping

    m = Mapping(world_size=4, tp_size=4, rank=1)
    assert isinstance(m, DeviceMeshTopologyImpl)

    resolve_calls = {"count": 0}
    pg = _FakePg(rank=1, ranks=[0, 1, 2, 3])

    def fake_get_mesh_dim(self, name):
        resolve_calls["count"] += 1
        return _FakeMeshDim(pg)

    monkeypatch.setattr(DeviceMeshTopologyImpl, "_get_mesh_dim_by_name", fake_get_mesh_dim)
    monkeypatch.setattr(
        DeviceMeshTopologyImpl, "_get_group_ranks", lambda self, pg_: list(pg_._ranks)
    )
    # Satisfy the memoization guard (device_mesh is not None) without lying
    # to torch.distributed (dynamo itself consults dist.is_initialized).
    monkeypatch.setattr(DeviceMeshTopologyImpl, "device_mesh", object())
    torch._dynamo.reset()
    return m, resolve_calls


def test_mesh_dim_memoized_once(mapping):
    m, resolve_calls = mapping
    assert m.tp_group == [0, 1, 2, 3]
    assert m.tp_rank == 1
    assert m.tp_group_pg.size() == 4
    assert resolve_calls["count"] == 1


def test_warm_reads_trace_under_fullgraph(mapping):
    m, _ = mapping
    _ = m.tp_group  # the eager warm model_engine performs before compiling

    def forward(x):
        return x * len(m.tp_group) + m.tp_rank

    compiled = torch.compile(forward, fullgraph=True, backend="eager")
    out = compiled(torch.ones(2))
    assert out[0].item() == 4 + 1


def test_cold_read_refused_under_fullgraph(mapping):
    m, _ = mapping

    def forward(x):
        return x * len(m.tp_group)

    compiled = torch.compile(forward, fullgraph=True, backend="eager")
    with pytest.raises(Exception, match="compiler.disable"):
        compiled(torch.ones(2))


def test_pickle_strips_process_group_caches(mapping):
    m, resolve_calls = mapping
    _ = m.tp_group
    restored = pickle.loads(pickle.dumps(m))
    assert not any(
        key.startswith(DeviceMeshTopologyImpl._MESH_DIM_CACHE_PREFIX) for key in restored.__dict__
    )
    # Re-resolves lazily on the restored object.
    assert restored.tp_group == [0, 1, 2, 3]
    assert resolve_calls["count"] == 2
