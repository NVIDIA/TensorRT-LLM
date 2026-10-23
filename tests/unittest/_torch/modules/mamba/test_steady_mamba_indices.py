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
"""Steady decode batches skip the Mamba V2 state-index setup (Qwen3.5 fast paths).

The skipped path must leave exactly the buffers the full path writes, and any change of requests, SSM base
blocks or dummy flags must take the full path again.
"""

import types

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch import q35_env
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManager,
    MambaHybridCacheManagerV2,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def fast_paths():
    """Yields a setter for the Qwen3.5 fast-path gate; restores it afterwards."""
    before = q35_env.fast_paths_enabled()
    yield q35_env.set_fast_paths
    q35_env.set_fast_paths(before)


def _manager(max_batch=4):
    """The attributes _setup_state_indices / _refresh_dummy_request_mask use."""
    m = types.SimpleNamespace(
        local_num_mamba_layers=1,
        ssm_layer_group_id=0,
        use_kda_replay_update=False,
        kv_cache_map={},
        _host_state_indices=torch.zeros(max_batch, dtype=torch.int32),
        _state_indices_buffer=DeviceCopyBuffer(
            torch.zeros(max_batch, dtype=torch.int32, device="cuda")
        ),
        _dummy_request_mask_host=torch.zeros(max_batch, dtype=torch.bool),
        _dummy_request_mask_buffer=DeviceCopyBuffer(
            torch.zeros(max_batch, dtype=torch.bool, device="cuda")
        ),
        _request_id_to_state_index={},
        _request_id_to_is_dummy={},
        _steady_state_index_key=None,
    )
    m._dummy_request_mask = m._dummy_request_mask_buffer.tensor
    m._refresh_dummy_request_mask = types.MethodType(
        MambaHybridCacheManager._refresh_dummy_request_mask, m
    )
    m._is_steady_state_index_batch = types.MethodType(
        MambaHybridCacheManagerV2._is_steady_state_index_batch, m
    )
    m.setup = types.MethodType(MambaHybridCacheManagerV2._setup_state_indices, m)
    return m


def _set_cache(m, rid, base):
    m.kv_cache_map[rid] = types.SimpleNamespace(get_ssm_block_base_index=lambda _gid, b=base: b)


def _req(rid, dummy=False):
    return types.SimpleNamespace(py_request_id=rid, is_dummy=dummy)


def _state(m):
    torch.cuda.synchronize()
    return (
        m._state_indices_buffer.tensor.cpu(),
        m._dummy_request_mask.cpu(),
        dict(m._request_id_to_state_index),
        dict(m._request_id_to_is_dummy),
    )


def _same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2] and a[3] == b[3]


# (requests, {rid: ssm base}) per step: arrivals, a moved SSM block, a dummy pad, a departure.
_STEPS = [
    ([_req(5)], {5: 40}),
    ([_req(5)], {5: 40}),
    ([_req(5), _req(7)], {5: 40, 7: 12}),
    ([_req(5), _req(7)], {5: 40, 7: 12}),
    ([_req(5), _req(7)], {5: 41, 7: 12}),
    ([_req(5), _req(7), _req(-1, dummy=True)], {5: 41, 7: 12, -1: 0}),
    ([_req(5), _req(7), _req(-1, dummy=True)], {5: 41, 7: 12, -1: 0}),
    ([_req(7)], {7: 12}),
    ([_req(7)], {7: 12}),
]


def test_steady_manager_matches_full_path(fast_paths):
    full, steady = _manager(), _manager()
    for requests, bases in _STEPS:
        for m in (full, steady):
            for rid, base in bases.items():
                _set_cache(m, rid, base)
        fast_paths(False)
        full.setup(requests)
        fast_paths(True)
        steady.setup(requests)
        assert _same(_state(full), _state(steady))


def test_steady_manager_skips_only_unchanged(fast_paths):
    fast_paths(True)
    m = _manager()
    _set_cache(m, 5, 40)
    m.setup([_req(5)])
    # A skipped step leaves a tampered host staging buffer alone (shows the skip) ...
    m._host_state_indices[0] = -7
    m.setup([_req(5)])
    assert int(m._host_state_indices[0]) == -7
    # ... and any change takes the full path again.
    _set_cache(m, 5, 41)
    m.setup([_req(5)])
    assert int(m._host_state_indices[0]) == 41
    # Without the fast paths every step uploads.
    fast_paths(False)
    m._host_state_indices[0] = -7
    m.setup([_req(5)])
    assert int(m._host_state_indices[0]) == 41


def test_kda_replay_never_skips(fast_paths):
    fast_paths(True)
    m = _manager()
    m.use_kda_replay_update = True
    m._kda_replay_suspended = set()
    m._relocate_kda_replay_slots = lambda old, new: None
    m._reset_kda_replay_slots = lambda slots: None
    _set_cache(m, 5, 40)
    m.setup([_req(5)])
    # KDA replay bookkeeping runs on every setup, so the steady skip is off.
    m._host_state_indices[0] = -7
    m.setup([_req(5)])
    assert int(m._host_state_indices[0]) == 40


def test_recurring_request_id_after_free_takes_full_path(fast_paths):
    """Warmup and CUDA-graph dummies are freed and re-added under the same id."""
    fast_paths(True)
    m = _manager()
    _set_cache(m, 0, 33)
    m.setup([_req(0)])
    # free_resources() prunes the per-request maps (the real one also drops the key; here only the maps).
    m._request_id_to_state_index.pop(0)
    m._request_id_to_is_dummy.pop(0)
    m.setup([_req(0)])
    assert m._request_id_to_state_index == {0: 33}
    assert m._request_id_to_is_dummy == {0: False}
