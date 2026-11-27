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
"""Host-tier prefix prefetch to the GPU for waiting context requests (RuntimeHooks.kv_gpu_prefetch).

The executor side: which requests are prefetched, the rank-agreed admission hold derived from the
bytes moved, its count-down, and the model opt-in. KVCacheManagerV2.prefetch_to_gpu itself is
covered in kv_cache_manager_v2_tests/test_kv_gpu_prefetch_runtime.py; its locality domain and
failure handling are checked here against a stub cache.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

import tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 as kvm
import tensorrt_llm._torch.pyexecutor.py_executor as pe
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


def _req(rid, first=True):
    return SimpleNamespace(py_request_id=rid, is_first_context_chunk=first)


def _executor(reqs, nbytes_by_rid, tp_size=1, peer_holds=None):
    kvc = SimpleNamespace(
        kv_cache_map={},
        prefetch_to_gpu=mock.Mock(side_effect=lambda r: nbytes_by_rid.get(r.py_request_id, 0)),
    )

    def allgather(holds):
        return [holds] + [peer_holds or holds for _ in range(tp_size - 1)]

    dist = SimpleNamespace(tp_size=tp_size, rank=0, tp_allgather=mock.Mock(side_effect=allgather))
    return SimpleNamespace(
        active_requests=reqs, kv_cache_manager=kvc, dist=dist, enable_attention_dp=False
    )


def test_hold_from_bytes_and_cap(monkeypatch):
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_MAX", 2)
    reqs = [_req(1), _req(2), _req(3)]
    ex = _executor(reqs, {1: 0, 2: 5_200_000_000, 3: 1})
    pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    assert reqs[0].py_kv_gpu_hold == 0  # already on the GPU: no hold
    assert reqs[1].py_kv_gpu_hold == 4  # ceil(5.2e9 / 1.5e9)
    assert not hasattr(reqs[2], "py_kv_gpu_hold")  # beyond the per-step cap
    assert ex.kv_cache_manager.prefetch_to_gpu.call_count == 2


def test_hold_clamped_and_counts_down(monkeypatch):
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_MAX", 2)
    r = _req(7)
    ex = _executor([r], {7: 100_000_000_000})
    pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    assert r.py_kv_gpu_hold == pe._KV_GPU_PREFETCH_MAX_HOLD
    for want in range(pe._KV_GPU_PREFETCH_MAX_HOLD - 1, -1, -1):
        pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)  # once per step
        assert r.py_kv_gpu_hold == want
    assert ex.kv_cache_manager.prefetch_to_gpu.call_count == 1  # prefetched once


@pytest.mark.parametrize("attention_dp, expected", [(False, (3, 1)), (True, (0, 1))])
def test_hold_is_max_over_ranks_unless_attention_dp(monkeypatch, attention_dp, expected):
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_MAX", 2)
    a, b = _req(1), _req(2)
    ex = _executor([a, b], {1: 0, 2: 1_000}, tp_size=2, peer_holds=[3, 0])
    ex.enable_attention_dp = attention_dp
    pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    assert (a.py_kv_gpu_hold, b.py_kv_gpu_hold) == expected
    assert ex.dist.tp_allgather.called is not attention_dp


@pytest.mark.parametrize(
    "opt_in,killed,armed", [(False, False, False), (True, False, True), (True, True, False)]
)
def test_prefetch_is_a_model_opt_in(monkeypatch, opt_in, killed, armed):
    # Off unless the model opts in; the kill switch wins over the opt-in.
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_KILLED", killed)
    assert pe._kv_gpu_prefetch_enabled(RuntimeHooks(kv_gpu_prefetch=opt_in)) is armed
    # The choice is the executor's own: arming the process-wide opt-ins never turns it on.
    pe._arm_runtime_opt_ins(RuntimeHooks(kv_gpu_prefetch=opt_in))
    assert pe.PyExecutor._kv_gpu_prefetch is False
    assert not hasattr(pe, "_KV_GPU_PREFETCH")


@pytest.mark.parametrize("armed", [False, True])
def test_prefetch_runs_only_when_armed(armed):
    ex = SimpleNamespace(
        _kv_gpu_prefetch=armed,
        _is_kv_manager_v2=True,
        kv_cache_manager=SimpleNamespace(enable_block_reuse=True, disk_prefetch_num_reqs=0),
        _prefetch_context_prefixes_to_gpu=mock.Mock(),
    )
    pe.PyExecutor._prefetch_for_context_requests(ex)
    assert ex._prefetch_context_prefixes_to_gpu.call_count == int(armed)


def _manager_view(kv_cache, num_locality_domains):
    """The KVCacheManagerV2 state prefetch_to_gpu reads, around a stub native cache."""
    view = SimpleNamespace(
        enable_block_reuse=True,
        impl=SimpleNamespace(create_kv_cache=mock.Mock(return_value=kv_cache)),
        num_locality_domains=num_locality_domains,
        is_draft=False,
        _context_reuse_tokens=lambda req: [1, 2, 3],
        _derive_reuse_salt=lambda salt: None,
        _kv_token_bytes_estimate=lambda: 100,
    )
    view._request_locality_domain_id = lambda req: kvm.KVCacheManagerV2._request_locality_domain_id(
        view, req
    )
    return view


def _stub_cache(prefetch):
    levels = [0] * (kvm.GPU_LEVEL + 2)
    levels[kvm.GPU_LEVEL + 1] = 4
    return SimpleNamespace(cached_tokens_by_level=levels, prefetch=prefetch, close=mock.Mock())


def _prefetch_req():
    return SimpleNamespace(
        py_request_id=7, cache_salt=None, lora_task_id=None, py_locality_domain_id=1
    )


def test_gpu_prefetch_uses_the_request_locality_domain():
    kv_cache = _stub_cache(mock.Mock(return_value=True))
    view = _manager_view(kv_cache, num_locality_domains=2)
    assert kvm.KVCacheManagerV2.prefetch_to_gpu(view, _prefetch_req()) == 400
    assert view.impl.create_kv_cache.call_args.kwargs["locality_domain_id"] == 1
    kv_cache.close.assert_called_once()


def test_gpu_prefetch_remote_failure_holds_nothing():
    kv_cache = _stub_cache(mock.Mock(side_effect=kvm.RemoteMemError("remote level retired")))
    view = _manager_view(kv_cache, num_locality_domains=1)
    req = _prefetch_req()
    req.py_locality_domain_id = None
    assert kvm.KVCacheManagerV2.prefetch_to_gpu(view, req) == 0
    kv_cache.close.assert_called_once()
