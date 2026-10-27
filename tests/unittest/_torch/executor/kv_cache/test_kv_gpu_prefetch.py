# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-tier prefix prefetch to the GPU for waiting context requests, with a rank-agreed admission hold."""

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
    return SimpleNamespace(active_requests=reqs, kv_cache_manager=kvc, dist=dist)


def test_hold_from_bytes_and_cap(monkeypatch):
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_MAX", 2)
    reqs = [_req(1), _req(2), _req(3)]
    ex = _executor(reqs, {1: 0, 2: 5_200_000_000, 3: 1})
    pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    assert reqs[0].py_kv_gpu_hold == 0  # already on GPU: no hold
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


def test_hold_is_max_over_ranks(monkeypatch):
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_MAX", 2)
    a, b = _req(1), _req(2)
    ex = _executor([a, b], {1: 0, 2: 1_000}, tp_size=2, peer_holds=[3, 0])
    pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    assert (a.py_kv_gpu_hold, b.py_kv_gpu_hold) == (3, 1)


def test_skips_non_first_chunks_and_admitted(monkeypatch):
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_MAX", 2)
    a, b = _req(1, first=False), _req(2)
    ex = _executor([a, b], {1: 10**9, 2: 10**9})
    ex.kv_cache_manager.kv_cache_map[2] = object()
    pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    assert ex.kv_cache_manager.prefetch_to_gpu.call_count == 0


def _kvm(tokens_by_level, prefetch_ok=True):
    cache = mock.Mock()
    cache.cached_tokens_by_level = tokens_by_level
    cache.prefetch.return_value = prefetch_ok
    impl = SimpleNamespace(create_kv_cache=mock.Mock(return_value=cache))
    self = SimpleNamespace(
        enable_block_reuse=True,
        impl=impl,
        _context_reuse_tokens=lambda r: [1, 2, 3],
        _derive_reuse_salt=lambda s: None,
        _kv_token_bytes_estimate=lambda: 1000,
    )
    return self, cache


def test_prefetch_only_when_off_gpu_and_always_closes():
    req = SimpleNamespace(cache_salt=None, lora_task_id=None)
    self, cache = _kvm([96, 0])
    assert kvm.KVCacheManagerV2.prefetch_to_gpu(self, req) == 0
    cache.prefetch.assert_not_called()
    cache.close.assert_called_once()
    self, cache = _kvm([32, 48, 16])
    assert kvm.KVCacheManagerV2.prefetch_to_gpu(self, req) == (48 + 16) * 1000
    cache.prefetch.assert_called_once_with(kvm.GPU_LEVEL)
    cache.close.assert_called_once()


def test_prefetch_failure_means_no_hold():
    req = SimpleNamespace(cache_salt=None, lora_task_id=None)
    self, cache = _kvm([0, 64], prefetch_ok=False)
    assert kvm.KVCacheManagerV2.prefetch_to_gpu(self, req) == 0
    cache.close.assert_called_once()


@pytest.mark.parametrize(
    "opt_in,killed,armed", [(False, False, False), (True, False, True), (True, True, False)]
)
def test_prefetch_is_a_model_opt_in(monkeypatch, opt_in, killed, armed):
    # Off unless the model opts in (RuntimeHooks.kv_gpu_prefetch); the kill switch wins over the opt-in.
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH", False)
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_KILLED", killed)
    pe._arm_runtime_opt_ins(RuntimeHooks(kv_gpu_prefetch=opt_in))
    assert pe._KV_GPU_PREFETCH is armed


@pytest.mark.parametrize("hold", [0, 1, 3])
def test_scheduler_guard_present(hold):
    # The scheduler skips a pending context request while py_kv_gpu_hold > 0.
    import inspect

    import tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 as sv2

    assert 'hold = getattr(req, "py_kv_gpu_hold", 0)' in inspect.getsource(sv2)
    assert "type(hold) is int and hold > 0" in inspect.getsource(sv2)
    assert (getattr(SimpleNamespace(py_kv_gpu_hold=hold), "py_kv_gpu_hold", 0) > 0) == (hold > 0)


def test_mock_requests_are_never_held():
    # Stock scheduler tests drive Mock requests: getattr on a Mock returns a Mock, which must not count as a hold.
    hold = getattr(mock.Mock(), "py_kv_gpu_hold", 0)
    assert not (type(hold) is int and hold > 0)


def test_counter_prints_every_200_steps_even_without_candidates(monkeypatch):
    log = mock.Mock()
    monkeypatch.setattr(pe, "logger", log)
    ex = _executor([], {})
    for _ in range(400):
        pe.PyExecutor._prefetch_context_prefixes_to_gpu(ex)
    lines = [c.args[0] for c in log.info.call_args_list]
    assert lines == [
        "[KVPF] steps 200 candidates 0 prefetched 0",
        "[KVPF] steps 400 candidates 0 prefetched 0",
    ]
    assert ex.kv_cache_manager.prefetch_to_gpu.call_count == 0


@pytest.mark.parametrize(
    "switch,killed,v2,reuse,want",
    [
        (True, False, True, True, "enabled"),
        (False, True, True, True, "disabled (kill switch)"),
        (False, False, True, True, "inactive (the model does not opt in)"),
        (True, False, False, True, "inactive (KV cache manager v1)"),
        (True, False, True, False, "inactive (block reuse off)"),
    ],
)
def test_startup_line_once_with_state(monkeypatch, switch, killed, v2, reuse, want):
    log = mock.Mock()
    monkeypatch.setattr(pe, "logger", log)
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH", switch)
    monkeypatch.setattr(pe, "_KV_GPU_PREFETCH_KILLED", killed)
    ex = SimpleNamespace(
        dist=SimpleNamespace(rank=0),
        _is_kv_manager_v2=v2,
        kv_cache_manager=SimpleNamespace(enable_block_reuse=reuse, disk_prefetch_num_reqs=0),
        _prefetch_context_prefixes_to_gpu=mock.Mock(),
    )
    ex._kv_gpu_prefetch_state = lambda: pe.PyExecutor._kv_gpu_prefetch_state(ex)
    for _ in range(3):
        pe.PyExecutor._prefetch_for_context_requests(ex)
    startup = [c.args[0] for c in log.info.call_args_list if "[KVPF] startup" in c.args[0]]
    assert len(startup) == 1 and startup[0].startswith(f"[KVPF] startup: {want} (")
    assert ex._prefetch_context_prefixes_to_gpu.call_count == (3 if switch and v2 and reuse else 0)
