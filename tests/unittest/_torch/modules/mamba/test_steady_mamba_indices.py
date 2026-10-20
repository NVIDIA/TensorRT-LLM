# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_STEADY_MAMBA_INDICES: a steady decode batch skips the Mamba state-index / dummy-mask uploads,
leaving exactly the buffers the full path would write (compared bitwise against the flag-off path), and any
change of requests, SSM base blocks or dummy flags takes the full path."""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    STEADY_MAMBA_INDICES_ENV,
    MambaHybridCacheManager,
    MambaHybridCacheManagerV2,
)


def _manager(max_batch=4):
    """The attributes _setup_state_indices / _refresh_dummy_request_mask use, on CPU tensors."""
    m = types.SimpleNamespace(
        local_num_mamba_layers=1,
        ssm_layer_group_id=0,
        kv_cache_map={},
        _host_state_indices=torch.zeros(max_batch, dtype=torch.int32),
        cuda_state_indices=torch.zeros(max_batch, dtype=torch.int32),
        _dummy_request_mask_host=torch.zeros(max_batch, dtype=torch.bool),
        _dummy_request_mask=torch.zeros(max_batch, dtype=torch.bool),
        _request_id_to_state_index={},
        _request_id_to_is_dummy={},
    )
    m._refresh_dummy_request_mask = types.MethodType(
        MambaHybridCacheManager._refresh_dummy_request_mask, m
    )
    m.setup = types.MethodType(MambaHybridCacheManagerV2._setup_state_indices, m)
    return m


def _set_cache(m, rid, base):
    m.kv_cache_map[rid] = types.SimpleNamespace(get_ssm_block_base_index=lambda _gid, b=base: b)


def _req(rid, dummy=False):
    return types.SimpleNamespace(py_request_id=rid, is_dummy=dummy)


def _state(m):
    return (
        m.cuda_state_indices.clone(),
        m._dummy_request_mask.clone(),
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


def test_steady_manager_matches_full_path(monkeypatch):
    full, steady = _manager(), _manager()
    for requests, bases in _STEPS:
        for m in (full, steady):
            for rid, base in bases.items():
                _set_cache(m, rid, base)
        monkeypatch.delenv(STEADY_MAMBA_INDICES_ENV, raising=False)
        full.setup(requests)
        monkeypatch.setenv(STEADY_MAMBA_INDICES_ENV, "1")
        steady.setup(requests)
        assert _same(_state(full), _state(steady))


def test_steady_manager_skips_only_unchanged(monkeypatch):
    monkeypatch.setenv(STEADY_MAMBA_INDICES_ENV, "1")
    m = _manager()
    _set_cache(m, 5, 40)
    m.setup([_req(5)])
    # Tamper with the device buffer: a skipped step leaves it alone (shows the skip) ...
    m.cuda_state_indices[0] = -7
    m.setup([_req(5)])
    assert int(m.cuda_state_indices[0]) == -7
    # ... and any change takes the full path again.
    _set_cache(m, 5, 41)
    m.setup([_req(5)])
    assert int(m.cuda_state_indices[0]) == 41
    # Negative control: with the flag off every step uploads.
    monkeypatch.delenv(STEADY_MAMBA_INDICES_ENV, raising=False)
    m.cuda_state_indices[0] = -7
    m.setup([_req(5)])
    assert int(m.cuda_state_indices[0]) == 41


def test_recurring_request_id_after_free_takes_full_path(monkeypatch):
    """Warmup and CUDA-graph dummies are freed and re-added under the same id (no stale-key KeyError)."""
    monkeypatch.setenv(STEADY_MAMBA_INDICES_ENV, "1")
    m = _manager()
    _set_cache(m, 0, 33)
    m.setup([_req(0)])
    # free_resources() prunes the per-request maps (the real one also drops the key; here only the maps).
    m._request_id_to_state_index.pop(0)
    m._request_id_to_is_dummy.pop(0)
    m.setup([_req(0)])
    assert m._request_id_to_state_index == {0: 33}
    assert m._request_id_to_is_dummy == {0: False}


def test_dummy_mask_refresh_skip(monkeypatch):
    monkeypatch.setenv(STEADY_MAMBA_INDICES_ENV, "1")
    m = _manager()
    m._refresh_dummy_request_mask([False, True])
    assert m._dummy_request_mask.tolist() == [False, True, False, False]
    m._dummy_request_mask[0] = True  # tamper
    m._refresh_dummy_request_mask([False, True])
    assert bool(m._dummy_request_mask[0])  # skipped
    m._refresh_dummy_request_mask([False, True, False])
    assert m._dummy_request_mask.tolist() == [False, True, False, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mamba2_metadata_steady_matches_full_path(monkeypatch):
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

    class _Kv:
        def __init__(self):
            self.indices = {}

        def get_state_indices(self, request_ids, is_padding):
            return [self.indices[r] for r in request_ids]

    def attn(kv, rids):
        n = len(rids)
        return types.SimpleNamespace(
            seq_lens=torch.ones(n, dtype=torch.int32),
            seq_lens_cuda=torch.ones(n, dtype=torch.int32, device="cuda"),
            num_contexts=0,
            num_ctx_tokens=0,
            kv_cache_manager=kv,
            request_ids=list(rids),
            kv_cache_params=types.SimpleNamespace(num_cached_tokens_per_seq=[100] * n),
        )

    full, steady = Mamba2Metadata(4, 128), Mamba2Metadata(4, 128)
    kv = _Kv()
    seq = [
        ([5], {5: 40}),
        ([5], {5: 40}),
        ([5, 7], {5: 40, 7: 12}),
        ([5, 7], {5: 40, 7: 12}),
        ([5, 7], {5: 41, 7: 12}),
        ([7], {7: 12}),
        ([7], {7: 12}),
    ]
    for rids, idx in seq:
        kv.indices = dict(idx)
        monkeypatch.delenv("TRTLLM_Q35_STEADY_MAMBA_INDICES", raising=False)
        full.prepare(attn(kv, rids))
        monkeypatch.setenv("TRTLLM_Q35_STEADY_MAMBA_INDICES", "1")
        steady.prepare(attn(kv, rids))
        torch.cuda.synchronize()
        n = len(rids)
        assert torch.equal(full.state_indices[:n].cpu(), steady.state_indices[:n].cpu())
        assert torch.equal(full.state_indices_long.cpu(), steady.state_indices_long.cpu())
        assert steady.state_indices_long.shape[0] == n
    # Negative control: a tampered device buffer survives a steady step only with the flag on.
    steady.state_indices[0] = -3
    steady.prepare(attn(kv, [7]))
    torch.cuda.synchronize()
    assert int(steady.state_indices[0]) == -3
    monkeypatch.delenv("TRTLLM_Q35_STEADY_MAMBA_INDICES", raising=False)
    steady.prepare(attn(kv, [7]))
    torch.cuda.synchronize()
    assert int(steady.state_indices[0]) == 12
