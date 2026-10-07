# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounds of the shape-keyed KDA prefill scratch caches.

``_buf_cache``, ``_padded_input_cache`` and ``_g_sentinel_cache`` are keyed by
the packed prefill token count, which changes on almost every executor
iteration, so each of them is LRU-bounded by ``_lru_put`` / ``_lru_touch``.
One ``_buf_cache`` entry holds about 1 GiB at Kimi K3 sizes and is allocated
after the KV-cache pool is sized, so the cap (``TLLM_KDA_BUF_CACHE_ENTRIES``,
default 2) is a real memory knob.

These tests call the cache getters directly and launch no KDA kernel, so they
run on any CUDA GPU. Releasing the scratch of entries evicted after a real
prefill launch is covered by ``test_kda_cache_soundness.py``.
"""

import gc

import pytest
import torch

# Small on purpose: the bound does not depend on the shape, and K3's H=96 at
# T=8192 would take about 1 GiB per entry.
H = 4
K_DIM = 128
V_DIM = 128
BT = 64
T0 = 1024
N_SHAPES = 16


def _op_module():
    # The module raises ImportError at import time without CuTe DSL / FlashInfer.
    return pytest.importorskip(
        "tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops", exc_type=ImportError
    )


def _settle() -> None:
    gc.collect()
    torch.cuda.synchronize()


def test_lru_put_evicts_least_recently_used() -> None:
    module = _op_module()
    cache = {}
    module._lru_put(cache, "a", 1, max_entries=2)
    module._lru_put(cache, "b", 2, max_entries=2)
    assert module._lru_touch(cache, "a") == 1
    assert module._lru_put(cache, "c", 3, max_entries=2) == 3
    assert list(cache) == ["a", "c"]


@pytest.mark.parametrize(
    ("value", "expected"), [(None, 2), ("5", 5), ("1", 1), ("0", 1), ("abc", 2), ("", 2)]
)
def test_scratch_cache_cap_from_env(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    module = _op_module()
    if value is None:
        monkeypatch.delenv("TLLM_KDA_BUF_CACHE_ENTRIES", raising=False)
    else:
        monkeypatch.setenv("TLLM_KDA_BUF_CACHE_ENTRIES", value)
    assert module._scratch_cache_max_entries() == expected


@pytest.fixture
def clean_caches():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    module = _op_module()
    caches = (module._buf_cache, module._padded_input_cache, module._g_sentinel_cache)
    saved = [dict(cache) for cache in caches]
    for cache in caches:
        cache.clear()
    _settle()
    yield module
    for cache, entries in zip(caches, saved):
        cache.clear()
        cache.update(entries)
    _settle()


def _get_buffers(module, device: torch.device, t: int):
    return module._get_buffers(
        device, torch.bfloat16, 1, t, H, K_DIM, V_DIM, t // BT, 1, BT, varlen=True
    )


def _buf_entry_bytes(t: int) -> int:
    """Bytes of the large tensors of one varlen ``_get_buffers`` entry.

    k_scaled, kg and q_scaled are [T + BT, H, K]; A_qk and A_kk are
    [T + BT, H, BT]; O_flat is [T + BT, H, V]; all bf16. The smaller tensors
    are covered by the slack of the budgets below.
    """
    return (t + BT) * H * 2 * (3 * K_DIM + 2 * BT + V_DIM)


@torch.no_grad()
def test_buf_cache_holds_at_most_cap_entries(clean_caches) -> None:
    module = clean_caches
    device = torch.device("cuda", torch.cuda.current_device())
    base = torch.cuda.memory_allocated(device)

    for i in range(N_SHAPES):
        _get_buffers(module, device, T0 + BT * i)
    _settle()

    cap = module._BUF_CACHE_MAX_ENTRIES
    assert len(module._buf_cache) <= cap
    live = torch.cuda.memory_allocated(device) - base
    budget = int(1.25 * cap * _buf_entry_bytes(T0 + BT * (N_SHAPES - 1))) + (8 << 20)
    assert live <= budget, (
        f"{live / 2**20:.1f} MiB live after {N_SHAPES} shapes, "
        f"budget {budget / 2**20:.1f} MiB for {cap} entries"
    )

    module._buf_cache.clear()
    _settle()
    leaked = torch.cuda.memory_allocated(device) - base
    assert leaked < (1 << 20), f"{leaked / 2**20:.1f} MiB survived clearing _buf_cache"


@torch.no_grad()
def test_buf_cache_keeps_recently_used_entry(clean_caches) -> None:
    module = clean_caches
    device = torch.device("cuda", torch.cuda.current_device())
    cap = module._BUF_CACHE_MAX_ENTRIES
    if cap < 2:
        pytest.skip("a recently used entry survives an insertion only with a cap of 2 or more")

    hot = _get_buffers(module, device, T0)
    for i in range(1, 2 * cap + 1):
        assert _get_buffers(module, device, T0) is hot
        _get_buffers(module, device, T0 + BT * i)
    assert _get_buffers(module, device, T0) is hot


@torch.no_grad()
def test_padded_input_and_g_sentinel_caches_are_bounded(clean_caches) -> None:
    module = clean_caches
    device = torch.device("cuda", torch.cuda.current_device())

    for i in range(N_SHAPES):
        t_padded = T0 + BT * i
        real_t = t_padded - 7
        module._get_g_sentinel_buffer(1, t_padded, H, K_DIM, torch.bfloat16, device, real_t)
        module._get_padded_input_buffers(
            1,
            t_padded,
            H,
            K_DIM,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            device,
            real_t,
        )

    assert len(module._g_sentinel_cache) <= module._G_SENTINEL_CACHE_MAX_ENTRIES
    assert len(module._padded_input_cache) <= module._PADDED_INPUT_CACHE_MAX_ENTRIES
