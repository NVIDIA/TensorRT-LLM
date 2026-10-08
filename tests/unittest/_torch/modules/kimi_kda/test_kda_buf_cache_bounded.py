# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounds of the shape-keyed KDA prefill scratch caches.

``_buf_cache``, ``_padded_input_cache`` and ``_g_sentinel_cache`` are keyed by
the packed prefill token count, which changes on almost every executor
iteration, so each of them is LRU-bounded by ``_lru_put`` / ``_lru_touch``.
One ``_buf_cache`` entry holds about 1 GiB at Kimi K3 sizes and is allocated
after the KV-cache pool is sized, so the cap (``TLLM_KDA_BUF_CACHE_ENTRIES``,
default 2) is a real memory knob.

Most tests here call the cache getters directly and launch no KDA kernel, so
they run on any CUDA GPU. Those cannot see the failure that matters most: the
K123 launch used to wrap the scratch in a module-level ``id()``-keyed CuTe
wrapper cache, and a wrapper holds a strong reference to its tensor, so every
evicted entry stayed alive. ``test_prefill_scratch_plateaus_over_distinct_token_counts``
drives a real prefill to cover that, and the two ownership tests guard the
layout that prevents it (one of them reads the source and needs no GPU).
"""

import ast
import gc
from pathlib import Path

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


_OP_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "tensorrt_llm"
    / "_torch"
    / "custom_ops"
    / "cute_dsl_kimi_k3_custom_ops.py"
)


def test_no_module_level_wrapper_cache_over_scratch() -> None:
    """The K123 launch must not wrap scratch through an ``id()``-keyed cache.

    Reads the source instead of importing it, so it runs without a GPU or the
    CuTe DSL. A module-level wrapper cache pins the storage of every tensor it
    has seen, which turns the ``_buf_cache`` LRU eviction into a no-op.
    """
    tree = ast.parse(_OP_SOURCE.read_text())
    launch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_fused_k123_inv"
    )
    called = {
        node.func.id
        for node in ast.walk(launch)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_ct_cached" not in called
    module_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_input_wrap_cache" not in module_names


@torch.no_grad()
def test_k123_wrappers_live_in_the_cache_entry(clean_caches) -> None:
    """The wrappers K123 consumes are owned by the ``_get_buffers`` entry."""
    module = clean_caches
    device = torch.device("cuda", torch.cuda.current_device())
    wrappers = _get_buffers(module, device, T0)[-1]
    for name in (
        "k123_ks_ct",
        "k123_kg_ct",
        "k123_qs_ct",
        "k123_beta_activated_ct",
        "k123_gk_ct",
        "k123_aqk_ct",
        "k123_akk_ct",
    ):
        assert name in wrappers, f"{name} must be owned by the _get_buffers entry"


# Real prefill over distinct token counts. The pre-fix code kept one full
# buffer set alive per distinct T; small T keeps that from exhausting the GPU
# before the assertion fires.
OP_NUM_HEADS = 96
OP_HEAD_DIM = 128
OP_T0 = 512
OP_N_SHAPES = 24


def _op_set_bytes(t: int) -> int:
    return (t + BT) * OP_NUM_HEADS * 2 * (3 * OP_HEAD_DIM + 2 * BT + OP_HEAD_DIM)


@torch.no_grad()
def test_prefill_scratch_plateaus_over_distinct_token_counts(clean_caches) -> None:
    pytest.importorskip("fla")
    from kda_prefill_test_utils import run_indexed_prefill

    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (
        KDAKernelDispatch,
        is_kda_optimized_supported,
    )

    if not is_kda_optimized_supported():
        pytest.skip("KDA optimized prefill needs SM100/SM103/SM107")
    module = clean_caches
    device = torch.device("cuda", torch.cuda.current_device())
    dispatch = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)
    assert dispatch.prefill_kernel_path == "optimized"
    generator = torch.Generator(device=device).manual_seed(0)
    a_log = torch.randn(OP_NUM_HEADS, generator=generator, device=device) * 0.5
    dt_bias = torch.randn(OP_NUM_HEADS * OP_HEAD_DIM, generator=generator, device=device) * 0.1

    def run(total_t: int) -> None:
        shape = (1, total_t, OP_NUM_HEADS, OP_HEAD_DIM)
        inputs = tuple(
            torch.randn(*shape, generator=generator, device=device).to(torch.bfloat16)
            for _ in range(4)
        ) + (torch.randn(1, total_t, OP_NUM_HEADS, generator=generator, device=device),)
        cu_seqlens = torch.tensor([0, total_t], dtype=torch.long, device=device)
        run_indexed_prefill(dispatch, (a_log, dt_bias), inputs, cu_seqlens)

    # Compile caches and the first buffer set must land before the baseline.
    for i in range(2):
        run(OP_T0 + BT * i)
    _settle()
    module._buf_cache.clear()
    _settle()
    base = torch.cuda.memory_allocated(device)

    for i in range(OP_N_SHAPES):
        run(OP_T0 + BT * i)
    _settle()
    live = torch.cuda.memory_allocated(device) - base

    cap = module._BUF_CACHE_MAX_ENTRIES
    # Clamp the cap so raising TLLM_KDA_BUF_CACHE_ENTRIES cannot hide a leak.
    budget = (min(cap, 4) + 2) * _op_set_bytes(OP_T0 + BT * (OP_N_SHAPES - 1)) + (128 << 20)
    assert len(module._buf_cache) <= cap
    assert live <= budget, (
        f"KDA prefill scratch grows with the token count: {live / 2**20:.0f} MiB live "
        f"after {OP_N_SHAPES} distinct T, budget {budget / 2**20:.0f} MiB for {cap} entries"
    )
