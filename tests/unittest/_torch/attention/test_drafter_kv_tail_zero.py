# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate of the drafter last-row context FMHA's K/V tail zeroing (drafter_last_row._zero_kv_tails)."""

import types
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import drafter_last_row
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.manager import FmhaManager
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

PAGE, HEADS, D = 128, 4, 128


@pytest.mark.cpu_only
@pytest.mark.parametrize("phase", [None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION])
@pytest.mark.parametrize("is_cross", [False, True])
@pytest.mark.parametrize("mask", [PredefinedAttentionMask.CAUSAL, PredefinedAttentionMask.FULL])
@pytest.mark.parametrize("parent_supported", [False, True])
def test_drafter_last_row_rejects_unsupported_requests_before_preprocessing(
    phase: FmhaPhase | None,
    is_cross: bool,
    mask: PredefinedAttentionMask,
    parent_supported: bool,
) -> None:
    """Only causal self-attention context reaches the inherited shape and backend checks."""
    backend = Mock(spec_set=["attn", "_is_supported_with_reason", "_logged_not_taken"])
    backend.attn = Mock(spec_set=TrtllmAttention)
    backend._logged_not_taken = False
    backend._is_supported_with_reason.return_value = (parent_supported, "test backend")
    metadata = Mock(spec_set=["is_cross"])
    metadata.is_cross = is_cross
    forward_args = AttentionForwardArgs(attention_mask=mask)
    q = torch.empty((1, HEADS * D))

    supported = drafter_last_row.DrafterLastRowContextFmha._is_supported(
        backend, q, None, None, metadata, forward_args, phase=phase
    )

    admitted = (
        phase is FmhaPhase.CONTEXT and not is_cross and mask is PredefinedAttentionMask.CAUSAL
    )
    assert supported is (admitted and parent_supported)
    if admitted:
        backend._is_supported_with_reason.assert_called_once_with(
            q, None, None, backend.attn, metadata, forward_args, phase=phase
        )
    else:
        backend._is_supported_with_reason.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("initialized", [False, True], ids=["deferred", "initialized"])
def test_mark_drafter_last_row_preserves_manager_lifecycle(initialized: bool) -> None:
    """A deferred backend can be marked before its FMHA manager is constructed."""
    backend = TrtllmAttention.__new__(TrtllmAttention)
    manager = None
    if initialized:
        manager = FmhaManager.__new__(FmhaManager)
        library = drafter_last_row.DrafterLastRowContextFmha.__new__(
            drafter_last_row.DrafterLastRowContextFmha
        )
        manager.fmha_libs = [library]
    backend._fmha_manager = manager
    draft = torch.nn.Module()
    draft.attn = backend
    assert not drafter_last_row.DrafterLastRowContextFmha._is_available(backend)
    assert drafter_last_row.mark_drafter_last_row_attention(draft) == 1
    assert drafter_last_row.DrafterLastRowContextFmha._is_available(backend)
    assert backend._fmha_manager is manager


def _pool_and_tables(kv_lens, seed=0):
    """A flat FP8 sub-block pool filled with NaN-coded and random bytes, and a shuffled [batch, 2, pages] table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [(n + PAGE - 1) // PAGE for n in kv_lens]
    num_blocks = 2 * sum(pages) + 6
    raw = torch.randint(
        0, 256, (num_blocks, HEADS, PAGE, D), dtype=torch.uint8, device="cuda", generator=g
    )
    raw[::3] = 0x7F
    perm = torch.randperm(num_blocks, device="cuda", generator=g)
    tables = torch.zeros(len(kv_lens), 2, max(pages), dtype=torch.int32, device="cuda")
    used = 0
    for b, n in enumerate(pages):
        tables[b, 0, :n] = perm[used : used + n]
        tables[b, 1, :n] = perm[used + n : used + 2 * n]
        used += 2 * n
    return raw.view(torch.float8_e4m3fn), tables


@pytest.mark.parametrize(
    "kv_lens",
    [[16390], [16], [32, 128, 129], [1, 255, 256, 257], [16384, 6]],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_only_the_partial_last_pages_tails_are_zeroed(kv_lens):
    pool, tables = _pool_and_tables(kv_lens)
    before = pool.view(torch.uint8).clone()
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    drafter_last_row._zero_kv_tails(pool, tables, lens, len(kv_lens))
    after = pool.view(torch.uint8)

    expected = before.clone()
    for b, n in enumerate(kv_lens):
        if n % PAGE:
            for half in (0, 1):
                expected[int(tables[b, half, (n - 1) // PAGE]), :, n % PAGE :, :] = 0
    assert torch.equal(after, expected)
    for b, n in enumerate(kv_lens):
        if n % PAGE:
            for half in (0, 1):
                tail = after[int(tables[b, half, (n - 1) // PAGE]), :, n % PAGE :, :]
                assert int(((tail & 0x7F) == 0x7F).sum()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rows_past_batch_are_untouched():
    pool, tables = _pool_and_tables([100, 200])
    before = pool.view(torch.uint8).clone()
    lens = torch.tensor([100, 200], dtype=torch.int32, device="cuda")
    drafter_last_row._zero_kv_tails(pool, tables, lens, 1)
    after = pool.view(torch.uint8)
    second = [int(tables[1, h, 1]) for h in (0, 1)]
    for blk in second:
        assert torch.equal(after[blk], before[blk])


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu_only),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_nvfp4_tail_zeroes_packed_data_and_nan_scales(device: str, batch: int) -> None:
    """Zero both NVFP4 pools without changing live tokens, full pages, or inactive requests."""
    lengths = [PAGE + 5, PAGE, 9]
    tables = torch.tensor(
        [[[3, 5], [7, 1]], [[6, 0], [2, 0]], [[4, 0], [8, 0]]],
        dtype=torch.int32,
        device=device,
    )
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    packed = torch.full((10, HEADS, PAGE, D // 2), 0x22, dtype=torch.uint8, device=device)
    scale_bytes = torch.full((10, HEADS, PAGE, D // 16), 0x7F, dtype=torch.uint8, device=device)
    scales = scale_bytes.view(torch.float8_e4m3fn)
    for b, length in enumerate(lengths):
        for token in range(length):
            for half in (0, 1):
                scale_bytes[int(tables[b, half, token // PAGE]), :, token % PAGE] = 0x38
    expected_data = packed.clone()
    expected_scales = scale_bytes.clone()
    for b, length in enumerate(lengths[:batch]):
        if length % PAGE:
            for half in (0, 1):
                block = int(tables[b, half, (length - 1) // PAGE])
                tail = slice(length % PAGE, None)
                expected_data[block, :, tail] = 0
                expected_scales[block, :, tail] = 0
                assert torch.isnan(scales[block, :, tail].float()).all()

    drafter_last_row._zero_kv_tails(packed, tables, lens, batch, scales)

    assert torch.equal(packed, expected_data)
    assert torch.equal(scale_bytes, expected_scales)
    for b, length in enumerate(lengths[:batch]):
        if length % PAGE:
            for half in (0, 1):
                block = int(tables[b, half, (length - 1) // PAGE])
                tail = slice(length % PAGE, None)
                data = packed[block, :, tail]
                assert torch.count_nonzero(data) == 0
                # Both E2M1 nibbles are zero, so dequantization must remain finite.
                expanded_scales = scales[block, :, tail].float().repeat_interleave(16, -1)
                dequantized = torch.zeros_like(expanded_scales) * expanded_scales
                assert torch.isfinite(dequantized).all()
                assert torch.count_nonzero(dequantized) == 0


@pytest.mark.parametrize(
    "manager,safe",
    [
        (types.SimpleNamespace(enable_block_reuse=False, enable_partial_reuse=True), True),
        (types.SimpleNamespace(enable_block_reuse=True, enable_partial_reuse=False), True),
        (types.SimpleNamespace(enable_block_reuse=True, enable_partial_reuse=True), False),
        (
            types.SimpleNamespace(
                enable_block_reuse=True, enable_partial_reuse=True, copy_on_partial_reuse=True
            ),
            True,
        ),
    ],
)
def test_tail_zeroing_needs_private_partial_pages(manager, safe):
    assert drafter_last_row._v_tail_zero_safe(manager) is safe


def _library(monkeypatch, enabled=True):
    monkeypatch.setattr(drafter_last_row, "_KV_TAIL_ZERO", enabled)
    lib = drafter_last_row.DrafterLastRowContextFmha.__new__(
        drafter_last_row.DrafterLastRowContextFmha
    )
    lib._kv_tail_safe = None
    return lib


def _zero_if_allowed(lib, manager, pool, tables, lens):
    """The attention path's own gate: zero the tails only when the library allows it for this KV cache."""
    if lib._kv_tail_zero(types.SimpleNamespace(kv_cache_manager=manager)):
        drafter_last_row._zero_kv_tails(pool, tables, lens, lens.shape[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_shared_partial_page_declines_and_changes_no_byte(monkeypatch):
    # Block and partial reuse on, no copy on partial reuse: the partial last page may belong to another request.
    shared = types.SimpleNamespace(enable_block_reuse=True, enable_partial_reuse=True)
    lib = _library(monkeypatch)
    pool, tables = _pool_and_tables([16390, 16])
    before = pool.view(torch.uint8).clone()
    lens = torch.tensor([16390, 16], dtype=torch.int32, device="cuda")
    _zero_if_allowed(lib, shared, pool, tables, lens)
    assert lib._kv_tail_safe is False
    assert torch.equal(pool.view(torch.uint8), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_kill_switch_changes_no_byte(monkeypatch):
    private = types.SimpleNamespace(enable_block_reuse=False, enable_partial_reuse=False)
    lib = _library(monkeypatch, enabled=False)
    pool, tables = _pool_and_tables([16390])
    before = pool.view(torch.uint8).clone()
    _zero_if_allowed(
        lib, private, pool, tables, torch.tensor([16390], dtype=torch.int32, device="cuda")
    )
    assert torch.equal(pool.view(torch.uint8), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_private_cache_zeroes_through_the_gate(monkeypatch):
    private = types.SimpleNamespace(enable_block_reuse=False, enable_partial_reuse=False)
    lib = _library(monkeypatch)
    pool, tables = _pool_and_tables([16390])
    _zero_if_allowed(
        lib, private, pool, tables, torch.tensor([16390], dtype=torch.int32, device="cuda")
    )
    for half in (0, 1):
        assert (
            int(
                pool.view(torch.uint8)[
                    int(tables[0, half, 16389 // PAGE]), :, 16390 % PAGE :
                ].count_nonzero()
            )
            == 0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("copy_partial", [False, True])
def test_v1_manager_config_controls_tail_zero_safety(copy_partial: bool) -> None:
    config = KvCacheConfig(
        max_tokens=64,
        enable_block_reuse=True,
        enable_partial_reuse=True,
        copy_on_partial_reuse=copy_partial,
    )
    manager = KVCacheManager(
        kv_cache_config=config,
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        mapping=Mapping(),
        dtype=DataType.FP8,
    )
    try:
        assert manager.enable_block_reuse and manager.enable_partial_reuse
        assert manager.copy_on_partial_reuse is copy_partial
        assert drafter_last_row._v_tail_zero_safe(manager) is copy_partial
    finally:
        manager.shutdown()
