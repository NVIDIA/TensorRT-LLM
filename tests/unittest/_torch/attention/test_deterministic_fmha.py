# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real runtime selection must keep attention invariant to unrelated requests."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import deterministic, flashinfer_trtllm_gen
from tensorrt_llm._torch.attention.backends.fmha.deterministic import DeterministicFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.fixture
def deterministic_launches(monkeypatch):
    """The phases the deterministic library launched; a fallback would launch none."""
    phases = []
    launch = DeterministicFmha._run_attention

    def recorded(self, phase, **kwargs):
        phases.append(phase)
        return launch(self, phase, **kwargs)

    monkeypatch.setattr(DeterministicFmha, "_run_attention", recorded)
    return phases


@pytest.fixture
def parallel_segments(monkeypatch):
    """For each launch of the kernel, whether it ran the key segments in parallel."""
    layouts = []
    kernel = deterministic.deterministic_attention

    def recorded(**kwargs):
        layouts.append(kernel(**kwargs))
        return layouts[-1]

    monkeypatch.setattr(deterministic, "deterministic_attention", recorded)
    return layouts


@pytest.mark.parametrize("sm, available", [(100, True), (103, False)])
def test_deterministic_library_serves_sm100_only(monkeypatch, sm, available):
    monkeypatch.setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1")
    monkeypatch.setattr(deterministic, "get_sm_version", lambda: sm)
    monkeypatch.setattr(flashinfer_trtllm_gen, "get_sm_version", lambda: sm)
    monkeypatch.setattr(flashinfer_trtllm_gen, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(DeterministicFmha, "_missing_fused_nanobind_ops", lambda: [])
    layer = SimpleNamespace(
        is_mla_enable=False,
        head_dim=128,
        num_heads=8,
        num_kv_heads=2,
        sparse_params=None,
        skip_correction_threshold=0.0,
    )
    assert DeterministicFmha.is_available(layer) is available


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bitwise equality of BF16 tensors: +0 and -0 differ, and a NaN equals itself."""
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)
    )


@pytest.mark.parametrize(
    "change, reason",
    [
        ({}, None),
        ({"query_dtype": torch.float16}, "query dtype"),
        ({"kv_dtype": DataType.FP8}, "KV cache dtype"),
        ({"is_cross": True}, "cross attention"),
        ({"beam_width": 2}, "beam width"),
        ({"tokens_per_block": 128}, "KV page size"),
        ({"num_heads": 66}, "query heads per KV head"),
        ({"use_spec_decoding": True}, "speculative decoding"),
        ({"attention_sinks": torch.zeros(8)}, "attention sinks"),
        ({"mask_type": AttentionMaskType.padding}, "mask type"),
        ({"attention_chunk_size": 16}, "chunked attention"),
        ({"attention_window_size": 64}, "sliding attention window"),
        ({"attention_window_size": None}, "sliding attention window"),
    ],
)
def test_requests_the_kernel_cannot_serve_are_left_to_the_fallback(change, reason):
    setup = dict(
        query_dtype=torch.bfloat16,
        kv_dtype=DataType.BF16,
        is_cross=False,
        beam_width=1,
        tokens_per_block=32,
        num_heads=8,
        use_spec_decoding=False,
        attention_sinks=None,
        mask_type=AttentionMaskType.causal,
        attention_chunk_size=0,
        attention_window_size=128,
    )
    setup.update(change)
    layer = SimpleNamespace(
        num_heads=setup["num_heads"],
        num_kv_heads=2,
        attention_chunk_size=setup["attention_chunk_size"],
    )
    fmha = DeterministicFmha.__new__(DeterministicFmha)
    fmha._attn_ref = lambda: layer
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(dtype=setup["kv_dtype"]),
        is_cross=setup["is_cross"],
        beam_width=setup["beam_width"],
        tokens_per_block=setup["tokens_per_block"],
        use_spec_decoding=setup["use_spec_decoding"],
        max_seq_len=128,
    )
    forward_args = SimpleNamespace(
        attention_sinks=setup["attention_sinks"],
        mask_type=setup["mask_type"],
        attention_window_size=setup["attention_window_size"],
    )
    query = torch.empty(1, 8 * 64, dtype=setup["query_dtype"])
    actual = fmha._unsupported_reason(query, metadata, forward_args)
    if reason is None:
        assert actual is None
    else:
        assert reason in actual


@pytest.mark.parametrize("contexts", [1, 3])
@pytest.mark.parametrize("generations", [1, 2])
def test_context_attention_ignores_generation_length_tail(contexts: int, generations: int) -> None:
    """The context phase must not launch kernels for the decode metadata tail."""
    if not torch.cuda.is_available() or get_sm_version() != 100:
        pytest.skip("Requires SM100")
    torch.manual_seed(20261006)
    heads, kv_heads, dim, page = 4, 2, 64, 32
    context_lengths = [3 + 2 * i for i in range(contexts)]
    num_context_tokens = sum(context_lengths)
    total = num_context_tokens + generations
    query = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(
        contexts + generations, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    tables = torch.arange(contexts + generations, device="cuda", dtype=torch.int32)[:, None]
    lengths = torch.tensor(context_lengths + [9] * generations, device="cuda", dtype=torch.int32)
    cu_q = (
        torch.tensor([0] + context_lengths + [1] * generations, device="cuda", dtype=torch.int32)
        .cumsum(0)
        .int()
    )
    output = torch.full_like(query, 123)
    expected = torch.empty_like(query[:num_context_tokens])
    common = dict(
        query=query[:num_context_tokens],
        kv_cache=(keys, values),
        block_tables=tables,
        batch_size=contexts,
        max_q_len=max(context_lengths),
        cum_seq_lens_q=cu_q,
        bmm1_scale=dim**-0.5,
        uses_shared_paged_kv_idx=True,
    )
    # Allocate valid backing storage for the decode tail so a buggy launch
    # deterministically overwrites the canary instead of poisoning CUDA.
    DeterministicFmha._run_attention(
        None,
        FmhaPhase.CONTEXT,
        **common,
        seq_lens=lengths,
        out=output[:num_context_tokens],
    )
    DeterministicFmha._run_attention(
        None,
        FmhaPhase.CONTEXT,
        **common,
        seq_lens=lengths[:contexts],
        out=expected,
    )
    _assert_same_bits(output[:num_context_tokens], expected)
    _assert_same_bits(
        output[num_context_tokens:], torch.full_like(output[num_context_tokens:], 123)
    )


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("page", [32, 64])
def test_deterministic_attention_runtime_phase_invariance(
    monkeypatch, deterministic_launches, dim, page
):
    if not torch.cuda.is_available() or get_sm_version() != 100:
        pytest.skip("Requires SM100")
    monkeypatch.setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1")
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    torch.manual_seed(71)
    length, heads, kv_heads = 129, 8, 2
    capacity = math.ceil(length / page) * page
    qkv = torch.randn(length, (heads + 2 * kv_heads) * dim, device="cuda", dtype=torch.bfloat16)
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=capacity, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=kv_heads,
        head_dim=dim,
        tokens_per_block=page,
        max_seq_len=capacity,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    try:
        manager.add_dummy_requests([0], [length])
        attention = TrtllmAttention(
            layer_idx=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim
        )
        results = []
        for context in (True, False):
            tokens = length if context else 1
            metadata = TrtllmAttention.Metadata(
                num_contexts=int(context),
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[length - tokens]
                ),
                seq_lens=torch.tensor([tokens], dtype=torch.int32),
                max_num_requests=1,
                max_num_tokens=length,
                kv_cache_manager=manager,
                request_ids=[0],
                prompt_lens=[length],
            )
            metadata.prepare()
            output = torch.empty(tokens, heads * dim, device="cuda", dtype=torch.bfloat16)
            forward_args = AttentionForwardArgs(
                output=output,
                is_fused_qkv=True,
                attention_input_type=AttentionInputType.context_only
                if context
                else AttentionInputType.generation_only,
            )
            attention.forward(
                qkv[-tokens:].clone(), None, None, metadata, forward_args=forward_args
            )
            results.append(output[-1].cpu())
            del metadata
        del attention
    finally:
        manager.shutdown()
    assert deterministic_launches == [FmhaPhase.CONTEXT, FmhaPhase.GENERATION]
    _assert_same_bits(results[0], results[1])


@pytest.mark.parametrize(
    "length, max_seq_len, parallel",
    [
        (513, None, [True, True]),
        (2049, None, [True, True]),
        (8193, None, [True, True]),
        # A long sequence limit: the partial results of 32 requests no longer fit
        # in the generation workspace, so that batch walks its segments serially.
        (513, 1 << 16, [True, False]),
    ],
)
@pytest.mark.parametrize("page", [32, 64])
def test_deterministic_attention_runtime_batch_invariance(
    monkeypatch, deterministic_launches, parallel_segments, length, max_seq_len, parallel, page
):
    if not torch.cuda.is_available() or get_sm_version() != 100:
        pytest.skip("Requires SM100")
    monkeypatch.setenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC", "1")
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    torch.manual_seed(20261005)
    heads, kv_heads, dim = 24, 4, 128
    pages = math.ceil(length / page)
    qkv = torch.randn(32, (heads + 2 * kv_heads) * dim, device="cuda", dtype=torch.bfloat16)
    cached = torch.randn(pages, 2, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for batch in (1, 32):
        manager = KVCacheManager(
            # The manager lowers the sequence limit to what its pool can hold.
            KvCacheConfig(
                max_tokens=max(batch * pages * page, max_seq_len or 0), enable_block_reuse=False
            ),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=kv_heads,
            head_dim=dim,
            tokens_per_block=page,
            max_seq_len=max_seq_len or pages * page,
            max_batch_size=batch,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )
        try:
            requests = list(range(batch))
            manager.add_dummy_requests(requests, [length] * batch)
            pool = manager.get_buffers(0)
            pool.zero_()
            indices = [i for i in manager.get_batch_cache_indices(requests, 0)[0] if i >= 0]
            pool.index_copy_(
                0, torch.tensor(indices, device="cuda"), cached.view(pages, *pool.shape[1:])
            )
            metadata = TrtllmAttention.Metadata(
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[length - 1] * batch
                ),
                seq_lens=torch.ones(batch, dtype=torch.int32),
                max_num_requests=batch,
                max_num_tokens=batch,
                kv_cache_manager=manager,
                request_ids=requests,
                prompt_lens=[length] * batch,
            )
            metadata.prepare()
            attention = TrtllmAttention(
                layer_idx=0, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim
            )
            output = torch.empty(batch, heads * dim, device="cuda", dtype=torch.bfloat16)
            forward_args = AttentionForwardArgs(
                output=output,
                is_fused_qkv=True,
                attention_input_type=AttentionInputType.generation_only,
            )
            attention.forward(qkv[:batch], None, None, metadata, forward_args=forward_args)
            outputs.append(output[0].cpu())
            del pool, metadata, attention
        finally:
            manager.shutdown()
    assert deterministic_launches == [FmhaPhase.GENERATION] * 2
    assert parallel_segments == parallel
    _assert_same_bits(outputs[0], outputs[1])
