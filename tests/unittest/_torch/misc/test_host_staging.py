# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned uploads remain correct while callers recycle pageable source storage."""

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch import host_staging
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.cpu_only
def test_host_staging_cpu_destination():
    store = {}
    source = torch.arange(24).view(4, 6)[:, ::2]
    destination = torch.empty_like(source)
    host_staging.copy_host_to_device(store, "test", destination, source)
    torch.testing.assert_close(destination, source, rtol=0, atol=0)
    assert not store
    host_staging.copy_host_to_device(store, "empty", destination[:0], source[:0])
    assert not store


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pinned_source", [False, True])
def test_host_staging_retains_inflight_sources(pinned_source):
    store, observed = {}, []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        for iteration, count in enumerate((8, 13, 4, 31, 7, 2, 31)):
            expected = torch.arange(count, dtype=torch.int64) + iteration * 100
            destination = torch.empty(count, dtype=torch.int64, device="cuda")
            if pinned_source:
                source = expected.pin_memory()
            else:
                source = torch.empty(count * 2, dtype=torch.int64)[::2]
                source.copy_(expected)
            host_staging.copy_host_to_device(store, "test", destination, source)
            if not pinned_source:
                source.fill_(-1)
            del source
            observed.append((destination, expected))
    stream.synchronize()
    for actual, expected in observed:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    if pinned_source:
        assert not store
    else:
        assert len(store) == 1
        ring = next(iter(store.values()))
        assert len(ring["buffers"]) == 2
        assert all(buffer.is_pinned() for buffer in ring["buffers"])
        assert all(buffer.numel() <= 62 for buffer in ring["buffers"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_host_staging_respects_unpinned_policy(monkeypatch):
    monkeypatch.setattr(host_staging, "prefer_pinned", lambda: False)
    source = torch.arange(13, dtype=torch.int32)
    destination = torch.empty_like(source, device="cuda")
    store = {}
    host_staging.copy_host_to_device(store, "test", destination, source)
    torch.testing.assert_close(destination.cpu(), source, rtol=0, atol=0)
    assert not store


_TOKENS_PER_BLOCK = 16
_MAX_SEQ_LEN = 256
_REQUESTS = 4


def _kv_cache_manager() -> KVCacheManager:
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=_REQUESTS * 2 * _MAX_SEQ_LEN),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_MAX_SEQ_LEN,
        max_batch_size=_REQUESTS,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )
    manager.add_dummy_requests(list(range(_REQUESTS)), [_MAX_SEQ_LEN // 2] * _REQUESTS)
    return manager


def _metadata(
    manager: KVCacheManager, uploader: host_staging.HostUploader | None
) -> TrtllmAttentionMetadata:
    metadata = TrtllmAttentionMetadata(
        max_num_requests=_REQUESTS,
        max_num_tokens=_REQUESTS * 8,
        kv_cache_manager=manager,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
    )
    metadata.host_uploader = uploader
    return metadata


def _prepare(metadata: TrtllmAttentionMetadata, step: int, num_seqs: int) -> None:
    metadata.request_ids = list(range(num_seqs))
    metadata.prompt_lens = [17 + 3 * row + step for row in range(num_seqs)]
    metadata.num_contexts = 0
    metadata.seq_lens = torch.full((num_seqs,), 1 + step % 3, dtype=torch.int)
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True,
        num_cached_tokens_per_seq=[40 + 5 * row + 2 * step for row in range(num_seqs)],
    )
    metadata.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trtllm_metadata_uploads_match_plain_copies():
    """Metadata prepared through a HostUploader while the host runs ahead of a busy
    stream holds the same device lengths as metadata prepared with plain copies."""
    manager = _kv_cache_manager()
    staged = _metadata(manager, host_staging.HostUploader())
    steps = list(enumerate((2, 4, 3, 4, 1, 4, 2)))
    observed = []
    busy = torch.cuda.Stream()
    with torch.cuda.stream(busy):
        for step, num_seqs in steps:
            # Earlier work still occupies the stream while the host prepares.
            torch.cuda._sleep(1_000_000)
            _prepare(staged, step, num_seqs)
            observed.append(
                (
                    staged.kv_lens_cuda[:num_seqs].clone(),
                    staged.prompt_lens_cuda[:num_seqs].clone(),
                )
            )
    busy.synchronize()
    # The plain copies read persistent host buffers, so they are prepared one step at a time.
    plain = _metadata(manager, None)
    for (step, num_seqs), (kv_lens, prompt_lens) in zip(steps, observed):
        _prepare(plain, step, num_seqs)
        torch.cuda.synchronize()
        assert torch.equal(kv_lens, plain.kv_lens_cuda[:num_seqs])
        assert torch.equal(prompt_lens, plain.prompt_lens_cuda[:num_seqs])
    manager.shutdown()
