# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned uploads remain correct while callers recycle pageable source storage."""

import pytest
import torch

from tensorrt_llm._torch import host_staging


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
