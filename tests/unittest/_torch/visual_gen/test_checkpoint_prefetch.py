# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-local checkpoint coverage and symmetric prefetch collectives."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.checkpoints import prefetch

pytestmark = pytest.mark.cpu_only


def _distributed(monkeypatch, rank, hosts, memory):
    calls = []
    monkeypatch.setattr(prefetch.dist, "is_available", lambda: True)
    monkeypatch.setattr(prefetch.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefetch.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(prefetch.dist, "get_world_size", lambda: len(hosts))
    monkeypatch.setattr(prefetch.socket, "gethostname", lambda: hosts[rank])
    monkeypatch.setattr(
        prefetch.psutil, "virtual_memory", lambda: SimpleNamespace(available=memory[rank])
    )

    def gather(output, value):
        assert value == (hosts[rank], memory[rank])
        calls.append("gather")
        output[:] = list(zip(hosts, memory))

    monkeypatch.setattr(prefetch.dist, "all_gather_object", gather)
    monkeypatch.setattr(prefetch.dist, "barrier", lambda: calls.append("barrier"))
    return calls


@pytest.fixture
def shards(tmp_path):
    paths = []
    for i in range(8):
        path = tmp_path / f"shard-{i}"
        path.write_bytes(bytes([i]) * 16)
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("interleaved", [False, True])
def test_prefetch_covers_every_shard_on_every_host(monkeypatch, shards, interleaved):
    hosts = [f"node-{i // 4}" for i in range(72)]
    if interleaved:
        hosts = [f"node-{i % 18}" for i in range(72)]
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "72")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    covered = {host: [] for host in hosts}
    for rank, host in enumerate(hosts):
        monkeypatch.setenv("LOCAL_RANK", str(rank % 4))
        calls = _distributed(monkeypatch, rank, hosts, [10000] * 72)
        monkeypatch.setattr(prefetch, "_prefetch_file", lambda path, _: covered[host].append(path))
        assert prefetch.prefetch_files_to_host_cache(shards, description="test")
        assert calls == ["gather", "barrier"]
    for paths in covered.values():
        assert sorted(paths) == sorted(shards)


def test_memory_limit_uses_only_actual_node_peers(monkeypatch, shards):
    hosts = ["a", "b", "a", "b"]
    memory = [10000, 1, 9000, 1]
    for rank in range(4):
        calls = _distributed(monkeypatch, rank, hosts, memory)
        reads = []
        monkeypatch.setattr(prefetch, "_prefetch_file", lambda path, _: reads.append(path))
        result = prefetch.prefetch_files_to_host_cache(shards, description="test")
        assert result == (hosts[rank] == "a")
        assert sorted(reads) == sorted(shards[rank // 2 :: 2] if result else [])
        assert calls == ["gather", "barrier"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_cache_hits_and_empty_inputs_keep_collective_sequence(monkeypatch, shards, cached, empty):
    calls = _distributed(monkeypatch, 0, ["a", "b"], [10000, 10000])
    cache = set(shards) if cached else set()
    reads = []
    monkeypatch.setattr(prefetch, "_prefetch_file", lambda path, _: reads.append(path))
    assert prefetch.prefetch_files_to_host_cache(
        [] if empty else shards, description="test", prefetched_paths=cache
    )
    assert sorted(reads) == sorted([] if cached or empty else shards)
    assert calls == ["gather", "barrier"]


def test_cache_filter_does_not_change_shard_ownership(monkeypatch, shards):
    reads = []
    for rank, cache in enumerate([{shards[0]}, {shards[2]}]):
        _distributed(monkeypatch, rank, ["a", "a"], [10000, 10000])
        assigned = shards[rank::2]
        pending = [path for path in assigned if path not in cache]
        rank_reads = []
        monkeypatch.setattr(prefetch, "_prefetch_file", lambda path, _: rank_reads.append(path))
        before = cache.copy()
        assert prefetch.prefetch_files_to_host_cache(
            shards, description="test", prefetched_paths=cache
        )
        assert sorted(rank_reads) == sorted(pending)
        assert cache == before | set(assigned)
        reads.extend(rank_reads)
    assert sorted(reads) == sorted(shards[1:])


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_read_failure_does_not_cache_unread_shards(monkeypatch, shards, ignore_errors):
    calls = _distributed(monkeypatch, 0, ["a", "b"], [10000, 10000])
    cache = set()

    def fail(path, description):
        raise OSError("injected read failure")

    monkeypatch.setattr(prefetch, "_prefetch_file", fail)
    if ignore_errors:
        assert not prefetch.prefetch_files_to_host_cache(
            shards, description="test", prefetched_paths=cache, ignore_errors=True
        )
    else:
        with pytest.raises(OSError, match="injected"):
            prefetch.prefetch_files_to_host_cache(
                shards, description="test", prefetched_paths=cache
            )
    assert cache == set()
    assert calls == ["gather", "barrier"]


def test_single_process_prefetch_reads_files(monkeypatch, shards):
    monkeypatch.setattr(prefetch.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(prefetch.psutil, "virtual_memory", lambda: SimpleNamespace(available=10000))
    cache = set()
    assert prefetch.prefetch_files_to_host_cache(
        shards + shards, description="test", prefetched_paths=cache
    )
    assert cache == set(shards)
    for path in shards:
        Path(path).unlink()
    assert prefetch.prefetch_files_to_host_cache(shards, description="test", prefetched_paths=cache)


def test_uneven_host_membership_ignores_stale_launcher_size(monkeypatch):
    hosts = ["a", "b", "a", "c", "a", "b"]
    memory = [1000, 400, 900, 1, 800, 500]
    expected = [(0, 3, 800), (0, 2, 400), (1, 3, 800), (0, 1, 1), (2, 3, 800), (1, 2, 400)]
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "72")
    for rank, context in enumerate(expected):
        _distributed(monkeypatch, rank, hosts, memory)
        assert prefetch._local_prefetch_context() == context
