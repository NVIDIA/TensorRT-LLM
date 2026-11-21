# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real NCCL coverage of cache/memory divergence between logical hosts."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.checkpoints import prefetch

from ._worker_pool import run_distributed


def _prefetch_worker(rank: int, world_size: int, directory: str) -> None:
    assert world_size == 2
    paths = [str(Path(directory) / f"shard-{i}") for i in range(8)]
    reads = []
    original_read = prefetch._prefetch_file

    def read(path: str, description: str) -> None:
        original_read(path, description)
        reads.append(Path(path).name)

    with (
        patch.object(prefetch.socket, "gethostname", return_value="same-node"),
        patch.object(
            prefetch.psutil, "virtual_memory", return_value=SimpleNamespace(available=10000)
        ),
        patch.object(prefetch, "_prefetch_file", side_effect=read),
    ):
        assert prefetch.prefetch_files_to_host_cache(paths, description="same node")
        assert sorted(reads) == [f"shard-{i}" for i in range(rank, 8, 2)]

    # Each GPU now represents a separate logical node. Only rank 0 skips
    # the first call, so the second call mixes a cache miss with a hit.
    cache = set()
    reads.clear()
    with (
        patch.dict(os.environ, {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"}),
        patch.object(prefetch.socket, "gethostname", return_value=f"node-{rank}"),
        patch.object(prefetch.psutil, "virtual_memory") as memory,
        patch.object(prefetch, "_prefetch_file", side_effect=read),
    ):
        memory.return_value = SimpleNamespace(available=1 if rank == 0 else 10000)
        result = prefetch.prefetch_files_to_host_cache(
            paths, description="first", prefetched_paths=cache
        )
        assert result == (rank == 1)
        assert len(reads) == (0 if rank == 0 else 8)
        reads.clear()
        memory.return_value = SimpleNamespace(available=10000)
        assert prefetch.prefetch_files_to_host_cache(
            paths, description="second", prefetched_paths=cache
        )
        assert len(reads) == (8 if rank == 0 else 0)
        assert cache == set(paths)
        # Exercise empty input versus uncached input on the same group.
        assert prefetch.prefetch_files_to_host_cache(
            [] if rank == 0 else paths, description="empty"
        )
    # A subsequent collective must still complete with the right value.
    result = torch.tensor(rank + 1, device=f"cuda:{rank}")
    dist.all_reduce(result)
    assert result.item() == 3


@pytest.mark.timeout(180)
def test_checkpoint_prefetch_divergent_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    for i in range(8):
        (tmp_path / f"shard-{i}").write_bytes(bytes([i]) * 16)
    run_distributed(
        2, _prefetch_worker, str(tmp_path), dist_backend="nccl", timeout=150, reuse=False
    )
