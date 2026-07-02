# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``tensorrt_llm._torch.visual_gen.utils`` (SequenceSharder, etc.)."""

import os
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Avoid leaking TLLM_DISABLE_MPI if other modules set it."""
    yield


# =============================================================================
# SequenceSharder — single-process (no distributed)
# =============================================================================


class TestSequenceSharderInactive:
    def test_shard_gather_identity_when_size_one(self):
        s = SequenceSharder(size=1, rank=0, group=None)
        assert not s.is_active
        x = torch.randn(2, 8, 4)
        assert s.shard(x, dim=1) is x
        assert s.gather(x, dim=1) is x
        assert s.gather(x, dim=1, unpad_to=3) is x

    def test_shard_none_returns_none(self):
        s = SequenceSharder(size=4, rank=1, group=None)
        assert s.shard(None, dim=1) is None

    def test_shard_rope_passthrough_when_inactive(self):
        s = SequenceSharder(size=1, rank=0, group=None)
        cos = torch.randn(1, 4, 8)
        rope = (cos, cos)
        assert s.shard_rope(rope, seq_len=4, seq_dim=1) is rope

    def test_disable_enable_no_collectives(self):
        s = SequenceSharder(size=4, rank=0, group=None)
        assert s.is_active
        s.disable()
        assert not s.is_active
        x = torch.randn(1, 8, 2)
        assert s.shard(x, dim=1) is x
        s.enable()
        assert s.is_active


class TestSequenceSharderShardSlices:
    """Active sharder: block slice math without ``gather``."""

    def test_shard_dim1_four_ranks(self):
        s = SequenceSharder(size=4, rank=2, group=None)
        x = torch.arange(24).view(2, 12, 1)
        part = s.shard(x, dim=1)
        assert part.shape == (2, 3, 1)
        assert torch.equal(part.squeeze(-1), torch.tensor([[6, 7, 8], [18, 19, 20]]))

    def test_expected_seq_len_skip_shard(self):
        s = SequenceSharder(size=4, rank=0, group=None)
        x = torch.randn(1, 7, 2)
        out = s.shard(x, dim=1, expected_seq_len=8)
        assert out.shape == (1, 7, 2)

    def test_not_divisible_raises(self):
        s = SequenceSharder(size=4, rank=0, group=None)
        x = torch.randn(1, 10, 2)
        with pytest.raises(ValueError, match="divisible"):
            s.shard(x, dim=1)


class TestSequenceSharderShardRope:
    def test_shard_rope_bsd_layout(self):
        s = SequenceSharder(size=2, rank=1, group=None)
        B, S, D = 1, 8, 16
        cos = torch.arange(B * S * D).view(B, S, D).float()
        sin = cos + 1000
        out = s.shard_rope((cos, sin), seq_len=S, seq_dim=1)
        assert out is not None
        oc, osin = out
        assert oc.shape == (1, 4, 16)
        assert torch.equal(oc, cos[:, 4:8].contiguous())

    def test_shard_rope_explicit_seq_dim_handles_square_layout(self):
        """``shard_rope`` requires an explicit ``seq_dim``. Square
        ``(B, S, S)`` layouts dispatch on the caller-supplied axis;
        the helper does not infer the sequence dimension."""
        s = SequenceSharder(size=2, rank=0, group=None)
        S = 8
        cos = torch.zeros(2, S, S)
        sin = torch.ones(2, S, S)
        out = s.shard_rope((cos, sin), seq_len=S, seq_dim=1)
        assert out is not None
        oc, _osin = out
        assert oc.shape == (2, 4, S)


class TestSequenceSharderFromVgm:
    def test_from_vgm_none(self):
        s = SequenceSharder.from_vgm(None)
        assert s.size == 1 and s.rank == 0 and not s.is_active

    def test_from_vgm_head_divisibility(self):
        # ``seq_group`` is a callable returning the process group on
        # the current ``SequenceSharder`` API; the stub uses a fresh
        # ``object()`` to stand in for a real ``ProcessGroup``.
        stub_group = object()
        vgm = SimpleNamespace(
            seq_size=4,
            seq_rank=0,
            seq_group=lambda: stub_group,
            ulysses_size=2,
        )
        SequenceSharder.from_vgm(vgm, num_attention_heads=8, num_kv_heads=4)
        with pytest.raises(ValueError, match="num_attention_heads"):
            SequenceSharder.from_vgm(vgm, num_attention_heads=7)


# =============================================================================
# SequenceSharder — distributed shard / gather round-trip (gloo, CPU)
# =============================================================================


def _dist_worker_shard_gather_combined(rank: int, world_size: int, port: int):
    """Single spawn: basic shard/gather round-trip and pad/gather/unpad."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        sharder = SequenceSharder(size=world_size, rank=rank, group=group)

        x = torch.arange(world_size * 5, dtype=torch.float32).view(1, world_size * 5, 1)
        local = sharder.shard(x, dim=1)
        assert local.shape == (1, 5, 1)
        exp = torch.arange(rank * 5, (rank + 1) * 5, dtype=torch.float32).view(1, 5, 1)
        assert torch.allclose(local, exp)
        full = sharder.gather(local, dim=1)
        assert full.shape == x.shape
        assert torch.allclose(full, x)

        original_len = 10
        xp = torch.ones(1, original_len, 2)
        local_p = sharder.shard(xp, dim=1, pad_to_multiple=True)
        restored = sharder.gather(local_p, dim=1, unpad_to=original_len)
        assert restored.shape == xp.shape
        assert torch.allclose(restored, xp)
    finally:
        dist.destroy_process_group()


def _spawn_entry_combined(rank: int, world_size: int, port: int):
    _dist_worker_shard_gather_combined(rank, world_size, port)


def _run_dist(world_size: int, entry: Callable[[int, int, int], None]):
    if not MODULES_AVAILABLE:
        pytest.skip("SequenceSharder import failed")
    port = get_free_port()
    mp.spawn(entry, args=(world_size, port), nprocs=world_size, join=True)


class TestSequenceSharderDistributed:
    def test_shard_gather_pad_unpad_four_ranks(self):
        _run_dist(4, _spawn_entry_combined)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
