# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side assertions for the native V2 sleep integration tests."""

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    CacheTier,
    GpuCacheTierConfig,
    HostCacheTierConfig,
)


def assert_v2_gpu_pool_bytes(replies: list[tuple[int, int]], expected: dict[int, int]) -> None:
    """Check native GPU pool bytes for every expected rank, with no missing or duplicate replies."""
    actual = dict(replies)
    assert len(replies) == len(expected) and actual.keys() == expected.keys(), (
        f"Expected one pool report per rank {sorted(expected)}, got {replies}"
    )
    for rank, num_bytes in actual.items():
        assert num_bytes == expected[rank], (
            f"Rank {rank}: expected {expected[rank]} native GPU pool bytes, got {num_bytes}"
        )


class V2SleepWorkerExtension:
    def assert_v2_cache_manager(self) -> None:
        assert isinstance(self.engine.kv_cache_manager, KVCacheManagerV2)

    def get_v2_gpu_pool_bytes(self) -> tuple[int, int]:
        """Return this worker's rank and current native GPU pool bytes across GPU tiers."""
        manager = self.engine.kv_cache_manager
        assert isinstance(manager, KVCacheManagerV2)
        num_bytes = sum(
            manager.impl.get_num_bytes(level, CacheTier.GPU_MEM)
            for level, tier in enumerate(manager.impl.init_config.cache_tiers)
            if isinstance(tier, GpuCacheTierConfig)
        )
        return self.global_rank, num_bytes

    def assert_v2_sleep_capacity(self, restore_mode: str) -> None:
        mode = self.engine.llm_args.sleep_config.restore_modes[ExecutorMemoryType.KV_CACHE]
        assert mode.name == restore_mode
        tiers = self.engine.kv_cache_manager.impl.init_config.cache_tiers
        gpu_levels = [i for i, tier in enumerate(tiers) if isinstance(tier, GpuCacheTierConfig)]
        gpu_quota = sum(tiers[i].quota for i in gpu_levels)
        host_quota = max(
            (
                tier.quota
                for tier in tiers[gpu_levels[-1] + 1 :]
                if isinstance(tier, HostCacheTierConfig)
            ),
            default=0,
        )
        assert host_quota >= gpu_quota, (gpu_quota, host_quota)

    def get_v2_prefix_match_lengths(self, prompts: list[list[int]]) -> list[int]:
        manager = self.engine.kv_cache_manager
        return [manager.probe_prefix_match_length(prompt) for prompt in prompts]


class V2SleepWorker(GenerationExecutorWorker):
    def setup_engine(self) -> None:
        super().setup_engine()
        assert isinstance(self.engine.kv_cache_manager, KVCacheManagerV2)
