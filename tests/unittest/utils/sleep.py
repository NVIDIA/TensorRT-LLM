# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side assertions for the native V2 sleep integration tests."""

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType
from tensorrt_llm.runtime.kv_cache_manager_v2 import GpuCacheTierConfig, HostCacheTierConfig


class V2SleepWorkerExtension:
    def assert_v2_cache_manager(self) -> None:
        assert isinstance(self.engine.kv_cache_manager, KVCacheManagerV2)

    def assert_v2_sleep_capacity(self, restore_mode: str) -> None:
        mode = self.engine.llm_args.sleep_config.restore_modes[ExecutorMemoryType.KV_CACHE]
        assert mode.name == restore_mode
        tiers = self.engine.kv_cache_manager.impl.init_config.cache_tiers
        gpu_quota = sum(tier.quota for tier in tiers if isinstance(tier, GpuCacheTierConfig))
        host_quota = sum(tier.quota for tier in tiers if isinstance(tier, HostCacheTierConfig))
        assert host_quota >= gpu_quota, (gpu_quota, host_quota)

    def get_v2_prefix_match_lengths(self, prompts: list[list[int]]) -> list[int]:
        manager = self.engine.kv_cache_manager
        return [manager.probe_prefix_match_length(prompt) for prompt in prompts]


class V2SleepWorker(GenerationExecutorWorker):
    def setup_engine(self) -> None:
        super().setup_engine()
        assert isinstance(self.engine.kv_cache_manager, KVCacheManagerV2)
