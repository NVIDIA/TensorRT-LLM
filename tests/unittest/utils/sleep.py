# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side assertions for the native V2 sleep integration tests."""

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.executor.worker import GenerationExecutorWorker


class V2SleepWorkerExtension:
    def assert_v2_cache_manager(self) -> None:
        assert isinstance(self.engine.kv_cache_manager, KVCacheManagerV2)


class V2SleepWorker(GenerationExecutorWorker):
    def setup_engine(self) -> None:
        super().setup_engine()
        assert isinstance(self.engine.kv_cache_manager, KVCacheManagerV2)
