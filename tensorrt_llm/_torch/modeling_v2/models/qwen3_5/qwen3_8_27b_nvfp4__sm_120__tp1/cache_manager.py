# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's KV cache manager: the Qwen3.5 hybrid manager without in-forward snapshots.

The built-in Gated-DeltaNet mixer implements the in-forward snapshot split (it writes the
state at a planned snapshot position P into staging rows while the chunk runs on to the
prompt end), and `Qwen35HybridCacheManager` declares that support. This target's prefill
writes only the chunk's final state, so the manager declares no support: the scheduler then
cuts a prompt at P into two context steps, and the state at P is the first step's final
state. Declared through `RuntimeHooks.kv_cache_manager_cls`.
"""

from tensorrt_llm._torch.modules.fla.cache_manager import Qwen35HybridCacheManager


class Qwen3827BSm120Tp1CacheManager(Qwen35HybridCacheManager):
    """`Qwen35HybridCacheManager` with no in-forward snapshot staging rows."""

    _supports_inforward_snapshot = False
