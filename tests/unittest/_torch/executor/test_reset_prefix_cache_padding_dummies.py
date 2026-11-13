# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""reset_prefix_cache with CUDA graph padding dummies on KVCacheManagerV2.

KVCacheManagerV2 refuses to clear reusable blocks while a KV cache is open, and a padding dummy
keeps one open between steps. An RL weight sync resets the prefix cache at its end, so the reset
releases the padding dummies first (the next padded step re-creates them).
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


class _Runner:
    def __init__(self, caches, events):
        self._caches = caches
        self._events = events

    def _padding_dummy_caches(self):
        return self._caches

    def release_padding_dummy(self, resource_manager, draft_len):
        self._events.append(("release", draft_len, resource_manager))
        for cache in self._caches:
            cache.pop(draft_len, None)
        return True


class _KvManager:
    def __init__(self, events):
        self._events = events

    def reset_reuse_state(self, on_cleared=None):
        assert all(not cache for cache in self.runner._padding_dummy_caches()), (
            "a padding dummy is still open"
        )
        self._events.append(("reset",))


def _executor(kv_manager_v2: bool, caches=None):
    events = []
    runner = _Runner(caches if caches is not None else ({0: object()},), events)
    kv = _KvManager(events)
    kv.runner = runner
    executor = SimpleNamespace(
        _is_kv_manager_v2=kv_manager_v2,
        model_engine=SimpleNamespace(cuda_graph_runner=runner, route_capture=None),
        resource_manager="resource-manager",
        kv_cache_manager=kv,
        enable_joint_kv_cache_reuse=False,
    )
    executor._release_padding_dummies_for_reuse_reset = lambda: (
        PyExecutor._release_padding_dummies_for_reuse_reset(executor)
    )
    return executor, events


def test_v2_reset_releases_the_padding_dummies_first():
    executor, events = _executor(True, caches=({0: object(), 3: object()}, {3: object()}))
    PyExecutor.reset_prefix_cache(executor)
    assert events == [
        ("release", 0, "resource-manager"),
        ("release", 3, "resource-manager"),
        ("reset",),
    ]


def test_v2_reset_without_padding_dummies_only_resets():
    executor, events = _executor(True, caches=({},))
    PyExecutor.reset_prefix_cache(executor)
    assert events == [("reset",)]


def test_v1_reset_keeps_the_padding_dummies():
    """The V1 manager has no open-cache precondition; its dummies stay cached."""
    caches = ({0: object()},)
    executor, events = _executor(False, caches=caches)
    executor.kv_cache_manager.reset_reuse_state = lambda on_cleared=None: events.append(("reset",))
    PyExecutor.reset_prefix_cache(executor)
    assert events == [("reset",)] and 0 in caches[0]


def test_engines_without_a_cuda_graph_runner_reset_as_before():
    executor, events = _executor(True)
    executor.model_engine.cuda_graph_runner = None
    executor.kv_cache_manager.reset_reuse_state = lambda on_cleared=None: events.append(("reset",))
    PyExecutor.reset_prefix_cache(executor)
    assert events == [("reset",)]
