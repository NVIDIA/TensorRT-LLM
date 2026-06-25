# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for proxy late-response handling after a hook terminate.

When a post-processor hook returns ``terminate`` the result is
marked done and popped from the
proxy's ``_results`` map, but the engine may still have in-flight responses for
the same ``client_id`` (abort is async). ``GenerationExecutorProxy.dispatch_result_task``
must drop those late responses via ``self._results.get(client_id)`` rather than
``pop``-ing a missing key and crashing the dispatch thread with a ``KeyError``.
"""

from types import SimpleNamespace

from tensorrt_llm.executor.proxy import GenerationExecutorProxy


class _FakeResultQueue:
    """Returns one queued item per ``get()`` call (mirrors the IPC queue)."""

    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0)


class _RecordingQueue:
    """Stand-in for a result's delivery queue (non-``_SyncQueue`` branch)."""

    def __init__(self):
        self.delivered = []

    def put(self, res):
        self.delivered.append(res)


def _make_proxy():
    # Avoid GenerationExecutorProxy.__init__ (it spawns workers); we only
    # exercise the pure dispatch logic with hand-set attributes.
    return object.__new__(GenerationExecutorProxy)


def test_late_response_after_terminate_is_dropped_without_keyerror():
    """A response for an already-popped client_id must be dropped, not crash."""
    proxy = _make_proxy()
    proxy._results = {}  # terminate already finalized + popped this client_id

    late = SimpleNamespace(client_id=999, has_error=False, result=SimpleNamespace(is_final=True))
    proxy.result_queue = _FakeResultQueue([late])

    # Must not raise KeyError; the dispatch loop stays alive (returns True).
    assert proxy.dispatch_result_task() is True


def test_final_response_is_delivered_and_popped():
    """A live, final response is delivered and removed from ``_results``.

    This establishes the exact condition under which a later duplicate becomes
    the 'late response' that the drop above guards against.
    """
    proxy = _make_proxy()
    queue = _RecordingQueue()
    proxy._results = {7: SimpleNamespace(queue=queue)}

    resp = SimpleNamespace(client_id=7, has_error=False, result=SimpleNamespace(is_final=True))
    proxy.result_queue = _FakeResultQueue([resp])

    assert proxy.dispatch_result_task() is True
    assert queue.delivered == [resp]
    assert 7 not in proxy._results  # popped on is_final
