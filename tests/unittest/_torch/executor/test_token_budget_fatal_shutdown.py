# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the fatal token-budget shutdown path.

When the prep-boundary token-budget fallback is *disabled*
(``enable_token_budget_fallback=False``), an over-budget batch reaches the
forward pass and ``_prepare_tp_inputs`` raises ``TokenBudgetExceededError``.
The executor must convert that into a fatal, server-terminating shutdown --
failing every active/queued request with the message and enqueuing a shutdown --
rather than letting the exception kill only the loop thread and leave the server
up but hanging. These tests drive ``PyExecutor._handle_errors`` /
``_handle_token_budget_error`` on a bare instance (``__new__``) with the minimal
collaborators stubbed; no GPU is touched.
"""

import unittest

from tensorrt_llm._torch.pyexecutor.error_classification import (
    ErrorBudget,
    TokenBudgetExceededError,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


class _FakeReq:
    def __init__(self, rid):
        self.py_request_id = rid
        self.py_client_id = rid
        self.state = None


class _EmptyRawQueue:
    def empty(self):
        return True


class _FakeExecQueue:
    def __init__(self):
        self.shutdown_enqueued = False

    def get_request_queue(self):
        return _EmptyRawQueue()

    def enqueue_shutdown_request(self):
        self.shutdown_enqueued = True


def _make_executor():
    ex = PyExecutor.__new__(PyExecutor)
    ex._error_budget = ErrorBudget()
    ex._fatal_error = None
    ex.is_shutdown = False
    ex.waiting_queue = []
    ex.executor_request_queue = _FakeExecQueue()
    ex.active_requests = []
    ex.gather_all_responses = False
    enqueued = []
    terminated = []
    ex._enqueue_responses = lambda items: enqueued.extend(items)
    ex._terminate_request = lambda r: terminated.append(r.py_request_id)
    return ex, enqueued, terminated


class TestTokenBudgetFatalShutdown(unittest.TestCase):
    def test_token_budget_error_terminates_server(self):
        # _handle_token_budget_error must fail ALL active requests with the
        # message, mark shutdown, and enqueue a shutdown request.
        ex, enqueued, terminated = _make_executor()
        ex.active_requests = [_FakeReq(1), _FakeReq(2)]

        ex._handle_token_budget_error(TokenBudgetExceededError("overshot by 100 tokens"))

        self.assertIsNotNone(ex._fatal_error)
        self.assertTrue(ex.is_shutdown)
        self.assertTrue(ex.executor_request_queue.shutdown_enqueued)
        self.assertEqual({rid for rid, _ in enqueued}, {1, 2})
        for _, resp in enqueued:
            self.assertIn("overshot by 100 tokens", resp.error_msg)
        self.assertEqual(set(terminated), {1, 2})
        self.assertEqual(ex.active_requests, [])

    def test_immediate_fatal_bypasses_error_budget(self):
        # immediate_fatal forces a fatal shutdown even with a pristine budget
        # and charge_budget=False (the budget is never consulted).
        ex, _, _ = _make_executor()
        ex.active_requests = [_FakeReq(1)]

        ex._handle_errors("boom", charge_budget=False, immediate_fatal=True)

        self.assertIsNotNone(ex._fatal_error)
        self.assertTrue(ex.is_shutdown)
        self.assertTrue(ex.executor_request_queue.shutdown_enqueued)

    def test_request_scoped_error_does_not_shutdown(self):
        # Guard against regressing the per-request path: a non-fatal,
        # budget-free request error must NOT trigger shutdown.
        ex, enqueued, terminated = _make_executor()
        req = _FakeReq(1)
        other = _FakeReq(2)
        ex.active_requests = [req, other]

        ex._handle_errors("bad input", requests=[req], charge_budget=False)

        self.assertIsNone(ex._fatal_error)
        self.assertFalse(ex.is_shutdown)
        self.assertFalse(ex.executor_request_queue.shutdown_enqueued)
        # Only the named request was failed; the other stays active.
        self.assertEqual([rid for rid, _ in enqueued], [1])
        self.assertEqual(ex.active_requests, [other])


if __name__ == "__main__":
    unittest.main()
