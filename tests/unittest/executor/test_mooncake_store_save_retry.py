# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Store fault and original worker-lifecycle tests; no torch import."""

import ast
import importlib.util
import os
import queue
import sys
import threading
import traceback
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store"
SPEC = importlib.util.spec_from_file_location("save_retry_cpu", SRC / "save_retry.py")
retry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = retry
SPEC.loader.exec_module(retry)


class Store:
    def __init__(self, results, probes=None, observe=None):
        self.results = iter(results)
        self.probes = iter(probes or [])
        self.puts = []
        self.probe_calls = []
        self.observe = observe

    def batch_put_from_multi_buffers(self, keys, addresses, sizes):
        self.puts.append((list(keys), list(addresses), list(sizes)))
        if self.observe:
            self.observe()
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result

    def batch_is_exist(self, keys):
        self.probe_calls.append(list(keys))
        result = next(self.probes, [0] * len(keys))
        if isinstance(result, Exception):
            raise result
        return result


def put(store, clock=lambda: 0.0):
    return retry.put_with_retries(store, ["a", "b"], [[11], [22]], [[4], [4]], clock=clock)


class RetryTests(unittest.TestCase):
    def test_healthy_is_one_put_without_retry_probe(self):
        s = Store([[0, 0]])
        r = put(s)
        self.assertEqual((r.put_attempts, r.failed_indices), (1, ()))
        self.assertEqual(len(s.puts), 1)
        self.assertEqual(s.probe_calls, [])

    def test_partial_retries_only_failed_key(self):
        s = Store([[0, -200], [0]])
        r = put(s)
        self.assertEqual([x[0] for x in s.puts], [["a", "b"], ["b"]])
        self.assertEqual(s.puts[1][1:], ([[22]], [[4]]))
        self.assertEqual((r.recovered_keys, r.retried_keys), (1, 1))

    def test_probe_recovers_unknown_write_without_duplicate_put(self):
        s = Store([RuntimeError("transport")], probes=[[1, 1]])
        r = put(s)
        self.assertEqual((len(s.puts), r.recovered_by_probe, r.failed_indices), (1, 2, ()))

    def test_all_fail_bounded_and_no_sleep(self):
        s = Store([[-200, -200]] * 3)
        r = put(s)
        self.assertEqual((len(s.puts), r.failed_indices), (3, (0, 1)))
        self.assertEqual(r.failure_codes, (-200,))

    def test_exception_retried_and_bounded(self):
        s = Store([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        r = put(s)
        self.assertEqual((r.exception_attempts, len(s.puts)), (3, 3))

    def test_empty_and_short_vector_cannot_certify_missing_keys(self):
        for first, second_keys in [([], ["a", "b"]), ([0], ["b"])]:
            s = Store([first, [0] * len(second_keys)])
            r = put(s)
            self.assertEqual(s.puts[1][0], second_keys)
            self.assertEqual(r.failed_indices, ())

    def test_noninteger_and_oversize_results_unknown(self):
        for result, retry_keys in [
            ([True, 0.0], ["a", "b"]),
            ([0, 0, 0], ["a", "b"]),
            (None, ["a", "b"]),
            (["0", 0], ["a"]),
        ]:
            s = Store([result, [0] * len(retry_keys)])
            r = put(s)
            self.assertEqual([x[0] for x in s.puts], [["a", "b"], retry_keys])
            self.assertGreater(r.invalid_results, 0)
            self.assertEqual(r.failed_indices, ())

    def test_deadline_prevents_new_retry_after_slow_first_call(self):
        times = iter([0.0, 0.2])
        s = Store([[-200, -200]])
        r = put(s, clock=lambda: next(times))
        self.assertEqual((r.put_attempts, r.failed_indices), (1, (0, 1)))

    def test_probe_consuming_budget_prevents_new_put(self):
        times = iter([0.0, 0.01, 0.2])
        s = Store([[-200, -200]])
        r = put(s, clock=lambda: next(times))
        self.assertEqual(r.put_attempts, 1)

    def test_probe_error_does_not_erase_unresolved_keys(self):
        s = Store([[-200, -200], [0, 0]], probes=[RuntimeError("probe")])
        r = put(s)
        self.assertEqual(r.failed_indices, ())
        self.assertEqual(r.exception_attempts, 1)

    def test_buffers_not_mutated_or_reallocated(self):
        addresses = [[11], [22]]
        sizes = [[4], [4]]
        s = Store([[-200, 0], [0]])
        retry.put_with_retries(s, ["a", "b"], addresses, sizes, clock=lambda: 0.0)
        self.assertIs(s.puts[1][1][0], addresses[0])
        self.assertEqual(addresses, [[11], [22]])
        self.assertEqual(sizes, [[4], [4]])

    def test_invalid_lengths_do_not_call_store(self):
        s = Store([])
        with self.assertRaises(ValueError):
            retry.put_with_retries(s, ["a"], [], [])
        self.assertEqual(s.puts, [])




if __name__ == "__main__":
    unittest.main()
