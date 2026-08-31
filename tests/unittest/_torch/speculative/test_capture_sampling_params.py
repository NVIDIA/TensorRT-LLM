# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the refactored non-greedy CUDA graph capture path
(TRTLLM-14874).

The advanced-sampling CUDA graph capture pass used to force the non-greedy
branch by mutating a ``_force_non_greedy_for_capture`` flag on the live
``SpecMetadata``. ``create_cuda_graph_metadata`` shallow-copied that flag into
every cached graph entry, and those copies were later reseated as the live
``spec_metadata`` on replay -- leaking the flag into serving and silently
rewriting every request's sampling params to the synthetic capture values
(temperature=0.7, top_k=50, top_p=0.9).

The fix drives capture with *real* non-greedy warmup ``SamplingParams``
(``KVCacheManager.add_dummy_requests(capture_sampling_params=...)``) instead
of a metadata flag: warmup requests carry genuine sampling params, so
``SpecMetadata._scan_one_model_sampling`` classifies them as non-greedy the
same way it would classify any real client request. There is no capture-only
mutable state left on ``SpecMetadata`` to leak, so these tests assert that
property rather than a specific teardown step.
"""

import types
import unittest

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import NON_GREEDY_CAPTURE_SAMPLING_PARAMS
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# The synthetic params the refactored capture path warms up with.
CAPTURE_TEMPERATURE = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.temperature
CAPTURE_TOP_K = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.top_k
CAPTURE_TOP_P = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.top_p


class TestAddDummyRequestsCaptureSamplingParams(unittest.TestCase):
    """`KVCacheManager.add_dummy_requests(capture_sampling_params=...)` must
    stamp the synthetic non-greedy values onto the dummy requests it builds,
    and must leave requests greedy when no capture params are supplied."""

    def _kv_cache_manager(self):
        return KVCacheManager(
            kv_cache_config=KvCacheConfig(max_tokens=256, enable_block_reuse=False),
            kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=2,
            num_kv_heads=2,
            head_dim=128,
            tokens_per_block=8,
            max_seq_len=64,
            max_batch_size=1,
            mapping=Mapping(),
        )

    def test_capture_sampling_params_stamped_onto_dummy_requests(self):
        kv_cache_manager = self._kv_cache_manager()
        try:
            requests = kv_cache_manager.add_dummy_requests(
                [0],
                token_nums=[8],
                capture_sampling_params=NON_GREEDY_CAPTURE_SAMPLING_PARAMS,
            )
            self.assertEqual(len(requests), 1)
            sampling_config = requests[0].sampling_config
            # Sampling params round-trip through a C++ binding as float32, so
            # compare with tolerance rather than exact equality.
            self.assertAlmostEqual(sampling_config.temperature[0], CAPTURE_TEMPERATURE, places=6)
            self.assertEqual(sampling_config.top_k, [CAPTURE_TOP_K])
            self.assertAlmostEqual(sampling_config.top_p[0], CAPTURE_TOP_P, places=6)
        finally:
            kv_cache_manager.shutdown()

    def test_no_capture_sampling_params_stays_greedy(self):
        kv_cache_manager = self._kv_cache_manager()
        try:
            requests = kv_cache_manager.add_dummy_requests([0], token_nums=[8])
            self.assertEqual(len(requests), 1)
            sampling_config = requests[0].sampling_config
            self.assertIsNone(sampling_config.temperature)
            self.assertIsNone(sampling_config.top_k)
            self.assertIsNone(sampling_config.top_p)
        finally:
            kv_cache_manager.shutdown()


def _request(temperature=None, top_k=None, top_p=None, min_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=[top_k] if top_k is not None else None,
            top_p=[top_p] if top_p is not None else None,
            min_p=[min_p] if min_p is not None else None,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def _fake_meta():
    return types.SimpleNamespace(
        runtime_draft_len=1, dummy_slot_row=0, group_all_greedy_sample=None
    )


def _scan(meta, requests):
    normalized, _ = SpecMetadata._scan_one_model_sampling(meta, requests)
    # Drop min_p and the trailing num_tokens; these tests are about the other three.
    return [entry[:3] for entry in normalized]


def _scan_with_min_p(meta, requests):
    """Like ``_scan`` but keeps min_p, which the tests below are about."""
    normalized, _ = SpecMetadata._scan_one_model_sampling(meta, requests)
    return [entry[:4] for entry in normalized]


class TestScanOneModelSamplingHonorsRealCaptureParams(unittest.TestCase):
    """`_scan_one_model_sampling` needs no capture-only override: warmup
    requests carrying real non-greedy params classify as non-greedy on their
    own, the same way any client request would."""

    def test_warmup_requests_with_real_capture_params_scan_non_greedy(self):
        meta = _fake_meta()
        warmup_requests = [
            _request(
                temperature=CAPTURE_TEMPERATURE, top_k=CAPTURE_TOP_K, top_p=CAPTURE_TOP_P, slot=None
            ),
            _request(
                temperature=CAPTURE_TEMPERATURE, top_k=CAPTURE_TOP_K, top_p=CAPTURE_TOP_P, slot=None
            ),
        ]
        normalized = _scan(meta, warmup_requests)
        self.assertEqual(
            normalized, [(CAPTURE_TEMPERATURE, CAPTURE_TOP_K, CAPTURE_TOP_P)] * len(warmup_requests)
        )
        self.assertFalse(meta.is_all_greedy_sample)

    def test_parameterless_warmup_requests_still_scan_greedy(self):
        # No SpecMetadata flag exists to force these non-greedy anymore: a
        # warmup pass that (mis-)builds dummy requests without capture params
        # must fall back to the ordinary greedy classification, not silently
        # capture the advanced-sampling graph.
        meta = _fake_meta()
        normalized = _scan(meta, [_request(slot=None), _request(slot=None)])
        self.assertTrue(all(temp != CAPTURE_TEMPERATURE for temp, _, _ in normalized))
        self.assertTrue(meta.is_all_greedy_sample)

    def test_capture_time_values_do_not_leak_into_a_later_serving_scan(self):
        # Regression for the original bug class: a SpecMetadata object (e.g.
        # a graph copy reseated as the live spec_metadata on replay) that was
        # previously scanned with capture-time params must not retain them --
        # every scan is a pure function of the requests passed in, so a
        # later scan with the client's real params must reflect only those.
        meta = _fake_meta()
        _scan(
            meta,
            [
                _request(
                    temperature=CAPTURE_TEMPERATURE,
                    top_k=CAPTURE_TOP_K,
                    top_p=CAPTURE_TOP_P,
                    slot=None,
                )
            ],
        )
        self.assertFalse(meta.is_all_greedy_sample)

        serving_normalized = _scan(meta, [_request(temperature=1.0, top_p=1.0, slot=1)])
        temp, top_k, top_p = serving_normalized[0]
        self.assertEqual(temp, 1.0)
        self.assertNotEqual(top_k, CAPTURE_TOP_K)
        self.assertEqual(top_p, 1.0)


class TestScanOneModelSamplingMinP(unittest.TestCase):
    """min_p's two silent-failure modes in the one-model scan.

    Both are invisible at op level -- the kernel is handed whatever the scan produced --
    and both degrade output rather than raising, so they get their own tests.
    """

    def test_min_p_only_request_is_not_greedy(self):
        """A request whose only knob is min_p must classify as NON-greedy.

        If it classified greedy it would take the argmax fast path and min_p would be
        dropped without a trace. ``is_all_greedy_sample`` is also part of the CUDA graph
        key, so the same mistake would select the argmax graph variant.
        """
        meta = _fake_meta()
        normalized = _scan_with_min_p(meta, [_request(min_p=0.05, slot=0)])
        self.assertFalse(meta.is_all_greedy_sample)
        # ... and the value must survive normalization rather than being reset to the
        # 0.0 disable sentinel the greedy branch would apply.
        self.assertEqual(normalized[0][3], 0.05)

    def test_min_p_one_is_explicit_greedy(self):
        """min_p == 1.0 keeps only the argmax, which SamplingParams documents as
        explicit greedy -- so the scan must agree and take the fast path."""
        meta = _fake_meta()
        _scan_with_min_p(meta, [_request(min_p=1.0, slot=0)])
        self.assertTrue(meta.is_all_greedy_sample)

    def test_absent_min_p_normalizes_to_the_disable_sentinel(self):
        """min_p's neutral value is 0.0, not 1.0 like top_p -- getting that backwards
        would filter every row down to the argmax."""
        meta = _fake_meta()
        normalized = _scan_with_min_p(meta, [_request(temperature=1.0, top_p=0.9, slot=0)])
        self.assertEqual(normalized[0][3], 0.0)

    def test_changing_only_min_p_invalidates_the_buffer_signature(self):
        """Two batches differing ONLY in min_p must refill the device buffers.

        The refill is skipped when the signature is unchanged, so leaving min_p out of it
        would let the second batch decode from the first batch's min_p -- silently, and
        only for requests that happen to follow one another.
        """
        meta = _fake_meta()
        meta._sampling_params_signature = [None, None]

        first, _ = SpecMetadata._scan_one_model_sampling(
            meta, [_request(temperature=1.0, min_p=0.05)]
        )
        SpecMetadata._sampling_params_buffers_need_update(meta, first)

        second, _ = SpecMetadata._scan_one_model_sampling(
            meta, [_request(temperature=1.0, min_p=0.5)]
        )
        need_request, need_expanded = SpecMetadata._sampling_params_buffers_need_update(
            meta, second
        )
        self.assertTrue(need_request)
        self.assertTrue(need_expanded)

    def test_identical_batches_still_skip_the_refill(self):
        """The counterpart: adding min_p to the signature must not defeat the caching
        that keeps a steady-state decode batch off the H2D path."""
        meta = _fake_meta()
        meta._sampling_params_signature = [None, None]

        first, _ = SpecMetadata._scan_one_model_sampling(
            meta, [_request(temperature=1.0, min_p=0.05)]
        )
        SpecMetadata._sampling_params_buffers_need_update(meta, first)

        second, _ = SpecMetadata._scan_one_model_sampling(
            meta, [_request(temperature=1.0, min_p=0.05)]
        )
        need_request, need_expanded = SpecMetadata._sampling_params_buffers_need_update(
            meta, second
        )
        self.assertFalse(need_request)
        self.assertFalse(need_expanded)


if __name__ == "__main__":
    unittest.main()
