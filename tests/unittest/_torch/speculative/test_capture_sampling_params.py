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
            self.assertAlmostEqual(sampling_config.temperature, CAPTURE_TEMPERATURE, places=6)
            self.assertEqual(sampling_config.top_k, CAPTURE_TOP_K)
            self.assertAlmostEqual(sampling_config.top_p, CAPTURE_TOP_P, places=6)
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


def _request(temperature=None, top_k=None, top_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=top_k,
            top_p=[top_p] if top_p is not None else None,
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
    # Drop the trailing num_tokens; only the sampling params matter here.
    return [entry[:3] for entry in normalized]


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


if __name__ == "__main__":
    unittest.main()
