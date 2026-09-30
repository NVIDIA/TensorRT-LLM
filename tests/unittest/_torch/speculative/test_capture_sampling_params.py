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

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import NON_GREEDY_CAPTURE_SAMPLING_PARAMS
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, KvCacheConfig
from tensorrt_llm.mapping import Mapping

# The synthetic params the refactored capture path warms up with.
CAPTURE_TEMPERATURE = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.temperature
CAPTURE_TOP_K = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.top_k
CAPTURE_TOP_P = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.top_p
CAPTURE_MIN_P = NON_GREEDY_CAPTURE_SAMPLING_PARAMS.min_p


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
            self.assertAlmostEqual(sampling_config.min_p, CAPTURE_MIN_P, places=6)
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
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def _fake_meta():
    return types.SimpleNamespace(
        runtime_draft_len=1, dummy_slot_row=0, group_all_greedy_sample=None
    )


def _scan(meta, requests):
    """The four sampling knobs, dropping the trailing num_tokens."""
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
                temperature=CAPTURE_TEMPERATURE,
                top_k=CAPTURE_TOP_K,
                top_p=CAPTURE_TOP_P,
                min_p=CAPTURE_MIN_P,
                slot=None,
            ),
            _request(
                temperature=CAPTURE_TEMPERATURE,
                top_k=CAPTURE_TOP_K,
                top_p=CAPTURE_TOP_P,
                min_p=CAPTURE_MIN_P,
                slot=None,
            ),
        ]
        normalized = _scan(meta, warmup_requests)
        self.assertEqual(
            normalized,
            [(CAPTURE_TEMPERATURE, CAPTURE_TOP_K, CAPTURE_TOP_P, CAPTURE_MIN_P)]
            * len(warmup_requests),
        )
        self.assertFalse(meta.is_all_greedy_sample)

    def test_parameterless_warmup_requests_still_scan_greedy(self):
        # No SpecMetadata flag exists to force these non-greedy anymore: a
        # warmup pass that (mis-)builds dummy requests without capture params
        # must fall back to the ordinary greedy classification, not silently
        # capture the advanced-sampling graph.
        meta = _fake_meta()
        normalized = _scan(meta, [_request(slot=None), _request(slot=None)])
        self.assertTrue(all(temp != CAPTURE_TEMPERATURE for temp, _, _, _ in normalized))
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
                    min_p=CAPTURE_MIN_P,
                    slot=None,
                )
            ],
        )
        self.assertFalse(meta.is_all_greedy_sample)

        serving_normalized = _scan(meta, [_request(temperature=1.0, top_p=1.0, slot=1)])
        temp, top_k, top_p, min_p = serving_normalized[0]
        self.assertEqual(temp, 1.0)
        self.assertNotEqual(top_k, CAPTURE_TOP_K)
        self.assertEqual(top_p, 1.0)
        self.assertNotEqual(min_p, CAPTURE_MIN_P)


def _populate_meta(mode, draft_len=1):
    """Stand-in with just enough of SpecMetadata to run populate_sampling_params_for_one_model.

    Same style as test_rejection_buffers_guard.py: SpecMetadata methods called unbound on a
    namespace, with the parts not under test stubbed out.
    """
    meta = types.SimpleNamespace(
        runtime_draft_len=draft_len,
        dummy_slot_row=0,
        group_all_greedy_sample=None,
        max_num_requests=4,
        max_draft_len=draft_len,
        max_total_draft_tokens=draft_len,
        is_spec_dec_tree=False,
        advanced_sampling_mode=mode,
        use_rejection_sampling=False,
        enable_penalty=False,
        batch_slot_ids=None,
        temperatures=None,
        top_ks=None,
        top_ps=None,
        min_ps=None,
        request_temperatures=None,
        request_top_ks=None,
        request_top_ps=None,
        request_min_ps=None,
        top_k_max=0,
        _sampling_params_signature=[None, None],
        spec_dec_mode=types.SimpleNamespace(use_one_engine=lambda: True),
        # Not under test.
        prepare_rejection_sampling_buffers=lambda: None,
        prepare_penalty_buffers=lambda: None,
        _populate_request_rng_state=lambda requests, normalized: None,
        _populate_penalty_params=lambda requests: None,
    )
    for name in (
        "_scan_one_model_sampling",
        "_sampling_params_buffers_need_update",
        "invalidate_sampling_params_cache",
    ):
        setattr(
            meta, name, (lambda fn: lambda *a, **k: fn(meta, *a, **k))(getattr(SpecMetadata, name))
        )
    return meta


def _context_request(**kwargs):
    """A request that has not started generating: its expanded span is one row, not
    ``draft_len + 1``."""
    request = _request(**kwargs)
    request.state = LlmRequestState.CONTEXT_INIT
    return request


@unittest.skipUnless(torch.cuda.is_available(), "populate allocates CUDA buffers")
class TestExpandedBufferLayout(unittest.TestCase):
    """The expanded per-token buffers are laid out by each request's token count: a
    context request occupies one row, a generation request ``draft_len + 1``. Each
    per-token filter is filled by its own pass over the same list, so a misalignment
    between them is a wrong filter on the wrong token rather than a crash.
    """

    DRAFT_LEN = 3

    def _expected_owner_per_token(self, requests):
        """Token index -> index of the request that owns it."""
        owners = []
        for i, request in enumerate(requests):
            span = (
                1 + self.DRAFT_LEN if request.state == LlmRequestState.GENERATION_IN_PROGRESS else 1
            )
            owners.extend(i for _ in range(span))
        return owners

    def _assert_aligned(self, meta, requests, min_ps, top_ps):
        owners = self._expected_owner_per_token(requests)
        for token, owner in enumerate(owners):
            self.assertAlmostEqual(
                meta.min_ps[token].item(),
                min_ps[owner],
                places=6,
                msg=f"token {token} should carry request {owner}'s min_p",
            )
            self.assertAlmostEqual(
                meta.top_ps[token].item(),
                top_ps[owner],
                places=6,
                msg=f"token {token} should carry request {owner}'s top_p",
            )
        return len(owners)

    def test_mixed_context_and_generation_batch(self):
        meta = _populate_meta(AdvancedSamplingMode.FULL, draft_len=self.DRAFT_LEN)
        min_ps = [0.1, 0.2, 0.3]
        top_ps = [0.7, 0.8, 0.9]
        requests = [
            _request(temperature=1.0, min_p=min_ps[0], top_p=top_ps[0], slot=0),
            _context_request(temperature=1.0, min_p=min_ps[1], top_p=top_ps[1], slot=1),
            _request(temperature=1.0, min_p=min_ps[2], top_p=top_ps[2], slot=2),
        ]
        SpecMetadata.populate_sampling_params_for_one_model(meta, requests)
        # 4 + 1 + 4: the context request in the middle shifts every later token.
        self.assertEqual(self._assert_aligned(meta, requests, min_ps, top_ps), 9)


if __name__ == "__main__":
    unittest.main()
