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


@unittest.skipUnless(torch.cuda.is_available(), "populate allocates CUDA buffers")
class TestMinPBufferFillIsGatedOnUniversal(unittest.TestCase):
    """Only UNIVERSAL reads the min_p buffers, so only UNIVERSAL should fill them.

    The cheap direction of a mistake here is wasted host work on every existing deploy.
    The expensive direction is silent: if UNIVERSAL stopped filling them, every request's
    min_p would read as the 0.0 sentinel and the filter would vanish without an error.
    """

    def test_universal_fills_the_min_p_buffers(self):
        meta = _populate_meta(AdvancedSamplingMode.UNIVERSAL)
        SpecMetadata.populate_sampling_params_for_one_model(
            meta, [_request(temperature=1.0, min_p=0.25)]
        )
        self.assertAlmostEqual(meta.request_min_ps[0].item(), 0.25, places=6)
        self.assertAlmostEqual(meta.min_ps[0].item(), 0.25, places=6)

    def test_full_leaves_them_at_the_disable_sentinel(self):
        # A min_p request cannot reach populate under FULL -- validate_request rejects it
        # -- so the buffers must stay at the 0.0 they were allocated with, and the fill
        # must not run.
        meta = _populate_meta(AdvancedSamplingMode.FULL)
        SpecMetadata.populate_sampling_params_for_one_model(
            meta, [_request(temperature=1.0, top_p=0.9)]
        )
        self.assertEqual(meta.request_min_ps[0].item(), 0.0)
        self.assertEqual(meta.min_ps[0].item(), 0.0)
        # The other buffers are still filled, i.e. the gate is narrow.
        self.assertAlmostEqual(meta.request_top_ps[0].item(), 0.9, places=6)


def _context_request(**kwargs):
    """A request that has not started generating: its expanded span is one row, not
    ``draft_len + 1``."""
    request = _request(**kwargs)
    request.state = LlmRequestState.CONTEXT_INIT
    return request


@unittest.skipUnless(torch.cuda.is_available(), "populate allocates CUDA buffers")
class TestMinPExpandsWithTheSameLayoutAsTopP(unittest.TestCase):
    """Hazard B4: the expanded per-token buffers are laid out by each request's token
    count, and a context request occupies one row where a generation request occupies
    ``draft_len + 1``.

    min_p is filled by a separate ``if fill_min_p`` branch from the one that fills top_p.
    Two branches walking the same list is only correct as long as they agree on the
    layout, and nothing in the types would catch them diverging -- the buffers are flat
    float32 and any misalignment is a silently wrong filter on the wrong token, not a
    crash.
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
        meta = _populate_meta(AdvancedSamplingMode.UNIVERSAL, draft_len=self.DRAFT_LEN)
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

    def test_layout_is_rebuilt_when_a_context_request_starts_generating(self):
        """The transition itself, with the sampling parameters held fixed.

        Only the token counts change, so the per-request buffers stay valid and only the
        expanded ones must be refilled. If the refill were keyed on the sampling values
        alone, min_p would keep the previous, shorter layout and every token after the
        transition would read its neighbour's filter.
        """
        meta = _populate_meta(AdvancedSamplingMode.UNIVERSAL, draft_len=self.DRAFT_LEN)
        min_ps = [0.1, 0.2]
        top_ps = [0.7, 0.8]

        def batch(second_is_generating):
            first = _request(temperature=1.0, min_p=min_ps[0], top_p=top_ps[0], slot=0)
            make = _request if second_is_generating else _context_request
            second = make(temperature=1.0, min_p=min_ps[1], top_p=top_ps[1], slot=1)
            return [first, second]

        before = batch(second_is_generating=False)
        SpecMetadata.populate_sampling_params_for_one_model(meta, before)
        self.assertEqual(self._assert_aligned(meta, before, min_ps, top_ps), 5)

        after = batch(second_is_generating=True)
        SpecMetadata.populate_sampling_params_for_one_model(meta, after)
        self.assertEqual(self._assert_aligned(meta, after, min_ps, top_ps), 8)

    def test_transition_invalidates_only_the_expanded_signature(self):
        """The refill decision itself, stated directly rather than through the buffers."""
        meta = _populate_meta(AdvancedSamplingMode.UNIVERSAL, draft_len=self.DRAFT_LEN)
        as_context = [_context_request(temperature=1.0, min_p=0.1, top_p=0.7, slot=0)]
        as_generation = [_request(temperature=1.0, min_p=0.1, top_p=0.7, slot=0)]

        normalized, _ = meta._scan_one_model_sampling(as_context)
        meta._sampling_params_buffers_need_update(normalized)

        normalized, _ = meta._scan_one_model_sampling(as_generation)
        need_request, need_expanded = meta._sampling_params_buffers_need_update(normalized)
        self.assertFalse(need_request, "sampling values did not change")
        self.assertTrue(need_expanded, "the token count -- and so the layout -- did")


if __name__ == "__main__":
    unittest.main()
