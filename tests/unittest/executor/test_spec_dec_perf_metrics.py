"""Unit tests for backfilling RequestPerfMetrics.speculative_decoding.

The PyTorch flow (TorchSampler) never calls
LlmRequest::updateNumTokensPerIteration — the only C++ writer of the
speculative_decoding perf-metrics section — so with
SamplingParams(return_perf_metrics=True) the section used to arrive zeroed
even when drafting ran. GenerationResultBase._maybe_fill_spec_dec_perf_metrics
backfills it from the cumulative totals the PyTorch executor attaches to the
response (LlmResult.spec_dec_totals). These tests exercise that client-side
fill logic directly (CPU-only; no engine needed).
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import GenerationResultBase


def _fill(spec_dec_totals, perf_metrics):
    """Invoke the fill helper with a minimal stand-in for the result object."""
    stub = SimpleNamespace(spec_dec_totals=spec_dec_totals)
    GenerationResultBase._maybe_fill_spec_dec_perf_metrics(stub, perf_metrics)


def test_fills_zeroed_section_from_totals():
    pm = tllm.RequestPerfMetrics()
    assert pm.speculative_decoding.total_draft_tokens == 0

    _fill((30, 40), pm)

    spec_dec = pm.speculative_decoding
    assert spec_dec.total_accepted_draft_tokens == 30
    assert spec_dec.total_draft_tokens == 40
    assert spec_dec.acceptance_rate == pytest.approx(0.75)


def test_noop_without_totals():
    pm = tllm.RequestPerfMetrics()

    _fill(None, pm)

    assert pm.speculative_decoding.total_accepted_draft_tokens == 0
    assert pm.speculative_decoding.total_draft_tokens == 0


def test_noop_when_section_already_populated():
    # TRT-engine / TRTLLMSampler paths populate the section runtime-side
    # (updateNumTokensPerIteration); the backfill must not overwrite it.
    pm = tllm.RequestPerfMetrics()
    spec_dec = tllm.SpeculativeDecodingMetrics()
    spec_dec.total_accepted_draft_tokens = 5
    spec_dec.total_draft_tokens = 10
    spec_dec.acceptance_rate = 0.5
    pm.speculative_decoding = spec_dec

    _fill((30, 40), pm)

    assert pm.speculative_decoding.total_accepted_draft_tokens == 5
    assert pm.speculative_decoding.total_draft_tokens == 10


def test_noop_on_nonpositive_drafted():
    pm = tllm.RequestPerfMetrics()

    _fill((0, 0), pm)

    assert pm.speculative_decoding.total_draft_tokens == 0


def test_survives_pickle_roundtrip():
    # Responses cross the worker/proxy IPC boundary pickled; the patched
    # metrics object must round-trip with the backfilled values.
    import pickle

    pm = tllm.RequestPerfMetrics()
    _fill((3, 4), pm)

    restored = pickle.loads(pickle.dumps(pm))
    assert restored.speculative_decoding.total_accepted_draft_tokens == 3
    assert restored.speculative_decoding.total_draft_tokens == 4
    assert restored.speculative_decoding.acceptance_rate == pytest.approx(0.75)
