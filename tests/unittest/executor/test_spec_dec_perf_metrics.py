# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for backfilling RequestPerfMetrics.speculative_decoding.

Nothing populates the speculative_decoding perf-metrics section runtime-side,
so with SamplingParams(return_perf_metrics=True) it arrives zeroed even when
drafting ran. GenerationResultBase._maybe_fill_spec_dec_perf_metrics backfills
it from the cumulative totals the PyTorch executor attaches to the response
(LlmResult.spec_dec_totals). These tests exercise that client-side fill logic
directly (CPU-only; no engine needed).
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import GenerationResultBase

# CI's CPU stages select tests with ``-m "cpu_only and not disabled"``; without
# this marker the file would be collected but never run there.
pytestmark = pytest.mark.cpu_only


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
    assert pm.speculative_decoding.acceptance_rate == pytest.approx(0.0)


def test_noop_on_nonpositive_drafted():
    pm = tllm.RequestPerfMetrics()

    _fill((0, 0), pm)

    assert pm.speculative_decoding.total_accepted_draft_tokens == 0
    assert pm.speculative_decoding.total_draft_tokens == 0
    assert pm.speculative_decoding.acceptance_rate == pytest.approx(0.0)


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
