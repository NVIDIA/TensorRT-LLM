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
"""CPU-only unit tests for the attention-perf pass/fail helpers.

The integration tests in ``test_attention_perf_module.py`` are all ``@_GPU``
skipped, so the pure gate/CV/golden-lookup logic that actually decides pass vs
fail never runs in CI on machines without the target GPU/arch. A regression in
the threshold formula (``_continuous_gate``) or the bootstrap semantics
(``_bootstrap_or_skip``) would therefore slip through unnoticed. These tests
exercise that math directly and run unconditionally (no CUDA required).
"""

from __future__ import annotations

import math

import pytest
import test_attention_perf_module as m

# --------------------------------------------------------------------------- #
# _continuous_gate: dict (variance-based) vs legacy scalar entries
# --------------------------------------------------------------------------- #


def test_continuous_gate_dict_k_sigma_dominates():
    # k_sigma * cv (4 * 0.03 = 0.12) > rel_floor (0.01) -> margin = 0.12
    baseline, margin, threshold = m._continuous_gate({"baseline_ms": 10.0, "cv": 0.03, "n": 5})
    assert baseline == 10.0
    assert margin == pytest.approx(m._DEFAULT_K_SIGMA * 0.03)
    assert threshold == pytest.approx(10.0 * (1 + m._DEFAULT_K_SIGMA * 0.03))


def test_continuous_gate_dict_rel_floor_dominates():
    # k_sigma * cv (4 * 0.0001 = 0.0004) < rel_floor default (0.01) -> floor wins
    baseline, margin, threshold = m._continuous_gate({"baseline_ms": 2.79, "cv": 0.0001, "n": 8})
    assert margin == pytest.approx(m._DEFAULT_REL_FLOOR)
    assert threshold == pytest.approx(2.79 * (1 + m._DEFAULT_REL_FLOOR))


def test_continuous_gate_dict_respects_explicit_overrides():
    baseline, margin, threshold = m._continuous_gate(
        {"baseline_ms": 1.0, "cv": 0.02, "n": 6, "k_sigma": 6.0, "rel_floor": 0.2}
    )
    # max(6 * 0.02 = 0.12, 0.2) -> floor override wins
    assert margin == pytest.approx(0.2)
    assert threshold == pytest.approx(1.2)


def test_continuous_gate_dict_missing_cv_defaults_to_floor():
    # No cv -> cv=0 -> k*cv=0 -> margin falls back to rel_floor.
    _, margin, _ = m._continuous_gate({"baseline_ms": 5.0})
    assert margin == pytest.approx(m._DEFAULT_REL_FLOOR)


def test_continuous_gate_legacy_scalar_uses_flat_fallback():
    baseline, margin, threshold = m._continuous_gate(3.0)
    assert baseline == 3.0
    assert margin == pytest.approx(m._LEGACY_CONT_THRESHOLD - 1.0)
    assert threshold == pytest.approx(3.0 * m._LEGACY_CONT_THRESHOLD)


# --------------------------------------------------------------------------- #
# _observed_cv: edge cases
# --------------------------------------------------------------------------- #


def test_observed_cv_empty_list_is_zero():
    assert m._observed_cv([], 1.0) == 0.0


def test_observed_cv_single_sample_is_zero():
    assert m._observed_cv([1.23], 1.23) == 0.0


def test_observed_cv_zero_median_is_zero():
    # median falsy -> guard returns 0.0 (no divide-by-zero).
    assert m._observed_cv([0.0, 0.0, 0.0], 0.0) == 0.0
    assert m._observed_cv([1.0, 2.0], None) == 0.0


def test_observed_cv_known_values():
    import statistics

    times = [1.0, 2.0, 3.0]
    median = 2.0
    expected = statistics.pstdev(times) / median
    assert m._observed_cv(times, median) == pytest.approx(expected)
    assert math.isfinite(m._observed_cv(times, median))


# --------------------------------------------------------------------------- #
# _golden_for / _bootstrap_or_skip: missing vs falsy-but-present goldens
# --------------------------------------------------------------------------- #


def test_golden_for_missing_case_returns_none():
    assert m._golden_for("no_such_case_id_xyz", "NoSuchGPU") is None


def test_bootstrap_or_skip_missing_golden_skips(monkeypatch):
    monkeypatch.setattr(m, "_golden_for", lambda *a, **k: None)
    with pytest.raises(pytest.skip.Exception):
        m._bootstrap_or_skip("case", "gpu", observed=123)


@pytest.mark.parametrize("falsy_golden", [False, 0])
def test_bootstrap_or_skip_falsy_but_present_golden_returned(monkeypatch, falsy_golden):
    # A blessed golden of False/0 (e.g. discrete launch_count sentinel) is a real
    # entry, NOT "missing" -> must be returned, not skipped.
    monkeypatch.setattr(m, "_golden_for", lambda *a, **k: falsy_golden)
    assert m._bootstrap_or_skip("case", "gpu", observed=1) == falsy_golden
