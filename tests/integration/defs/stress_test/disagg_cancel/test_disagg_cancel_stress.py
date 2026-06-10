# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Marathon-style cancellation stress tests for disagg KV transfer.

Parametrized pytest entry point for the cancellation stress suite;
the actual work happens in ``harness.py``. Marathon YAMLs live in
``configs/`` next to this file, one per backend-knob combination
(KV cache manager x transceiver runtime).

Additional marathon configurations land here as the suite expands;
each one is added to ``_MARATHON_CONFIGS`` below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import DisaggCancellationStressHarness, StressConfig

_CONFIG_DIR = Path(__file__).parent / "configs"

# Marathon configurations exercised by the parametrized test below.
# Add a new entry here to wire an additional YAML into the suite.
_MARATHON_CONFIGS: list[str] = [
    "marathon_cpp_v1_deepseek.yaml",
]


def test_all_marathon_yamls_parse_and_validate() -> None:
    """Every ``marathon_*.yaml`` in ``configs/`` must parse and validate.

    Catches YAML-syntax and schema-validity regressions in marathon
    configs that are not yet parametrized into ``_MARATHON_CONFIGS``
    (e.g. configs whose canary references are still being recorded).
    Without it, broken-but-not-yet-wired YAMLs sit undetected until
    the day they are parametrized, which can be PRs away.
    """
    marathon_yamls = sorted(_CONFIG_DIR.glob("marathon_*.yaml"))
    assert marathon_yamls, f"no marathon_*.yaml configs found under {_CONFIG_DIR}"
    for path in marathon_yamls:
        StressConfig.from_yaml_path(path)


@pytest.mark.parametrize("config_filename", _MARATHON_CONFIGS)
def test_disagg_cancellation_marathon(config_filename: str) -> None:
    """Drive a long-running disagg cancellation marathon and assert pass criteria.

    Current scope: only what the already-implemented thread bodies
    can contribute. The marathon entry point exists; the marathon
    *content* lands incrementally as setup / pass-criteria wiring is
    completed:

    - lifecycle plumbing (setup -> start -> wait -> stop ->
      collect_results, fail-fast event propagation, dict-shape
      contract).
    - log-pattern fail-fast — a hard-zero pattern in any worker log
      trips ``failure_reason`` via the log_scanner thread
      (component-level coverage in ``test_log_scanner.py``).

    Marathon pass criteria not yet enforced here (will land alongside
    their owning result aggregation in follow-up changes): canary error
    rate, recovery time after each injection, KV-cache utilization
    growth bound, injection-schedule completeness, sustained load
    throughput. Until those land, this test passes trivially after
    the lifecycle smoke completes; the value at this stage is that
    the entry point and result-dict contract are pinned down so the
    follow-up commits can extend in place rather than restructure.
    """
    config_path = _CONFIG_DIR / config_filename
    assert config_path.exists(), (
        f"Marathon config not found: {config_path}. "
        "Did you regenerate the symlinks or rename the YAML?"
    )

    harness = DisaggCancellationStressHarness(config_path)
    try:
        harness.setup()
        harness.start()
        # setup() is still a stub, so no server endpoint is bound.
        # The load thread exits and signals ``stop_event`` on that
        # no-endpoint path, which lets this lifecycle smoke complete
        # almost instantly. Once setup launches a real cluster, the
        # timeout becomes ``stress_config.duration_min`` plus a safety
        # margin.
        clean = harness.wait_until_done(timeout_s=10.0)
        assert clean is True, (
            f"wait_until_done did not return cleanly; failure_reason={harness.failure_reason!r}"
        )
    finally:
        harness.stop()

    results = harness.collect_results()
    # Skeleton stage: no real pass criteria yet. Just confirm the
    # collector returns the expected shape so future commits can
    # extend in place.
    assert "canary_records" in results
    assert "load_records" in results
    assert "kv_utilization_samples" in results
    assert "injection_events" in results
    assert results["failure_reason"] is None, (
        f"Harness tripped fail-fast in skeleton run: {results['failure_reason']!r}"
    )
