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

import textwrap
import threading
import time
from pathlib import Path
from typing import Any

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
    """Drive the configured disagg stress mode and assert current pass criteria."""
    config_path = _CONFIG_DIR / config_filename
    assert config_path.exists(), (
        f"Marathon config not found: {config_path}. "
        "Did you regenerate the symlinks or rename the YAML?"
    )

    harness = DisaggCancellationStressHarness(config_path)
    try:
        harness.setup()
        harness.start()
        timeout_s = float(harness.config.duration_min) * 60.0 + 300.0
        clean = harness.wait_until_done(timeout_s=timeout_s)
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
        f"Harness tripped fail-fast: {results['failure_reason']!r}"
    )
    if harness.config.is_log_only:
        assert any(
            record.get("mode") == "log_only" and record.get("success")
            for record in results["canary_records"]
        ), "log_only mode completed without a successful server probe"


def _write_mode_yaml(tmp_path: Path, stress_config: str) -> Path:
    """Write a minimal marathon YAML for mode-level harness tests."""
    yaml_path = tmp_path / "mode.yaml"
    content = textwrap.dedent(
        """\
        hostname: localhost
        model: dummy
        backend: pytorch
        context_servers: {}
        generation_servers: {}
        stress_config:
        """
    )
    content += textwrap.indent(textwrap.dedent(stress_config).strip(), "  ") + "\n"
    yaml_path.write_text(content)
    return yaml_path


@pytest.mark.parametrize("mode", ["log_only", "full_cancel_poison"])
def test_stress_config_accepts_supported_modes(tmp_path: Path, mode: str) -> None:
    """Both supported mode strings should parse and expose helper predicates."""
    cfg = StressConfig.from_yaml_path(
        _write_mode_yaml(
            tmp_path,
            f"""\
            mode: {mode}
            duration_min: 1
            kv_cache_manager: v1
            transceiver: cpp
            """,
        )
    )

    assert cfg.mode == mode
    assert cfg.is_log_only is (mode == "log_only")
    assert cfg.is_full_cancel_poison is (mode == "full_cancel_poison")


def test_stress_config_rejects_unknown_mode(tmp_path: Path) -> None:
    """Typos in mode must fail during YAML validation."""
    with pytest.raises(ValueError, match="mode must be one of"):
        StressConfig.from_yaml_path(
            _write_mode_yaml(
                tmp_path,
                """\
                mode: accidental
                duration_min: 1
                kv_cache_manager: v1
                transceiver: cpp
                """,
            )
        )


def test_log_only_thread_sends_probe_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regular-protection mode should require at least one clean probe."""
    h = DisaggCancellationStressHarness(
        _write_mode_yaml(
            tmp_path,
            """\
            mode: log_only
            duration_min: 1
            kv_cache_manager: v1
            transceiver: cpp
            log_only_probe:
              interval_s: 0.01
              max_tokens: 8
              request_timeout_s: 1
            log_scan:
              hard_zero_patterns:
                - "Broken promise"
            """,
        ),
        load_duration_s=0.03,
    )
    h.bind_server_endpoint("http://127.0.0.1:8000", "test-model")
    h._marathon_start_monotonic = time.monotonic()

    calls: list[dict[str, Any]] = []

    def fake_probe(**kwargs: Any) -> tuple[list[int], None, None]:
        calls.append(kwargs)
        return [1, 2], None, None

    monkeypatch.setattr(h, "_send_log_only_probe", fake_probe)

    thread = threading.Thread(target=h._log_only_thread_body, name="test-log-only", daemon=True)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert h.stop_event.is_set()
    assert not h.failed_event.is_set()
    assert calls
    assert any(record["success"] for record in h._canary_records)
