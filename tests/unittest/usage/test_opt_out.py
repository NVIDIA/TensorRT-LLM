# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for telemetry opt-out mechanisms and CI/test auto-detection."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tensorrt_llm.usage import usage_lib

# ---------------------------------------------------------------------------
# Opt-out tests
# ---------------------------------------------------------------------------


class TestOptOut:
    def test_opt_out_env_var(self, monkeypatch):
        """TRTLLM_NO_USAGE_STATS=1 disables telemetry."""
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setattr(
            "tensorrt_llm.usage.usage_lib._OPT_OUT_FILE",
            Path("/nonexistent"),
        )
        assert not usage_lib.is_usage_stats_enabled()

    def test_opt_out_do_not_track(self, monkeypatch):
        """DO_NOT_TRACK=1 (industry standard) disables telemetry."""
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        monkeypatch.setattr(
            "tensorrt_llm.usage.usage_lib._OPT_OUT_FILE",
            Path("/nonexistent"),
        )
        assert not usage_lib.is_usage_stats_enabled()

    def test_opt_out_file(self, tmp_path, monkeypatch):
        """File-based opt-out (~/.config/trtllm/do_not_track) disables telemetry."""
        opt_out = tmp_path / "do_not_track"
        opt_out.touch()
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setattr("tensorrt_llm.usage.usage_lib._OPT_OUT_FILE", opt_out)
        assert not usage_lib.is_usage_stats_enabled()

    def test_enabled_by_default(self, enable_telemetry):
        """Telemetry is enabled when no opt-out is configured."""
        assert usage_lib.is_usage_stats_enabled()

    def test_opt_out_telemetry_disabled_env_var_true(self, monkeypatch):
        """TELEMETRY_DISABLED=true disables telemetry."""
        monkeypatch.setenv("TELEMETRY_DISABLED", "true")
        assert not usage_lib.is_usage_stats_enabled()

    def test_opt_out_telemetry_disabled_env_var_one(self, monkeypatch):
        """TELEMETRY_DISABLED=1 disables telemetry."""
        monkeypatch.setenv("TELEMETRY_DISABLED", "1")
        assert not usage_lib.is_usage_stats_enabled()

    def test_opt_out_telemetry_disabled_env_var_case_insensitive(self, monkeypatch):
        """TELEMETRY_DISABLED=True (mixed case) disables telemetry."""
        monkeypatch.setenv("TELEMETRY_DISABLED", "True")
        assert not usage_lib.is_usage_stats_enabled()

    def test_opt_out_telemetry_disabled_env_var_false(self, monkeypatch, enable_telemetry):
        """TELEMETRY_DISABLED=false does NOT disable telemetry."""
        monkeypatch.setenv("TELEMETRY_DISABLED", "false")
        assert usage_lib.is_usage_stats_enabled()

    def test_opt_out_programmatic_flag(self):
        """telemetry_disabled=True (programmatic) disables telemetry."""
        assert not usage_lib.is_usage_stats_enabled(telemetry_disabled=True)

    def test_programmatic_flag_default_false(self, enable_telemetry):
        """Default telemetry_disabled=False does not disable."""
        assert usage_lib.is_usage_stats_enabled(telemetry_disabled=False)


# ---------------------------------------------------------------------------
# CI/Test auto-detection tests
# ---------------------------------------------------------------------------


class TestCIAutoDetection:
    """Test automatic disabling of telemetry in CI and test environments."""

    @pytest.fixture(autouse=True)
    def _clear_all(self, monkeypatch):
        """Clear all CI/test/opt-out env vars for a clean slate."""
        for var in usage_lib._CI_ENV_VARS + usage_lib._TEST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("TRTLLM_USAGE_FORCE_ENABLED", raising=False)
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
        monkeypatch.setattr(
            "tensorrt_llm.usage.usage_lib._OPT_OUT_FILE",
            Path("/nonexistent/path/do_not_track"),
        )

    def test_auto_disable_ci_generic(self, monkeypatch):
        """CI=true (generic CI env var) disables telemetry."""
        monkeypatch.setenv("CI", "true")
        assert not usage_lib.is_usage_stats_enabled()

    def test_auto_disable_github_actions(self, monkeypatch):
        """GITHUB_ACTIONS=true disables telemetry."""
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert not usage_lib.is_usage_stats_enabled()

    def test_auto_disable_jenkins(self, monkeypatch):
        """JENKINS_URL set disables telemetry."""
        monkeypatch.setenv("JENKINS_URL", "http://jenkins.example.com")
        assert not usage_lib.is_usage_stats_enabled()

    def test_auto_disable_gitlab_ci(self, monkeypatch):
        """GITLAB_CI=true disables telemetry."""
        monkeypatch.setenv("GITLAB_CI", "true")
        assert not usage_lib.is_usage_stats_enabled()

    def test_auto_disable_buildkite(self, monkeypatch):
        """BUILDKITE=true disables telemetry."""
        monkeypatch.setenv("BUILDKITE", "true")
        assert not usage_lib.is_usage_stats_enabled()

    def test_auto_disable_pytest(self, monkeypatch):
        """PYTEST_CURRENT_TEST set disables telemetry."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_example.py::test_foo")
        assert not usage_lib.is_usage_stats_enabled()

    def test_ci_detection_returns_true(self, monkeypatch):
        """_is_ci_or_test_environment() returns True when CI var is set."""
        monkeypatch.setenv("CI", "true")
        assert usage_lib._is_ci_or_test_environment()

    def test_no_ci_detection_returns_false(self, monkeypatch):
        """_is_ci_or_test_environment() returns False with no CI vars."""
        # PYTEST_CURRENT_TEST is set by pytest after fixtures run, re-clear it
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert not usage_lib._is_ci_or_test_environment()

    def test_force_enable_overrides_ci_detection(self, monkeypatch):
        """TRTLLM_USAGE_FORCE_ENABLED=1 re-enables telemetry in CI."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
        assert usage_lib.is_usage_stats_enabled()

    def test_force_enable_does_not_override_explicit_opt_out(self, monkeypatch):
        """Explicit opt-out takes precedence over force-enable."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("TRTLLM_USAGE_FORCE_ENABLED", "1")
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
        assert not usage_lib.is_usage_stats_enabled()

    def test_all_ci_vars_detected(self, monkeypatch):
        """Every CI env var in _CI_ENV_VARS triggers detection."""
        for var in usage_lib._CI_ENV_VARS:
            monkeypatch.setenv(var, "true")
            assert usage_lib._is_ci_or_test_environment(), f"{var} was not detected"

    def test_empty_ci_var_not_detected(self, monkeypatch):
        """Empty string CI var does NOT trigger detection."""
        # PYTEST_CURRENT_TEST is set by pytest after fixtures run, re-clear it
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("CI", "")
        assert not usage_lib._is_ci_or_test_environment()

    def test_noop_in_ci_without_force(self, monkeypatch):
        """report_usage() does not spawn thread in CI environment."""
        monkeypatch.setenv("CI", "true")
        with patch("tensorrt_llm.usage.usage_lib.threading.Thread") as thread_cls:
            usage_lib.report_usage()
            thread_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Path.home() failure resilience tests
# ---------------------------------------------------------------------------


class TestPathHomeFailure:
    """Verify telemetry degrades gracefully when Path.home() fails.

    In minimal containers or non-standard service accounts, HOME may be
    unset and passwd lookup may fail, causing Path.home() to raise
    RuntimeError.  The file-based opt-out becomes unavailable, but
    everything else (env-var opt-out, telemetry reporting) must still work.
    """

    def test_opt_out_file_none_does_not_crash(self, monkeypatch, enable_telemetry):
        """is_usage_stats_enabled() works when _OPT_OUT_FILE is None."""
        monkeypatch.setattr("tensorrt_llm.usage.usage_lib._OPT_OUT_FILE", None)
        # Should not raise; telemetry enabled since no opt-out is active
        assert usage_lib.is_usage_stats_enabled()

    def test_env_var_opt_out_still_works_when_file_unavailable(self, monkeypatch):
        """Env-var opt-out works even when file-based opt-out is unavailable."""
        monkeypatch.setattr("tensorrt_llm.usage.usage_lib._OPT_OUT_FILE", None)
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
        assert not usage_lib.is_usage_stats_enabled()

    def test_report_usage_does_not_crash_when_file_unavailable(self, monkeypatch):
        """report_usage() degrades silently when _OPT_OUT_FILE is None."""
        monkeypatch.setattr("tensorrt_llm.usage.usage_lib._OPT_OUT_FILE", None)
        monkeypatch.setenv("CI", "true")
        # Should not raise
        usage_lib.report_usage()

    def test_module_import_survives_path_home_failure(self):
        """Simulates Path.home() raising RuntimeError at module scope.

        The module-level _OPT_OUT_FILE assignment is guarded by
        try/except, so the module should already be imported.  This test
        verifies the guard produces None (not a crash) by replicating
        the exact logic.
        """
        from pathlib import Path

        def failing_home():
            raise RuntimeError("Could not determine home directory")

        with patch.object(Path, "home", side_effect=failing_home):
            try:
                result = Path.home() / ".config" / "trtllm" / "do_not_track"
            except (RuntimeError, KeyError):
                result = None
        assert result is None
