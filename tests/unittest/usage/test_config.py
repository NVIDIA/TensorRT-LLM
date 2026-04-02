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
"""Tests for tensorrt_llm.usage.config -- canonical location for telemetry types."""

import pytest


class TestTelemetryConfigLocation:
    """Verify TelemetryConfig and UsageContext live in tensorrt_llm.usage.config."""

    def test_import_telemetry_config_from_usage_config(self):
        """TelemetryConfig must be importable from tensorrt_llm.usage.config."""
        from tensorrt_llm.usage import config

        assert hasattr(config, "TelemetryConfig")

    def test_import_usage_context_from_usage_config(self):
        """UsageContext must be importable from tensorrt_llm.usage.config."""
        from tensorrt_llm.usage import config

        assert hasattr(config, "UsageContext")

    def test_telemetry_config_defaults(self):
        """TelemetryConfig defaults: disabled=False, usage_context=UNKNOWN."""
        from tensorrt_llm.usage import config

        tc = config.TelemetryConfig()
        assert tc.disabled is False
        assert tc.usage_context == config.UsageContext.UNKNOWN

    def test_usage_context_values(self):
        """UsageContext enum has all expected members."""
        from tensorrt_llm.usage import config

        expected = {"UNKNOWN", "LLM_CLASS", "CLI_SERVE", "CLI_BENCH", "CLI_EVAL"}
        actual = {e.name for e in config.UsageContext}
        assert expected == actual

    def test_usage_context_string_values(self):
        """UsageContext members have correct string values."""
        from tensorrt_llm.usage import config

        assert config.UsageContext.UNKNOWN.value == "unknown"
        assert config.UsageContext.LLM_CLASS.value == "llm_class"
        assert config.UsageContext.CLI_SERVE.value == "cli_serve"
        assert config.UsageContext.CLI_BENCH.value == "cli_bench"
        assert config.UsageContext.CLI_EVAL.value == "cli_eval"

    def test_telemetry_config_disabled_flag(self):
        """TelemetryConfig(disabled=True) sets the flag."""
        from tensorrt_llm.usage import config

        tc = config.TelemetryConfig(disabled=True)
        assert tc.disabled is True

    def test_telemetry_config_with_context(self):
        """TelemetryConfig accepts usage_context parameter."""
        from tensorrt_llm.usage import config

        tc = config.TelemetryConfig(usage_context=config.UsageContext.CLI_SERVE)
        assert tc.usage_context == config.UsageContext.CLI_SERVE

    def test_telemetry_config_rejects_extra_fields(self):
        """TelemetryConfig(extra='forbid') raises ValidationError on unknown fields."""
        from pydantic import ValidationError

        from tensorrt_llm.usage import config

        with pytest.raises(ValidationError):
            config.TelemetryConfig(unknown_field=1)


class TestBackwardCompatibility:
    """Verify types are still importable from llm_args for backward compat."""

    def test_telemetry_config_importable_from_llm_args(self):
        """TelemetryConfig must still be importable from llm_args."""
        from tensorrt_llm.llmapi import llm_args

        assert hasattr(llm_args, "TelemetryConfig")

    def test_usage_context_importable_from_llm_args(self):
        """UsageContext must still be importable from llm_args."""
        from tensorrt_llm.llmapi import llm_args

        assert hasattr(llm_args, "UsageContext")

    def test_same_types_both_locations(self):
        """Types from both locations must be the same class."""
        from tensorrt_llm.llmapi import llm_args
        from tensorrt_llm.usage import config

        assert config.TelemetryConfig is llm_args.TelemetryConfig
        assert config.UsageContext is llm_args.UsageContext
